# Copyright 2026 The Orbax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Composite Checkpoint Manager with P2P syncing and Persistent Fallback."""

import shutil
import threading
import time
from typing import Any, Iterable, Sequence, final

from absl import logging
from etils import epath
from etils import epy
import jax
from orbax.checkpoint import abstract_checkpoint_manager
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.multihost import multislice
from orbax.checkpoint.experimental.emergency import path as emergency_path
from orbax.checkpoint.experimental.emergency.p2p import args as p2p_args_lib
from orbax.checkpoint.experimental.emergency.p2p import constants
from orbax.checkpoint.experimental.emergency.p2p import local
from orbax.checkpoint.experimental.emergency.p2p import options as options_lib
from orbax.checkpoint.experimental.emergency.p2p import peer_selector
from orbax.checkpoint.experimental.emergency.p2p import persistent
from orbax.checkpoint.experimental.emergency.p2p import protocol
from orbax.checkpoint.experimental.emergency.p2p import service
from typing_extensions import override
from typing_extensions import Self

PyTree = Any


class _P2PSubsystem:
  """Manages P2P checkpoint synchronization and discovery."""

  def __init__(
      self,
      local_directory: epath.Path,
      global_mesh: jax.sharding.Mesh,
      replica_axis_index: int | None,
      local_manager: local.LocalCheckpointManager,
      process_index: int,
  ):
    self._local_directory = local_directory
    self._global_mesh = global_mesh
    self._replica_axis_index = replica_axis_index
    self._local_manager = local_manager
    self._process_index = process_index
    self._registry_stale = True

    self._p2p_node: service.P2PNode | None = None
    self._peer_selector: peer_selector.PeerSelector | None = None

    self._bkg_fetch_step: int = -1
    self._bkg_fetch_event = threading.Event()
    self._bkg_fetch_success: bool = False

    self._start_p2p_subsystem()
    self._start_background_fetch()

  def _start_p2p_subsystem(self):
    """Initializes the P2P node and starts background discovery."""
    # 1. Synchronous Setup: Bind ports immediately so self._p2p_node is valid.
    logging.info('Initializing P2P sidecar')

    self._p2p_node = service.P2PNode(
        directory=self._local_directory,
    )
    self._p2p_node.start()
    logging.info(
        'P2P sidecar successfully bound to %s:%d',
        self._p2p_node.ip,
        self._p2p_node.port,
    )

    # 2. Synchronous Setup: Block on global handshake.
    self._sync_registry()

  def _sync_registry(self):
    """Broadcasts local holdings and updates the peer selector."""
    stored_idx, my_steps = self._local_manager.scan_stored_steps()

    if my_steps:
      logging.info(
          'P2P Discovery: Process %d holding steps %s previously stored by %s',
          self._process_index,
          my_steps,
          stored_idx,
      )
    assert self._p2p_node is not None
    my_info = protocol.PeerDiscoveryInfo(
        ip=self._p2p_node.ip,
        port=self._p2p_node.port,
        process_index=stored_idx,
        steps=list(my_steps),
    )

    all_infos_dicts = emergency_path.sync_global_data(my_info.to_dict())

    # PeerSelector handles strict typing and validation
    self._peer_selector = peer_selector.PeerSelector(
        global_mesh=self._global_mesh,
        replica_axis_index=self._replica_axis_index,
        raw_metadata_list=all_infos_dicts,
    )
    self._registry_stale = False

  def mark_registry_stale(self):
    self._registry_stale = True

  def sync_registry_if_stale(self):
    if self._registry_stale:
      logging.info('P2P registry is stale, re-syncing.')
      self._sync_registry()

  def _start_background_fetch(self):
    """Starts a background thread to fetch the latest step from peers."""
    latest_step = self.get_latest_complete_step()
    if (
        latest_step is not None
        and latest_step not in self._local_manager.all_steps()
    ):
      logging.info('Prefetching latest step %d in background.', latest_step)
      self._bkg_fetch_step = latest_step
      threading.Thread(
          target=self._run_background_fetch, args=(latest_step,)
      ).start()

  def _run_background_fetch(self, step: int):
    """Worker for background fetch of latest step."""
    try:
      self._bkg_fetch_success = self._fetch_from_peers(step)
    except (OSError, RuntimeError):
      logging.exception('P2P background fetch task for step %d failed.', step)
      self._bkg_fetch_success = False
    finally:
      self._bkg_fetch_event.set()

  def _fetch_from_peers(self, step: int) -> bool:
    """Finds the optimal peer and downloads the shard."""
    assert self._peer_selector is not None
    peer_info = self._peer_selector.get_source_peer(step, self._process_index)
    if not peer_info:
      logging.warning(
          'Step %d found in P2P registry, but no source peer found for my'
          ' shard (%d).',
          step,
          self._process_index,
      )
      return False
    assert self._p2p_node is not None

    # TODO(exlin): optimization to process when peer is localhost

    return self._p2p_node.fetch_shard_from_peer(
        peer_info.ip, peer_info.port, step, peer_info.process_index
    )

  def get_latest_complete_step(self) -> int | None:
    """Returns the latest step that is complete in the P2P network."""
    assert self._peer_selector is not None
    return self._peer_selector.get_latest_complete_step()

  def get_all_steps_from_peers(self) -> list[int]:
    """Returns all steps available in any peer in P2P network."""
    assert self._peer_selector is not None
    return self._peer_selector.get_all_steps()

  def has_shard_for_step(self, step: int) -> bool:
    """Checks if this process's shard for a step exists in P2P."""
    assert self._peer_selector is not None
    return (
        self._peer_selector.get_source_peer(step, self._process_index)
        is not None
    )

  def fetch(self, step: int) -> bool:
    """Fetches from peers, or waits for background fetch if it's in progress."""
    if step == self._bkg_fetch_step:
      logging.info(
          'Waiting for P2P background fetch of step %d to complete...', step
      )
      if not self._bkg_fetch_event.wait(timeout=600):
        logging.error('P2P background fetch for step %d timed out.', step)
        return False
      else:
        return self._bkg_fetch_success
    else:
      return self._fetch_from_peers(step)

  def close(self):
    """Closes the P2P subsystem."""
    if self._bkg_fetch_step != -1 and not self._bkg_fetch_event.is_set():
      if not self._bkg_fetch_event.wait(timeout=600):
        logging.error('P2P background fetch task timed out during close.')
    if self._p2p_node:
      self._p2p_node.stop()


@final
class CheckpointManager(
    abstract_checkpoint_manager.AbstractCheckpointManager, epy.ContextManager
):
  """P2P local checkpointing with persistent storage failover.

  Restoration Strategy:
    1. Check Local Disk (Fastest)
    2. Query P2P Network (Fast, Intra-cluster)
    3. Fallback to Persistent Storage (Slow, GCS/S3)
  """

  def __init__(
      self,
      global_mesh: jax.sharding.Mesh,
      abstract_state: PyTree | None,
      local_directory: epath.PathLike,
      persistent_directory: epath.PathLike | None = None,
      *,
      options: options_lib.CheckpointManagerOptions | None = None,
  ):
    """Initializes the P2P Checkpoint Manager."""
    self._local_directory = epath.Path(local_directory)
    self._global_mesh = global_mesh
    self._process_index = multihost.process_index()
    self._abstract_state = abstract_state

    # 1. Parse and Validate Options
    self._options = options or options_lib.CheckpointManagerOptions()
    self._validate_options(persistent_directory is not None)

    self._replica_axis_index = self._options.replica_axis_index
    self._replica_id = multislice.process_replica_id(
        self._process_index,
        self._global_mesh,
        replica_axis_index=self._replica_axis_index,
    )

    # 2. Initialize Internal Managers
    self._local_manager = local.LocalCheckpointManager(
        self._local_directory,
        self._global_mesh,
        options=self._options,
    )

    self._persistent_manager: persistent.PersistentCheckpointManager | None = (
        None
    )
    if persistent_directory:
      self._persistent_manager = persistent.PersistentCheckpointManager(
          directory=persistent_directory,
          global_mesh=global_mesh,
          replica_axis_index=self._replica_axis_index,
          options=self._options,
      )

    # 3. Initialize P2P Networking & Logic
    self._p2p = _P2PSubsystem(
        local_directory=self._local_directory,
        global_mesh=self._global_mesh,
        replica_axis_index=self._replica_axis_index,
        local_manager=self._local_manager,
        process_index=self._process_index,
    )

  @property
  def directory(self) -> epath.Path:
    if self._persistent_manager:
      return self._persistent_manager.directory
    return self._local_manager.directory

  def _validate_options(self, has_persistent: bool):
    if not has_persistent:
      return

    l_interval = self._options.local.save_interval_steps
    p_interval = self._options.persistent.save_interval_steps

    if l_interval > p_interval:
      raise ValueError(
          f'Local save interval ({l_interval}) must be more frequent'
          f' than persistent interval ({p_interval}).'
      )
    if p_interval % l_interval != 0:
      raise ValueError(
          f'Persistent interval ({p_interval}) must be a multiple of'
          f' local interval ({l_interval}).'
      )

  # --- Abstract Manager Implementation ---

  @override
  def all_steps(self, read: bool = False) -> Sequence[int]:
    self._p2p.sync_registry_if_stale()
    all_steps = set(self._local_manager.all_steps(read))
    all_steps.update(self._p2p.get_all_steps_from_peers())
    if self._persistent_manager:
      all_steps.update(self._persistent_manager.all_steps(read))
    return sorted(list(all_steps))

  @override
  def latest_step(self) -> int | None:
    self._p2p.sync_registry_if_stale()

    # We intentionally use the step returned by P2P regardless of whether a
    # newer step is available in persistent storage. This is because we assume
    # P2P is more efficient overall for catching up to the latest step, even
    # if persistent storage has a newer step available.
    # TODO(exlin): Revisit if P2P should always be prioritized over
    # persistent storage for latest_step.
    step = self._p2p.get_latest_complete_step()
    logging.info('P2P latest_step=%s', step)

    if step is None and self._persistent_manager:
      step = self._persistent_manager.latest_step()
      logging.info('Persistent latest_step=%s', step)

    return step

  @override
  def best_step(self) -> int | None:
    return self._local_manager.best_step()

  @override
  def reload(self):
    """Reloads the checkpoint manager and its components.

    This method refreshes the local and persistent managers and marks the P2P
    registry as stale, forcing a re-sync on the next access.
    """
    self._p2p.mark_registry_stale()
    self._local_manager.reload()
    if self._persistent_manager:
      self._persistent_manager.reload()

  @override
  def reached_preemption(self, step: int) -> bool:
    return multihost.reached_preemption(step)

  @override
  def should_save(self, step: int) -> bool:
    return self._local_manager.should_save(step)

  @override
  def delete(self, step: int):
    # mark it stale regardless of result to simplify logics
    self._p2p.mark_registry_stale()
    self._local_manager.delete(step)
    if self._persistent_manager:
      self._persistent_manager.delete(step)

  @override
  def save(
      self,
      step: int,
      args: p2p_args_lib.Composite,
      *,
      force: bool = False,
  ) -> bool:
    # mark it stale regardless of result to simplify logics
    self._p2p.mark_registry_stale()
    p2p_saved = self._local_manager.save(step, args=args, force=force)

    persistent_saved = False
    if self._persistent_manager:
      persistent_saved = self._persistent_manager.save(
          step, args=args, force=force
      )

    return p2p_saved or persistent_saved

  def _restore_from_persistent_storage(
      self, step: int, args: p2p_args_lib.Composite
  ) -> Any:
    """Restores from persistent storage."""
    assert self._persistent_manager is not None
    logging.info('Restoring step %d from persistent storage.', step)
    return self._persistent_manager.restore(step, args=args)

  def _restore_from_local_or_p2p(
      self, step: int, args: p2p_args_lib.Composite
  ) -> Any:
    """Restores from local storage or P2P network."""
    logging.info('Attempting to restore step %d from local or P2P.', step)
    if step in self._local_manager.all_steps():
      logging.info('Step %d found in local storage.', step)
      return self._local_manager.restore(step, args=args)
    else:
      logging.info('Step %d not found locally, fetching from P2P.', step)
      p2p_restore_dir = self._local_directory / constants.P2P_RESTORE_DIR_NAME
      try:
        if self._p2p.fetch(step):
          return self._local_manager.restore(
              step, args=args, directory=p2p_restore_dir
          )
        else:
          raise FileNotFoundError(
              f'Failed to fetch step {step} from P2P network.'
          )
      finally:
        if p2p_restore_dir.exists():
          logging.info(
              'Removing P2P restore directory: %s after restoration is'
              ' complete',
              p2p_restore_dir,
          )
          try:
            shutil.rmtree(str(p2p_restore_dir))
          except OSError as e:
            logging.exception('Failed to remove P2P restore directory: %s', e)

  @override
  def restore(
      self, step: int | None, args: p2p_args_lib.Composite | None
  ) -> p2p_args_lib.Composite | None:
    if args is None:
      raise ValueError('The `args` parameter is required for restore.')

    start_time = time.time()
    # 1. Registry Sync: Ensure P2P registry is current.
    logging.info('Registry Sync - ensuring P2P registry is current.')
    self._p2p.sync_registry_if_stale()

    use_persistent = False
    if step is None:
      step = self._p2p.get_latest_complete_step()
      logging.info('P2P latest_step=%s', step)

    if step is None and self._persistent_manager:
      step = self._persistent_manager.latest_step()
      logging.info('Persistent latest_step=%s', step)
      if step is not None:
        use_persistent = True

    if step is None:
      raise FileNotFoundError(
          'No steps found in either local/persistent storage or P2P registry.'
      )

    logging.info('Targeting restore step: %d', step)

    # 2. Try P2P/Local Restore
    restored = None
    restore_source = 'Unknown'
    if not use_persistent:
      if self._p2p.has_shard_for_step(step):
        try:
          restored = self._restore_from_local_or_p2p(step, args)
          restore_source = 'P2P'
        except (OSError, ValueError, FileNotFoundError) as e:
          logging.exception('Local/P2P restore for step %d failed: %s', step, e)
      else:
        logging.warning(
            'Step %d not available in P2P network, falling back to'
            ' persistent storage.',
            step,
        )

    # 3. Coordinated Fallback to Persistent Storage
    if self._persistent_manager:
      # If any host failed local/P2P restore, all hosts must use persistent.
      fallback_to_persistent = 1 if restored is None else 0
      any_host_needs_fallback_list = multihost.global_max(
          [fallback_to_persistent]
      )
      if any_host_needs_fallback_list and any_host_needs_fallback_list[0]:
        logging.warning(
            'At least one host failed Local/P2P restore or step not available'
            ' in P2P. All hosts falling back to persistent storage.'
        )
        restored = self._restore_from_persistent_storage(step, args)
        restore_source = 'Persistent Storage'

    if restored is not None:
      logging.info(
          'Restoration finished using %s in %.2fs',
          restore_source,
          time.time() - start_time,
      )
      return restored

    raise FileNotFoundError(f'All restore strategies failed for step {step}.')

  @override
  def item_metadata(self, step: int) -> Any:
    return self._local_manager.item_metadata(step)

  @override
  def metadata(self, step: int | None = None) -> Any:
    return self._local_manager.metadata(step)

  @override
  def metrics(self, step: int) -> PyTree | None:
    return None

  @override
  def wait_until_finished(self):
    self._local_manager.wait_until_finished()
    if self._persistent_manager:
      self._persistent_manager.wait_until_finished()

  @override
  def check_for_errors(self):
    self._local_manager.check_for_errors()
    if self._persistent_manager:
      self._persistent_manager.check_for_errors()

  @override
  def close(self):
    self._p2p.close()
    self._local_manager.close()
    if self._persistent_manager:
      self._persistent_manager.close()

  def __contextmanager__(self) -> Iterable[Self]:
    try:
      yield self
    finally:
      self.close()
