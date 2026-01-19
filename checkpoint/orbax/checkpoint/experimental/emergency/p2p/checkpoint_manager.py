# Copyright 2025 The Orbax Authors.
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

"""Composite Checkpoint Manager handling P2P syncing with optional Persistent Fallback."""

import threading
import time
from typing import Any, Iterable, Mapping, Optional, Sequence, Union, final

from absl import logging
from etils import epath
from etils import epy
import jax
from orbax.checkpoint import abstract_checkpoint_manager
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.multihost import multislice
from orbax.checkpoint.experimental.emergency import checkpoint_manager as emc
from orbax.checkpoint.experimental.emergency import path as emergency_path
from orbax.checkpoint.experimental.emergency.p2p import constants
from orbax.checkpoint.experimental.emergency.p2p import local
from orbax.checkpoint.experimental.emergency.p2p import peer_selector
from orbax.checkpoint.experimental.emergency.p2p import persistent
from orbax.checkpoint.experimental.emergency.p2p import protocol
from orbax.checkpoint.experimental.emergency.p2p import service
from orbax.checkpoint.experimental.emergency.p2p.args import Composite
from typing_extensions import override
from typing_extensions import Self

PyTree = Any


@final
class CheckpointManager(
    abstract_checkpoint_manager.AbstractCheckpointManager, epy.ContextManager
):
  """Orchestrates P2P local checkpointing with optional persistent storage failover.

  Restoration Strategy:
    1. Check Local Disk (Fastest)
    2. Query P2P Network (Fast, Intra-cluster)
    3. Fallback to Persistent Storage (Slow, GCS/S3)
  """

  def __init__(
      self,
      local_directory: epath.PathLike,
      global_mesh: jax.sharding.Mesh,
      abstract_state: PyTree | None,
      persistent_directory: epath.PathLike | None = None,
      *,
      options: emc.CheckpointManagerOptions | None = None,
  ):
    """Initializes the P2P Checkpoint Manager."""
    self._local_directory = epath.Path(local_directory)
    self._global_mesh = global_mesh
    self._process_index = multihost.process_index()
    self._abstract_state = abstract_state

    # 1. Parse and Validate Options
    self._emc_options = options or emc.CheckpointManagerOptions()
    self._validate_options(persistent_directory is not None)

    self._replica_axis_index = self._emc_options.replica_axis_index
    self._replica_id = multislice.process_replica_id(
        self._process_index,
        self._global_mesh,
        replica_axis_index=self._replica_axis_index,
    )

    # 2. Initialize Internal Managers
    self._local_manager = local.LocalCheckpointManager(
        self._local_directory,
        self._global_mesh,
        options=self._emc_options,
    )

    self._persistent_manager: persistent.PersistentCheckpointManager | None = (
        None
    )
    if persistent_directory:
      self._persistent_manager = persistent.PersistentCheckpointManager(
          directory=persistent_directory,
          global_mesh=global_mesh,
          replica_axis_index=self._replica_axis_index,
          options=self._emc_options,
      )

    # 3. Initialize P2P Networking & Logic
    self._start_p2p_subsystem()

    self._bkg_fetch_step: int = -1
    self._bkg_fetch_event = threading.Event()
    self._bkg_fetch_success: bool = False
    self._start_background_fetch()

  def _start_background_fetch(self):
    """Starts a background thread to fetch the latest step from peers."""
    latest_step = self._peer_selector.get_latest_complete_step()
    if (
        latest_step is not None
        and latest_step not in self._local_manager.all_steps()
    ):
      logging.info('Prefetching latest step %d in background.', latest_step)
      self._bkg_fetch_step = latest_step
      threading.Thread(
          target=self._run_background_fetch, args=(latest_step,)
      ).start()

  @property
  def directory(self) -> epath.Path:
    if self._persistent_manager:
      return self._persistent_manager.directory
    return self._local_manager.directory

  def _validate_options(self, has_persistent: bool):
    if not has_persistent:
      return

    l_interval = self._emc_options.local.save_interval_steps
    p_interval = self._emc_options.persistent.save_interval_steps

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
    if (
        peer_info.ip == self._p2p_node.ip
        and peer_info.port == self._p2p_node.port
    ):
      return True

    return self._p2p_node.fetch_shard_from_peer(
        peer_info.ip, peer_info.port, step, peer_info.process_index
    )

  # --- Abstract Manager Implementation ---

  @override
  def all_steps(self, read: bool = False) -> Sequence[int]:
    return self._local_manager.all_steps(read)

  @override
  def latest_step(self) -> int | None:
    return self._local_manager.latest_step()

  @override
  def best_step(self) -> int | None:
    return self._local_manager.best_step()

  @override
  def reload(self):
    self._local_manager.reload()

  @override
  def reached_preemption(self, step: int) -> bool:
    return multihost.reached_preemption(step)

  @override
  def should_save(self, step: int) -> bool:
    return self._local_manager.should_save(step)

  @override
  def delete(self, step: int):
    self._local_manager.delete(step)
    if self._persistent_manager:
      self._persistent_manager.delete(step)

  @override
  def save(self, step: int, args: Composite, *, force: bool = False) -> bool:
    p2p_saved = self._local_manager.save(step, args=args, force=force)

    persistent_saved = False
    if self._persistent_manager:
      persistent_saved = self._persistent_manager.save(
          step, args=args, force=force
      )

    return p2p_saved or persistent_saved

  @override
  def restore(
      self, step: int | None, args: Composite | None = None
  ) -> Union[Any, Mapping[str, Any], Composite, None]:

    if step is None:
      step = self._local_manager.latest_step()
      if step is None:
        assert self._peer_selector is not None
        step = self._peer_selector.get_latest_complete_step()

    if step is None:
      logging.warning('No restore step found in local storage or P2P registry.')
      return None

    logging.info('Targeting restore step: %d', step)
    start_time = time.time()

    # Strategy A: Local Restore
    if step in self._local_manager.all_steps():
      logging.info('Strategy A - Found locally. Restoring...')
      try:
        res = self._local_manager.restore(step)
        logging.info(
            'Local restore finished in %.2fs', time.time() - start_time
        )
        return res
      except (OSError, ValueError) as e:
        logging.exception('Local restore failed: %s', e)

    # Strategy B: P2P Network Restore
    logging.info('Strategy B - Not found locally. Attempting P2P fetch...')

    fetch_succeeded = False
    if step == self._bkg_fetch_step:
      logging.info(
          'Waiting for P2P background fetch of step %d to complete...', step
      )
      if not self._bkg_fetch_event.wait(timeout=600):
        logging.error('P2P background fetch for step %d timed out.', step)
      else:
        fetch_succeeded = self._bkg_fetch_success
    else:
      source_peer_exists = self._peer_selector.get_source_peer(
          step,
          self._process_index,
      )
      if source_peer_exists:
        fetch_succeeded = self._fetch_from_peers(step)
      else:
        logging.info(
            'Step %d for %d not found in peers, unable to fetch via P2P.',
            step,
            self._process_index,
        )

    if fetch_succeeded:
      p2p_restore_dir = self._local_directory / constants.P2P_RESTORE_DIR_NAME
      try:
        res = self._local_manager.restore(step, directory=p2p_restore_dir)
        logging.info(
            'P2P restore finished in %.2fs',
            time.time() - start_time,
        )
        return res
      except (OSError, ValueError) as e:
        logging.exception('P2P restore failed after download: %s', e)

    # Strategy C: Persistent Storage Fallback
    if self._persistent_manager:
      logging.warning(
          'Strategy C - P2P failed. Falling back to persistent storage.'
      )
      restore_args = args
      if not restore_args:
        restore_args = Composite(
            state=self._abstract_state,
        )

      return self._persistent_manager.restore(step, args=restore_args)

    logging.error('All restore strategies failed for step %d.', step)
    return None

  @override
  def item_metadata(self, step: int) -> Any:
    return self._local_manager.item_metadata(step)

  @override
  def metadata(self, step: Optional[int] = None) -> Any:
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
    if self._bkg_fetch_step != -1 and not self._bkg_fetch_event.is_set():
      if not self._bkg_fetch_event.wait(timeout=600):
        logging.error('P2P background fetch task timed out during close.')
    self._local_manager.close()
    if self._persistent_manager:
      self._persistent_manager.close()
    if self._p2p_node:
      self._p2p_node.stop()

  def __contextmanager__(self) -> Iterable[Self]:
    try:
      yield self
    finally:
      self.close()
