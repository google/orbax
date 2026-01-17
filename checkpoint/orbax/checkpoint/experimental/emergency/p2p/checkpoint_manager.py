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

"""Composite CheckpointManager for P2P coordination and persistent failover.


WARNING: This class is experimental; do not use without specific approval.
"""

import functools
import time
from typing import Any, Iterable, Optional, Sequence

from absl import logging
from etils import epath
from etils import epy
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from orbax.checkpoint import abstract_checkpoint_manager
from orbax.checkpoint import args as args_lib
from orbax.checkpoint import checkpoint_manager
from orbax.checkpoint import checkpoint_utils
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.multihost import multislice
from orbax.checkpoint._src.serialization import type_handler_registry
from orbax.checkpoint._src.serialization import type_handlers
from orbax.checkpoint.experimental.emergency import checkpoint_manager as emc
from p2p import p2p_manager
from typing_extensions import override
from typing_extensions import Self

PyTree = checkpoint_manager.PyTree
RootMetadata = checkpoint_manager.RootMetadata
StepMetadata = checkpoint_manager.StepMetadata

_PRIMARY_REPLICA_ID = 0


def _create_persistent_handler(
    multiprocessing_options: checkpoint_manager.MultiprocessingOptions,
) -> ocp.PyTreeCheckpointHandler:
  """Factory for creating a persistent checkpoint handler."""
  registry = type_handler_registry.create_type_handler_registry(
      (
          jax.Array,
          type_handlers.ArrayHandler(
              primary_host=multiprocessing_options.primary_host,
              replica_id=0,
              use_replica_parallel=False,
          ),
      ),
  )
  return ocp.PyTreeCheckpointHandler(
      use_ocdbt=True,
      use_zarr3=True,
      multiprocessing_options=multiprocessing_options,
      type_handler_registry=registry,
  )


class CheckpointManager(
    abstract_checkpoint_manager.AbstractCheckpointManager, epy.ContextManager
):
  """Orchestrates P2P local checkpointing with persistent storage failover.

  ### Design & Assumptions

  1.  **Local Superset Assumption:** We assume that the P2P (local) storage is
      faster, cheaper, and strictly a "superset" of the persistent storage
      regarding the *presence* of steps. Any step saved to persistent storage
      is assumed to also be present in local P2P storage.

  2.  **Metadata Delegation:** Based on (1), methods that query checkpoint
      existence (`all_steps`, `latest_step`, `best_step`) delegate **only** to
      the `_p2p_manager`. This avoids the high latency of polling remote file
      systems (like GCS/S3) during the training loop.

  3.  **Failover Restoration:** While metadata is queried locally, `restore()`
      is robust. It attempts to load from P2P storage first, but if that fails
      (e.g., due to preemption wiping local disks), it falls back to the
      slower `_restore_from_persistent` path.
  """

  def __init__(
      self,
      local_directory: epath.PathLike,
      persistent_directory: epath.PathLike,
      global_mesh: jax.sharding.Mesh,
      abstract_state: PyTree,
      *,
      options: Optional[emc.CheckpointManagerOptions] = None,
  ):
    options = options or emc.CheckpointManagerOptions()
    self._global_mesh = global_mesh
    self._abstract_state = abstract_state
    self._replica_axis_index = options.replica_axis_index
    self._persistent_directory = epath.Path(persistent_directory)

    # Topology Info
    self._replica_id = multislice.process_replica_id(
        multihost.process_index(),
        self._global_mesh,
        replica_axis_index=self._replica_axis_index,
    )
    self._in_primary_slice = multislice.in_replica(
        multihost.process_index(),
        global_mesh,
        replica_axis_index=self._replica_axis_index,
        replica_id=_PRIMARY_REPLICA_ID,
    )

    # 1. Initialize Persistent Manager (Primary Slice Only)
    self._persistent_checkpoint_manager = None
    self._persistent_options = None

    if self._in_primary_slice:
      self._init_persistent_manager(options)

    # 2. Initialize Internal P2P Manager (All Slices)
    self._p2p_manager = p2p_manager._P2PCheckpointManager(
        local_directory,
        global_mesh,
        self._replica_id,
        options=options,
    )

  def _init_persistent_manager(self, options: emc.CheckpointManagerOptions):
    """Initializes the persistent manager on the primary slice."""
    replica_devices = multislice.replica_devices(
        self._global_mesh,
        replica_axis_index=self._replica_axis_index,
        replica_id=_PRIMARY_REPLICA_ID,
    )
    primary_host = multislice.primary_process_in_replica(
        self._global_mesh,
        replica_axis_index=self._replica_axis_index,
        replica_id=_PRIMARY_REPLICA_ID,
    )
    active_processes = multihost.unique_processes_from_devices(replica_devices)

    logging.info(
        'Initializing Persistent Manager. Primary Host: %s, Active Processes:'
        ' %s',
        primary_host,
        active_processes,
    )

    mp_options = checkpoint_manager.MultiprocessingOptions(
        primary_host=primary_host,
        active_processes=active_processes,
        barrier_sync_key_prefix='persistent',
    )
    self._persistent_options = checkpoint_manager.CheckpointManagerOptions(
        create=False,
        multiprocessing_options=mp_options,
        step_name_format=options.step_name_format,
    )
    self._persistent_checkpoint_manager = (
        ocp.checkpoint_manager.CheckpointManager(
            self._persistent_directory,
            options=self._persistent_options,
            item_handlers=_create_persistent_handler(mp_options),
        )
    )

  @property
  def directory(self) -> epath.Path:
    if self._persistent_checkpoint_manager:
      return self._persistent_checkpoint_manager.directory
    return self._p2p_manager.directory

  @override
  def all_steps(self, read: bool = False) -> Sequence[int]:
    # Delegated solely to P2P manager.
    return self._p2p_manager.all_steps(read)

  @override
  def latest_step(self) -> Optional[int]:
    # Delegated solely to P2P manager.
    return self._p2p_manager.latest_step()

  @override
  def best_step(self) -> Optional[int]:
    # Delegated solely to P2P manager.
    return self._p2p_manager.best_step()

  @override
  def reload(self):
    if self._persistent_checkpoint_manager:
      self._persistent_checkpoint_manager.reload()
    self._p2p_manager.reload()

  @override
  def reached_preemption(self, step: int) -> bool:
    return multihost.reached_preemption(step)

  @override
  def should_save(self, step: int) -> bool:
    # We rely on the P2P manager's logic for saving frequency.
    return self._p2p_manager.should_save(step)

  @override
  def delete(self, step: int):
    # Deletion must happen on both to maintain consistency.
    if self._persistent_checkpoint_manager:
      self._persistent_checkpoint_manager.delete(step)
    self._p2p_manager.delete(step)

  @override
  def save(
      self, step: int, args: args_lib.Composite, *, force: bool = False
  ) -> bool:
    # Attempt save on P2P (always) and Persistent (if available/configured).
    # The P2P manager is the primary indicator of success.
    p2p_saved = self._p2p_manager.save(step, args=args.state, force=force)

    persistent_saved = False
    if self._persistent_checkpoint_manager:
      persistent_saved = self._persistent_checkpoint_manager.save(
          step, args=args.state, force=force
      )

    return p2p_saved or persistent_saved

  @override
  def restore(
      self,
      step: Optional[int],
      args: args_lib.Composite | None = None,
  ) -> Any:
    """Restores state, preferring P2P/local storage with failover to persistent."""
    try:
      # Attempt P2P restore first
      ret = self._p2p_manager.restore(step, args)
      if ret:
        logging.info('Restored from P2P storage (step=%s).', step)
        return ret
    # pylint: disable=broad-exception-caught
    except Exception as e:
      # pylint: enable=broad-exception-caught
      logging.warning(
          'Failed to restore from P2P storage: %s. Attempting persistent'
          ' restore.',
          e,
      )

    return self._restore_from_persistent(step)

  @override
  def item_metadata(self, step: int) -> Any:
    return None

  @override
  def metadata(self, step: int | None = None) -> Any:
    return None

  @override
  def metrics(self, step: int) -> Optional[PyTree]:
    return None

  @override
  def wait_until_finished(self):
    self._p2p_manager.wait_until_finished()
    if self._persistent_checkpoint_manager:
      self._persistent_checkpoint_manager.wait_until_finished()

  @override
  def check_for_errors(self):
    if self._persistent_checkpoint_manager:
      self._persistent_checkpoint_manager.check_for_errors()
    self._p2p_manager.check_for_errors()

  @override
  def close(self):
    if self._persistent_checkpoint_manager:
      self._persistent_checkpoint_manager.close()
    self._p2p_manager.close()

  def __contextmanager__(self) -> Iterable[Self]:
    try:
      yield self
    finally:
      self.close()

  def _get_single_slice_shardings(self) -> PyTree:
    """Calculates sharding specs for a single slice (slice 0)."""
    target_slice_id = 0
    slice_devices = multislice.replica_devices(
        self._global_mesh,
        replica_id=target_slice_id,
        replica_axis_index=self._replica_axis_index,
    )
    single_slice_mesh_shape = [
        1 if i == self._replica_axis_index else d
        for i, d in enumerate(self._global_mesh.devices.shape)
    ]
    slice_mesh = jax.sharding.Mesh(
        slice_devices.reshape(single_slice_mesh_shape),
        self._global_mesh.axis_names,
    )

    def _get_sharding(arr):
      return jax.sharding.NamedSharding(slice_mesh, arr.sharding.spec)

    return jax.tree.map(_get_sharding, self._abstract_state)

  def _load_from_primary(
      self, step: int, directory: epath.Path, single_slice_shardings: PyTree
  ) -> PyTree:
    """Loads checkpoint from persistent storage on the primary slice."""
    step_dir = self._persistent_options.step_name_format.find_step(
        directory, step
    ).path

    primary_slice_devices = multislice.replica_devices(
        self._global_mesh,
        replica_axis_index=self._replica_axis_index,
        replica_id=_PRIMARY_REPLICA_ID,
    )
    mp_options = checkpoint_manager.MultiprocessingOptions(
        primary_host=multislice.primary_process_in_replica(
            self._global_mesh,
            replica_axis_index=self._replica_axis_index,
            replica_id=_PRIMARY_REPLICA_ID,
        ),
        active_processes=multihost.unique_processes_from_devices(
            primary_slice_devices
        ),
        barrier_sync_key_prefix='persistent_primary_slice_restore',
    )

    restore_args_obj = args_lib.PyTreeRestore(
        item=self._abstract_state,
        restore_args=checkpoint_utils.construct_restore_args(
            self._abstract_state, single_slice_shardings
        ),
    )

    handler = _create_persistent_handler(mp_options)
    try:
      return handler.restore(step_dir / 'default', args=restore_args_obj)
    except FileNotFoundError as e:
      raise FileNotFoundError(
          f'No checkpoint found for step {step} in persistent storage.'
      ) from e

  def _init_zeros(
      self, shape_dtypes: Sequence[Any], single_slice_shardings: PyTree
  ) -> PyTree:
    """Initializes zero-filled arrays on non-primary slices."""
    # Flatten shardings for jitted function output spec
    shardings_tuple = tuple(jax.tree.flatten(single_slice_shardings)[0])

    @functools.partial(jax.jit, static_argnums=0, out_shardings=shardings_tuple)
    def create_zeros(shape_dtype_tup):
      return jax.tree.map(
          lambda sd: jnp.zeros(sd.shape, dtype=sd.dtype), shape_dtype_tup
      )

    return create_zeros(tuple(shape_dtypes))

  def _restore_from_persistent(
      self, step: int, directory: Optional[epath.PathLike] = None
  ) -> Any:
    logging.info(
        'p2p.CheckpointManager: Restoring step=%s from persistent storage.',
        step,
    )
    restore_path = epath.Path(directory or self._persistent_directory)

    # 1. Prepare Metadata
    shape_dtypes, tree_defs = jax.tree.flatten(self._abstract_state)
    single_slice_shardings = self._get_single_slice_shardings()

    # 2. Load or Initialize
    if self._in_primary_slice:
      logging.info('Primary Slice: Loading from disk...')
      restored_pytree = self._load_from_primary(
          step, restore_path, single_slice_shardings
      )
      in_tree = tuple(jax.tree.flatten(restored_pytree)[0])
    else:
      logging.info('Non-Primary Slice: Initializing zeros...')
      zeros_pytree = self._init_zeros(shape_dtypes, single_slice_shardings)
      in_tree = tuple(jax.tree.flatten(zeros_pytree)[0])

    # 3. Sync and Broadcast
    multihost.sync_global_processes('persistent_restore_pre_broadcast')
    start_broadcast = time.time()

    shared_states, _ = multislice.broadcast_one_replica_to_all(
        in_tree,
        self._global_mesh,
        replica_axis_index=self._replica_axis_index,
        is_source=self._in_primary_slice,
    )

    logging.info(
        'Broadcast complete in %.2fs (slice_id=%s)',
        time.time() - start_broadcast,
        self._replica_id,
    )

    return jax.tree.unflatten(tree_defs, shared_states)
