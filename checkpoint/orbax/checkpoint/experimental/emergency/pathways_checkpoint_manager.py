# Copyright 2024 The Orbax Authors.
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

"""A class providing emergency checkpoint management.


This class is experimental; do not use without specific approval.

NOTE: All classes within this module should be called across all *relevant*
processes. CheckpointManager is designed to be created and called across
*every* process. LocalCheckpointManager is designed to be created and called
across every process within *non-primary* slices. Similarly, a CheckpointManager
intended to work only with the persistent checkpoint on the primary slice should
always be called across all processes within the primary slice.
"""

import dataclasses
import time
from typing import Any, Iterable, Optional, Sequence

from absl import logging
from etils import epath
from etils import epy
import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint import abstract_checkpoint_manager
from orbax.checkpoint import args as args_lib
from orbax.checkpoint import checkpoint_manager
from orbax.checkpoint import multihost
from orbax.checkpoint import SaveArgs
from orbax.checkpoint import type_handlers
from orbax.checkpoint import utils
from orbax.checkpoint._src.handlers import pytree_checkpoint_handler
from orbax.checkpoint.experimental.emergency import checkpoint_manager as emergency_checkpoint_manager
from orbax.checkpoint.logging import abstract_logger
from orbax.checkpoint.logging import standard_logger
from orbax.checkpoint.logging import step_statistics
from orbax.checkpoint.multihost import multislice
from typing_extensions import Self  # for Python version < 3.11


PyTree = checkpoint_manager.PyTree
CheckpointHandler = checkpoint_manager.CheckpointHandler
P = jax.sharding.PartitionSpec
PyTreeCheckpointHandler = pytree_checkpoint_handler.PyTreeCheckpointHandler
unique_barrier_key = multihost.utils._unique_barrier_key  # pylint: disable=protected-access

_PROCESS_METADATA_FOLDER = 'process_metadata'
_PROCESS_METADATA_FILE_NAME = 'process_metadata.json'
_GLOBAL_PROCESS_METADATA_FILE_NAME = 'global_process_metadata.json'
_MESH_METADATA_FILE_NAME = 'mesh_metadata.json'

_PRIMARY_REPLICA_ID = 0
_SECONDARY_REPLICA_ID = 1


class _PathwaysLocalCheckpointManager(checkpoint_manager.CheckpointManager):
  """A checkpoint manager that checkpoints to local storage.

  Attributes:
    device_array: an ndarray representing all the devices running
      LocalCheckpointManager in the same global jax Mesh, importantly the first
      axis of the device_array is assumed to be the direction of device slices
      across which the Data Parallelism is happening.
  """

  # TODO: b/330585086 - Allow configuration of global mesh describing slices.
  # Validate against global meshes used for arrays in state.
  def __init__(
      self,
      directory: epath.PathLike,
      # TODO: b/330585086 - Support arbitrary items beyond state. We will have
      # to evaluate whether arbitrary items can be a good fit for local
      # checkpointing, given restore+broadcast requirements.
      state_handler: CheckpointHandler,
      global_mesh: jax.sharding.Mesh,
      *,
      options: Optional[
          emergency_checkpoint_manager.CheckpointManagerOptions
      ] = None,
      metadata: Optional[dict[str, Any]] = None,
      logger: Optional[abstract_logger.AbstractLogger] = None,
  ):
    # TODO: b/330585086 - Fully support options.
    options = options or emergency_checkpoint_manager.CheckpointManagerOptions()
    self._global_mesh = global_mesh
    self._replica_axis_index = options.replica_axis_index

    self._active_processes = set([jax.process_index(backend='proxy')])
    local_options = checkpoint_manager.CheckpointManagerOptions(
        save_interval_steps=options.local.save_interval_steps,
        max_to_keep=options.local.max_to_keep,
        step_name_format=options.step_name_format,
        create=False,
        cleanup_tmp_directories=options.cleanup_tmp_directories,
        async_options=options.async_options,
        multiprocessing_options=checkpoint_manager.MultiprocessingOptions(
            primary_host=None,
            active_processes=self._active_processes,
            barrier_sync_key_prefix='local',
        ),
        enable_async_checkpointing=options.enable_async_checkpointing,
        read_only=options.local.read_only,
        single_host_load_and_broadcast=False,
    )
    self._logger = logger or standard_logger.StandardLogger()
    self._coordination_timeout_secs = (
        options.multiprocessing_options
        or emergency_checkpoint_manager.MultiprocessingOptions()
    ).coordination_timeout_secs
    super().__init__(
        directory,
        options=local_options,
        metadata=metadata,
        item_handlers=state_handler,
        logger=self._logger,
    )
    self._max_to_keep = options.local.max_to_keep
    self._local_options = options.local
    self._steps = None

  def local_host_steps(self, read: bool) -> Sequence[int]:
    """Returns steps known to local host."""
    # List of steps present in individual host storage.

    # TO DO: Make a proper function to read all steps from local storage.
    local_steps = list(super().all_steps(read))

    # local_steps = [6]
    logging.info(
        'Found steps: %s in local host storage: %s.',
        local_steps,
        self.directory,
    )

    if len(local_steps) > self._max_to_keep:
      raise AssertionError(
          f' local_step on host {multihost.process_index()} exceeded'
          f' `max_to_keep` {self._max_to_keep}'
      )

    return emergency_checkpoint_manager.pad_steps(
        local_steps, self._max_to_keep
    )

  def all_steps(self, read: bool = False) -> Sequence[int]:
    """Returns all steps tracked by the manager.

    Includes steps located in local as well as persistent storage.

    Args:
      read: If True, forces a read directly from the storage location.
        Otherwise, a cached result can be returned.

    Returns:
      A sequence of steps (integers)
    """
    if self._steps is None:
      local_steps = set(self.local_host_steps(read))
      return local_steps
    return self._steps

  def latest_step(self) -> Optional[int]:
    """Returns the latest step saved in the local storage.

    Returns None if no steps have been saved.

    Returns:
      A step (int) or None if no steps are present.
    """
    if self._steps is None:
      self._steps = list(self.all_steps())

    logging.info('****** self._steps from local ckpt manager: %s', self._steps)

    return max(self._steps) if self._steps else None

  def save(
      self,
      step: int,
      args: Optional[args_lib.CheckpointArgs] = None,
      metrics: Optional[PyTree] = None,
      force: Optional[bool] = False,
  ) -> bool:
    """Saves the checkpoint at the given step."""
    write_test_metadata = True
    if write_test_metadata:
      logging.info('****** writing test metadata')
      hello_world_test_metadata = 'Hello World'
      metadata_item = jnp.array(
          list(hello_world_test_metadata.encode('utf-8')), dtype=jnp.uint8
      )
      metadata_save_args = jax.tree.map(
          lambda _: SaveArgs(chunk_byte_size=2 * 1024**3), metadata_item
      )
      metadata_params = {'test': 'txt'}
      logging.info('****** Args is: %s', args)

      devices = jax.devices()
      logging.info('****** jax devices is: %s', devices)
      logging.info('****** metadata_item is on: %s', metadata_item.devices())
      logging.info('****** putting array on device: %s', devices)
      logging.info(
          '****** metadata_item is NOW on: %s', metadata_item.devices()
      )

      jax.device_put_replicated(metadata_item, devices)

      logging.info('****** Args is: %s', args)
      logging.info('****** Dir of Args is: %s', dir(args))
      logging.info('****** Dir of Args.item is: %s', dir(args.item))
      logging.info('****** type of Args.item is: %s', type(args.item))
      logging.info('****** Args.save_args is: %s', args.save_args)
      logging.info('****** step is: %s', step)

      logging.info('****** metadata_item is: %s', metadata_item)
      logging.info('****** type of metadata_item is: %s', type(metadata_item))
      logging.info('****** metadata_save_args is: %s', metadata_save_args)
      logging.info('****** metadata_params is: %s', metadata_params)

      args.item = metadata_item
      args.save_args = metadata_save_args
      args.params = metadata_params
    else:
      logging.info('****** not writing test metadata')

    saved = super().save(step, args=args, metrics=metrics, force=force)
    logging.info('****** result of saved: %s', saved)
    if saved:
      # the assumption is that super.save() calls latest_step() and the steps
      # cache is updated
      if self._steps is None:
        logging.info('the steps cache should not be empty after save()')
        self._steps = list(self.all_steps())
      self._steps.append(step)
      self._steps = self._steps[-self._max_to_keep :]

    return saved

  def reload(self):
    """Reloads internal properties.

    refreshes the cached list of globally available local checkpointed steps.
    """
    super().reload()
    self._steps = None


# (TO DO): Break this out into a separate file.
class PathwaysCheckpointManager(
    abstract_checkpoint_manager.AbstractCheckpointManager, epy.ContextManager
):
  """Provides both checkpoint management and emergency checkpointings.

  This class composes a local and a persistent checkpoint managers. The local
  manager saves checkpoints frequently to a fast local storage (like RAMFS).
  When a complete checkpoint exists at least one slice, restoration is possible,
  and the slice broadcasts the checkpoint to others. Additionally, the
  persistent manager checkpoints less frequently to a remote file system (e.g.,
  GCS),
  providing a fail-safe if local checkpoints become unavailable due to issues
  like hardware failure or preemption.

  Usage::
    options = CheckpointManagerOptions(
        local=LocalCheckpointOptions(save_interval_steps=2, max_to_keep=2),
        persistent=PersistentCheckpointOptions(
            save_interval_steps=5, max_to_keep=3
        ),
        enable_async_checkpointing=use_async,
    )
    return PathwaysCheckpointManager(
        local_directory=local_directory,
        persistent_directory=persistent_directory,
        global_mesh=global_mesh,
        abstract_state=abstract_state,
        options=options,
        local_state_handler=local_checkpoint_handler(),
    )
  """

  # TODO: b/330585086 - Allow configuration of global mesh describing slices.
  # Validate against global meshes used for arrays in state.
  def __init__(
      self,
      local_directory: epath.PathLike,
      persistent_directory: epath.PathLike,
      global_mesh: jax.sharding.Mesh,
      abstract_state: PyTree,  # a single PyTree describing the state structure
      # TODO: b/330585086 - Support arbitrary items beyond state. We will have
      # to evaluate whether arbitrary items can be a good fit for local
      # checkpointing, given restore+broadcast requirements.
      local_state_handler: CheckpointHandler,
      *,
      options: Optional[
          emergency_checkpoint_manager.CheckpointManagerOptions
      ] = None,
      metadata: Optional[dict[str, Any]] = None,
      logger: Optional[abstract_logger.AbstractLogger] = None,
  ):
    self._local_directory = epath.Path(local_directory)
    self._persistent_directory = epath.Path(persistent_directory)
    self._logger = logger or standard_logger.StandardLogger()
    # TODO: b/330585086 - Fully support options.
    options = options or emergency_checkpoint_manager.CheckpointManagerOptions()
    self._global_mesh = global_mesh
    logging.info('****** global mesh is: %s', self._global_mesh)

    emergency_checkpoint_manager.maybe_save_process_metadata(
        self._persistent_directory, self._global_mesh
    )

    self._abstract_state = abstract_state
    logging.info('****** Abstract state is: %s', self._abstract_state)

    self._replica_axis_index = options.replica_axis_index

    self._local_state_handler = local_state_handler
    self._options = options
    self._metadata = metadata

    if global_mesh.devices.shape[0] > 1:
      logging.info(
          '****** global_mesh.devices[1].flat[0].process_index is: %s',
          global_mesh.devices[1].flat[0].process_index,
      )
    self._persistent_max_to_keep = self._options.persistent.max_to_keep
    self._local_max_to_keep = self._options.local.max_to_keep
    self._coordination_timeout_secs = (
        options.multiprocessing_options
        or emergency_checkpoint_manager.MultiprocessingOptions()
    ).coordination_timeout_secs

    self._persistent_checkpoint_manager = (
        self._make_persistent_checkpoint_manager()
    )
    self._local_checkpoint_manager = self._make_local_checkpoint_manager()

    logging.info(
        'Created emergency.PathwaysCheckpointManager with'
        ' process_index=%d, jax.process_index=%d',
        multihost.process_index(),
        jax.process_index(),
    )

  def _make_persistent_checkpoint_manager(self):
    all_devices = self._global_mesh.devices.flatten()
    first_slice_devices = []
    for d in all_devices:
      if d.slice_index == 0:
        first_slice_devices.append(d)

    active_processes = set(first_slice_devices)
    persistent_multiprocessing_options = (
        checkpoint_manager.MultiprocessingOptions(
            primary_host=jax.process_index(backend='proxy'),
            active_processes=active_processes,
            barrier_sync_key_prefix='persistent',
        )
    )
    persistent_options = checkpoint_manager.CheckpointManagerOptions(
        save_interval_steps=self._options.persistent.save_interval_steps,
        max_to_keep=self._persistent_max_to_keep,
        step_name_format=self._options.step_name_format,
        create=False,
        cleanup_tmp_directories=self._options.cleanup_tmp_directories,
        async_options=self._options.async_options,
        multiprocessing_options=persistent_multiprocessing_options,
        enable_async_checkpointing=self._options.enable_async_checkpointing,
    )
    return checkpoint_manager.CheckpointManager(
        self._persistent_directory,
        options=persistent_options,
        metadata=self._metadata,
        item_handlers=PyTreeCheckpointHandler(
            # use_ocdbt=True,
            use_ocdbt=False,
            use_zarr3=True,
            multiprocessing_options=persistent_multiprocessing_options,
        ),
        logger=self._logger,
    )

  # TO DO: Need to use this instead of the persistent checkpoint manager.
  # because it has some helper functions in it that need to be used.
  def _make_local_checkpoint_manager(self):
    return _PathwaysLocalCheckpointManager(
        self._local_directory,
        self._local_state_handler,
        global_mesh=self._global_mesh,
        options=self._options,
        metadata=self._metadata,
        logger=self._logger,
    )

  @property
  def directory(self) -> epath.Path:
    raise NotImplementedError()

  @property
  def global_mesh(self) -> jax.sharding.Mesh:
    return self._global_mesh

  def all_steps(self, read: bool = False) -> Sequence[int]:
    """Returns all steps tracked by the manager.

    Includes steps located in local as well as persistent storage.

    Args:
      read: If True, forces a read directly from the storage location.
        Otherwise, a cached result can be returned.

    Returns:
      A sequence of steps (integers)
    """
    logging.info('Retrieving all steps.')
    local_steps = [-1] * self._local_max_to_keep
    persistent_steps = [-1] * self._persistent_max_to_keep

    # TO DO: Make this correct. later
    # persistent_steps = list(
    #     self._persistent_checkpoint_manager.all_steps(read=read)
    # )
    persistent_steps = list([0])

    if len(persistent_steps) > self._persistent_max_to_keep:
      # TODO: b/330585086 - for now we assume that
      # persistent_checkpoint_manager.all_steps returns an array with length
      # smaller than max_to_keep
      raise AssertionError(
          f'persistent_step on host {multihost.process_index()} exceeded'
          f' `max_to_keep` {self._persistent_max_to_keep}'
      )
    persistent_steps = emergency_checkpoint_manager.pad_steps(
        persistent_steps, self._persistent_max_to_keep
    )
    local_steps = emergency_checkpoint_manager.pad_steps(
        list(self._local_checkpoint_manager.all_steps(read)),
        self._local_max_to_keep,
    )

    logging.info('****** local_steps %s', local_steps)
    logging.info('****** persistent_steps %s', persistent_steps)

    local_steps = np.asarray(
        multihost.broadcast_one_to_all(
            local_steps,
            is_source=True,
        )
    )

    persistent_steps = np.asarray(
        multihost.broadcast_one_to_all(
            persistent_steps,
            is_source=True,
            # is_source=multihost.process_index()
            # == self._persistent_primary_host,
        )
    )

    logging.info('****** local_steps %s', local_steps)
    logging.info('****** persistent_steps %s', persistent_steps)

    return [
        x
        for x in set(np.concatenate((local_steps, persistent_steps)))
        if x != -1
    ]

  def latest_step(self) -> Optional[int]:
    """Returns the latest step saved.

    Includes steps located in local as well as persistent storage.

    Returns None if no steps have been saved.

    Returns:
      A step (int) or None if no steps are present.
    """
    logging.info('Retrieving latest step.')
    latest_step_persistent = self._persistent_checkpoint_manager.latest_step()
    latest_step_local = self._local_checkpoint_manager.latest_step()
    logging.info('Got latest step from persistent %s', latest_step_persistent)
    logging.info('Got latest step from local %s', latest_step_local)
    if latest_step_persistent is None:
      latest_step_persistent = -1
    if latest_step_local is None:
      latest_step_local = -1
    latest_step = max(latest_step_persistent, latest_step_local)

    if latest_step is None:
      latest_step = -1

    return latest_step if latest_step != -1 else None

  def best_step(self) -> Optional[int]:
    """Returns the best step saved, as defined by `options.best_fn`.

    Includes steps located in local as well as persistent storage.

    Returns None if no steps have been saved.

    Returns:
      A step (int) or None if no steps are present.
    """
    raise NotImplementedError(
        'Metrics tracking not yet implemented for emergency.CheckpointManager.'
    )

  def reload(self):
    """Performs disk reads to ensure internal properties are up to date."""
    self._persistent_checkpoint_manager.reload()
    self._local_checkpoint_manager.reload()

  def reached_preemption(self, step: int) -> bool:
    """Returns True if a preemption sync point has been reached."""
    return utils.reached_preemption(step)

  def _global_max(self, value: int) -> int:
    """Returns the global max of a local value across all devices as a scalar."""
    return value

  def should_save(self, step: int) -> bool:
    """Returns True if a checkpoint should be saved for the current step.

    This depends the previous step and save interval.

    Args:
      step: int

    Returns:
      True if the checkpoint should be saved.
    """
    logging.info('Checking should_save at step: %d.', step)

    # This needs to be corrected for the appropriate step.
    should_save_persistent = self._persistent_checkpoint_manager.should_save(
        step
    )
    should_save_local = self._local_checkpoint_manager.should_save(step)
    should_save = should_save_persistent or should_save_local

    return bool(self._global_max(int(should_save)))

  def delete(self, step: int):
    """Deletes a step checkpoint."""
    raise NotImplementedError(
        'Delete not yet implemented for emergency.CheckpointManager.'
    )

  def save(
      self,
      step: int,
      args: Optional[args_lib.CheckpointArgs] = None,
      metrics: Optional[PyTree] = None,
      force: Optional[bool] = False,
  ) -> bool:
    """Returns True no matter if a checkpoint is saved or not."""
    # TODO: b/330608746 - implement save op on different slices
    # For now just save to local.
    # logging.info('****** Maybe saving at step %d (persistent).', step)
    # _ = self._persistent_checkpoint_manager.save(
    #     step, args=args, metrics=metrics, force=force
    # )
    logging.info('****** Maybe saving at step %d (local).', step)
    _ = self._local_checkpoint_manager.save(
        step, args=args, metrics=metrics, force=force
    )
    logging.info('****** successfully saved test metadata')

    # global_max is costing a lot and it's not worth it to keep return value
    # correct across processes. directly returning true.
    # return bool(self._global_max(int(saved)))
    return True

  def _find_slice_with_complete_checkpoint(self, step: int) -> int:
    """Return the slice id which has the step."""
    # if self.in_primary_slice:
    #   # No steps can be found in local storage, since this
    #   # is the primary slice.
    #   local_steps = set()
    # else:

    # TO DO: This needs to be updated to look for all the local host steps.
    local_steps = set(self._local_checkpoint_manager.local_host_steps(True))

    # Here we need to determine which slice has the step.
    # For now, let's hardcode this to 1
    slice_id = -1
    if step in local_steps:
      slice_id = 1
    return slice_id

  def _restore_from_lcm(
      self,
      step: int,
      restoring_slice_id: int,
      args: Optional[args_lib.CheckpointArgs] = None,
      directory: Optional[epath.PathLike] = None,
  ) -> Any:
    logging.info(
        'emergency.CheckpointManager: restoring step=%s from local checkpoint'
        ' using slice_id: %s',
        step,
        restoring_slice_id,
    )
    step_stats = step_statistics.EmergencyRestoreStepStatistics()
    step_stats.checkpoint_manager_start_time = time.time()
    step_stats.step = step

    # Commented out for pathways path.
    # is_restoring_slice = restoring_slice_id == self._slice_id
    # step_stats.is_restoring_slice = is_restoring_slice
    # step_stats.in_primary_slice = self.in_primary_slice

    # shape_dtypes
    _, tree_defs = jax.tree.flatten(self._abstract_state)

    logging.info('****** tree_defs: %s', tree_defs)

    def _get_single_slice_sharding(
        mesh: jax.sharding.Mesh,
        pspec: jax.sharding.PartitionSpec,
    ):
      # Replaced for pathways path.
      # slice_devices = np.asarray([self._global_mesh.devices[self._slice_id]])
      slice_devices = np.asarray(
          [self._global_mesh.devices[restoring_slice_id]]
      )

      slice_mesh = jax.sharding.Mesh(slice_devices, mesh.axis_names)
      logging.info('****** got slice mesh: %s', slice_mesh)
      ss_sharding = jax.sharding.NamedSharding(slice_mesh, pspec)
      logging.info('****** got single slice sharding: %s', ss_sharding)
      return ss_sharding

    single_slice_shardings = jax.tree.map(
        lambda arr: _get_single_slice_sharding(
            mesh=arr.sharding.mesh,
            pspec=arr.sharding.spec,
        ),
        self._abstract_state,
    )

    # single_replica_shardings_tuple
    _ = jax.tree.flatten(single_slice_shardings)[0]

    # This is indeed restoring, but we need to treat it like a persistent
    # checkpoint
    # Instead of a local checkpoint.
    # if is_restoring_slice:
    logging.vlog(
        1, 'emergency.CheckpointManager: restoring from local checkpoint.'
    )
    ss_args = jax.tree.map(
        lambda ss_shard, arr: type_handlers.ArrayRestoreArgs(
            sharding=ss_shard,
            global_shape=arr.shape,  # single-slice sharding
        ),
        single_slice_shardings,
        self._abstract_state,
    )
    logging.info('****** got single slice restore args: %s', ss_args)
    # restore_directory = self._local_checkpoint_manager._get_read_step_directory(  # pylint: disable=protected-access
    #     step, epath.Path(directory or self._local_directory)
    # )
    restore_directory = (
        self._local_directory / f'{step}.orbax-checkpoint-tmp-24'
    )
    step_stats.directory = str(restore_directory)

    # Directly use CheckpointHandler to restore. This is undesirable, but
    # allows us to avoid barrier issues that occur when calling
    # LocalCheckpointManager a different number of times on the non-primary
    # slices, which leads to
    # _module_unique_count getting out of sync.
    restore_path = (
        restore_directory
        / f'{checkpoint_manager.DEFAULT_ITEM_NAME}.orbax-checkpoint-tmp-27'
    )
    logging.vlog(
        1,
        'Restoring from %s',
        restore_directory / checkpoint_manager.DEFAULT_ITEM_NAME,
    )
    step_stats.checkpointer_start_time = time.time()
    single_slice_pytree = self._local_state_handler.restore(
        restore_path,
        args=dataclasses.replace(args, restore_args=ss_args),
    )
    logging.info('****** Single slice pytree: %s', single_slice_pytree)
    step_stats.checkpointer_duration_secs = (
        time.time() - step_stats.checkpointer_start_time
    )
    in_tree = tuple(jax.tree.flatten(single_slice_pytree)[0])

    # This part may not be necessary.
    # else:
    #   logging.vlog(
    #       1,
    #       'emergency.CheckpointManager: secondary slice, create zeros and'
    #       ' wait for broacast.',
    #   )

    #   @functools.partial(
    #       jax.jit,
    #       static_argnums=0,
    #       out_shardings=tuple(single_replica_shardings_tuple),
    #   )
    #   def create_zeros(shape_dtype_tup):
    #     return jax.tree.map(
    #         lambda sd: jnp.zeros(sd.shape, dtype=sd.dtype), shape_dtype_tup
    #     )

    #   zeros_pytree = create_zeros(tuple(shape_dtypes))
    #   in_tree = tuple(zeros_pytree)

    start_broadcast = time.time()

    logging.info('****** Starting broadcast with in_tree: %s', in_tree)

    # This function will need some updates
    shared_states, _ = multislice.broadcast_one_replica_to_all(
        in_tree,
        self._global_mesh,
        replica_axis_index=self._replica_axis_index,
        # is_source=is_restoring_slice,
        is_source=True,
    )
    broadcast_elapsed_s = time.time() - start_broadcast
    jax.monitoring.record_event_duration_secs(
        '/orbax/emergency/checkpoint/read/broadcast_duration_secs',
        broadcast_elapsed_s,
    )
    step_stats.broadcast_start_time = start_broadcast
    step_stats.broadcast_duration_secs = broadcast_elapsed_s
    step_stats.checkpoint_manager_duration_secs = (
        time.time() - step_stats.checkpoint_manager_start_time
    )
    self._logger.log_entry(dataclasses.asdict(step_stats))

    logging.info('Finished broadcasting in %.2f', broadcast_elapsed_s)
    return jax.tree.unflatten(tree_defs, shared_states)

  def _restore_from_persistent_cm(
      self,
      step: int,
      args: Optional[args_lib.CheckpointArgs] = None,
      directory: Optional[epath.PathLike] = None,
  ) -> Any:
    logging.info(
        'emergency.CheckpointManager: restoring step=%s from persistent'
        ' checkpoint in directory=%s',
        step,
        directory or self._persistent_directory,
    )

    # Create a temporarily read-only PersistentCheckpointManager that will
    # synchronize the restoration with global processes.
    persistent_options = checkpoint_manager.CheckpointManagerOptions(
        step_name_format=self._options.step_name_format,
        create=False,
        cleanup_tmp_directories=False,
        read_only=True,
        enable_async_checkpointing=False,
        multiprocessing_options=checkpoint_manager.MultiprocessingOptions(
            barrier_sync_key_prefix='persistent_global',
        ),
    )
    with checkpoint_manager.CheckpointManager(
        self._persistent_directory,
        options=persistent_options,
        metadata=self._metadata,
        item_handlers=PyTreeCheckpointHandler(
            use_ocdbt=True,
            use_zarr3=True,
        ),
    ) as pcm:
      return pcm.restore(step, args=args, directory=directory)

  def restore(
      self,
      step: int,
      args: Optional[args_lib.CheckpointArgs] = None,
      directory: Optional[epath.PathLike] = None,
  ) -> Any:
    logging.info('Restoring at step %d.', step)
    restoring_slice_id = self._find_slice_with_complete_checkpoint(step)
    if restoring_slice_id > -1:
      logging.info(
          '****** Restoring from LCM with restoring slice id %d',
          restoring_slice_id,
      )
      # restore from LCM
      return self._restore_from_lcm(
          step=step,
          restoring_slice_id=restoring_slice_id,
          args=args,
          directory=directory,
      )

    return self._restore_from_persistent_cm(
        step=step, args=args, directory=directory
    )

  def item_metadata(self, step: int) -> Any:
    raise NotImplementedError(
        'Item metadata not yet implemented for emergency.CheckpointManager.'
    )

  def metadata(self) -> dict[str, Any]:
    """Returns CheckpointManager level metadata if present, empty otherwise."""
    raise NotImplementedError(
        'Metadata not yet implemented for emergency.CheckpointManager.'
    )

  def metrics(self, step: int) -> Optional[PyTree]:
    """Returns metrics for step, if present."""
    raise NotImplementedError(
        'Metrics not yet implemented for emergency.CheckpointManager.'
    )

  def wait_until_finished(self):
    """Blocks until any incomplete save operations are completed.

    Note that this method will typically be a no-op if all checkpointers are
    synchronous, since old checkpoints are already cleaned up immediately after
    completing `save`, and there is no background thread to wait for.

    If some checkpointers are of type AsyncCheckpointer, however, this method
    will wait until each of these checkpointers is finished.
    """
    logging.info('Waiting for checkpoint to complete.')
    # if self.in_primary_slice:
    self._persistent_checkpoint_manager.wait_until_finished()
    # else:
    self._local_checkpoint_manager.wait_until_finished()

  def check_for_errors(self):
    """Checks for any outstanding errors in completed asynchronous save operations.

    Delegates to underlying Checkpointer.
    """
    # if self.in_primary_slice:
    self._persistent_checkpoint_manager.check_for_errors()
    # else:
    self._local_checkpoint_manager.check_for_errors()

  def close(self):
    """Waits for outstanding operations to finish and closes Checkpointers."""
    logging.info('Closing CheckpointManager.')
    # if self.in_primary_slice:
    self._persistent_checkpoint_manager.close()
    # else:
    self._local_checkpoint_manager.close()

  def __contextmanager__(
      self,
  ) -> Iterable[Self]:
    try:
      yield self
    finally:
      self.close()
