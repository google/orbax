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

import collections
import dataclasses
import functools
import itertools
import operator
import time
from typing import Any, Iterable, Optional, Sequence, Set, Union

from absl import logging
from etils import epath
from etils import epy
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint import abstract_checkpoint_manager
from orbax.checkpoint import args as args_lib
from orbax.checkpoint import checkpoint_manager
from orbax.checkpoint import multihost
from orbax.checkpoint import type_handlers
from orbax.checkpoint import utils
from orbax.checkpoint.path import step as step_lib
from typing_extensions import Self  # for Python version < 3.11


PyTree = checkpoint_manager.PyTree
CheckpointHandler = checkpoint_manager.CheckpointHandler
P = jax.sharding.PartitionSpec

_module_unique_count = itertools.count()


@dataclasses.dataclass
class LocalCheckpointOptions:
  """Optional CheckpointManager arguments for saving local checkpoints.

  save_interval_steps:
    The interval at which checkpoints should be saved to local storage.
    Ensures checkpoints will only be saved every m steps. Defaults to 10.
  max_to_keep:
    Specifies the maximum number of local checkpoints to
    keep. Older checkpoints are removed. When set, no more than `max_to_keep`
    checkpoints will be present at any one time. This option has a slightly
    different meaning than it normally does in Orbax: this should be treated
    as a hard cap on the number of checkpoints concurrently present, rather
    than a threshold beyond which checkpoints start to be deleted.
  """

  save_interval_steps: int = 10
  max_to_keep: int = 2
  read_only: bool = False


@dataclasses.dataclass
class PersistentCheckpointOptions:
  """Optional CheckpointManager arguments for saving persistent checkpoints.

  save_interval_steps:
    The interval at which checkpoints should be saved to persistent storage.
    Ensures checkpoints will only be saved every n steps. Defaults to 1000.
  max_to_keep:
    If provided, specifies the maximum number of persistent checkpoints to
    keep. Older checkpoints are removed. By default, does not remove any old
    checkpoints. Must be None or non-negative. When set, checkpoints
    may be considered for deletion when there are more than `max_to_keep`
    checkpoints present.
  """

  save_interval_steps: int = 1000
  max_to_keep: Optional[int] = None


@dataclasses.dataclass
class CheckpointManagerOptions:
  """Optional arguments for CheckpointManager.

  local:
    Options relevant to the local checkpoints.
    See `LocalCheckpointOptions`.
  persistent:
    Options relevant to the persistent checkpoints. See
    `PersistentCheckpointOptions`.
  step_name_format:
    NameFormat to build or find steps under input root directory. If provided,
    `step_prefix`, `step_format_fixed_length` are ignored.
  cleanup_tmp_directories:
    If True, cleans up any existing temporary directories
    on CheckpointManager creation.
  enable_async_checkpointing:
    Enable async saving.
  async_options:
    Used to configure properties of async behavior. See above.
  """

  local: LocalCheckpointOptions = dataclasses.field(
      default_factory=LocalCheckpointOptions
  )
  persistent: PersistentCheckpointOptions = dataclasses.field(
      default_factory=PersistentCheckpointOptions
  )

  step_name_format: Optional[step_lib.NameFormat] = None
  cleanup_tmp_directories: bool = False
  enable_async_checkpointing: bool = True
  async_options: Optional[checkpoint_manager.AsyncOptions] = None


def _process_slice_id(
    process_index: int, global_mesh: jax.sharding.Mesh
) -> int:
  """Returns the slice id that the process_index belongs to."""
  for slice_id, device_slice in enumerate(global_mesh.devices):
    if process_index in _pid_in_slice(device_slice):
      return slice_id

  return -1


def _pid_in_slice(device_slice: np.ndarray) -> np.ndarray:
  pid = np.vectorize(lambda d: d.process_index)
  return pid(device_slice)


def in_slice(process_index: int, device_slice: np.ndarray) -> bool:
  return process_index in _pid_in_slice(device_slice)


def in_primary_slice(
    process_index: int, global_mesh: jax.sharding.Mesh
) -> bool:
  """Returns true if host is in primary slice (the first slice)."""
  primary_slice = global_mesh.devices[0]

  return in_slice(process_index, primary_slice)


def _unique_processes_from_devices(device_array: np.ndarray) -> Set[int]:
  pid = np.vectorize(lambda d: d.process_index)
  return set(pid(device_array).flat)


def _local_slice_devices(devices_array: np.ndarray) -> np.ndarray:
  for device_slice in devices_array:
    if in_slice(multihost.process_index(), device_slice):
      return device_slice
  raise ValueError(
      f'process_index {multihost.process_index()} does not exist in provided'
      ' `global_mesh`'
  )


def _pad_steps(steps, target):
  return steps + [-1] * (target - len(steps))


class _LocalCheckpointManager(checkpoint_manager.CheckpointManager):
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
      device_array: np.ndarray,
      *,
      options: Optional[CheckpointManagerOptions] = None,
      metadata: Optional[dict[str, Any]] = None,
  ):
    # TODO: b/330585086 - Fully support options.
    options = options or CheckpointManagerOptions()
    local_options = checkpoint_manager.CheckpointManagerOptions(
        save_interval_steps=options.local.save_interval_steps,
        max_to_keep=options.local.max_to_keep,
        step_name_format=options.step_name_format,
        create=False,
        cleanup_tmp_directories=options.cleanup_tmp_directories,
        async_options=options.async_options,
        multiprocessing_options=checkpoint_manager.MultiprocessingOptions(
            primary_host=None,
            active_processes=_unique_processes_from_devices(device_array),
            barrier_sync_key_prefix='local',
        ),
        enable_async_checkpointing=options.enable_async_checkpointing,
        read_only=options.local.read_only,
    )
    self._options = local_options
    self._device_array = device_array

    super().__init__(
        directory,
        options=local_options,
        metadata=metadata,
        item_handlers=state_handler,
    )
    self._max_to_keep = options.local.max_to_keep

  def _global_list_union(
      self, steps: Sequence[int], devices: np.ndarray
  ) -> np.ndarray:
    slice_mesh = jax.sharding.Mesh(
        devices.reshape(
            devices.size // jax.local_device_count(), jax.local_device_count()
        ),
        ['host', 'dev'],
    )

    g_arr = multihost_utils.host_local_array_to_global_array(
        np.asarray(steps), slice_mesh, P('host')
    )

    result_arr = jax.jit(
        lambda x: x,
        out_shardings=jax.sharding.NamedSharding(slice_mesh, P()),
    )(g_arr)

    return np.asarray(result_arr.addressable_data(0))

  def _global_list_union_interslice(self, steps: Sequence[int]) -> Set[int]:
    barrier_processes = self._options.multiprocessing_options.active_processes
    barrier_processes = [
        multihost.utils._runtime_to_distributed_process_id(runtime_id)  # pylint: disable=protected-access
        for runtime_id in barrier_processes
    ]

    client = multihost.utils._get_jax_distributed_client()  # pylint: disable=protected-access
    dir_key = f'steps/{next(_module_unique_count)}/'
    dir_key = multihost.utils._unique_barrier_key(dir_key) + '/'  # pylint: disable=protected-access
    key = dir_key + str(multihost.process_index())
    client.key_value_set(key, ','.join([str(s) for s in steps]))

    barrier_key = 'broadcast_interslice_' + str(next(_module_unique_count))
    barrier_key = multihost.utils._unique_barrier_key(barrier_key)  # pylint: disable=protected-access
    client.wait_at_barrier(
        barrier_key, process_ids=barrier_processes, timeout_in_ms=10000
    )

    per_slice_steps = client.key_value_dir_get(dir_key)
    per_slice_steps = [
        set([int(s) for s in v.split(',')]) for _, v in per_slice_steps if v
    ]
    return functools.reduce(operator.ior, per_slice_steps, set([]))

  def _common_steps_global(self, steps: Sequence[int]) -> np.ndarray:
    """Returns common steps across all slices.

    A step is considered as "common step" if it is known to any slice.

    The slice is assumed to be diveded along the first axis, i.e. slice0 =
    global_mesh.devices[0]

    Args:
      steps: a list of steps known to all hosts on a slice
    """
    unioned_steps = self._global_list_union_interslice(steps)
    return np.asarray(list(unioned_steps))

  def common_steps_within_slice(self, steps: Sequence[int]) -> np.ndarray:
    """Returns common steps within one slices.

    A step is considered "common step" if it is known to every host in the slice

    The slice is assumed to be diveded along the first axis, i.e. slice0 =
    global_mesh.devices[0]

    This function will pad return array to len(steps) with -1's indicating no
    step.

    Args:
      steps: a list of known steps on host
    """

    devices = _local_slice_devices(self._device_array)
    slice_process_count = devices.size // jax.local_device_count()
    unioned_steps = self._global_list_union(steps, devices)

    def count_and_filter(steps, num_process):
      count = collections.Counter(steps)
      return np.asarray([k for k in count if count[k] == num_process])

    result = count_and_filter(unioned_steps, slice_process_count)

    # here len(result) will be <= len(steps) because there are at most
    # len(steps) unique step number that appeared `slice_process_count` times in
    # an array with size [len(steps), slice_process_count]
    if len(result) > len(steps):
      raise AssertionError(
          f' len(result steps) {result} exceeded length of input steps {steps}'
      )

    return np.asarray(_pad_steps(list(result), len(steps)))

  def local_host_steps(self, read: bool) -> Sequence[int]:
    """Returns steps known to local host."""
    # List of steps present in individual host storage.
    local_steps = list(super().all_steps(read))

    if len(local_steps) > self._max_to_keep:
      raise AssertionError(
          f' local_step on host {multihost.process_index()} exceeded'
          f' `max_to_keep` {self._max_to_keep}'
      )

    return _pad_steps(local_steps, self._max_to_keep)

  def all_steps(self, read: bool = False) -> Sequence[int]:
    """Returns all steps tracked by the manager.

    Includes steps located in local as well as persistent storage.

    Args:
      read: If True, forces a read directly from the storage location.
        Otherwise, a cached result can be returned.

    Returns:
      A sequence of steps (integers)
    """
    local_steps = self.local_host_steps(read)

    common_steps_on_slice = self.common_steps_within_slice(local_steps)

    steps = self._common_steps_global(common_steps_on_slice)

    return [x for x in steps if x != -1]

  def latest_step(self) -> Optional[int]:
    """Returns the latest step saved in the local storage.

    Returns None if no steps have been saved.

    Returns:
      A step (int) or None if no steps are present.
    """
    local_all_steps = self.all_steps()
    return max(local_all_steps) if local_all_steps else None


class CheckpointManager(
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
      persistent_state_handler: CheckpointHandler,
      *,
      options: Optional[CheckpointManagerOptions] = None,
      metadata: Optional[dict[str, Any]] = None,
  ):
    # TODO: b/330585086 - Fully support options.
    options = options or CheckpointManagerOptions()
    self._global_mesh = global_mesh
    self._abstract_state = abstract_state
    self._slice_id = _process_slice_id(
        multihost.process_index(), self._global_mesh
    )

    self._local_directory = local_directory
    self._local_state_handler = local_state_handler
    self._persistent_directory = persistent_directory
    self._persistent_state_handler = persistent_state_handler
    self._options = options
    self._metadata = metadata
    self._persistent_primary_host = global_mesh.devices[0].flat[0].process_index
    self._local_primary_host = (
        global_mesh.devices[1].flat[0].process_index
        if global_mesh.devices.shape[0] > 1
        else None
    )
    if self._local_primary_host is None:
      raise AssertionError(
          'to use this CheckpointManager, at least 3 data-parallel slices are'
          ' needed.'
      )

    self.in_primary_slice = in_primary_slice(
        multihost.process_index(), global_mesh
    )
    self._persistent_max_to_keep = self._options.persistent.max_to_keep
    self._local_max_to_keep = self._options.local.max_to_keep

    persistent_multiprocessing_options = (
        checkpoint_manager.MultiprocessingOptions(
            primary_host=self._persistent_primary_host,
            active_processes=_unique_processes_from_devices(
                self._global_mesh.devices[0]
            ),
            barrier_sync_key_prefix='persistent',
        )
    )
    if self.in_primary_slice:
      persistent_options = checkpoint_manager.CheckpointManagerOptions(
          save_interval_steps=self._options.persistent.save_interval_steps,
          max_to_keep=self._persistent_max_to_keep,
          step_name_format=self._options.step_name_format,
          create=False,
          cleanup_tmp_directories=self._options.cleanup_tmp_directories,
          async_options=self._options.async_options,
          multiprocessing_options=persistent_multiprocessing_options,
          enable_async_checkpointing=options.enable_async_checkpointing,
      )
      self._persistent_checkpoint_manager = (
          checkpoint_manager.CheckpointManager(
              self._persistent_directory,
              options=persistent_options,
              metadata=self._metadata,
              item_handlers=self._persistent_state_handler,
          )
      )
    else:
      self._local_checkpoint_manager = _LocalCheckpointManager(
          self._local_directory,
          self._local_state_handler,
          device_array=global_mesh.devices[1:],
          options=self._options,
          metadata=self._metadata,
      )

    logging.info(
        'Created emergency.CheckpointManager with slice_id=%s', self._slice_id
    )

  @property
  def directory(self) -> epath.Path:
    raise NotImplementedError()

  def _data_per_individual_slice(
      self, data: Sequence[Union[bool, int, float]]
  ) -> np.ndarray:
    """Broadcasts its own data and collect data from all other slices.

    This function assumes the slice is divided along the first axis (i.e. slice0
    = global_mesh.devices[0]). `Data` from hosts within a slice must be
    identical, as well as the data dimensions between slices. If these
    assumptions are not met, the function may return indeterministic results.

    Args:
      data: a list of bool/int/float.

    Returns:
      a np.ndarray and its index corresponding to the slice id.
    """
    g_arr = multihost_utils.host_local_array_to_global_array(
        np.asarray(data),
        self._global_mesh,
        P(self._global_mesh.axis_names[0]),  # assume first axis is the slice
    )

    result_arr = jax.jit(
        lambda x: x,
        out_shardings=jax.sharding.NamedSharding(self._global_mesh, P()),
    )(g_arr)

    return np.asarray(result_arr.addressable_data(0))

  def all_steps(self, read: bool = False) -> Sequence[int]:
    """Returns all steps tracked by the manager.

    Includes steps located in local as well as persistent storage.

    Args:
      read: If True, forces a read directly from the storage location.
        Otherwise, a cached result can be returned.

    Returns:
      A sequence of steps (integers)
    """
    local_steps = [-1] * self._local_max_to_keep
    persistent_steps = [-1] * self._persistent_max_to_keep
    if self.in_primary_slice:
      persistent_steps = list(
          self._persistent_checkpoint_manager.all_steps(read=read)
      )
      if len(persistent_steps) > self._persistent_max_to_keep:
        # TODO: b/330585086 - for now we assume that
        # persistent_checkpoint_manager.all_steps returns an array with length
        # smaller than max_to_keep
        raise AssertionError(
            f'persistent_step on host {multihost.process_index()} exceeded'
            f' `max_to_keep` {self._persistent_max_to_keep}'
        )
      persistent_steps = _pad_steps(
          persistent_steps, self._persistent_max_to_keep
      )
    else:
      local_steps = _pad_steps(
          list(self._local_checkpoint_manager.all_steps(read)),
          self._local_max_to_keep,
      )

    local_steps = np.asarray(
        multihost.broadcast_one_to_all(
            local_steps,
            is_source=multihost.process_index() == self._local_primary_host,
        )
    )

    persistent_steps = np.asarray(
        multihost.broadcast_one_to_all(
            persistent_steps,
            is_source=multihost.process_index()
            == self._persistent_primary_host,
        )
    )

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
    if self.in_primary_slice:
      latest_step = self._persistent_checkpoint_manager.latest_step()
    else:
      latest_step = self._local_checkpoint_manager.latest_step()

    if latest_step is None:
      latest_step = -1

    latest_step = self._global_max(latest_step)
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
    if self._local_primary_host:
      self._persistent_checkpoint_manager.reload()
    else:
      self._local_checkpoint_manager.reload()

  def reached_preemption(self, step: int) -> bool:
    """Returns True if a preemption sync point has been reached."""
    return utils.reached_preemption(step)

  def _global_max(self, value: Any) -> Any:
    """Returns the global max of a local value across all devices as a scalar."""

    device_array = self._global_mesh.devices
    slice_mesh = jax.sharding.Mesh(
        device_array.reshape(
            device_array.size // jax.local_device_count(),
            jax.local_device_count(),
        ),
        ['host', 'dev'],
    )

    g_arr = multihost_utils.host_local_array_to_global_array(
        np.asarray([value]), slice_mesh, P('host')
    )

    result_arr = jax.jit(
        jnp.max,
        out_shardings=jax.sharding.NamedSharding(slice_mesh, P()),
    )(g_arr)

    return result_arr.addressable_data(0)

  def should_save(self, step: int) -> bool:
    """Returns True if a checkpoint should be saved for the current step.

    This depends the previous step and save interval.

    Args:
      step: int

    Returns:
      True if the checkpoint should be saved.
    """
    if self.in_primary_slice:
      should_save = self._persistent_checkpoint_manager.should_save(step)
    else:
      should_save = self._local_checkpoint_manager.should_save(step)
    return self._global_max(should_save)

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
    # TODO: b/330608746 - implement save op on different slices
    if self.in_primary_slice:
      saved = self._persistent_checkpoint_manager.save(
          step, args=args, metrics=metrics, force=force
      )
    else:
      saved = self._local_checkpoint_manager.save(
          step, args=args, metrics=metrics, force=force
      )
    return self._global_max(saved)

  def _find_slice_with_complete_checkpoint(self, step: int) -> int:
    """Return the slice id which has the step."""
    if self.in_primary_slice:
      steps_in_slice = np.asarray([], dtype=int)
    else:
      local_steps = self._local_checkpoint_manager.local_host_steps(True)
      steps_in_slice = self._local_checkpoint_manager.common_steps_within_slice(
          local_steps
      )

    has_step_in_this_slice = step in steps_in_slice

    has_steps = self._data_per_individual_slice(
        [has_step_in_this_slice]
    ).tolist()

    logging.debug('has_steps=%s', has_steps)

    try:
      return has_steps.index(True)
    except ValueError:
      # not present in lcm
      return -1

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

    is_restoring_slice = restoring_slice_id == self._slice_id

    shape_dtypes, tree_defs = jax.tree_util.tree_flatten(self._abstract_state)

    def _get_single_slice_sharding(
        mesh: jax.sharding.Mesh,
        pspec: jax.sharding.PartitionSpec,
    ):
      slice_devices = np.asarray([self._global_mesh.devices[self._slice_id]])
      slice_mesh = jax.sharding.Mesh(slice_devices, mesh.axis_names)
      ss_sharding = jax.sharding.NamedSharding(slice_mesh, pspec)
      return ss_sharding

    single_slice_shardings = jax.tree_util.tree_map(
        lambda arr: _get_single_slice_sharding(
            mesh=arr.sharding.mesh,
            pspec=arr.sharding.spec,
        ),
        self._abstract_state,
    )
    single_replica_shardings_tuple = jax.tree_util.tree_flatten(
        single_slice_shardings
    )[0]

    if is_restoring_slice:
      logging.debug(
          'emergency.CheckpointManager: restoring from local checkpoint.'
      )
      ss_args = jax.tree_util.tree_map(
          lambda ss_shard, arr: type_handlers.ArrayRestoreArgs(
              sharding=ss_shard,
              global_shape=arr.shape,  # sigle-slice sharding
          ),
          single_slice_shardings,
          self._abstract_state,
      )
      restore_directory = (
          self._local_checkpoint_manager._get_read_step_directory(  # pylint: disable=protected-access
              step, epath.Path(directory or self._local_directory)
          )
      )
      # Directly use CheckpointHandler to restore. This is undesirable, but
      # allows us to avoid barrier issues that occur when calling
      # LocalCheckpointManager a different number of times on the non-primary
      # slices, which leads to
      # _module_unique_count getting out of sync.
      logging.debug(
          'Restoring from %s',
          restore_directory / checkpoint_manager.DEFAULT_ITEM_NAME,
      )
      single_slice_pytree = self._local_state_handler.restore(
          restore_directory / checkpoint_manager.DEFAULT_ITEM_NAME,
          args=dataclasses.replace(args, restore_args=ss_args),
      )
      in_tree = tuple(jax.tree_util.tree_flatten(single_slice_pytree)[0])
    else:
      logging.debug(
          'emergency.CheckpointManager: secondary slice, create zeros and'
          ' wait for broacast.'
      )

      @functools.partial(
          jax.jit,
          static_argnums=0,
          out_shardings=tuple(single_replica_shardings_tuple),
      )
      def create_zeros(shape_dtype_tup):
        return jax.tree_util.tree_map(
            lambda sd: jnp.zeros(sd.shape, dtype=sd.dtype), shape_dtype_tup
        )

      zeros_pytree = create_zeros(tuple(shape_dtypes))
      in_tree = tuple(zeros_pytree)

    start_broadcast = time.time()
    shared_states = utils.broadcast_one_replica_to_all(
        in_tree,
        self._global_mesh,
        tuple(single_replica_shardings_tuple),
        0,
        is_source=is_restoring_slice,
    )
    broadcast_elapsed_s = time.time() - start_broadcast
    jax.monitoring.record_event_duration_secs(
        '/orbax/emergency/checkpoint/read/broadcast_duration_secs',
        broadcast_elapsed_s,
    )
    logging.info('Finished broadcasting in %.2f', broadcast_elapsed_s)

    return jax.tree_util.tree_unflatten(tree_defs, shared_states)

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
        async_options=self._options.async_options,
        read_only=True,
        multiprocessing_options=checkpoint_manager.MultiprocessingOptions(
            barrier_sync_key_prefix='persistent_global',
        ),
    )
    with checkpoint_manager.CheckpointManager(
        self._persistent_directory,
        options=persistent_options,
        metadata=self._metadata,
        item_handlers=self._persistent_state_handler,
    ) as pcm:
      return pcm.restore(step, args=args, directory=directory)

  def restore(
      self,
      step: int,
      args: Optional[args_lib.CheckpointArgs] = None,
      directory: Optional[epath.PathLike] = None,
  ) -> Any:

    restoring_slice_id = self._find_slice_with_complete_checkpoint(step)
    if restoring_slice_id > -1:
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
    if self.in_primary_slice:
      self._persistent_checkpoint_manager.wait_until_finished()
    else:
      self._local_checkpoint_manager.wait_until_finished()

  def check_for_errors(self):
    """Checks for any outstanding errors in completed asynchronous save operations.

    Delegates to underlying Checkpointer.
    """
    if self.in_primary_slice:
      self._persistent_checkpoint_manager.check_for_errors()
    else:
      self._local_checkpoint_manager.check_for_errors()

  def close(self):
    """Waits for outstanding operations to finish and closes Checkpointers."""
    if self.in_primary_slice:
      self._persistent_checkpoint_manager.close()
    else:
      self._local_checkpoint_manager.close()

  def __contextmanager__(
      self,
  ) -> Iterable[Self]:
    try:
      yield self
    finally:
      self.close()
