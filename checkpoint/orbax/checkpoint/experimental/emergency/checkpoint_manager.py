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
import json
import operator
import time
from typing import Any, Iterable, Optional, Sequence, Set

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
from orbax.checkpoint import pytree_checkpoint_handler
from orbax.checkpoint import type_handlers
from orbax.checkpoint import utils
from orbax.checkpoint.experimental.emergency import multihost as emergency_multihost
from orbax.checkpoint.logging import abstract_logger
from orbax.checkpoint.logging import standard_logger
from orbax.checkpoint.multihost import multislice
from orbax.checkpoint.path import step as step_lib
from typing_extensions import Self  # for Python version < 3.11


PyTree = checkpoint_manager.PyTree
CheckpointHandler = checkpoint_manager.CheckpointHandler
P = jax.sharding.PartitionSpec
PyTreeCheckpointHandler = pytree_checkpoint_handler.PyTreeCheckpointHandler

_PROCESS_METADATA_FOLDER = 'process_metadata'
_PROCESS_METADATA_FILE_NAME = 'process_metadata.json'
_GLOBAL_PROCESS_METADATA_FILE_NAME = 'global_process_metadata.json'
_MESH_METADATA_FILE_NAME = 'mesh_metadata.json'


def _write_process_metadata(path: epath.Path, mesh: jax.sharding.Mesh):
  """Write process metadata to the given path."""
  logging.info('Saving process index metadata at %s', path)

  if multihost.process_index() == 0:
    path.mkdir(parents=False, exist_ok=False)
    runtime_to_distributed_ids = multihost.utils.runtime_to_distributed_ids()
    (path / _GLOBAL_PROCESS_METADATA_FILE_NAME).write_text(
        json.dumps(runtime_to_distributed_ids)
    )
    (path / _MESH_METADATA_FILE_NAME).write_text(
        json.dumps([int(id) for id in mesh.device_ids.flatten()])
    )
  multihost.sync_global_processes('create_process_metadata')


def _read_process_metadata(path: epath.Path):
  """Read process metadata from the given path."""
  logging.info('Loading process index metadata from %s', path)

  runtime_to_distributed_ids = json.loads(
      (path / _GLOBAL_PROCESS_METADATA_FILE_NAME).read_text()
  )
  device_ids = json.loads((path / _MESH_METADATA_FILE_NAME).read_text())
  return runtime_to_distributed_ids, device_ids


def _maybe_save_process_metadata(
    path: epath.Path, global_mesh: jax.sharding.Mesh
) -> bool:
  """Saves process metadata if it does not already exist."""
  metadata_folder = path / _PROCESS_METADATA_FOLDER
  if metadata_folder.exists():
    return False
  # All processes must check the folder before proceeding. Otherwise the
  # primary process may create the folder from scratch before another
  # process has a chance to check it.
  multihost.sync_global_processes('check_process_metadata_folder')
  _write_process_metadata(metadata_folder, global_mesh)
  return True


def should_restore_mesh_from_metadata(path: epath.Path) -> bool:
  metadata_path = path / _PROCESS_METADATA_FOLDER
  return metadata_path.exists()


def consistent_restore_mesh_from_metadata(
    path: epath.Path, global_mesh: jax.sharding.Mesh
) -> jax.sharding.Mesh:
  """Create a mesh consistent with the saved metadata."""
  metadata_path = path / _PROCESS_METADATA_FOLDER
  runtime_to_distributed_ids, device_ids = _read_process_metadata(metadata_path)
  assert isinstance(device_ids, list)
  logging.info(
      'From process metadata, runtime_to_distributed_ids=%s',
      runtime_to_distributed_ids,
  )
  logging.info('From process metadata, device_ids=%s', device_ids)
  consistent_mesh = emergency_multihost.consistent_restore_mesh(
      global_mesh, device_ids, runtime_to_distributed_ids
  )
  logging.info(
      'Created consistent mesh with device_ids=%s',
      consistent_mesh.device_ids.flatten(),
  )
  return consistent_mesh


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
  read_only:
    If True, the local checkpoint manager will not save any checkpoints.
  interslice_coordination_timeout:
    The timeout in seconds for interslice coordination. Essentially, this should
    represent the maximum amount of time that different slices may be "out of
    sync" by.
  """

  save_interval_steps: int = 10
  max_to_keep: int = 2
  read_only: bool = False
  interslice_coordination_timeout: int = 60


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

  step_name_format: Optional[step_lib.NameFormat[step_lib.Metadata]] = None
  cleanup_tmp_directories: bool = False
  enable_async_checkpointing: bool = True
  async_options: Optional[checkpoint_manager.AsyncOptions] = None


def _pad_steps(steps, target):
  return steps + [-1] * (target - len(steps))


def _global_list_union(
    values: Sequence[int], devices: np.ndarray
) -> np.ndarray:
  """Unions the provided values across the given devices."""
  num_hosts = devices.size // jax.local_device_count()
  num_devices_per_host = jax.local_device_count()
  slice_mesh = jax.sharding.Mesh(
      devices.reshape(num_hosts, num_devices_per_host),
      ['host', 'dev'],
  )

  sdas = []
  for d in jax.local_devices():
    sdas.append(
        jax.device_put(np.asarray(values).reshape((1, 1, len(values))), d)
    )
  sharding = jax.sharding.NamedSharding(slice_mesh, P('host', 'dev'))
  # TODO(cpgaffney): Use jax.make_array_from_process_local_data.
  g_arr = jax.make_array_from_single_device_arrays(
      (num_hosts, num_devices_per_host, len(values)), sharding, sdas
  )

  result_arr = jax.jit(
      lambda x: x,
      out_shardings=jax.sharding.NamedSharding(slice_mesh, P()),
  )(g_arr)

  return np.asarray(result_arr.addressable_data(0)).flatten()


def _global_max(value: int, devices: np.ndarray) -> int:
  """Returns the global max of a local value across given devices as a scalar."""
  unioned_values = _global_list_union([value], devices)
  return np.max(unioned_values)


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
      logger: Optional[abstract_logger.AbstractLogger] = None,
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
            active_processes=multihost.unique_processes_from_devices(
                device_array
            ),
            barrier_sync_key_prefix='local',
        ),
        enable_async_checkpointing=options.enable_async_checkpointing,
        read_only=options.local.read_only,
        single_host_load_and_broadcast=False,
    )
    self._logger = logger or standard_logger.StandardLogger()
    super().__init__(
        directory,
        options=local_options,
        metadata=metadata,
        item_handlers=state_handler,
        logger=self._logger,
    )
    self._max_to_keep = options.local.max_to_keep
    self._local_options = options.local
    self._device_array = device_array

  def _global_list_union_interslice(self, steps: Sequence[int]) -> Set[int]:
    """Shares a list of steps across slices.

    Args:
      steps: Sequence of slice-local steps.

    Returns:
      A set of steps that are known to all slices.
    """
    barrier_processes = self._options.multiprocessing_options.active_processes
    barrier_processes = list(barrier_processes)

    client = multihost.utils._get_jax_distributed_client()  # pylint: disable=protected-access
    dir_key = (
        f'steps/{multihost.counters.interslice_steps_broadcast_counter()}/'
    )
    dir_key = multihost.utils._unique_barrier_key(dir_key) + '/'  # pylint: disable=protected-access
    key = dir_key + str(multihost.process_index())
    client.key_value_set(key, ','.join([str(s) for s in steps]))

    barrier_key = 'broadcast_interslice_' + str(
        multihost.counters.interslice_steps_broadcast_counter()
    )
    barrier_key = multihost.utils._unique_barrier_key(barrier_key)  # pylint: disable=protected-access
    client.wait_at_barrier(
        barrier_key,
        process_ids=barrier_processes,
        timeout_in_ms=self._local_options.interslice_coordination_timeout
        * 1000,
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

    devices = multislice.local_slice_devices(self._device_array)
    slice_device_count = devices.size
    unioned_steps = _global_list_union(steps, devices)

    def count_and_filter(steps, num_devices):
      count = collections.Counter(steps)
      return np.asarray([k for k in count if count[k] == num_devices])

    result = count_and_filter(unioned_steps, slice_device_count)

    # here len(result) will be <= len(steps) because there are at most
    # len(steps) unique step number that appeared `slice_process_count` times in
    # an array with size [len(steps), slice_process_count]
    if len(result) > len(steps):
      raise AssertionError(
          f' len(result steps) {result} exceeded length of input steps {steps}'
      )

    logging.info('After intra-slice broadcast, found steps: %s.', result)
    return np.asarray(_pad_steps(list(result), len(steps)))

  def local_host_steps(self, read: bool) -> Sequence[int]:
    """Returns steps known to local host."""
    # List of steps present in individual host storage.
    local_steps = list(super().all_steps(read))
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
      *,
      options: Optional[CheckpointManagerOptions] = None,
      metadata: Optional[dict[str, Any]] = None,
      logger: Optional[abstract_logger.AbstractLogger] = None,
  ):
    self._local_directory = epath.Path(local_directory)
    self._persistent_directory = epath.Path(persistent_directory)
    self._logger = logger or standard_logger.StandardLogger()
    # TODO: b/330585086 - Fully support options.
    options = options or CheckpointManagerOptions()
    self._global_mesh = global_mesh
    _maybe_save_process_metadata(self._persistent_directory, self._global_mesh)

    self._abstract_state = abstract_state
    self._slice_id = multislice.process_slice_id(
        multihost.process_index(), self._global_mesh
    )

    self._local_state_handler = local_state_handler
    self._options = options
    self._metadata = metadata
    self._persistent_primary_host = multihost.runtime_to_distributed_process_id(
        global_mesh.devices[0].flat[0].process_index
    )
    self._local_primary_host = None
    if global_mesh.devices.shape[0] > 1:
      self._local_primary_host = multihost.runtime_to_distributed_process_id(
          global_mesh.devices[1].flat[0].process_index
      )
    if self._local_primary_host is None:
      raise AssertionError(
          'To use this CheckpointManager, at least 2 data-parallel slices are'
          ' needed.'
      )

    self.in_primary_slice = multislice.in_primary_slice(
        multihost.process_index(), global_mesh
    )
    self._persistent_max_to_keep = self._options.persistent.max_to_keep
    self._local_max_to_keep = self._options.local.max_to_keep

    persistent_multiprocessing_options = (
        checkpoint_manager.MultiprocessingOptions(
            primary_host=self._persistent_primary_host,
            active_processes=multihost.unique_processes_from_devices(
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
              item_handlers=PyTreeCheckpointHandler(
                  use_ocdbt=True,
                  use_zarr3=True,
                  primary_host=self._persistent_primary_host,
              ),
              logger=self._logger,
          )
      )
    else:
      self._local_checkpoint_manager = _LocalCheckpointManager(
          self._local_directory,
          self._local_state_handler,
          device_array=global_mesh.devices[1:],
          options=self._options,
          metadata=self._metadata,
          logger=self._logger,
      )

    logging.info(
        'Created emergency.CheckpointManager with slice_id=%d,'
        ' process_index=%d, jax.process_index=%d',
        self._slice_id,
        multihost.process_index(),
        jax.process_index(),
    )

  @property
  def directory(self) -> epath.Path:
    raise NotImplementedError()

  @property
  def global_mesh(self) -> jax.sharding.Mesh:
    return self._global_mesh

  def _data_per_individual_slice(
      self, data: int
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
    local_values = _global_list_union([data], self._global_mesh.devices)
    assert len(local_values) == len(jax.devices())
    num_slices = self._global_mesh.devices.shape[0]
    num_devices_per_slice = jax.device_count() // num_slices
    values_per_slice = local_values.reshape((num_slices, num_devices_per_slice))
    # Check that all rows have the same values.
    assert (values_per_slice[:, 1:] == values_per_slice[:, :-1]).all()
    return values_per_slice[:, 0].flatten()

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
    logging.info('Retrieving latest step.')
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
    if self.in_primary_slice:
      self._persistent_checkpoint_manager.reload()
    else:
      self._local_checkpoint_manager.reload()

  def reached_preemption(self, step: int) -> bool:
    """Returns True if a preemption sync point has been reached."""
    return utils.reached_preemption(step)

  def _global_max(self, value: int) -> int:
    """Returns the global max of a local value across all devices as a scalar."""
    return _global_max(value, np.asarray(jax.devices()))

  def should_save(self, step: int) -> bool:
    """Returns True if a checkpoint should be saved for the current step.

    This depends the previous step and save interval.

    Args:
      step: int

    Returns:
      True if the checkpoint should be saved.
    """
    logging.info('Checking should_save at step: %d.', step)
    if self.in_primary_slice:
      should_save = self._persistent_checkpoint_manager.should_save(step)
    else:
      should_save = self._local_checkpoint_manager.should_save(step)
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
    # TODO: b/330608746 - implement save op on different slices
    if self.in_primary_slice:
      logging.info('Maybe saving at step %d (persistent).', step)
      saved = self._persistent_checkpoint_manager.save(
          step, args=args, metrics=metrics, force=force
      )
    else:
      logging.info('Maybe saving at step %d (local).', step)
      saved = self._local_checkpoint_manager.save(
          step, args=args, metrics=metrics, force=force
      )

    return bool(self._global_max(int(saved)))

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

    has_steps = self._data_per_individual_slice(has_step_in_this_slice).tolist()

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

    shape_dtypes, tree_defs = jax.tree.flatten(self._abstract_state)

    def _get_single_slice_sharding(
        mesh: jax.sharding.Mesh,
        pspec: jax.sharding.PartitionSpec,
    ):
      slice_devices = np.asarray([self._global_mesh.devices[self._slice_id]])
      slice_mesh = jax.sharding.Mesh(slice_devices, mesh.axis_names)
      ss_sharding = jax.sharding.NamedSharding(slice_mesh, pspec)
      return ss_sharding

    single_slice_shardings = jax.tree.map(
        lambda arr: _get_single_slice_sharding(
            mesh=arr.sharding.mesh,
            pspec=arr.sharding.spec,
        ),
        self._abstract_state,
    )
    single_replica_shardings_tuple = jax.tree.flatten(single_slice_shardings)[0]

    if is_restoring_slice:
      logging.debug(
          'emergency.CheckpointManager: restoring from local checkpoint.'
      )
      ss_args = jax.tree.map(
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
      in_tree = tuple(jax.tree.flatten(single_slice_pytree)[0])
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
        return jax.tree.map(
            lambda sd: jnp.zeros(sd.shape, dtype=sd.dtype), shape_dtype_tup
        )

      zeros_pytree = create_zeros(tuple(shape_dtypes))
      in_tree = tuple(zeros_pytree)

    start_broadcast = time.time()
    shared_states, _ = multislice.broadcast_one_replica_to_all(
        in_tree,
        self._global_mesh,
        tuple(single_replica_shardings_tuple),
        0,
        is_restoring_slice,
    )
    broadcast_elapsed_s = time.time() - start_broadcast
    jax.monitoring.record_event_duration_secs(
        '/orbax/emergency/checkpoint/read/broadcast_duration_secs',
        broadcast_elapsed_s,
    )

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
    logging.info('Closing CheckpointManager.')
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
