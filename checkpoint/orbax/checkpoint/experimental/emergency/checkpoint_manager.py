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
import enum
import functools
import json
import operator
import time
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Set

from absl import logging
from etils import epath
from etils import epy
import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint import abstract_checkpoint_manager
from orbax.checkpoint import args as args_lib
from orbax.checkpoint import checkpoint_manager
from orbax.checkpoint import utils
from orbax.checkpoint._src.handlers import pytree_checkpoint_handler
from orbax.checkpoint._src.multihost import counters
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.multihost import multislice
from orbax.checkpoint._src.path import step as step_lib
from orbax.checkpoint._src.serialization import type_handlers
from orbax.checkpoint.experimental.emergency import multihost as emergency_multihost
from orbax.checkpoint.logging import abstract_logger
from orbax.checkpoint.logging import standard_logger
from orbax.checkpoint.logging import step_statistics
from typing_extensions import Self  # for Python version < 3.11


PyTree = checkpoint_manager.PyTree
CheckpointHandler = checkpoint_manager.CheckpointHandler
P = jax.sharding.PartitionSpec
PyTreeCheckpointHandler = pytree_checkpoint_handler.PyTreeCheckpointHandler
unique_barrier_key = multihost._unique_barrier_key  # pylint: disable=protected-access

_PROCESS_METADATA_FOLDER = 'process_metadata'
_PROCESS_METADATA_FILE_NAME = 'process_metadata.json'
_GLOBAL_PROCESS_METADATA_FILE_NAME = 'global_process_metadata.json'
_MESH_METADATA_FILE_NAME = 'mesh_metadata.json'

_PRIMARY_REPLICA_ID = 0
_SECONDARY_REPLICA_ID = 1


def _write_process_metadata(path: epath.Path, mesh: jax.sharding.Mesh):
  """Write process metadata to the given path."""
  logging.info('Saving process index metadata at %s', path)

  if multihost.process_index() == 0:
    path.mkdir(parents=False, exist_ok=False)
    runtime_to_distributed_ids = (
        emergency_multihost.runtime_to_distributed_ids()
    )
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


def _should_restore_mesh_from_metadata(path: epath.Path) -> bool:
  metadata_path = path / _PROCESS_METADATA_FOLDER
  return metadata_path.exists()


def _consistent_restore_mesh_from_metadata(
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


def local_checkpoint_handler() -> PyTreeCheckpointHandler:
  """Create a PyTreeCheckpointHandler for local checkpoints."""
  local_registry = type_handlers.create_type_handler_registry(
      (
          jax.Array,
          type_handlers.ArrayHandler(primary_host=None, replica_id=None),
      ),
  )
  return PyTreeCheckpointHandler(
      use_ocdbt=True,
      use_zarr3=True,
      multiprocessing_options=checkpoint_manager.MultiprocessingOptions(
          primary_host=None,
      ),
      type_handler_registry=local_registry,
  )


def _local_checkpoint_handler(
    multiprocessing_options: checkpoint_manager.MultiprocessingOptions,
) -> PyTreeCheckpointHandler:
  """Create a PyTreeCheckpointHandler for local checkpoints."""
  if multiprocessing_options.primary_host is not None:
    raise ValueError(
        'multiprocessing_options.primary_host must be set to None for local'
        ' checkpoints.'
    )
  local_registry = type_handlers.create_type_handler_registry(
      (
          jax.Array,
          type_handlers.ArrayHandler(primary_host=None, replica_id=None),
      ),
  )
  return PyTreeCheckpointHandler(
      use_ocdbt=True,
      use_zarr3=True,
      multiprocessing_options=multiprocessing_options,
      type_handler_registry=local_registry,
  )


def _persistent_checkpoint_handler(
    multiprocessing_options: checkpoint_manager.MultiprocessingOptions,
) -> PyTreeCheckpointHandler:
  """Create a PyTreeCheckpointHandler for local checkpoints."""
  # TODO(b/372291557) Selection of replica_id=0 could be problematic if we can't
  # guarantee that the primary slice (in which the primary process is present)
  # always has the shard with shard.replica_id=0 for all available arrays.
  registry = type_handlers.create_type_handler_registry(
      (
          jax.Array,
          type_handlers.ArrayHandler(
              primary_host=multiprocessing_options.primary_host, replica_id=0
          ),
      ),
  )
  return PyTreeCheckpointHandler(
      use_ocdbt=True,
      use_zarr3=True,
      multiprocessing_options=multiprocessing_options,
      type_handler_registry=registry,
  )


@dataclasses.dataclass
class LocalCheckpointOptions:
  """Optional CheckpointManager arguments for saving local checkpoints.

  save_interval_steps:
    The interval at which checkpoints should be saved to local storage.
    Ensures checkpoints will only be saved every m steps. Defaults to 10.
  max_to_keep:
    Specifies the maximum number of local checkpoints to
    keep aside from the one currently being written. Older checkpoints are
    removed. When set, no more than (`max_to_keep` + 1) checkpoints will be
    present at any one time.
  read_only:
    If True, the local checkpoint manager will not save any checkpoints.
  should_save_fn:
    Predicate callable to check if given step can be saved. This callable
    accepts step number and optional latest step number as param and returns
    bool. If present then `save_interval_steps` and `save_on_steps` options are
    ignored.
  """

  save_interval_steps: int = 10
  max_to_keep: int = 1
  read_only: bool = False
  should_save_fn: Optional[Callable[[int, Optional[int]], bool]] = None

  debug_use_full_global_mesh: bool = False


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
  should_save_fn:
    Predicate callable to check if given step can be saved. This callable
    accepts step number and optional latest step number as param and returns
    bool. If present then `save_interval_steps` and `save_on_steps` options are
    ignored.
  """

  save_interval_steps: int = 1000
  max_to_keep: Optional[int] = None
  should_save_fn: Optional[Callable[[int, Optional[int]], bool]] = None


@dataclasses.dataclass
class MultiprocessingOptions:
  """Options used to configure multiprocessing behavior.

  coordination_timeout_secs: The timeout in seconds for inter-process
    coordination. Essentially, this should represent the maximum amount of time
    that different processes can be "out of sync" by.
  """

  coordination_timeout_secs: int = 120


@dataclasses.dataclass
class CheckpointManagerOptions:
  """Optional arguments for CheckpointManager.

  local:
    Options relevant to the local checkpoints.
    See `LocalCheckpointOptions`.
  persistent:
    Options relevant to the persistent checkpoints. See
    `PersistentCheckpointOptions`.
  replica_axis_index:
    The index of the replica axis in the global mesh.
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

  replica_axis_index: int = 0

  step_name_format: step_lib.NameFormat[step_lib.Metadata] = (
      step_lib.standard_name_format()
  )
  cleanup_tmp_directories: bool = False
  enable_async_checkpointing: bool = True
  async_options: Optional[checkpoint_manager.AsyncOptions] = None
  multiprocessing_options: Optional[MultiprocessingOptions] = None


class _BarrierIdentifier(enum.Enum):
  """Identifies the barrier being run."""

  GLOBAL_MAX = 'global_max'
  LOCAL_ALL_STEPS = 'local_all_steps'
  FIND_COMPLETE_SLICE = 'find_complete_slice'

  def get_counter(self) -> str:
    if self.name == self.GLOBAL_MAX.name:
      return counters.global_max_broadcast_counter()
    elif self.name == self.LOCAL_ALL_STEPS.name:
      return counters.local_all_steps_broadcast_counter()
    elif self.name == self.FIND_COMPLETE_SLICE.name:
      return counters.find_complete_slice_broadcast_counter()
    else:
      raise ValueError(f'Unknown barrier identifier: {self.name}')


def _common_values_per_slice(
    per_process_values: Dict[int, Set[int]],
    global_mesh: jax.sharding.Mesh,
    *,
    replica_axis_index: int,
) -> Dict[int, Set[int]]:
  """Obtains values shared in common across all processes in each slice.

  Args:
    per_process_values: A mapping of process index to a list of values local to
      that process.
    global_mesh: The global mesh.
    replica_axis_index: The index of the replica axis in the global mesh.

  Returns:
    A mapping of slice index to a set of values shared in common across all
    processes in that slice. A value appearing in one process but not another
    in the same slice will not appear in the output.
  """
  total_num_slices = multislice.slice_count(
      global_mesh, replica_axis_index=replica_axis_index
  )
  num_processes_per_slice = (
      global_mesh.devices.size // total_num_slices // jax.local_device_count()
  )
  per_slice_values = collections.defaultdict(list)
  for pid, values in per_process_values.items():
    slice_id = multislice.process_slice_id(
        pid, global_mesh, replica_axis_index=replica_axis_index
    )
    per_slice_values[slice_id].extend(values)

  for slice_id, values in per_slice_values.items():
    counter = collections.Counter(values)
    common_values = [
        k for k in counter if counter[k] == num_processes_per_slice
    ]
    # Here `len(common_values)`` will be less than or equal to `len(values)`
    # because a value can only appear in `common_values` if it occurs
    # `num_processes_per_slice` times in `values`.
    if len(common_values) > len(values):
      raise AssertionError(
          f' len(common_values) ({common_values}) exceeded length of input'
          f' values ({values}).'
      )
    per_slice_values[slice_id] = common_values

  return {k: set(v) for k, v in per_slice_values.items()}


def _pad_steps(steps, target):
  return steps + [-1] * (target - len(steps))


def _process_local_to_global(
    values: Set[int],
    barrier_processes: Set[int],
    *,
    timeout: int,
    barrier_id: _BarrierIdentifier,
    slice_id: Optional[int] = None,
) -> Dict[int, Set[int]]:
  """Shares a sequence of host-local integers across given processes.

  Args:
    values: A set of local values. Each process has its own set of values.
    barrier_processes: A set of processes to share the set of values with.
    timeout: The timeout in seconds for inter-process coordination.
    barrier_id: Barrier identifier.
    slice_id: The slice id. Only needed if multiple slices need to run the same
      barrier in parallel, but only sync intra-slice, not inter-slice.

  Returns:
    A mapping of process index to the sequence of local values on that process.
    The result will have an entry for every process in `barrier_processes`.
  """
  barrier_name = (
      f'{barrier_id.name}_{slice_id}' if slice_id else barrier_id.name
  )
  # Need to include barrier processes because there can be identical calls, just
  # with different participating processes.
  barrier_processes_as_string = ''.join(str(p) for p in barrier_processes)
  barrier_name_and_id = (
      f'{barrier_name}_{barrier_id.get_counter()}_{barrier_processes_as_string}'
  )

  client = multihost.get_jax_distributed_client()
  broadcast_dir_key = f'broadcast_{barrier_name_and_id}/'
  broadcast_dir_key = unique_barrier_key(broadcast_dir_key) + '/'
  broadcast_key = broadcast_dir_key + str(multihost.process_index())
  client.key_value_set(broadcast_key, ','.join([str(s) for s in values]))

  barrier_key = f'barrier_{barrier_name_and_id}'
  barrier_key = unique_barrier_key(barrier_key)
  logging.vlog(
      1,
      '[process=%s] Waiting at barrier %s',
      multihost.process_index(),
      barrier_key,
  )
  logging.vlog(
      1,
      '[process=%s] Barrier processes: %s',
      multihost.process_index(),
      barrier_processes,
  )
  client.wait_at_barrier(
      barrier_key,
      process_ids=list(barrier_processes),
      timeout_in_ms=timeout * 1000,
  )

  per_process_values = {
      int(k.split('/')[-1]): {int(s) for s in v.split(',')} if v else set()
      for k, v in client.key_value_dir_get(broadcast_dir_key)
  }
  assert set(per_process_values.keys()) == barrier_processes
  return per_process_values


def _all_devices_excepting_slice(
    devices: np.ndarray,
    *,
    replica_id: int = 0,
    replica_axis_index: int = 0,
) -> np.ndarray:
  if hasattr(jax.devices()[0], 'slice_index'):
    get_slice_id = np.vectorize(lambda x: x.slice_index)
    return devices[get_slice_id(devices) != replica_id]
  else:
    return np.delete(devices, replica_id, axis=replica_axis_index)


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
  # TODO: b/330585086 - Support arbitrary items beyond state. We will have
  # to evaluate whether arbitrary items can be a good fit for local
  # checkpointing, given restore+broadcast requirements.
  def __init__(
      self,
      directory: epath.PathLike,
      global_mesh: jax.sharding.Mesh,
      *,
      primary_replica_id: int = _PRIMARY_REPLICA_ID,
      options: Optional[CheckpointManagerOptions] = None,
      metadata: Optional[dict[str, Any]] = None,
      logger: Optional[abstract_logger.AbstractLogger] = None,
  ):
    # TODO: b/330585086 - Fully support options.
    options = options or CheckpointManagerOptions()
    self._global_mesh = global_mesh
    self._replica_axis_index = options.replica_axis_index

    devices = np.asarray(self._global_mesh.devices)
    # Select all devices except those belonging to the primary replica.
    if not options.local.debug_use_full_global_mesh:
      devices = _all_devices_excepting_slice(
          devices,
          replica_id=primary_replica_id,
          replica_axis_index=self._replica_axis_index,
      )

    self._active_processes = multihost.unique_processes_from_devices(devices)
    multiprocessing_options = checkpoint_manager.MultiprocessingOptions(
        primary_host=None,
        active_processes=self._active_processes,
        barrier_sync_key_prefix='local',
    )
    local_options = checkpoint_manager.CheckpointManagerOptions(
        save_interval_steps=options.local.save_interval_steps,
        max_to_keep=options.local.max_to_keep,
        step_name_format=options.step_name_format,
        should_save_fn=options.local.should_save_fn,
        create=False,
        # we always clean up local tmp directories explicitly
        cleanup_tmp_directories=False,
        async_options=options.async_options,
        multiprocessing_options=multiprocessing_options,
        enable_async_checkpointing=options.enable_async_checkpointing,
        read_only=options.local.read_only,
        single_host_load_and_broadcast=False,
        # enable_background_delete set to False to ensure gc is done before save
        enable_background_delete=False,
    )
    self._logger = logger or standard_logger.StandardLogger()
    self._coordination_timeout_secs = (
        options.multiprocessing_options or MultiprocessingOptions()
    ).coordination_timeout_secs
    super().__init__(
        directory,
        options=local_options,
        metadata=metadata,
        item_handlers=_local_checkpoint_handler(multiprocessing_options),
        logger=self._logger,
    )

    # Remove steps that might be left over from previous runs.
    steps_to_remove = self._get_old_steps_to_remove()
    self._checkpoints = [
        info for info in self._checkpoints if info.step not in steps_to_remove
    ]
    self._checkpoint_deleter.delete_steps(steps_to_remove)

    # Set additional properties.
    self._max_to_keep = options.local.max_to_keep
    self._local_options = options.local
    self._steps = list(self.all_steps(read=True))

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
      logging.error(
          'local_step on host %d exceeded `max_to_keep` %d',
          multihost.process_index(),
          self._max_to_keep,
      )

    return _pad_steps(local_steps, self._max_to_keep)

  def all_steps(self, read: bool = False) -> Sequence[int]:
    """Returns all steps tracked by the manager.

    Includes steps located in local storage.

    Args:
      read: If True, forces a read directly from the storage location.
        Otherwise, a cached result can be returned.

    Returns:
      A sequence of steps (integers)
    """
    if read:
      local_steps = set(self.local_host_steps(read))
      # Per-process mapping of the local steps that each process knows about.
      per_process_steps = _process_local_to_global(
          local_steps,
          self._active_processes,
          timeout=self._coordination_timeout_secs,
          barrier_id=_BarrierIdentifier.LOCAL_ALL_STEPS,
      )
      slice_id = multislice.process_slice_id(
          multihost.process_index(),
          self._global_mesh,
          replica_axis_index=self._replica_axis_index,
      )
      per_slice_steps = _common_values_per_slice(
          per_process_steps,
          self._global_mesh,
          replica_axis_index=self._replica_axis_index,
      )
      logging.info(
          'After broadcast, found steps %s shared between local slice'
          ' processes.',
          per_slice_steps[slice_id],
      )
      steps = functools.reduce(operator.ior, per_slice_steps.values(), set())
      self._steps = [x for x in steps if x != -1]
    return self._steps

  def latest_step(self) -> Optional[int]:
    """Returns the latest step saved in the local storage.

    Returns None if no steps have been saved.

    Returns:
      A step (int) or None if no steps are present.
    """
    return max(self._steps) if self._steps else None

  def save(
      self,
      step: int,
      args: Optional[args_lib.CheckpointArgs] = None,
      metrics: Optional[PyTree] = None,
      force: Optional[bool] = False,
  ) -> bool:
    """Saves the checkpoint at the given step."""
    saved = super().save(step, args=args, metrics=metrics, force=force)
    if saved:
      self._steps.append(step)
      self._steps = self._steps[-self._max_to_keep :]
    return saved

  def reload(self):
    """Reloads internal properties.

    refreshes the cached list of globally available local checkpointed steps.
    """
    super().reload()
    self._steps = list(self.all_steps(read=True))


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

  Usage::
    options = CheckpointManagerOptions(
        local=LocalCheckpointOptions(save_interval_steps=2, max_to_keep=2),
        persistent=PersistentCheckpointOptions(
            save_interval_steps=5, max_to_keep=3
        ),
        enable_async_checkpointing=use_async,
    )
    return CheckpointManager(
        local_directory=local_directory,
        persistent_directory=persistent_directory,
        global_mesh=global_mesh,
        abstract_state=abstract_state,
        options=options,
    )
  """

  # TODO: b/330585086 - Allow configuration of global mesh describing slices.
  # Validate against global meshes used for arrays in state.
  # TODO: b/330585086 - Support arbitrary items beyond state. We will have
  # to evaluate whether arbitrary items can be a good fit for local
  # checkpointing, given restore+broadcast requirements.
  def __init__(
      self,
      local_directory: epath.PathLike,
      persistent_directory: epath.PathLike,
      global_mesh: jax.sharding.Mesh,
      abstract_state: PyTree,  # a single PyTree describing the state structure
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
    self._replica_axis_index = options.replica_axis_index
    self._global_mesh = global_mesh
    logging.info(
        'Configured emergency.CheckpointManager with replica_axis_index=%d,'
        ' corresponding to "%s" in the global mesh.',
        self._replica_axis_index,
        self._global_mesh.axis_names[self._replica_axis_index],
    )
    _maybe_save_process_metadata(self._persistent_directory, self._global_mesh)

    self._abstract_state = abstract_state
    self._slice_id = multislice.process_slice_id(
        multihost.process_index(),
        self._global_mesh,
        replica_axis_index=self._replica_axis_index,
    )
    self._options = options
    self._metadata = metadata

    self._persistent_max_to_keep = self._options.persistent.max_to_keep
    self._local_max_to_keep = self._options.local.max_to_keep
    self._coordination_timeout_secs = (
        options.multiprocessing_options or MultiprocessingOptions()
    ).coordination_timeout_secs

    if len(global_mesh.devices.shape) <= self._replica_axis_index:
      raise AssertionError(
          f'replica_axis_index {self._replica_axis_index} is out of bound for'
          f' global_mesh.devices.shape {global_mesh.devices.shape}'
      )
    if (
        multislice.slice_count(
            global_mesh, replica_axis_index=self._replica_axis_index
        )
        <= 1
    ):
      raise AssertionError(
          'To use this CheckpointManager, at least 2 data-parallel replicas are'
          ' needed.'
      )

    primary_replica_id = _PRIMARY_REPLICA_ID
    secondary_replica_id = _SECONDARY_REPLICA_ID

    self._persistent_primary_host = multislice.primary_process_in_slice(
        self._global_mesh,
        replica_axis_index=self._replica_axis_index,
        replica_id=primary_replica_id,
    )
    self._local_primary_host = multislice.primary_process_in_slice(
        self._global_mesh,
        replica_axis_index=self._replica_axis_index,
        replica_id=secondary_replica_id,
    )
    self.in_primary_slice = multislice.in_slice(
        multihost.process_index(),
        global_mesh,
        replica_axis_index=self._replica_axis_index,
        replica_id=primary_replica_id,
    )

    if self.in_primary_slice:
      persistent_multiprocessing_options = (
          checkpoint_manager.MultiprocessingOptions(
              primary_host=self._persistent_primary_host,
              active_processes=multihost.unique_processes_from_devices(
                  multislice.slice_devices(
                      self._global_mesh,
                      replica_axis_index=self._replica_axis_index,
                      replica_id=primary_replica_id,
                  )
              ),
              barrier_sync_key_prefix='persistent',
          )
      )
      self._persistent_checkpoint_manager = (
          self._make_persistent_checkpoint_manager(
              persistent_multiprocessing_options
          )
      )
    else:
      self._local_checkpoint_manager = self._make_local_checkpoint_manager(
          primary_replica_id
      )

    self._local_steps = []
    self._persistent_steps = []
    # clean up tmp directories in ram
    self._cleanup_local_tmp_directories()

    # Initialize step cache.
    self.all_steps(read=True)

    # Compile function for global broadcast.
    slice_mesh = jax.sharding.Mesh(
        np.asarray(jax.devices()).reshape(
            multihost.process_count(), jax.local_device_count()
        ),
        ['host', 'dev'],
    )
    self._global_broadcast_fn = jax.jit(
        lambda x: x,
        out_shardings=jax.sharding.NamedSharding(slice_mesh, P()),
    )

    logging.info(
        'Created emergency.CheckpointManager with slice_id=%d,'
        ' process_index=%d, jax.process_index=%d',
        self._slice_id,
        multihost.process_index(),
        jax.process_index(),
    )

  def _cleanup_local_tmp_directories(self):
    logging.info(
        'Cleaning up existing temporary directories at %s.',
        self._local_directory,
    )
    tmp_files = step_lib.tmp_checkpoints(self._local_directory)
    for tmp_file in tmp_files:
      logging.info('Deleting temporary checkpoint: %s.', tmp_file)
      (self._local_directory / tmp_file).rmtree()

  def _make_persistent_checkpoint_manager(
      self,
      persistent_multiprocessing_options: checkpoint_manager.MultiprocessingOptions,
  ) -> checkpoint_manager.CheckpointManager:
    persistent_options = checkpoint_manager.CheckpointManagerOptions(
        save_interval_steps=self._options.persistent.save_interval_steps,
        max_to_keep=self._persistent_max_to_keep,
        step_name_format=self._options.step_name_format,
        create=False,
        cleanup_tmp_directories=self._options.cleanup_tmp_directories,
        async_options=self._options.async_options,
        multiprocessing_options=persistent_multiprocessing_options,
        enable_async_checkpointing=self._options.enable_async_checkpointing,
        should_save_fn=self._options.persistent.should_save_fn,
    )
    return checkpoint_manager.CheckpointManager(
        self._persistent_directory,
        options=persistent_options,
        metadata=self._metadata,
        item_handlers=_persistent_checkpoint_handler(
            persistent_multiprocessing_options
        ),
        logger=self._logger,
    )

  def _make_local_checkpoint_manager(
      self, primary_replica_id: int = _PRIMARY_REPLICA_ID
  ) -> _LocalCheckpointManager:
    return _LocalCheckpointManager(
        self._local_directory,
        global_mesh=self._global_mesh,
        primary_replica_id=primary_replica_id,
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
    if read:
      per_slice_local_steps = self._get_per_slice_local_steps()
      self._local_steps = list(set.union(*per_slice_local_steps.values()))
      self._persistent_steps = step_lib.checkpoint_steps(
          self._persistent_directory
      )
    return list(set(self._local_steps) | set(self._persistent_steps))

  def latest_step(self) -> Optional[int]:
    """Returns the latest step saved.

    Includes steps located in local as well as persistent storage.

    Returns None if no steps have been saved.

    Returns:
      A step (int) or None if no steps are present.
    """
    logging.info('Retrieving latest step.')
    all_steps = self.all_steps()
    return max(all_steps) if all_steps else None

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

  def _global_max(self, values: list[int]) -> list[int]:
    """Returns the global max of local values across all processes as a scalar.

    Uses JAX-based broadcasting to ensure greater performance at scale.

    Args:
      values: A list of integers.

    Args:
      values: Max of the values across all processes.
    """
    num_hosts = multihost.process_count()
    num_devices_per_host = jax.local_device_count()
    slice_mesh = jax.sharding.Mesh(
        np.asarray(jax.devices()).reshape(num_hosts, num_devices_per_host),
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
    result_arr = self._global_broadcast_fn(g_arr)
    result_arr = np.asarray(result_arr.addressable_data(0))

    # Checks that for every host, values are equal across local devices.
    assert (
        np.sum(result_arr, axis=1) / num_devices_per_host == result_arr[:, 0, :]
    ).all()
    # Select values from first device and compute max for each value across
    # hosts.
    return list(np.max(result_arr[:, 0, :], axis=0).astype(int))

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
    return bool(self._global_max([int(should_save)])[0])

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
    persistent_saved = False
    local_saved = False
    if self.in_primary_slice:
      logging.info('Maybe saving at step %d (persistent).', step)
      persistent_saved = self._persistent_checkpoint_manager.save(
          step, args=args, metrics=metrics, force=force
      )
    else:
      logging.info('Maybe saving at step %d (local).', step)
      local_saved = self._local_checkpoint_manager.save(
          step, args=args, metrics=metrics, force=force
      )

    start = time.time()
    saved = tuple(
        bool(e)
        for e in self._global_max([int(persistent_saved), int(local_saved)])
    )
    persistent_saved, local_saved = saved
    logging.info('Broadcast `saved` bool in %f seconds.', time.time() - start)

    if persistent_saved:
      self._persistent_steps.append(step)
      if self._persistent_max_to_keep is not None:
        self._persistent_steps = self._persistent_steps[
            -self._persistent_max_to_keep :
        ]
    if local_saved:
      self._local_steps.append(step)
      self._local_steps = self._local_steps[-self._local_max_to_keep :]

    return persistent_saved or local_saved

  def _get_per_slice_local_steps(self) -> Dict[int, Set[int]]:
    local_steps = set(step_lib.checkpoint_steps(self._local_directory))
    logging.info(
        'Found steps: %s in local host storage: %s.',
        local_steps,
        self._local_directory,
    )

    # The steps that each process actually has in local storage.
    per_process_steps = _process_local_to_global(
        local_steps,
        set(range(jax.process_count())),
        timeout=self._coordination_timeout_secs,
        barrier_id=_BarrierIdentifier.FIND_COMPLETE_SLICE,
    )
    logging.vlog(1, 'per_process_steps=%s', per_process_steps)
    per_slice_steps = _common_values_per_slice(
        per_process_steps,
        self._global_mesh,
        replica_axis_index=self._replica_axis_index,
    )
    logging.vlog(1, 'per_slice_steps=%s', per_slice_steps)
    return per_slice_steps

  def _find_slice_with_complete_local_checkpoint(self, step: int) -> int:
    """Return the slice id which has the step."""
    per_slice_steps = self._get_per_slice_local_steps()

    for slice_id, steps in per_slice_steps.items():
      if step in steps:
        return slice_id
    return -1

  def _restore_from_local(
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

    is_restoring_slice = restoring_slice_id == self._slice_id
    step_stats.is_restoring_slice = is_restoring_slice
    step_stats.in_primary_slice = self.in_primary_slice

    shape_dtypes, tree_defs = jax.tree.flatten(self._abstract_state)

    def _get_single_slice_sharding(
        mesh: jax.sharding.Mesh,
        pspec: jax.sharding.PartitionSpec,
        replica_id: int,
        replica_axis_index: int,
    ):
      slice_devices = multislice.slice_devices(
          mesh,
          replica_id=replica_id,
          replica_axis_index=replica_axis_index,
      )
      single_slice_mesh_shape = [
          1 if i == replica_axis_index else d
          for i, d in enumerate(mesh.devices.shape)
      ]
      slice_mesh = jax.sharding.Mesh(
          slice_devices.reshape(single_slice_mesh_shape), mesh.axis_names
      )
      return jax.sharding.NamedSharding(slice_mesh, pspec)

    original_single_slice_shardings = jax.tree.map(
        lambda arr: _get_single_slice_sharding(
            mesh=arr.sharding.mesh,
            pspec=arr.sharding.spec,
            replica_id=self._slice_id,
            replica_axis_index=self._replica_axis_index,
        ),
        self._abstract_state,
    )
    original_single_slice_shardings_tuple = tuple(
        jax.tree.flatten(original_single_slice_shardings)[0]
    )

    if is_restoring_slice:
      logging.vlog(
          1, 'emergency.CheckpointManager: restoring from local checkpoint.'
      )
      # Each device has its own specific local shard.
      # Naive example:
      #  Before restart: device X (with id 0) saves shard 0.
      #  After restart:  device X (with new id 1) reads shard 0.
      # To ensure the same device X (with different software ids) reads the
      # same locally available shard and represent it in the correct index
      # within the global jax.Array, we use the `restore_mesh`.

      # More context on `restore_mesh`:
      # 1. We can think of mesh as being backed by a list of devices.
      # 2. Default mesh follows the default device id order [0, ..., n-1]. Or
      #    the user may permute it according to their needs.
      # 3. After restart, the user will construct the same software mesh as (2).
      # 4. But a given hardware device may change its id because of scheduler
      #    or runtime quirks.
      # 5. Goal: construct the mesh with the same hardware device order as
      #    before restart, that may not follow the current software ids.
      # 5. Thus, we shuffle the device order within the mesh by checking how
      #    each device's software ids changed across restarts.
      restore_mesh = self._global_mesh
      if _should_restore_mesh_from_metadata(self._persistent_directory):
        logging.info(
            'Found consistent_restore_mesh, using it for local restoration'
        )
        restore_mesh = _consistent_restore_mesh_from_metadata(
            self._persistent_directory, self._global_mesh
        )

      restoring_processes = multihost.unique_processes_from_devices(
          multislice.slice_devices(
              restore_mesh,
              replica_id=self._slice_id,
              replica_axis_index=self._replica_axis_index,
          )
      )
      multiprocessing_options = checkpoint_manager.MultiprocessingOptions(
          primary_host=None,
          active_processes=restoring_processes,
          barrier_sync_key_prefix='local_restoring_slice',
      )
      local_state_handler = _local_checkpoint_handler(multiprocessing_options)

      restore_single_slice_shardings = jax.tree.map(
          lambda arr: _get_single_slice_sharding(
              mesh=restore_mesh,
              pspec=arr.sharding.spec,
              replica_id=self._slice_id,
              replica_axis_index=self._replica_axis_index,
          ),
          self._abstract_state,
      )
      restore_args = jax.tree.map(
          lambda restore_shard, arr: type_handlers.ArrayRestoreArgs(
              sharding=restore_shard,
              global_shape=arr.shape,  # single-slice sharding
          ),
          restore_single_slice_shardings,
          self._abstract_state,
      )
      restore_directory = self._options.step_name_format.find_step(
          epath.Path(directory or self._local_directory), step
      ).path
      step_stats.directory = str(restore_directory)

      # Directly use CheckpointHandler to restore. This is undesirable, but
      # allows us to avoid barrier issues that occur when calling
      # LocalCheckpointManager a different number of times on the non-primary
      # slices, which leads to
      # _module_unique_count getting out of sync.
      logging.vlog(
          1,
          'Restoring from %s',
          restore_directory / checkpoint_manager.DEFAULT_ITEM_NAME,
      )
      step_stats.checkpointer_start_time = time.time()
      single_slice_pytree = local_state_handler.restore(
          restore_directory / checkpoint_manager.DEFAULT_ITEM_NAME,
          args=dataclasses.replace(args, restore_args=restore_args),
      )
      step_stats.checkpointer_duration_secs = (
          time.time() - step_stats.checkpointer_start_time
      )
      in_tree = tuple(jax.tree.flatten(single_slice_pytree)[0])
      if not np.array_equal(
          restore_mesh.device_ids, self._global_mesh.device_ids
      ):
        # User-provided mesh is usually optimized for performance.
        # But we permuted the original mesh so that we can read each locally
        # available shard correctly. This may cause performance issues.

        # Thus, we re-shard the array to follow the original mesh and layout.
        in_tree = self._consistent_restore_mesh_to_global_mesh(
            in_tree, original_single_slice_shardings_tuple
        )
    else:
      logging.vlog(
          1,
          'emergency.CheckpointManager: secondary slice, create zeros and'
          ' wait for broacast.',
      )

      @functools.partial(
          jax.jit,
          static_argnums=0,
          out_shardings=original_single_slice_shardings_tuple,
      )
      def create_zeros(shape_dtype_tup):
        return jax.tree.map(
            lambda sd: jnp.zeros(sd.shape, dtype=sd.dtype), shape_dtype_tup
        )

      zeros_pytree = create_zeros(tuple(shape_dtypes))
      in_tree = tuple(zeros_pytree)

    multihost.sync_global_processes('local_restore_pre_broadcast')

    start_broadcast = time.time()
    shared_states, _ = multislice.broadcast_one_replica_to_all(
        in_tree,
        self._global_mesh,
        replica_axis_index=self._replica_axis_index,
        is_source=is_restoring_slice,
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

  def _consistent_restore_mesh_to_global_mesh(
      self,
      original_state: PyTree,
      desired_slice_shardings,
  ) -> Any:
    """Transfers from consistent restore mesh to global mesh."""
    logging.info('Transferring from consistent restore mesh to global mesh')

    start_transfer = time.time()
    resharded_state = jax.device_put(
        original_state, desired_slice_shardings, donate=True
    )
    transfer_elapsed_s = time.time() - start_transfer
    logging.info(
        'Finished transferring from consistent restore mesh to global mesh'
        ' in %.2fs',
        transfer_elapsed_s,
    )
    jax.monitoring.record_event_duration_secs(
        '/orbax/emergency/checkpoint/read/transfer_global_shard_duration_secs',
        transfer_elapsed_s,
    )

    return resharded_state

  def _restore_from_persistent(
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
      try:
        return pcm.restore(step, args=args, directory=directory)
      except FileNotFoundError as e:
        raise FileNotFoundError(
            'No steps found in either local or persistent storage when'
            f' requesting restoration of step {step}.'
        ) from e

  def restore(
      self,
      step: int,
      args: Optional[args_lib.CheckpointArgs] = None,
      directory: Optional[epath.PathLike] = None,
  ) -> Any:
    if step is None:
      raise ValueError('Step must be provided for restore. Found None.')
    logging.info('Restoring at step %d.', step)
    restoring_slice_id = self._find_slice_with_complete_local_checkpoint(step)
    if restoring_slice_id > -1:
      # restore from LCM
      return self._restore_from_local(
          step=step,
          restoring_slice_id=restoring_slice_id,
          args=args,
          directory=directory,
      )

    return self._restore_from_persistent(
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
    self.wait_until_finished()
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
