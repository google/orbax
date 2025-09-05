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

"""A class providing emergency checkpoint management.


WARNING: This class is experimental; do not use without specific approval.

NOTE: All classes within this module should be called across all *relevant*
processes. CheckpointManager is designed to be created and called across
*every* process. LocalCheckpointManager is designed to be created and called
across every process within *non-primary* slices. Similarly, a CheckpointManager
intended to work only with the persistent checkpoint on the primary slice should
always be called across all processes within the primary slice.
"""

import asyncio
import collections
import dataclasses
import functools
import time
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Set, Union
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
from orbax.checkpoint import checkpoint_utils
from orbax.checkpoint._src.checkpoint_managers import save_decision_policy as save_decision_policy_lib
from orbax.checkpoint._src.handlers import handler_registration
from orbax.checkpoint._src.handlers import pytree_checkpoint_handler
from orbax.checkpoint._src.logging import abstract_logger
from orbax.checkpoint._src.logging import standard_logger
from orbax.checkpoint._src.logging import step_statistics
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.multihost import multislice
from orbax.checkpoint._src.path import step as step_lib
from orbax.checkpoint._src.serialization import type_handlers
from orbax.checkpoint.experimental.emergency import local_checkpoint_data_debugging
from orbax.checkpoint.experimental.emergency import mesh_consistency
from orbax.checkpoint.experimental.emergency import process_metadata_checkpoint_handler
from typing_extensions import override
from typing_extensions import Self  # for Python version < 3.11


PyTree = checkpoint_manager.PyTree
CheckpointHandler = checkpoint_manager.CheckpointHandler
CheckpointHandlersDict = Dict[str, CheckpointHandler]
P = jax.sharding.PartitionSpec
PyTreeCheckpointHandler = pytree_checkpoint_handler.PyTreeCheckpointHandler
ProcessMetadataCheckpointHandler = (
    process_metadata_checkpoint_handler.ProcessMetadataCheckpointHandler
)
ChunkId = local_checkpoint_data_debugging.ChunkId
get_present_and_missing_chunks = (
    local_checkpoint_data_debugging.get_present_and_missing_chunks
)
RootMetadata = checkpoint_manager.RootMetadata
StepMetadata = checkpoint_manager.StepMetadata

_PRIMARY_REPLICA_ID = 0
_SECONDARY_REPLICA_ID = 1
_STATE_ITEM_NAME = 'state'
_PROCESS_METADATA_NAME = 'process_metadata'
_DATASET_ITEM_NAME = 'dataset'


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
          type_handlers.ArrayHandler(
              primary_host=None,
              replica_id=None,
              use_replica_parallel=False,
          ),
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
              primary_host=multiprocessing_options.primary_host,
              replica_id=0,
              use_replica_parallel=False,
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
  save_decision_policy: An object used to determine when a checkpoint should be
    saved. If provided, overrides any other options dealing with this subject,
    including `save_interval_steps`, `save_on_steps`, and `should_save_fn`, and
    is the sole means of determining when a checkpoint should be saved. If not
    provided, these other options are used instead. Prefer to use this option
    over others.
  """

  save_interval_steps: int = 10
  max_to_keep: int = 1
  read_only: bool = False
  should_save_fn: Optional[Callable[[int, Optional[int]], bool]] = None
  save_decision_policy: Optional[
      save_decision_policy_lib.SaveDecisionPolicy
  ] = None

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
  keep_period:
    If set, any existing checkpoints matching checkpoint_step % keep_period == 0
    will not be deleted.
  should_save_fn:
    Predicate callable to check if given step can be saved. This callable
    accepts step number and optional latest step number as param and returns
    bool. If present then `save_interval_steps` and `save_on_steps` options are
    ignored.
  save_decision_policy: An object used to determine when a checkpoint should be
    saved. If provided, overrides any other options dealing with this subject,
    including `save_interval_steps`, `save_on_steps`, and `should_save_fn`, and
    is the sole means of determining when a checkpoint should be saved. If not
    provided, these other options are used instead. Prefer to use this option
    over others.
  """

  save_interval_steps: int = 1000
  max_to_keep: Optional[int] = None
  keep_period: Optional[int] = None
  should_save_fn: Optional[Callable[[int, Optional[int]], bool]] = None
  save_decision_policy: Optional[
      save_decision_policy_lib.SaveDecisionPolicy
  ] = None


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
  use_shard_map_broadcast: bool = True
  single_host_load_and_broadcast: bool = True


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
  total_num_replicas = multislice.replica_count(
      global_mesh, replica_axis_index=replica_axis_index
  )
  num_processes_per_replica = (
      global_mesh.devices.size // total_num_replicas // jax.local_device_count()
  )
  per_replica_values = collections.defaultdict(list)
  for pid, values in per_process_values.items():
    replica_id = multislice.process_replica_id(
        pid, global_mesh, replica_axis_index=replica_axis_index
    )
    per_replica_values[replica_id].extend(values)

  for replica_id, values in per_replica_values.items():
    counter = collections.Counter(values)
    common_values = [
        k for k in counter if counter[k] == num_processes_per_replica
    ]
    # Here `len(common_values)`` will be less than or equal to `len(values)`
    # because a value can only appear in `common_values` if it occurs
    # `num_processes_per_slice` times in `values`.
    if len(common_values) > len(values):
      raise AssertionError(
          f' len(common_values) ({common_values}) exceeded length of input'
          f' values ({values}).'
      )
    per_replica_values[replica_id] = common_values

  return {k: set(v) for k, v in per_replica_values.items()}


def _global_max(values: list[int]) -> list[int]:
  """Computes the global max of a list of values across all hosts."""
  host_mesh = jax.sharding.Mesh(
      np.asarray(jax.devices()).reshape(
          multihost.process_count(), jax.local_device_count()
      ),
      ['host', 'dev'],
  )
  sharding = jax.sharding.NamedSharding(host_mesh, P('host', None))
  local_array = np.array([values], dtype=np.int32)
  # Create the global array, which is sharded across hosts.
  global_array = jax.make_array_from_process_local_data(sharding, local_array)

  @jax.jit
  @functools.partial(
      jax.shard_map, mesh=host_mesh, in_specs=P('host', None), out_specs=P()
  )
  def reduce_max_fn(x):
    return jax.lax.pmax(x, axis_name='host')

  max_values_array = reduce_max_fn(global_array).squeeze(axis=0)
  return list(np.asarray(max_values_array).astype(int))


class _LocalCheckpointManager(checkpoint_manager.CheckpointManager):
  """A checkpoint manager that checkpoints to local storage."""

  # TODO: b/330585086 - Allow configuration of global mesh describing slices.
  # Validate against global meshes used for arrays in state.
  # TODO: b/330585086 - Support arbitrary items beyond state. We will have
  # to evaluate whether arbitrary items can be a good fit for local
  # checkpointing, given restore+broadcast requirements.
  def __init__(
      self,
      directory: epath.PathLike,
      global_mesh: jax.sharding.Mesh,
      replica_id: int,
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

    if options.local.debug_use_full_global_mesh:
      all_processes = multihost.unique_processes_from_devices(
          np.asarray(self._global_mesh.devices)
      )
      local_replica_processes = all_processes
      barrier_sync_key_prefix = 'local-all'
    else:
      replica_devices = multislice.replica_devices(
          self._global_mesh,
          replica_id=replica_id,
          replica_axis_index=self._replica_axis_index,
      )
      local_replica_processes = multihost.unique_processes_from_devices(
          replica_devices
      )
      barrier_sync_key_prefix = f'local-{replica_id}'
    logging.vlog(
        1,
        'local_replica_processes for replica_id=%d: %r',
        replica_id,
        local_replica_processes,
    )
    multiprocessing_options = checkpoint_manager.MultiprocessingOptions(
        primary_host=None,
        active_processes=local_replica_processes,
        barrier_sync_key_prefix=barrier_sync_key_prefix,
    )
    local_options = checkpoint_manager.CheckpointManagerOptions(
        save_interval_steps=options.local.save_interval_steps,
        max_to_keep=options.local.max_to_keep,
        step_name_format=options.step_name_format,
        should_save_fn=options.local.should_save_fn,
        save_decision_policy=options.local.save_decision_policy,
        create=False,
        # we always clean up local tmp directories explicitly
        cleanup_tmp_directories=False,
        multiprocessing_options=multiprocessing_options,
        enable_async_checkpointing=options.enable_async_checkpointing,
        read_only=options.local.read_only,
        single_host_load_and_broadcast=False,
        # enable_background_delete set to False to ensure gc is done before save
        enable_background_delete=False,
        save_root_metadata=False,
        enable_per_process_directory_creation=True,
    )
    super().__init__(
        directory,
        options=local_options,
        metadata=metadata,
        item_handlers=dict(
            state=_local_checkpoint_handler(multiprocessing_options),
            process_metadata=ProcessMetadataCheckpointHandler,
        ),
        logger=logger,
    )
    self._run_initial_garbage_collection()

  def _run_initial_garbage_collection(self):
    """Remove steps that might be left over from previous runs."""
    steps_to_remove = self._get_old_steps_to_remove()
    self._checkpoints.delete_if(lambda info: info.step in steps_to_remove)
    self._checkpoint_deleter.delete_steps(steps_to_remove)

  @override
  def latest_step(self) -> Optional[int]:
    # Do not use the latest step from the local checkpoint manager.
    return None


def _get_persistent_options(
    options: CheckpointManagerOptions,
    multiprocessing_options: checkpoint_manager.MultiprocessingOptions,
) -> checkpoint_manager.CheckpointManagerOptions:
  """Get options for persistent checkpoint manager."""
  return checkpoint_manager.CheckpointManagerOptions(
      save_interval_steps=options.persistent.save_interval_steps,
      max_to_keep=options.persistent.max_to_keep,
      keep_period=options.persistent.keep_period,
      step_name_format=options.step_name_format,
      create=False,
      cleanup_tmp_directories=options.cleanup_tmp_directories,
      async_options=options.async_options,
      multiprocessing_options=multiprocessing_options,
      enable_async_checkpointing=options.enable_async_checkpointing,
      should_save_fn=options.persistent.should_save_fn,
      save_root_metadata=False,
      save_decision_policy=options.persistent.save_decision_policy,
  )


class _MultisliceCheckpointManager(
    abstract_checkpoint_manager.AbstractCheckpointManager, epy.ContextManager
):
  """Provides both checkpoint management and emergency checkpointings.

  This class is an implementation layer for handling multislice checkpointing.

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
    return _MultisliceCheckpointManager(
        local_directory=local_directory,
        persistent_directory=persistent_directory,
        global_mesh=global_mesh,
        abstract_state=abstract_state,
        options=options,
    )
  """

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
    if not self._local_directory.exists():
      raise FileNotFoundError(
          f'Local directory {self._local_directory} must be created by the'
          ' caller.'
      )
    if not self._persistent_directory.exists():
      raise FileNotFoundError(
          f'Persistent directory {self._persistent_directory} must be created'
          ' by the caller.'
      )

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

    self._abstract_state = abstract_state
    self._slice_id = multislice.process_replica_id(
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
    self._slice_count = multislice.replica_count(
        self._global_mesh, replica_axis_index=self._replica_axis_index
    )

    if len(global_mesh.devices.shape) <= self._replica_axis_index:
      raise AssertionError(
          f'replica_axis_index {self._replica_axis_index} is out of bound for'
          f' global_mesh.devices.shape {global_mesh.devices.shape}'
      )
    if self._slice_count <= 1:
      raise AssertionError(
          'To use this CheckpointManager, at least 2 data-parallel replicas are'
          ' needed.'
      )

    primary_replica_id = _PRIMARY_REPLICA_ID
    secondary_replica_id = _SECONDARY_REPLICA_ID

    self._persistent_primary_host = multislice.primary_process_in_replica(
        self._global_mesh,
        replica_axis_index=self._replica_axis_index,
        replica_id=primary_replica_id,
    )
    self._local_primary_host = multislice.primary_process_in_replica(
        self._global_mesh,
        replica_axis_index=self._replica_axis_index,
        replica_id=secondary_replica_id,
    )
    self._in_primary_slice = multislice.in_replica(
        multihost.process_index(),
        global_mesh,
        replica_axis_index=self._replica_axis_index,
        replica_id=primary_replica_id,
    )

    if self._in_primary_slice:
      persistent_multiprocessing_options = (
          checkpoint_manager.MultiprocessingOptions(
              primary_host=self._persistent_primary_host,
              active_processes=multihost.unique_processes_from_devices(
                  multislice.replica_devices(
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

    logging.info(
        'Created emergency.CheckpointManager with slice_id=%d,'
        ' process_index=%d, jax.process_index=%d',
        self._slice_id,
        multihost.process_index(),
        jax.process_index(),
    )
    logging.vlog(1, 'Local devices: %s', jax.local_devices())

  def _cleanup_local_tmp_directories(self):
    logging.info(
        'Cleaning up existing temporary directories at %s.',
        self._local_directory,
    )
    tmp_paths = step_lib.all_temporary_paths(self._local_directory)
    for tmp_path in tmp_paths:
      logging.info('Deleting temporary checkpoint: %s.', tmp_path)
      tmp_path.get().rmtree()

  def _make_persistent_checkpoint_manager(
      self,
      persistent_multiprocessing_options: checkpoint_manager.MultiprocessingOptions,
  ) -> checkpoint_manager.CheckpointManager:
    return checkpoint_manager.CheckpointManager(
        self._persistent_directory,
        options=_get_persistent_options(
            self._options, persistent_multiprocessing_options
        ),
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
        replica_id=self._slice_id,
        primary_replica_id=primary_replica_id,
        options=self._options,
        metadata=self._metadata,
        logger=self._logger,
    )

  @property
  def directory(self) -> epath.Path:
    return self._persistent_directory

  @property
  def in_primary_slice(self) -> bool:
    return self._in_primary_slice

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
      if (
          step_lib.is_standard_name_format(self._options.step_name_format)
          and self._options.single_host_load_and_broadcast
      ):
        optimized_name_format = (
            step_lib.single_host_load_and_broadcast_name_format(
                self._options.step_name_format
            )
        )
      else:
        logging.warning(
            'Step name format is not optimized. This may result in a slow'
            ' find_all operation.'
        )
        optimized_name_format = self._options.step_name_format
      self._persistent_steps = [
          metadata.step
          for metadata in optimized_name_format.find_all(
              self._persistent_directory
          )
      ]
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
    return multihost.reached_preemption(step)

  def should_save(self, step: int) -> bool:
    """Returns True if a checkpoint should be saved for the current step.

    This depends the previous step and save interval.

    Args:
      step: int

    Returns:
      True if the checkpoint should be saved.
    """
    logging.info('Checking should_save at step: %d.', step)

    last_checkpoint_step = self.latest_step()
    # Ensure current step is between the last step and next step (accounting for
    # save interval).
    if last_checkpoint_step is not None and last_checkpoint_step >= step:
      return False

    if self.in_primary_slice:
      should_save = self._persistent_checkpoint_manager.should_save(step)
    else:
      should_save = self._local_checkpoint_manager.should_save(step)
    return bool(_global_max([int(should_save)])[0])

  def delete(self, step: int):
    """Deletes a step checkpoint."""
    raise NotImplementedError(
        'Delete not yet implemented for emergency.CheckpointManager.'
    )

  def save(
      self,
      step: int,
      args: args_lib.Composite,
      *,
      force: bool = False,
  ) -> bool:
    """Returns True if a checkpoint was saved either locally or persistently."""
    multihost.sync_global_processes(
        multihost.unique_barrier_key(
            'CheckpointManager:save_start',
            prefix='emergency_checkpoint_manager',
        ),
        record_event_name=(
            '/jax/orbax/write/checkpoint_start_sync_duration_secs'
        ),
    )

    if not isinstance(args, args_lib.Composite):
      raise ValueError(
          f'Expected args must be a Composite object, but got {type(args)}.'
      )

    if _STATE_ITEM_NAME not in args.keys():
      raise ValueError(
          f'{_STATE_ITEM_NAME} is a required key and should be'
          ' specified by the user.'
      )

    if _PROCESS_METADATA_NAME in args.keys():
      raise ValueError(
          f'{_PROCESS_METADATA_NAME} is a reserved key and should not be'
          ' specified by the user.'
      )

    # TODO: b/330608746 - implement save op on different slices
    persistent_saved = False
    local_saved = False
    if self.in_primary_slice:
      logging.info(
          'Maybe saving at step %d (persistent) to %s.',
          step,
          self._persistent_checkpoint_manager.directory,
      )
      persistent_saved = self._persistent_checkpoint_manager.save(
          step, args=args.state, force=force
      )
    else:
      logging.info(
          'Maybe saving at step %d (local) to %s.',
          step,
          self._local_checkpoint_manager.directory,
      )

      args_dict = dict(args.items())
      args_dict[_PROCESS_METADATA_NAME] = (
          process_metadata_checkpoint_handler.ProcessMetadataSaveArgs(
              global_mesh=self._global_mesh
          )
      )
      args = args_lib.Composite(**args_dict)

      local_saved = self._local_checkpoint_manager.save(
          step, args=args, force=force
      )

    start = time.time()
    saved = tuple(
        bool(e) for e in _global_max([int(persistent_saved), int(local_saved)])
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
    """Gets the set of steps present in each slice from all hosts."""
    local_steps = set(step_lib.checkpoint_steps(self._local_directory))
    logging.info(
        'Found steps: %s in local host storage: %s.',
        local_steps,
        self._local_directory,
    )

    num_local_steps = len(local_steps)
    max_num_local_steps = _global_max([num_local_steps])[0]
    # Pad the local steps so all hosts have an array of the same length.
    padded_local_steps = list(local_steps) + [-1] * (
        max_num_local_steps - num_local_steps
    )
    local_steps_per_process_array = np.array(
        [multihost.process_index()] + padded_local_steps, dtype=np.int32
    )

    # Use all_gather to collect the arrays from every host.
    global_steps_per_process = multihost_utils.process_allgather(
        local_steps_per_process_array, tiled=False
    )

    # The rest of the logic works on the gathered NumPy array.
    per_process_steps = {}
    for process_and_steps in global_steps_per_process:
      per_process_steps[process_and_steps[0]] = set(
          s for s in process_and_steps[1:] if s != -1
      )
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

  def _get_single_slice_sharding(
      self,
      mesh: jax.sharding.Mesh,
      pspec: jax.sharding.PartitionSpec,
  ):
    """Get sharding for a single slice."""
    slice_devices = multislice.replica_devices(
        mesh,
        replica_id=self._slice_id,
        replica_axis_index=self._replica_axis_index,
    )
    single_slice_mesh_shape = [
        1 if i == self._replica_axis_index else d
        for i, d in enumerate(mesh.devices.shape)
    ]
    slice_mesh = jax.sharding.Mesh(
        slice_devices.reshape(single_slice_mesh_shape), mesh.axis_names
    )
    return jax.sharding.NamedSharding(slice_mesh, pspec)

  def _restore_from_local(
      self,
      step: int,
      restoring_slice_id: int,
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
    original_single_slice_shardings = jax.tree.map(
        lambda arr: self._get_single_slice_sharding(
            mesh=arr.sharding.mesh,
            pspec=arr.sharding.spec,
        ),
        self._abstract_state,
    )
    original_single_slice_shardings_tuple = tuple(
        jax.tree.flatten(original_single_slice_shardings)[0]
    )

    # Debug logging for sharding information.
    if logging.vlog_is_on(1):
      logging.vlog(
          1,
          'Debugging global restore_args based on abstract state. This uses the'
          ' user-provided mesh, and is not actually used for restoration.',
      )
      local_checkpoint_data_debugging.print_devices_indices_debug_info(
          checkpoint_utils.construct_restore_args(self._abstract_state)
      )
      logging.vlog(
          1,
          'Debugging single-slice restore_args based on abstract state. This'
          ' uses the user-provided mesh, and is not actually used for'
          ' restoration.',
      )
      local_checkpoint_data_debugging.print_devices_indices_debug_info(
          checkpoint_utils.construct_restore_args(
              self._abstract_state, original_single_slice_shardings
          )
      )

    if is_restoring_slice:
      logging.vlog(
          1, 'emergency.CheckpointManager: restoring from local checkpoint.'
      )
      restore_directory = self._options.step_name_format.find_step(
          epath.Path(directory or self._local_directory), step
      ).path
      step_stats.directory = str(restore_directory)
      (
          previous_distributed_to_device_ids,
          previous_device_ids,
      ) = ProcessMetadataCheckpointHandler().restore(
          restore_directory / _PROCESS_METADATA_NAME,
          process_metadata_checkpoint_handler.ProcessMetadataRestoreArgs(),
      )
      restore_mesh = mesh_consistency.consistent_restore_mesh_from_metadata(
          self._global_mesh,
          multihost.distributed_to_device_ids(),
          previous_distributed_to_device_ids=previous_distributed_to_device_ids,
          previous_device_ids=previous_device_ids,
      )
      restoring_processes = multihost.unique_processes_from_devices(
          multislice.replica_devices(
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
          lambda arr: self._get_single_slice_sharding(
              mesh=restore_mesh,
              pspec=arr.sharding.spec,
          ),
          self._abstract_state,
      )
      single_slice_restore_args = checkpoint_utils.construct_restore_args(
          self._abstract_state, restore_single_slice_shardings
      )

      # Directly use CheckpointHandler to restore. This is undesirable, but
      # allows us to avoid barrier issues that occur when calling
      # LocalCheckpointManager a different number of times on the non-primary
      # slices, which leads to
      # _module_unique_count getting out of sync.
      logging.vlog(
          1,
          'Restoring from %s',
          restore_directory / _STATE_ITEM_NAME,
      )
      if logging.vlog_is_on(1):
        logging.vlog(
            1,
            'Debugging single-slice restore_args used for restoration.',
        )
        asyncio.run(
            local_checkpoint_data_debugging.print_chunk_debug_info(
                restore_directory / _STATE_ITEM_NAME,
                single_slice_restore_args,
            )
        )
        local_checkpoint_data_debugging.print_devices_indices_debug_info(
            single_slice_restore_args
        )

      step_stats.checkpointer_start_time = time.time()
      args = args_lib.PyTreeRestore(
          item=self._abstract_state,
          restore_args=checkpoint_utils.construct_restore_args(
              self._abstract_state
          ),
      )
      single_slice_pytree = local_state_handler.restore(
          restore_directory / _STATE_ITEM_NAME,
          args=dataclasses.replace(
              args, restore_args=single_slice_restore_args
          ),
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
        in_tree = mesh_consistency.consistent_restore_mesh_to_global_mesh(
            in_tree, original_single_slice_shardings_tuple
        )
    else:
      logging.vlog(
          1,
          'emergency.CheckpointManager: non-primary slice, create zeros and'
          ' wait for broadcast.',
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
        use_shard_map=self._options.use_shard_map_broadcast,
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

  def _restore_from_persistent(
      self,
      step: int,
      directory: Optional[epath.PathLike] = None,
  ) -> Any:
    logging.info(
        'emergency.CheckpointManager: restoring step=%s from persistent'
        ' checkpoint in directory=%s',
        step,
        directory or self._persistent_directory,
    )
    step_stats = step_statistics.EmergencyRestoreStepStatistics()
    step_stats.checkpoint_manager_start_time = time.time()
    step_stats.step = step
    step_stats.is_restoring_slice = self.in_primary_slice
    step_stats.in_primary_slice = self.in_primary_slice

    shape_dtypes, tree_defs = jax.tree.flatten(self._abstract_state)
    single_slice_shardings = jax.tree.map(
        lambda arr: self._get_single_slice_sharding(
            mesh=arr.sharding.mesh,
            pspec=arr.sharding.spec,
        ),
        self._abstract_state,
    )
    single_slice_shardings_tuple = tuple(
        jax.tree.flatten(single_slice_shardings)[0]
    )

    if self.in_primary_slice:
      logging.info(
          'emergency.CheckpointManager: restoring from persistent checkpoint.'
      )
      restore_directory = self._options.step_name_format.find_step(
          epath.Path(directory or self._persistent_directory), step
      ).path
      step_stats.directory = str(restore_directory)

      primary_slice_processes = multihost.unique_processes_from_devices(
          multislice.replica_devices(
              self._global_mesh,
              replica_axis_index=self._replica_axis_index,
              replica_id=_PRIMARY_REPLICA_ID,
          )
      )
      primary_host = multislice.primary_process_in_replica(
          self._global_mesh,
          replica_axis_index=self._replica_axis_index,
          replica_id=_PRIMARY_REPLICA_ID,
      )
      multiprocessing_options = checkpoint_manager.MultiprocessingOptions(
          primary_host=primary_host,
          active_processes=primary_slice_processes,
          barrier_sync_key_prefix='persistent_primary_slice_restore',
      )

      single_slice_restore_args = checkpoint_utils.construct_restore_args(
          self._abstract_state, single_slice_shardings
      )
      args = args_lib.PyTreeRestore(
          item=self._abstract_state,
          restore_args=single_slice_restore_args,
      )

      handler = _persistent_checkpoint_handler(multiprocessing_options)
      try:
        step_stats.checkpointer_start_time = time.time()
        restored_pytree = handler.restore(
            restore_directory / 'default',
            args=args,
        )
        step_stats.checkpointer_duration_secs = (
            time.time() - step_stats.checkpointer_start_time
        )
        in_tree = tuple(jax.tree.flatten(restored_pytree)[0])
        logging.info(
            'Finished restoring from persistent checkpoint on primary replica.'
        )
      except FileNotFoundError as e:
        raise FileNotFoundError(
            'No steps found in either local or persistent storage when'
            f' requesting restoration of step {step}.'
        ) from e
    else:
      logging.info(
          'emergency.CheckpointManager: non-primary slice, create zeros and'
          ' wait for broadcast.',
      )

      @functools.partial(
          jax.jit,
          static_argnums=0,
          out_shardings=single_slice_shardings_tuple,
      )
      def create_zeros(shape_dtype_tup):
        return jax.tree.map(
            lambda sd: jnp.zeros(sd.shape, dtype=sd.dtype), shape_dtype_tup
        )

      zeros_pytree = create_zeros(tuple(shape_dtypes))
      in_tree = tuple(zeros_pytree)

    multihost.sync_global_processes('persistent_restore_pre_broadcast')

    start_broadcast = time.time()
    shared_states, _ = multislice.broadcast_one_replica_to_all(
        in_tree,
        self._global_mesh,
        replica_axis_index=self._replica_axis_index,
        is_source=self.in_primary_slice,
        use_shard_map=self._options.use_shard_map_broadcast,
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

    logging.info(
        'Finished broadcasting during persistent restore in %.2f on slice %s',
        broadcast_elapsed_s,
        self._slice_id,
    )

    return jax.tree.unflatten(tree_defs, shared_states)

  def restore(
      self,
      step: Optional[int],
      args: args_lib.Composite | None = None,
  ) -> Any:
    del args
    if step is None:
      step = self.latest_step()
      if step is None:
        raise FileNotFoundError(
            'No steps found in persistent or local storage.'
        )
    logging.info('Restoring at step %d.', step)
    restoring_slice_id = self._find_slice_with_complete_local_checkpoint(step)
    if restoring_slice_id > -1:
      # restore from LCM
      return self._restore_from_local(
          step=step,
          restoring_slice_id=restoring_slice_id,
      )

    return self._restore_from_persistent(step=step)

  def item_metadata(self, step: int) -> Any:
    raise NotImplementedError(
        'Item metadata not yet implemented for emergency.CheckpointManager.'
    )

  def metadata(self, step: int | None = None) -> RootMetadata | StepMetadata:
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
      item_handlers: Optional[
          Union[CheckpointHandler, CheckpointHandlersDict]
      ] = None,
      handler_registry: Optional[
          handler_registration.CheckpointHandlerRegistry
      ] = None,
      persistent_non_replicated_directory: Optional[epath.PathLike] = None,
  ):
    options = options or CheckpointManagerOptions()
    self._global_mesh = global_mesh
    self._abstract_state = abstract_state
    self._slice_count = multislice.replica_count(
        global_mesh, replica_axis_index=options.replica_axis_index
    )
    checkpoint_manager._create_root_directory(
        persistent_directory,
        multiprocessing_options=checkpoint_manager.MultiprocessingOptions(),
    )
    if self._slice_count <= 0:
      raise ValueError(
          'Slice count must be positive, but got'
          f' {self._slice_count} for mesh {global_mesh}.'
      )
    elif self._slice_count == 1:
      del local_directory
      self._checkpoint_manager = checkpoint_manager.CheckpointManager(
          persistent_directory,
          options=_get_persistent_options(
              options, checkpoint_manager.MultiprocessingOptions()
          ),
          metadata=metadata,
          logger=logger,
      )
    else:
      self._checkpoint_manager = _MultisliceCheckpointManager(
          local_directory=local_directory,
          persistent_directory=persistent_directory,
          global_mesh=global_mesh,
          abstract_state=abstract_state,
          options=options,
          metadata=metadata,
          logger=logger,
      )
    multiprocessing_options = checkpoint_manager.MultiprocessingOptions(
        primary_host=0,
        barrier_sync_key_prefix='non_replicated',
    )
    self._non_replicated_checkpoint_manager = None
    if persistent_non_replicated_directory is not None:
      self._non_replicated_checkpoint_manager = (
          checkpoint_manager.CheckpointManager(
              directory=persistent_non_replicated_directory,
              options=_get_persistent_options(options, multiprocessing_options),
              metadata=metadata,
              handler_registry=handler_registry,
              item_handlers=item_handlers,
          )
      )

  @property
  def directory(self) -> epath.Path:
    return self._checkpoint_manager.directory

  @property
  def in_primary_slice(self) -> bool:
    if self._slice_count == 1:
      return True
    else:
      assert isinstance(self._checkpoint_manager, _MultisliceCheckpointManager)
      return self._checkpoint_manager.in_primary_slice

  @property
  def global_mesh(self) -> jax.sharding.Mesh:
    return self._global_mesh

  def all_steps(self, read: bool = False) -> Sequence[int]:
    return self._checkpoint_manager.all_steps(read=read)

  def latest_step(self) -> Optional[int]:
    return self._checkpoint_manager.latest_step()

  def best_step(self) -> Optional[int]:
    raise NotImplementedError(
        'Metrics tracking not yet implemented for emergency.CheckpointManager.'
    )

  def reload(self):
    return self._checkpoint_manager.reload()

  def reached_preemption(self, step: int) -> bool:
    return self._checkpoint_manager.reached_preemption(step)

  def should_save(self, step: int) -> bool:
    return self._checkpoint_manager.should_save(step)

  def delete(self, step: int):
    """Deletes a step checkpoint."""
    raise NotImplementedError(
        'Delete not yet implemented for emergency.CheckpointManager.'
    )

  def save(
      self,
      step: int,
      args: args_lib.Composite,
      *,
      force: bool = False,
  ) -> bool:
    if _DATASET_ITEM_NAME in args.keys():
      if self._non_replicated_checkpoint_manager is None:
        raise ValueError(
            'Non-replicated checkpoint manager is None, but dataset was'
            f' provided at step {step}.'
        )
      else:
        self._non_replicated_checkpoint_manager.save(
            step, args=args.dataset, force=force
        )
        args_dict = dict(args.items())
        args_dict.pop(_DATASET_ITEM_NAME)
        args = args_lib.Composite(**args_dict)
    return self._checkpoint_manager.save(step, args=args, force=force)

  def restore(
      self,
      step: int | None,
      args: args_lib.Composite | None = None,
  ) -> Any:
    restored_dataset = None
    if args and _DATASET_ITEM_NAME in args.keys():
      if self._non_replicated_checkpoint_manager is None:
        raise ValueError(
            'Non-replicated checkpoint manager is None, but dataset was'
            f' requested to be restored at step {step}.'
        )
      else:
        restored_dataset = self._non_replicated_checkpoint_manager.restore(
            step=step, args=args.dataset
        )
    del args
    args = args_lib.Composite(
        state=args_lib.PyTreeRestore(
            item=self._abstract_state,
            restore_args=checkpoint_utils.construct_restore_args(
                self._abstract_state
            ),
        )
    )
    if isinstance(self._checkpoint_manager, _MultisliceCheckpointManager):
      restore = self._checkpoint_manager.restore(step, args=args)
    else:
      restore = self._checkpoint_manager.restore(step, args=args).state
    if restored_dataset:
      return args_lib.Composite(
          state=restore,
          dataset=restored_dataset,
      )
    return args_lib.Composite(
        state=restore,
    )

  def item_metadata(self, step: int) -> Any:
    raise NotImplementedError(
        'Item metadata not implemented for emergency.CheckpointManager.'
    )

  def metadata(self, step: int | None = None) -> RootMetadata | StepMetadata:
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
    if self._non_replicated_checkpoint_manager is not None:
      self._non_replicated_checkpoint_manager.wait_until_finished()
    return self._checkpoint_manager.wait_until_finished()

  def check_for_errors(self):
    if self._non_replicated_checkpoint_manager is not None:
      self._non_replicated_checkpoint_manager.check_for_errors()
    return self._checkpoint_manager.check_for_errors()

  def close(self):
    if self._non_replicated_checkpoint_manager is not None:
      self._non_replicated_checkpoint_manager.close()
    return self._checkpoint_manager.close()

  def __contextmanager__(
      self,
  ) -> Iterable[Self]:
    try:
      yield self
    finally:
      self.close()
