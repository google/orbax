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

"""ReplicatorCheckpointManager for emergency checkpoint. See details below.

WARNING: Do not use without specific approval. The API and implementation are
subject to change without notice.
"""

import dataclasses
from typing import Any, Callable, Iterable, List, Sequence, Tuple

from absl import logging
from etils import epath
from etils import epy
import jax
from orbax.checkpoint import abstract_checkpoint_manager
from orbax.checkpoint import args as args_lib
from orbax.checkpoint import checkpoint_manager
from orbax.checkpoint import checkpoint_utils
from orbax.checkpoint._src.checkpoint_managers import preservation_policy as preservation_policy_lib
from orbax.checkpoint._src.handlers import handler_registration
from orbax.checkpoint._src.handlers import pytree_checkpoint_handler
from orbax.checkpoint._src.metadata import sharding as sharding_metadata
from orbax.checkpoint._src.metadata import value as value_metadata
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.serialization import type_handler_registry
from orbax.checkpoint._src.serialization import type_handlers
from orbax.checkpoint.experimental.emergency import mesh_consistency
from orbax.checkpoint.experimental.emergency import process_metadata_checkpoint_handler
from orbax.checkpoint.experimental.emergency.multi_tier_checkpointing import colocated_utils
from orbax.checkpoint.path import step as step_lib
from typing_extensions import Self


PyTree = Any
DefaultCheckpointHandlerRegistry = (
    handler_registration.DefaultCheckpointHandlerRegistry
)
PyTreeCheckpointHandler = pytree_checkpoint_handler.PyTreeCheckpointHandler
RootMetadata = checkpoint_manager.RootMetadata
StepMetadata = checkpoint_manager.StepMetadata
ProcessMetadataCheckpointHandler = (
    process_metadata_checkpoint_handler.ProcessMetadataCheckpointHandler
)


_STATE_ITEM_NAME = 'state'
_PROCESS_METADATA_NAME = 'process_metadata'
_DATASET_ITEM_NAME = 'dataset'


def _restore_sharding_from_metadata(
    leaf: Any,
    global_mesh: jax.sharding.Mesh,
) -> jax.sharding.Sharding | None:
  """Builds explicit restore shardings from checkpoint metadata.

  When callers use bare `PyTreeRestore()`, Orbax falls back to reading per-leaf
  sharding files at restore time. Reconstructing restore shardings once from the
  already-loaded tree metadata keeps restore on the standard backend path while
  avoiding that slower and less explicit fallback.

  Named shardings are rebuilt against the live global mesh using the saved
  partition spec. Single-device shardings are rebuilt from the exact saved
  device string because those leaves are not mesh-relative.

  Args:
    leaf: The array metadata leaf.
    global_mesh: The global mesh.

  Returns:
    The restored sharding or None.
  """
  if not isinstance(leaf, value_metadata.ArrayMetadata):
    return None
  if leaf.sharding is None:
    raise ValueError(
        'ArrayMetadata for restore must include sharding metadata.'
    )

  if isinstance(leaf.sharding, sharding_metadata.NamedShardingMetadata):
    return jax.sharding.NamedSharding(
        global_mesh,
        jax.sharding.PartitionSpec(*leaf.sharding.partition_spec),
    )
  if isinstance(leaf.sharding, sharding_metadata.SingleDeviceShardingMetadata):
    # Single-device metadata already carries the exact local device string that
    # saved the leaf. Reconstruct that specific device instead of collapsing all
    # such leaves onto global_mesh.devices.flat[0], which would silently change
    # the restore contract for scalar or host-local leaves.
    return leaf.sharding.to_jax_sharding()

  raise ValueError(
      'Unsupported sharding metadata for restore:'
      f' {type(leaf.sharding)}'
  )


def _local_checkpoint_handler(
    multiprocessing_options: checkpoint_manager.MultiprocessingOptions,
    distributed_to_device_ids_fn: (
        Callable[[], List[List[int]]] | None
    ) = None,
) -> Tuple[PyTreeCheckpointHandler, ProcessMetadataCheckpointHandler]:
  """Create a PyTreeCheckpointHandler for local checkpoints."""
  if multiprocessing_options.primary_host is not None:
    raise ValueError(
        'multiprocessing_options.primary_host must be set to None for local'
        ' checkpoints.'
    )
  local_registry = type_handler_registry.create_type_handler_registry(
      (
          jax.Array,
          type_handlers.ArrayHandler(
              primary_host=None, replica_id=None, use_replica_parallel=False
          ),
      ),
  )
  pytree_handler = PyTreeCheckpointHandler(
      use_ocdbt=True,
      use_zarr3=True,
      multiprocessing_options=multiprocessing_options,
      type_handler_registry=local_registry,
  )
  metadata_handler = ProcessMetadataCheckpointHandler(
      multiprocessing_options=multiprocessing_options,
      distributed_to_device_ids_fn=distributed_to_device_ids_fn,
  )
  return pytree_handler, metadata_handler


@dataclasses.dataclass
class ReplicatorCheckpointManagerOptions:
  save_interval_steps: int = 1
  step_name_format: step_lib.NameFormat[step_lib.Metadata] | None = None
  should_save_fn: Callable[[int, int | None], bool] | None = None
  preservation_policy: preservation_policy_lib.PreservationPolicy | None = None


def _get_checkpoint_manager_options(
    options: ReplicatorCheckpointManagerOptions,
    multiprocessing_options: checkpoint_manager.MultiprocessingOptions,
) -> checkpoint_manager.CheckpointManagerOptions:
  """Get options for checkpoint manager."""
  per_process_directory_creation = multiprocessing_options.primary_host is None
  return checkpoint_manager.CheckpointManagerOptions(
      save_interval_steps=options.save_interval_steps,
      step_name_format=options.step_name_format,
      should_save_fn=options.should_save_fn,
      multiprocessing_options=multiprocessing_options,
      create=True,
      cleanup_tmp_directories=False,  # Handled separately below.
      enable_background_delete=True,
      enable_async_checkpointing=True,
      preservation_policy=options.preservation_policy,
      enable_per_process_directory_creation=per_process_directory_creation,
  )


class ReplicatorCheckpointManager(
    abstract_checkpoint_manager.AbstractCheckpointManager, epy.ContextManager
):
  """ReplicatorCheckpointManager.

  This class is intended for use in emergency checkpointing, but the system must
  conform to restrictive assumptions in order to work. Consider using
  `emergency.checkpoint_manager.CheckpointManager` if these assumptions are not
  met.

  This class assumes that an independent service is responsible for reflecting
  any checkpoints saved in local storage (e.g. RAMFS) to a persistent global
  storage (e.g. GCS). Thus, this class only saves checkpoints to process-local
  storage. Each process saves addressable data to its own storage location, on
  every process.

  When restoring a checkpoint, this class assumes that the independent
  replicator service will ensure consistency across all process-local storages.
  This means that if a restart causes data loss on one or more processes, the
  replicator must ensure that the corresponding data from a process with intact
  data is copied to the process with data loss. Specifically, the processes must
  be "peers" in the sense that they share the same index into the global array
  data.

  Users can control a few properties of the checkpointing behavior like the save
  interval. The period at which checkpoints are persisted to global storage, or
  the period at which they are
  garbage-collected, must be controlled by the replicator service.

  Attributes:
    local_directory: The local directory used for saving replicated states.
    persistent_directory: Optional. The top-level directory in persistent
      storage (e.g., GCS) used for saving non-replicated states like dataset
      iterators. Handlers for these states must be registered either in the
      `handler_registry` or in the `args.register_with_handler` decorator. This
      should only be used if you need to save states that are not replicated
      across processes.
    handler_registry: Optional. A registry for custom checkpoint handlers. Use
      this to register handlers for non-replicated (e.g., dataset iterator)
      states.
  """

  def __init__(
      self,
      local_directory: epath.Path,
      options: ReplicatorCheckpointManagerOptions | None = None,
      *,
      global_mesh: jax.sharding.Mesh,
      persistent_directory: epath.Path | None = None,
      handler_registry: (
          handler_registration.CheckpointHandlerRegistry | None
      ) = None,
      _distributed_to_device_ids_fn: (
          Callable[[], List[List[int]]] | None
      ) = None,
      _is_sidecar: bool = False,
  ):
    self._global_mesh = global_mesh
    options = options or ReplicatorCheckpointManagerOptions()
    multiprocessing_options = checkpoint_manager.MultiprocessingOptions(
        primary_host=None
    )
    self._options = options
    self._local_directory = local_directory
    self._is_sidecar = _is_sidecar
    if self._is_sidecar and _distributed_to_device_ids_fn is None:
      # Sidecars do not initialize JAX distributed client. Use a pathways-safe
      # local mapping by default if caller did not inject one.
      _distributed_to_device_ids_fn = (
          lambda: colocated_utils.compute_distributed_to_device_ids(
              jax.devices()
          )
      )
    self._get_distributed_to_device_ids = (
        _distributed_to_device_ids_fn or multihost.distributed_to_device_ids
    )
    self._step_name_format = (
        options.step_name_format or step_lib.standard_name_format()
    )
    self._options = dataclasses.replace(
        self._options,
        step_name_format=self._step_name_format,
    )
    self._init_standard(
        local_directory,
        options,
        multiprocessing_options,
        persistent_directory,
        handler_registry,
    )

  def _init_standard(
      self,
      local_directory,
      options,
      multiprocessing_options,
      persistent_directory,
      handler_registry,
  ):
    """Initializes the standard backend path."""
    is_sidecar = self._is_sidecar
    # Sidecars own only a subset of global devices, so collectives-based
    # barriers do not work there. Treat each sidecar as its own active process.
    self._active_processes = None
    sidecar_mp_options = multiprocessing_options
    if is_sidecar:
      self._active_processes = {jax.process_index()}
      sidecar_mp_options = dataclasses.replace(
          multiprocessing_options,
          primary_host=None,
          active_processes=self._active_processes,
      )
      epath.Path(local_directory).mkdir(parents=True, exist_ok=True)
    [state_handler, process_metadata_handler] = _local_checkpoint_handler(
        sidecar_mp_options,
        distributed_to_device_ids_fn=(
            self._get_distributed_to_device_ids if is_sidecar else None
        ),
    )
    self._state_handler = state_handler
    self._process_metadata_handler = process_metadata_handler
    replicator_options = _get_checkpoint_manager_options(
        options, sidecar_mp_options
    )
    if is_sidecar:
      # `CheckpointManager` rejects `create=True` when `active_processes` is
      # set, so the sidecar creates the root directory explicitly above.
      replicator_options = dataclasses.replace(
          replicator_options, create=False
      )

    self._local_handler_registry = DefaultCheckpointHandlerRegistry()
    self._local_handler_registry.add(None, args_lib.PyTreeSave, state_handler)
    self._local_handler_registry.add(
        None, args_lib.PyTreeRestore, state_handler
    )
    self._local_handler_registry.add(
        None,
        process_metadata_checkpoint_handler.ProcessMetadataSaveArgs,
        process_metadata_handler,
    )
    self._local_handler_registry.add(
        None,
        process_metadata_checkpoint_handler.ProcessMetadataRestoreArgs,
        process_metadata_handler,
    )
    self._impl = checkpoint_manager.CheckpointManager(
        local_directory,
        options=replicator_options,
        handler_registry=self._local_handler_registry,
    )
    non_replicated_multiprocessing_options = (
        checkpoint_manager.MultiprocessingOptions(
            barrier_sync_key_prefix='non_replicated',
        )
    )
    non_replicated_options = _get_checkpoint_manager_options(
        options, non_replicated_multiprocessing_options
    )
    self._persistent_checkpoint_manager = None
    if persistent_directory is not None:
      self._persistent_checkpoint_manager = (
          checkpoint_manager.CheckpointManager(
              persistent_directory,
              options=non_replicated_options,
              handler_registry=handler_registry,
          )
      )
    self._run_initial_garbage_collection()

  def _run_initial_garbage_collection(self):
    """Remove steps that might be left over from previous runs."""
    logging.info('Running initial garbage collection at %s.', self.directory)
    logging.info('Cleaning up existing temporary directories.')
    tmp_paths = step_lib.all_temporary_paths(self.directory)
    logging.info('Found tmp files: %s', tmp_paths)
    for tmp_path in tmp_paths:
      tmp_path.get().rmtree()

  @property
  def directory(self) -> epath.Path:
    return self._impl.directory

  @property
  def global_mesh(self) -> jax.sharding.Mesh:
    return self._global_mesh

  def all_steps(self, read: bool = False) -> Sequence[int]:
    return self._impl.all_steps(read=read)

  def latest_step(self) -> int | None:
    return self._impl.latest_step()

  def best_step(self) -> int | None:
    raise NotImplementedError()

  def reload(self):
    return self._impl.reload()

  def reached_preemption(self, step: int) -> bool:
    return self._impl.reached_preemption(step)

  def should_save(self, step: int) -> bool:
    return self._impl.should_save(step)

  def is_saving_in_progress(self) -> bool:
    """Returns whether a checkpoint save is currently in progress."""
    return self._impl.is_saving_in_progress()

  def delete(self, step: int):
    if self._persistent_checkpoint_manager is not None:
      self._persistent_checkpoint_manager.delete(step)
    return self._impl.delete(step)

  def _validate_and_standardize_args(
      self,
      args: args_lib.Composite,
  ) -> args_lib.Composite:
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
    for k, a in args.items():
      if not isinstance(a, args_lib.CheckpointArgs):
        raise TypeError(
            'Expected CheckpointArgs, got'
            f' {type(a).__name__}'
        )
      if (
          self._local_handler_registry is not None
          and not self._local_handler_registry.has(None, a)
          and k != _DATASET_ITEM_NAME
      ):
        raise ValueError(
            f'{type(a)} is not supported by this CheckpointManager. This is'
            ' likely because it does not yet implement support for local'
            ' checkpointing.'
        )
    return args

  def save(
      self,
      step: int,
      args: args_lib.Composite,
      *,
      force: bool = False,
  ) -> bool:
    args = self._validate_and_standardize_args(args)
    return self._standard_save(step, args, force=force)

  def _standard_save(
      self,
      step: int,
      args: args_lib.Composite,
      *,
      force: bool = False,
  ) -> bool:
    """Standard (multi-controller) save path."""
    multihost.sync_global_processes(
        multihost.unique_barrier_key(
            'CheckpointManager:save_start',
            prefix='replicator_checkpoint_manager',
        ),
        processes=self._active_processes,
        record_event_name=(
            '/jax/orbax/write/checkpoint_start_sync_duration_secs'
        ),
    )
    process_metadata_args = (
        process_metadata_checkpoint_handler.ProcessMetadataSaveArgs(
            global_mesh=self._global_mesh,
        )
    )
    if (
        self._persistent_checkpoint_manager is not None
        and _DATASET_ITEM_NAME in args.keys()
    ):
      self._persistent_checkpoint_manager.save(
          step,
          args=args_lib.Composite(
              **{
                  _DATASET_ITEM_NAME: (
                      args[_DATASET_ITEM_NAME]
                  )
              }
          ),
          force=force,
      )

    args_dict = dict(args.items())
    args_dict[_PROCESS_METADATA_NAME] = process_metadata_args
    args_dict.pop(_DATASET_ITEM_NAME, None)
    args = args_lib.Composite(**args_dict)
    return self._impl.save(step, args=args, force=force)

  def _get_mesh_consistent_args(
      self,
      previous_distributed_to_device_ids: List[List[int]],
      previous_device_ids: List[int],
      args: args_lib.Composite,
  ) -> Tuple[args_lib.Composite, args_lib.Composite]:
    current_distributed_to_device_ids = self._get_distributed_to_device_ids()

    restore_mesh = mesh_consistency.consistent_restore_mesh_from_metadata(
        self._global_mesh,
        current_distributed_to_device_ids,
        previous_distributed_to_device_ids=previous_distributed_to_device_ids,
        previous_device_ids=previous_device_ids,
    )

    def _replace_sharding(arg: type_handlers.ArrayRestoreArgs):
      sharding = arg.sharding
      if isinstance(sharding, sharding_metadata.ShardingMetadata):
        sharding = sharding.to_jax_sharding()
      elif (
          sharding is None
          and arg.mesh is not None
          and arg.mesh_axes is not None
      ):
        sharding = jax.sharding.NamedSharding(arg.mesh, arg.mesh_axes)
      if sharding is None:
        raise ValueError(
            'ArrayRestoreArgs must provide sharding or (mesh, mesh_axes).'
        )
      if isinstance(sharding, jax.sharding.SingleDeviceSharding):
        return arg
      if not isinstance(sharding, jax.sharding.NamedSharding):
        raise ValueError(
            'ArrayRestoreArgs sharding must be a NamedSharding or'
            f' SingleDeviceSharding, but got {type(sharding)}.'
        )
      return dataclasses.replace(
          arg,
          sharding=jax.sharding.NamedSharding(
              mesh=restore_mesh, spec=sharding.spec
          ),
      )

    original_args = args
    consistent_args = {}
    for k, a in args.items():
      if isinstance(a, args_lib.PyTreeRestore):
        if a.restore_args is None:
          # Allow inference-based restore args. Mesh-consistency remapping
          # requires explicit per-leaf shardings and is skipped in this mode.
          consistent_args[k] = a
          continue
        consistent_args[k] = dataclasses.replace(
            a, restore_args=jax.tree.map(_replace_sharding, a.restore_args)
        )
    return (
        original_args,
        args_lib.Composite(**consistent_args),
    )

  def _materialize_restore_args_from_metadata(
      self,
      step: int,
      args: args_lib.Composite,
  ) -> args_lib.Composite:
    """Expands inference-based restore args into explicit metadata-backed args.

    This keeps restore semantics generic: callers may still request plain
    `PyTreeRestore()`, but the backend converts that into explicit restore args
    once from checkpoint metadata so mesh-consistency remapping can run and the
    restore path does not need to re-read sharding files leaf by leaf.

    This still happens inside the standard backend restore flow.

    Args:
      step: The checkpoint step.
      args: The composite restore args.

    Returns:
      The materialized restore args.
    """
    updated_args = {}
    for key, item_args in args.items():
      if (
          not isinstance(item_args, args_lib.PyTreeRestore)
          or item_args.restore_args is not None
      ):
        updated_args[key] = item_args
        continue

      # Reuse handler metadata instead of maintaining a second restore-spec tree
      # in controller code.
      step_directory = self._step_name_format.find_step(
          self.directory, step
      ).path
      item_metadata = self._state_handler.metadata(step_directory / key)
      sharding_tree = jax.tree.map(
          lambda leaf: _restore_sharding_from_metadata(leaf, self._global_mesh),
          item_metadata.tree,
      )
      restore_args = checkpoint_utils.construct_restore_args(
          item_metadata.tree,
          sharding_tree=sharding_tree,
      )
      updated_args[key] = dataclasses.replace(
          item_args,
          item=item_metadata.tree,
          restore_args=restore_args,
      )
    return args_lib.Composite(**updated_args)

  def _get_mesh_consistent_result(
      self,
      original_args: args_lib.Composite,
      consistent_result: args_lib.Composite,
      *,
      default_item_mode: bool,
  ):
    result = {}
    for k, a in original_args.items():
      item_result = consistent_result[k]
      if isinstance(a, args_lib.PyTreeRestore):
        if a.restore_args is None:
          # Inference-based restore: preserve handler-returned sharding/layout.
          result[k] = item_result
          continue
        original_shardings = jax.tree.map(
            lambda arg: arg.sharding, a.restore_args
        )
        item_result = mesh_consistency.consistent_restore_mesh_to_global_mesh(
            item_result, original_shardings
        )
      result[k] = item_result

    result = args_lib.Composite(**result)
    if default_item_mode:
      assert len(result) == 1
      return result[_STATE_ITEM_NAME]
    else:
      return result

  def restore(
      self,
      step: int | None,
      args: args_lib.Composite,
  ) -> Any:
    if step is None:
      step = self.latest_step()
      if step is None:
        raise FileNotFoundError(
            f'No steps found in {self.directory}.'
        )

    args = self._validate_and_standardize_args(args)
    return self._standard_restore(step, args)

  def _standard_restore(
      self,
      step: int,
      args: args_lib.Composite,
  ) -> Any:
    """Standard (multi-controller) restore path."""
    default_item_mode = (
        checkpoint_manager
        .determine_default_item_mode_from_args(args)
    )
    args = self._validate_and_standardize_args(args)
    return self._standard_restore_single_step(
        step, args, default_item_mode=default_item_mode
    )

  def _standard_restore_single_step(
      self,
      step: int,
      args: args_lib.Composite,
      *,
      default_item_mode: bool,
  ) -> Any:
    """Restores exactly one checkpoint step."""
    process_metadata_args = args_lib.Composite(
        **{
            _PROCESS_METADATA_NAME: (
                process_metadata_checkpoint_handler
                .ProcessMetadataRestoreArgs()
            )
        }
    )
    process_metadata_restored = self._impl.restore(
        step, args=process_metadata_args
    )
    previous_distributed_to_device_ids, previous_device_ids = (
        process_metadata_restored[
            _PROCESS_METADATA_NAME
        ]
    )
    # Expand inference-based restore args before mesh-consistency remapping.
    args = self._materialize_restore_args_from_metadata(step, args)
    original_args, consistent_args = (
        self._get_mesh_consistent_args(
            previous_distributed_to_device_ids,
            previous_device_ids,
            args,
        )
    )
    restored = self._impl.restore(
        step, args=consistent_args
    )

    if (
        self._persistent_checkpoint_manager is not None
        and _DATASET_ITEM_NAME in args.keys()
    ):
      restored_dataset = (
          self._persistent_checkpoint_manager.restore(
              step,
              args=args_lib.Composite(
                  **{
                      _DATASET_ITEM_NAME: (
                          args[_DATASET_ITEM_NAME]
                      )
                  }
              ),
          )
      )
      args_dict = dict(restored.items())
      args_dict[_DATASET_ITEM_NAME] = (
          restored_dataset[_DATASET_ITEM_NAME]
      )
      restored = args_lib.Composite(**args_dict)

    return self._get_mesh_consistent_result(
        original_args,
        restored,
        default_item_mode=default_item_mode,
    )

  def item_metadata(self, step: int) -> Any:
    return self._impl.item_metadata(step)

  def metadata(self, step: int | None = None) -> RootMetadata | StepMetadata:
    return self._impl.metadata(step)

  def metrics(self, step: int) -> PyTree | None:
    raise NotImplementedError()

  def wait_until_finished(self):
    return self._impl.wait_until_finished()

  def check_for_errors(self):
    return self._impl.check_for_errors()

  def close(self):
    if self._persistent_checkpoint_manager is not None:
      self._persistent_checkpoint_manager.close()
    return self._impl.close()

  def __contextmanager__(
      self,
  ) -> Iterable[Self]:
    try:
      yield self
    finally:
      self.close()
