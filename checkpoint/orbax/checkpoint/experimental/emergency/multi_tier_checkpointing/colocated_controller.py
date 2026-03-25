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

"""Controller-side colocated orchestration for Pathways MTC."""

from collections.abc import Mapping
import time
from typing import Any, Callable, Optional

from absl import logging
from etils import epath
import jax
import jax.numpy as jnp
from orbax.checkpoint import args as args_lib
from orbax.checkpoint import checkpoint_manager
from orbax.checkpoint._src.handlers import handler_registration
from orbax.checkpoint._src.metadata import sharding as sharding_metadata
from orbax.checkpoint._src.multihost import colocated_transport
from orbax.checkpoint._src.multihost import dispatchers
from orbax.checkpoint._src.serialization import type_handlers
from orbax.checkpoint.experimental.emergency.multi_tier_checkpointing import (
    colocated_utils,
)


PyTree = Any
_STATE_ITEM_NAME = 'state'
_DATASET_ITEM_NAME = 'dataset'
_LATEST_STEP_RETRY_TIMEOUT_SECS = 30
_RETRIABLE_COLOCATED_CALL_EXCEPTIONS = (
    jax.errors.JaxRuntimeError,
    RuntimeError,
)


def _step_arr_on_state_devices(step: int, state: PyTree) -> jax.Array:
  """Creates a scalar step array on the same device list as `state`."""
  leaves = jax.tree.leaves(state)
  if not leaves:
    raise ValueError(
        'colocated save/restore requires a non-empty pytree of jax.Array '
        'leaves, but got an empty pytree.'
    )
  non_array_types = {
      type(x).__name__ for x in leaves if not isinstance(x, jax.Array)
  }
  if non_array_types:
    raise TypeError(
        'colocated save/restore requires all pytree leaves to be jax.Array. '
        f'Found non-array leaf types: {sorted(non_array_types)}.'
    )
  first_leaf = leaves[0]
  first_device_list_key = colocated_utils.device_list_signature(
      first_leaf.sharding
  )
  for leaf in leaves[1:]:
    if (
        colocated_utils.device_list_signature(leaf.sharding)
        != first_device_list_key
    ):
      raise ValueError(
          'colocated save/restore requires all pytree leaves to share the same '
          'device list.'
      )
  first_sharding = first_leaf.sharding
  if isinstance(first_sharding, jax.sharding.NamedSharding):
    scalar_sharding = jax.sharding.NamedSharding(
        first_sharding.mesh, jax.sharding.PartitionSpec()
    )
  elif isinstance(first_sharding, jax.sharding.SingleDeviceSharding):
    scalar_sharding = first_sharding
  else:
    raise TypeError(
        'colocated save/restore requires NamedSharding or '
        'SingleDeviceSharding on state arrays, '
        f'but got {type(first_sharding).__name__}.'
    )
  return jax.device_put(jnp.array(step, dtype=jnp.int32), scalar_sharding)


def _scalar_on_dummy(
    value: Any, dummy: jax.Array, *, dtype: Any
) -> jax.Array:
  return jax.device_put(jnp.asarray(value, dtype=dtype), dummy.sharding)


def _single_mapping_child(tree: PyTree) -> Any | None:
  if isinstance(tree, Mapping) and len(tree) == 1:
    return next(iter(tree.values()))
  return None


def _scalar_spec_like(
    arg_spec: jax.ShapeDtypeStruct, dtype: jnp.dtype
) -> jax.ShapeDtypeStruct:
  return jax.ShapeDtypeStruct((), dtype=dtype, sharding=arg_spec.sharding)


class ColocatedController:
  """Controller wrapper around the worker-side checkpoint backend."""

  def __init__(
      self,
      *,
      local_directory: epath.Path,
      global_mesh: jax.sharding.Mesh,
      options: Any,
      persistent_directory: Optional[epath.Path],
      handler_registry: Optional[
          handler_registration.CheckpointHandlerRegistry
      ],
      checkpoint_manager_options_fn: Callable[
          ..., checkpoint_manager.CheckpointManagerOptions
      ],
  ) -> None:
    # Lazy import to break circular dependency with
    # sidecar_worker_checkpoint_manager.
    from orbax.checkpoint.experimental.emergency.multi_tier_checkpointing import (  # pylint: disable=g-import-not-at-top
        sidecar_worker_checkpoint_manager,
    )

    self._local_directory = local_directory
    colocated_transport.install_pathways_colocated_serialization_patch()
    worker_manager_cls = (
        sidecar_worker_checkpoint_manager.WorkerCheckpointManager
    )
    self._worker_manager: Any = worker_manager_cls(
        local_directory=str(local_directory),
        mesh_shape=tuple(global_mesh.devices.shape),
        mesh_axis_names=tuple(global_mesh.axis_names),
        mesh_axis_types=tuple(global_mesh.axis_types),
        save_interval_steps=options.save_interval_steps,
    )
    # Derive colocated CPU devices from the caller's mesh order so controller
    # specialization order matches the worker mesh reconstruction.
    self._colocated_cpu_devices = (
        colocated_transport.unique_colocated_cpu_devices(
            tuple(global_mesh.devices.flat)
        )
    )
    if not self._colocated_cpu_devices:
      raise ValueError(
          'Pathways colocated checkpointing requires at least one colocated '
          'CPU device.'
      )
    self._colocated_cpu_ids = frozenset(
        d.id for d in self._colocated_cpu_devices
    )
    self._dummy = dispatchers.get_dummy_input_array(self._colocated_cpu_devices)
    self._specialize_scalar_worker_calls()

    self._persistent_checkpoint_manager = None
    if persistent_directory is not None:
      non_replicated_options = checkpoint_manager_options_fn(
          checkpoint_manager.MultiprocessingOptions(
              barrier_sync_key_prefix='non_replicated',
          )
      )
      self._persistent_checkpoint_manager = (
          checkpoint_manager.CheckpointManager(
              persistent_directory,
              options=non_replicated_options,
              handler_registry=handler_registry,
          )
      )

    self._worker_save_call = None

  @property
  def directory(self) -> epath.Path:
    """Returns the local checkpoint directory."""
    return self._local_directory

  def latest_step(self) -> Optional[int]:
    """Returns the latest local step, or `None` if no step exists."""
    attempt = 0
    deadline = time.time() + _LATEST_STEP_RETRY_TIMEOUT_SECS
    while True:
      attempt += 1
      try:
        result = self._worker_manager.latest_step(self._dummy)
        jax.block_until_ready(result)
        value = colocated_utils.require_unanimous_scalar_result(
            result, op_name='latest_step'
        )
        latest_step = int(value)
        return latest_step if latest_step != 0 else None
      except _RETRIABLE_COLOCATED_CALL_EXCEPTIONS as e:
        if time.time() > deadline:
          raise RuntimeError(
              'latest_step failed after retry budget'
              f' ({_LATEST_STEP_RETRY_TIMEOUT_SECS}s).'
          ) from e
        logging.vlog(
            1,
            'latest_step transient failure on attempt=%s (%s), retrying...',
            attempt,
            e,
        )
        time.sleep(1)

  def should_save(self, step: int) -> bool:
    """Returns whether workers want to save `step`."""
    step_arr = _scalar_on_dummy(step, self._dummy, dtype=jnp.int32)
    result = self._worker_manager.should_save(step_arr)
    jax.block_until_ready(result)
    value = colocated_utils.require_unanimous_scalar_result(
        result, op_name='should_save'
    )
    return bool(value)

  def is_saving_in_progress(self) -> bool:
    """Returns whether any worker still has async save work in flight."""
    result = self._worker_manager.is_saving_in_progress(self._dummy)
    jax.block_until_ready(result)
    value = colocated_utils.require_unanimous_scalar_result(
        result, op_name='is_saving_in_progress'
    )
    return bool(value)

  def save(
      self,
      step: int,
      args: args_lib.Composite,
      *,
      force: bool = False,
  ) -> bool:
    """Saves replicated state and returns whether a new save occurred."""
    state = self._prepare_state_for_save(args)
    step_arr = _step_arr_on_state_devices(step, state)
    force_arr = jax.device_put(
        jnp.array(force, dtype=jnp.bool_), step_arr.sharding
    )
    result = self._invoke_specialized_call(
        op_name='save',
        cache_attr='_worker_save_call',
        build_call=lambda: self._get_worker_save_call(
            step_arr, force_arr, state
        ),
        call=lambda fn: fn(step_arr, force_arr, state),
    )
    value = colocated_utils.require_unanimous_scalar_result(
        result, op_name='save'
    )
    saved = bool(value)
    if saved:
      self._save_persistent_dataset(step, args, force=force)
    return saved

  def restore(
      self,
      step: Optional[int],
      args: args_lib.Composite,
      *,
      default_item_mode: bool,
  ) -> Any:
    """Restores state and remaps worker-local shardings back to caller specs."""
    state_restore_args = args[_STATE_ITEM_NAME]
    if not isinstance(state_restore_args, args_lib.PyTreeRestore):
      raise ValueError(
          'Expected PyTreeRestore for state restore args in colocated mode, '
          f'got {type(state_restore_args).__name__}.'
      )
    if state_restore_args.restore_args is None:
      raise NotImplementedError(
          'colocated restore currently requires explicit restore_args.'
      )
    if state_restore_args.item is not None:
      restore_treedef = jax.tree.structure(state_restore_args.restore_args)
      item_treedef = jax.tree.structure(state_restore_args.item)
      if restore_treedef != item_treedef:
        raise ValueError(
            'colocated restore requires restore_args and item to share the '
            'same pytree structure.'
        )

    resolved_step = step
    if resolved_step is None and _DATASET_ITEM_NAME in args.keys():
      resolved_step = self.latest_step()
      if resolved_step is None:
        raise FileNotFoundError(f'No steps found in {self.directory}.')

    target_shardings = self._prepare_restore_target_shardings(
        state_restore_args
    )
    restore_step = -1 if resolved_step is None else resolved_step
    step_arr = _scalar_on_dummy(restore_step, self._dummy, dtype=jnp.int32)
    # Restore is metadata-driven on the worker side. Avoid sending a large
    # synthetic restore-spec tree across colocated Python just to describe the
    # output contract.
    try:
      result = self._worker_manager.restore_infer(step_arr)
      jax.block_until_ready(result)
    except _RETRIABLE_COLOCATED_CALL_EXCEPTIONS as e:
      logging.vlog(
          1,
          'restore_infer dispatch failed (%s: %s); retrying once',
          type(e).__name__,
          e,
      )
      result = self._worker_manager.restore_infer(step_arr)
      jax.block_until_ready(result)
    colocated_utils.assert_arrays_on_platform(
        result,
        expected_platform='cpu',
        tree_name='restored_state_from_workers',
    )
    state = self._rebuild_restored_state(
        result, state_restore_args.item
    )
    if jax.tree.structure(state) != jax.tree.structure(target_shardings):
      raise ValueError(
          'colocated restore produced a pytree structure that does not match '
          'restore_args.'
      )
    target_specs = jax.tree.map(
        lambda leaf, sharding: jax.ShapeDtypeStruct(
            leaf.shape, leaf.dtype, sharding=sharding
        ),
        state,
        target_shardings,
    )
    state = colocated_transport.to_final_specs(state, target_specs)
    return self._finalize_restore_result(
        step=resolved_step,
        args=args,
        state=state,
        default_item_mode=default_item_mode,
    )

  def wait_until_finished(self) -> None:
    """Blocks until worker-side async save work finishes."""
    result = self._worker_manager.wait_until_finished(self._dummy)
    jax.block_until_ready(result)

  def close(self) -> None:
    """Closes worker-side and persistent managers."""
    if self._persistent_checkpoint_manager is not None:
      self._persistent_checkpoint_manager.close()
    result = self._worker_manager.close(self._dummy)
    jax.block_until_ready(result)

  def _specialize_scalar_worker_calls(self) -> None:
    """Specializes scalar worker calls for colocated Python."""
    def _specialize(worker_call: Any, dtype: jnp.dtype) -> Any:
      return worker_call.specialize(
          out_specs_fn=lambda arg_spec: _scalar_spec_like(arg_spec, dtype),
          devices=self._colocated_cpu_devices,
      )

    self._worker_manager.latest_step = _specialize(
        self._worker_manager.latest_step, jnp.int32
    )
    self._worker_manager.should_save = _specialize(
        self._worker_manager.should_save, jnp.bool_
    )
    self._worker_manager.wait_until_finished = _specialize(
        self._worker_manager.wait_until_finished, jnp.bool_
    )
    self._worker_manager.is_saving_in_progress = _specialize(
        self._worker_manager.is_saving_in_progress, jnp.bool_
    )
    self._worker_manager.close = _specialize(
        self._worker_manager.close, jnp.bool_
    )

  def _get_worker_save_call(
      self,
      step_arr: jax.Array,
      force_arr: jax.Array,
      state: PyTree,
  ) -> Any:
    """Returns specialized call for worker save."""
    if self._worker_save_call is not None:
      return self._worker_save_call
    in_specs = (
        (
            colocated_transport.shape_dtype_struct_for_array(step_arr),
            colocated_transport.shape_dtype_struct_for_array(force_arr),
            jax.tree.map(
                colocated_transport.shape_dtype_struct_for_array, state
            ),
        ),
        {},
    )
    colocated_utils.assert_specs_on_allowed_cpu_ids(
        in_specs,
        allowed_ids=self._colocated_cpu_ids,
        tree_name='save_in_specs_for_colocated',
    )
    self._worker_save_call = self._worker_manager.save.specialize(
        in_specs=in_specs,
        out_specs_fn=lambda step_spec, _force_spec, _state_spec: (
            jax.ShapeDtypeStruct(
                (), dtype=jnp.bool_, sharding=step_spec.sharding
            )
        ),
        devices=self._colocated_cpu_devices,
    )
    return self._worker_save_call

  def _save_persistent_dataset(
      self,
      step: int,
      args: args_lib.Composite,
      *,
      force: bool,
  ) -> None:
    """Saves persistent dataset if present in args."""
    if _DATASET_ITEM_NAME not in args.keys():
      return
    if self._persistent_checkpoint_manager is None:
      raise NotImplementedError(
          'colocated save does not support dataset state without a '
          'persistent_directory.'
      )
    self._persistent_checkpoint_manager.save(
        step,
        args=args_lib.Composite(
            **{_DATASET_ITEM_NAME: args[_DATASET_ITEM_NAME]}
        ),
        force=force,
    )

  def _prepare_state_for_save(self, args: args_lib.Composite) -> PyTree:
    """Prepares state for colocated save, ensuring it is on CPU."""
    state_save_args = args[_STATE_ITEM_NAME]
    if not isinstance(state_save_args, args_lib.PyTreeSave):
      raise ValueError(
          'colocated save requires state args of type PyTreeSave, got '
          f'{type(state_save_args).__name__}.'
      )
    state = colocated_transport.to_colocated_python(state_save_args.item)
    colocated_utils.assert_arrays_on_platform(
        state,
        expected_platform='cpu',
        tree_name='state_for_colocated_save',
    )
    colocated_utils.assert_arrays_on_allowed_cpu_ids(
        state,
        allowed_ids=self._colocated_cpu_ids,
        tree_name='state_for_colocated_save',
    )
    return state

  def _prepare_restore_target_shardings(
      self,
      state_restore_args: args_lib.PyTreeRestore,
  ) -> PyTree:
    """Resolves target shardings for restore."""
    def _resolve_sharding(ra: Any) -> jax.sharding.Sharding:
      if not isinstance(ra, type_handlers.ArrayRestoreArgs):
        raise TypeError(
            'Colocated restore requires all restore_args leaves to be '
            'ArrayRestoreArgs, got '
            f'{type(ra).__name__}.'
        )

      sharding = ra.sharding
      if isinstance(sharding, sharding_metadata.ShardingMetadata):
        sharding = sharding.to_jax_sharding()
      elif (
          sharding is None
          and ra.mesh is not None
          and ra.mesh_axes is not None
      ):
        sharding = jax.sharding.NamedSharding(ra.mesh, ra.mesh_axes)
      if sharding is None:
        raise ValueError(
            'ArrayRestoreArgs must provide sharding or (mesh, mesh_axes) '
            f'for colocated restore. Got: {ra}'
        )
      if not isinstance(sharding, jax.sharding.Sharding):
        raise ValueError(
            'ArrayRestoreArgs sharding must resolve to jax.sharding.Sharding, '
            f'got {type(sharding).__name__}.'
        )
      return sharding

    return jax.tree.map(_resolve_sharding, state_restore_args.restore_args)

  def _rebuild_restored_state(
      self,
      restored_state: PyTree,
      template_state: Optional[PyTree],
  ) -> PyTree:
    """Rebuilds restored state to match template structure."""
    if template_state is None:
      return restored_state
    if jax.tree.structure(restored_state) == jax.tree.structure(template_state):
      return restored_state

    # Worker inference may return one checkpoint-state wrapper above the caller
    # template. Support only that narrow compatibility case.
    child = _single_mapping_child(restored_state)
    if child is not None and jax.tree.structure(child) == jax.tree.structure(
        template_state
    ):
      return child

    raise ValueError(
        'colocated restore produced a pytree structure that does not match the '
        'caller template structure.'
    )

  def _invoke_specialized_call(
      self,
      *,
      op_name: str,
      cache_attr: str,
      build_call: Callable[[], Any],
      call: Callable[[Any], Any],
  ) -> Any:
    """Invokes specialized call with retry logic."""
    specialized_call = build_call()
    try:
      result = call(specialized_call)
      jax.block_until_ready(result)
      return result
    except _RETRIABLE_COLOCATED_CALL_EXCEPTIONS as e:
      logging.vlog(
          1,
          '%s dispatch failed (%s: %s); rebuilding specialization once',
          op_name,
          type(e).__name__,
          e,
      )
      setattr(self, cache_attr, None)
      specialized_call = build_call()
      result = call(specialized_call)
      jax.block_until_ready(result)
      return result

  def _finalize_restore_result(
      self,
      *,
      step: Optional[int],
      args: args_lib.Composite,
      state: PyTree,
      default_item_mode: bool,
  ) -> Any:
    """Finalizes restore result, combining state and dataset if needed."""
    result = state
    if _DATASET_ITEM_NAME in args.keys():
      if self._persistent_checkpoint_manager is None:
        raise NotImplementedError(
            'colocated restore does not support dataset state without a '
            'persistent_directory.'
        )
      if step is None:
        raise ValueError(
            'colocated restore must resolve an explicit step before restoring '
            'dataset state.'
        )
      restored_dataset = self._persistent_checkpoint_manager.restore(
          step,
          args=args_lib.Composite(
              **{_DATASET_ITEM_NAME: args[_DATASET_ITEM_NAME]}
          ),
      )
      if not default_item_mode:
        result = args_lib.Composite(
            state=state,
            dataset=restored_dataset[_DATASET_ITEM_NAME],
        )

    if default_item_mode:
      return state
    if isinstance(result, args_lib.Composite):
      return result
    return args_lib.Composite(state=state)
