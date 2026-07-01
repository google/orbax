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
import dataclasses
import math
import threading
import time
from typing import Any, Callable

from absl import logging
from etils import epath
import jax
import jax.numpy as jnp
from orbax.checkpoint import args as args_lib
from orbax.checkpoint import checkpoint_manager
from orbax.checkpoint._src.handlers import handler_registration
from orbax.checkpoint._src.metadata import sharding as sharding_metadata
from orbax.checkpoint._src.metadata import tree as tree_metadata
from orbax.checkpoint._src.multihost import colocated_transport
from orbax.checkpoint._src.multihost import dispatchers
from orbax.checkpoint._src.serialization import type_handlers
from orbax.checkpoint._src.tree import utils as tree_utils
from orbax.checkpoint.experimental.emergency.multi_tier_checkpointing import (
    colocated_utils,
)
from orbax.checkpoint.experimental.emergency.multi_tier_checkpointing import (
    pathways_topology,
)


PyTree = Any
_STATE_ITEM_NAME = 'state'
_DATASET_ITEM_NAME = 'dataset'
_LATEST_STEP_RETRY_TIMEOUT_SECS = 30
_RETRIABLE_COLOCATED_CALL_EXCEPTIONS = (
    jax.errors.JaxRuntimeError,
    RuntimeError,
)


def _array_tree_summary(tree: PyTree) -> str:
  """Returns a compact shape/dtype-only summary without materializing arrays.

  Non-array leaves are ignored because this helper is only used for debug
  logging. Save-path validation happens outside logging-gated code.

  Args:
    tree: The PyTree payload block to summarize.

  Returns:
    A compact shape/dtype-only summary wrapper.
  """
  array_leaves = [
      leaf for leaf in jax.tree.leaves(tree) if isinstance(leaf, jax.Array)
  ]
  total_bytes = 0
  platforms = set()
  samples = []
  for leaf in array_leaves:
    total_bytes += math.prod(leaf.shape) * leaf.dtype.itemsize
    platforms.update(device.platform for device in leaf.sharding.device_set)
    if len(samples) < 3:
      samples.append(
          f'shape={leaf.shape}, dtype={leaf.dtype}, '
          f'sharding={type(leaf.sharding).__name__}'
      )
  gib = total_bytes / (1024**3)
  return (
      f'array_leaves={len(array_leaves)}, estimated_bytes={total_bytes} '
      f'({gib:.2f} GiB), platforms={sorted(platforms)}, '
      f'samples={samples}'
  )


def _run_lifecycle_ops(*ops: Callable[[], None]) -> None:
  """Runs all lifecycle operations and re-raises the first failure."""
  first_error = None
  for op in ops:
    try:
      op()
    except Exception as e:  # pylint: disable=broad-exception-caught
      if first_error is None:
        first_error = e
      else:
        logging.exception(
            'Additional colocated checkpoint lifecycle operation failed: %s', e
        )
  if first_error is not None:
    raise first_error


def _validate_colocated_options(options: Any) -> None:
  """Rejects RCM options not yet supported by worker-side colocated MTC."""
  unsupported_options = [
      name
      for name in ('step_name_format', 'should_save_fn', 'preservation_policy')
      if getattr(options, name, None) is not None
  ]
  if unsupported_options:
    raise NotImplementedError(
        'Pathways colocated MTC does not support custom '
        'ReplicatorCheckpointManagerOptions for '
        f'{unsupported_options}.'
    )


@dataclasses.dataclass(frozen=True)
class _ExpectedLeaf:
  """Metadata extracted from a state tree's expected leaf layout."""

  shape: tuple[int, ...] | None
  dtype: Any | None


@dataclasses.dataclass
class _PendingSave:
  """Controller-side anchors for an in-flight colocated worker save."""

  step: int
  force: bool
  result: Any
  keepalive: tuple[Any, ...] | None
  dataset_args: args_lib.Composite | None
  start_time: float
  handoff_thread: threading.Thread | None = None
  handoff_done: bool = False
  handoff_saved: bool | None = None
  handoff_error: BaseException | None = None
  saved: bool | None = None
  drain_lock: threading.Lock = dataclasses.field(
      default_factory=threading.Lock, init=False, repr=False
  )


class _ControllerSaveHandoffLifecycle:
  """Owns Pathways controller-side handoff state for one worker save.

  Worker sidecars already delegate checkpoint I/O and async finalization to the
  regular Orbax/RCM save path. This helper only tracks the controller-owned
  colocated call result and payload anchors until the sidecar reports that it
  accepted ownership of the save payload.
  """

  def __init__(
      self,
      *,
      worker_wait_until_finished: Callable[[], None],
      save_persistent_dataset: Callable[..., None],
  ) -> None:
    self._lock = threading.RLock()
    self._pending_save: _PendingSave | None = None
    self._worker_wait_until_finished = worker_wait_until_finished
    self._save_persistent_dataset = save_persistent_dataset

  @property
  def pending_save(self) -> _PendingSave | None:
    with self._lock:
      return self._pending_save

  @pending_save.setter
  def pending_save(self, pending_save: _PendingSave | None) -> None:
    with self._lock:
      self._pending_save = pending_save

  def has_pending_save(self) -> bool:
    with self._lock:
      return self._pending_save is not None

  def has_undrained_pending_save(self) -> bool:
    with self._lock:
      pending_save = self._pending_save
      return pending_save is not None and pending_save.saved is None

  def start_handoff_drain(self, pending_save: _PendingSave) -> None:
    """Publishes `pending_save` and starts its background handoff drain."""

    def _drain_handoff() -> None:
      try:
        saved = self.await_handoff(pending_save)
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception(
            'Pathways colocated MTC save step=%s: async worker handoff failed.',
            pending_save.step,
        )
        self.release_keepalive(pending_save, 'sidecar save handoff failed')
        handoff_error = e.with_traceback(None)
        with self._lock:
          pending_save.handoff_error = handoff_error
          pending_save.handoff_done = True
        return
      with self._lock:
        pending_save.handoff_saved = saved
        pending_save.handoff_done = True
        if not saved or pending_save.dataset_args is None:
          pending_save.saved = saved

    thread = threading.Thread(
        name=f'pathways_mtc_save_handoff_{pending_save.step}',
        target=_drain_handoff,
        daemon=True,
    )
    with self._lock:
      self._pending_save = pending_save
      pending_save.handoff_thread = thread
    thread.start()

  def await_handoff(self, pending_save: _PendingSave) -> bool:
    """Waits for sidecar save result and releases controller anchors."""
    block_start = time.time()
    logging.vlog(
        1,
        'Pathways colocated MTC save step=%s: draining async worker handoff.',
        pending_save.step,
    )
    jax.block_until_ready(pending_save.result)
    logging.vlog(
        1,
        'Pathways colocated MTC save step=%s: async worker handoff ready '
        'elapsed=%.3fs total_elapsed=%.3fs.',
        pending_save.step,
        time.time() - block_start,
        time.time() - pending_save.start_time,
    )
    try:
      value = colocated_utils.require_unanimous_scalar_result(
          pending_save.result, op_name='save_handoff'
      )
      if not isinstance(value, bool):
        raise TypeError(
            'save_handoff: expected unanimous worker result to be a bool'
            f' scalar, got {type(value).__name__}: {value!r}.'
        )
      return value
    finally:
      self.release_keepalive(
          pending_save, 'sidecar save result accepted ownership'
      )

  def join_handoff(self, pending_save: _PendingSave, *, wait: bool) -> bool:
    """Returns whether the pending save handoff has completed."""
    handoff_thread = pending_save.handoff_thread
    if handoff_thread is not None:
      if wait:
        handoff_thread.join()
      elif handoff_thread.is_alive():
        return False
    with self._lock:
      return pending_save.handoff_done

  def drain(self, *, wait: bool = True) -> bool | None:
    """Drains a pending worker save result, if one exists."""
    with self._lock:
      pending_save = self._pending_save
    if pending_save is None:
      return None
    return self._drain(pending_save, wait=wait)

  def _drain(self, pending_save: _PendingSave, *, wait: bool) -> bool | None:
    """Drains `pending_save` without re-reading current lifecycle state."""
    with pending_save.drain_lock:
      if pending_save.saved is not None:
        return pending_save.saved
      if not self.join_handoff(pending_save, wait=wait):
        return None
      if pending_save.handoff_error is not None:
        raise pending_save.handoff_error
      saved = pending_save.handoff_saved
      if saved is None:
        return None
      if saved:
        self._save_persistent_dataset(
            pending_save.step,
            pending_save.dataset_args,
            force=pending_save.force,
        )
      with self._lock:
        pending_save.saved = saved
      logging.vlog(
          1,
          'Pathways colocated MTC save step=%s: drained async save saved=%s.',
          pending_save.step,
          saved,
      )
      return saved

  def check_handoff_error(self) -> None:
    """Surfaces completed handoff errors without starting finalization work."""
    with self._lock:
      pending_save = self._pending_save
    if pending_save is None:
      return
    if not self.join_handoff(pending_save, wait=False):
      return
    with self._lock:
      handoff_error = pending_save.handoff_error
    if handoff_error is not None:
      raise handoff_error

  def release_keepalive(
      self, pending_save: _PendingSave, reason: str
  ) -> None:
    """Releases controller payload anchors for the pending worker save."""
    with self._lock:
      if pending_save.keepalive is None:
        return
      pending_save.keepalive = None
    logging.info(
        'Pathways colocated MTC save step=%s: released controller async save '
        'payload anchors: %s.',
        pending_save.step,
        reason,
    )

  def finish(self) -> bool | None:
    """Finishes pending worker async save work and releases payload anchors."""
    with self._lock:
      pending_save = self._pending_save
    if pending_save is None:
      return None
    saved = self._drain(pending_save, wait=True)
    try:
      if pending_save.saved:
        wait_start = time.time()
        logging.info(
            'Pathways colocated MTC save step=%s: waiting for worker-side'
            ' async save finalization.',
            pending_save.step,
        )
        self._worker_wait_until_finished()
        logging.info(
            'Pathways colocated MTC save step=%s: worker-side async save'
            ' finalizer wait returned wait_elapsed=%.3fs'
            ' pending_save_age=%.3fs. This is controller observation time, not'
            ' payload anchor lifetime or worker save duration.',
            pending_save.step,
            time.time() - wait_start,
            time.time() - pending_save.start_time,
        )
      return saved
    finally:
      step = pending_save.step
      self.release_keepalive(pending_save, 'pending save lifecycle finished')
      with self._lock:
        if self._pending_save is pending_save:
          self._pending_save = None
      logging.info(
          'Pathways colocated MTC save step=%s: finished pending async save'
          ' lifecycle.',
          step,
      )


def _step_arr_on_state_devices(step: int, state: PyTree) -> jax.Array:
  """Creates a scalar step array on the same device list as `state`."""
  leaves = jax.tree.leaves(state)
  if not leaves:
    raise ValueError(
        'colocated save/restore requires a non-empty pytree of state or '
        'sharding leaves, but got an empty pytree.'
    )

  def _resolve_sharding(x: Any) -> jax.sharding.Sharding:
    if isinstance(x, jax.sharding.Sharding):
      return x
    if isinstance(x, jax.Array):
      return x.sharding
    raise TypeError(
        'colocated save/restore requires all pytree leaves to be jax.Array or '
        f'jax.sharding.Sharding. Found leaf type: {type(x).__name__}.'
    )

  shardings = [_resolve_sharding(x) for x in leaves]
  first_sharding = shardings[0]
  first_device_list_key = colocated_utils.device_list_signature(
      first_sharding
  )
  for sharding in shardings[1:]:
    if (
        colocated_utils.device_list_signature(sharding)
        != first_device_list_key
    ):
      raise ValueError(
          'colocated save/restore requires all pytree leaves to share the same '
          'device list.'
      )
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


def _state_mapping_child(tree: PyTree) -> Any | None:
  if isinstance(tree, Mapping) and _STATE_ITEM_NAME in tree:
    return tree[_STATE_ITEM_NAME]
  return None


def _scalar_spec_like(
    arg_spec: jax.ShapeDtypeStruct, dtype: jnp.dtype
) -> jax.ShapeDtypeStruct:
  return jax.ShapeDtypeStruct((), dtype=dtype, sharding=arg_spec.sharding)


def _call_worker_restore_infer(
    worker_manager: Any,
    *,
    state_cpu_devices: tuple[jax.Device, ...],
    step_arr: jax.Array,
    partial_restore_arr: jax.Array,
) -> PyTree:
  """Calls worker restore inference with runtime-inferred output specs."""

  in_specs = (
      (
          colocated_transport.shape_dtype_struct_for_array(step_arr),
          colocated_transport.shape_dtype_struct_for_array(partial_restore_arr),
      ),
      {},
  )
  restore_call = worker_manager.restore_infer.specialize(
      in_specs=in_specs,
      devices=state_cpu_devices,
  )
  return _invoke_restore_call_with_retry(
      restore_call, step_arr, partial_restore_arr
  )


def _invoke_restore_call_with_retry(
    restore_call: Callable[..., PyTree],
    step_arr: jax.Array,
    partial_restore_arr: jax.Array,
) -> PyTree:
  """Invokes restore and retries once for transient colocated failures."""
  try:
    result = restore_call(step_arr, partial_restore_arr)
    jax.block_until_ready(result)
    return result
  except _RETRIABLE_COLOCATED_CALL_EXCEPTIONS as e:
    logging.vlog(
        1,
        'restore_infer dispatch failed (%s: %s); retrying once',
        type(e).__name__,
        e,
    )
    result = restore_call(step_arr, partial_restore_arr)
    jax.block_until_ready(result)
    return result


def _serialize_for_colocated_transport(tree: PyTree) -> PyTree:
  """Canonicalizes a PyTree before it crosses the colocated boundary."""
  return tree_metadata.serialize_tree(
      tree, tree_metadata.PYTREE_METADATA_OPTIONS
  )


def _try_deserialize_from_colocated_transport(
    restored_state: PyTree,
    template_state: PyTree,
) -> PyTree | None:
  try:
    return tree_utils.deserialize_tree(restored_state, template_state)
  except (AttributeError, IndexError, KeyError, TypeError, ValueError):
    return None


def _is_restore_spec_leaf(value: Any) -> bool:
  """Returns whether the value is a valid restore spec leaf."""
  return isinstance(
      value,
      (
          jax.Array,
          jax.ShapeDtypeStruct,
          type_handlers.ArrayRestoreArgs,
      ),
  )


def _expected_leaf_from_template(value: Any) -> _ExpectedLeaf | None:
  """Returns expected shape/dtype based on a restore spec template leaf."""
  if isinstance(value, (jax.Array, jax.ShapeDtypeStruct)):
    return _ExpectedLeaf(tuple(value.shape), value.dtype)
  if isinstance(value, type_handlers.ArrayRestoreArgs):
    shape = None
    if value.global_shape is not None:
      shape = tuple(value.global_shape)
    return _ExpectedLeaf(shape, value.dtype)
  shape = getattr(value, 'shape', None)
  dtype = getattr(value, 'dtype', None)
  if shape is not None or dtype is not None:
    return _ExpectedLeaf(None if shape is None else tuple(shape), dtype)
  return None


def _validate_restored_shape_dtype(
    *,
    restored_state: PyTree,
    template_state: PyTree,
) -> None:
  """Validates restored leaves against caller-provided shape/dtype specs."""
  restored_leaves = jax.tree.leaves(restored_state)
  template_leaves = jax.tree.leaves(
      template_state, is_leaf=_is_restore_spec_leaf
  )
  if len(restored_leaves) != len(template_leaves):
    raise ValueError(
        'colocated restore produced a different number of leaves than the '
        f'restore template: restored={len(restored_leaves)}, '
        f'template={len(template_leaves)}.'
    )

  for index, (restored_leaf, template_leaf) in enumerate(
      zip(restored_leaves, template_leaves)
  ):
    expected_leaf = _expected_leaf_from_template(template_leaf)
    if expected_leaf is None:
      continue
    restored_shape = getattr(restored_leaf, 'shape', None)
    if (
        expected_leaf.shape is not None
        and tuple(restored_shape or ()) != expected_leaf.shape
    ):
      raise ValueError(
          'colocated restore produced a leaf with an unexpected shape: '
          f'leaf={index}, restored_shape={restored_shape}, '
          f'expected_shape={expected_leaf.shape}.'
      )
    restored_dtype = getattr(restored_leaf, 'dtype', None)
    if (
        expected_leaf.dtype is not None
        and restored_dtype != expected_leaf.dtype
    ):
      raise ValueError(
          'colocated restore produced a leaf with an unexpected dtype: '
          f'leaf={index}, restored_dtype={restored_dtype}, '
          f'expected_dtype={expected_leaf.dtype}.'
      )


def _validate_restore_template_compatibility(
    *,
    item: PyTree,
    restore_args: PyTree,
) -> None:
  """Checks explicit restore_args do not conflict with the restore item."""
  item_leaves = jax.tree.leaves(item, is_leaf=_is_restore_spec_leaf)
  restore_arg_leaves = jax.tree.leaves(
      restore_args, is_leaf=_is_restore_spec_leaf
  )
  for index, (item_leaf, restore_arg_leaf) in enumerate(
      zip(item_leaves, restore_arg_leaves)
  ):
    item_expected = _expected_leaf_from_template(item_leaf)
    restore_arg_expected = _expected_leaf_from_template(restore_arg_leaf)
    if item_expected is None or restore_arg_expected is None:
      continue
    if (
        restore_arg_expected.shape is not None
        and item_expected.shape is not None
        and restore_arg_expected.shape != item_expected.shape
    ):
      raise ValueError(
          'colocated restore does not support restore_args shape transforms; '
          'item and restore_args shape must agree. '
          f'leaf={index}, item_shape={item_expected.shape}, '
          f'restore_args_shape={restore_arg_expected.shape}.'
      )
    if (
        restore_arg_expected.dtype is not None
        and item_expected.dtype is not None
        and restore_arg_expected.dtype != item_expected.dtype
    ):
      raise ValueError(
          'colocated restore does not support restore_args dtype casting; '
          'item and restore_args dtype must agree. '
          f'leaf={index}, item_dtype={item_expected.dtype}, '
          f'restore_args_dtype={restore_arg_expected.dtype}.'
      )


def _prepare_restore_target_shardings(
    state_restore_args: args_lib.PyTreeRestore,
) -> PyTree:
  """Resolves target shardings for restore."""

  def _resolve_restore_arg_sharding(ra: Any) -> jax.sharding.Sharding:
    if not isinstance(ra, type_handlers.ArrayRestoreArgs):
      raise TypeError(
          'Colocated restore requires all restore_args leaves to be '
          f'ArrayRestoreArgs, got {type(ra).__name__}.'
      )
    sharding = ra.sharding
    if isinstance(sharding, sharding_metadata.ShardingMetadata):
      sharding = sharding.to_jax_sharding()
    elif sharding is None and ra.mesh is not None and ra.mesh_axes is not None:
      sharding = jax.sharding.NamedSharding(ra.mesh, ra.mesh_axes)
    if not isinstance(sharding, jax.sharding.Sharding):
      raise ValueError(
          'ArrayRestoreArgs sharding must be a jax.sharding.Sharding, '
          f'got {type(sharding).__name__}.'
      )
    return sharding

  def _resolve_item_sharding(leaf: Any) -> jax.sharding.Sharding:
    sharding = getattr(leaf, 'sharding', None)
    if sharding is None:
      raise ValueError(
          'Colocated restore requires item leaves to provide sharding when '
          f'restore_args is not provided. Got: {leaf!r}'
      )
    if isinstance(sharding, sharding_metadata.ShardingMetadata):
      sharding = sharding.to_jax_sharding()
    if not isinstance(sharding, jax.sharding.Sharding):
      raise ValueError(
          'PyTreeRestore.item sharding must be a jax.sharding.Sharding, '
          f'got {type(sharding).__name__}.'
      )
    return sharding

  if state_restore_args.restore_args is not None:
    return jax.tree.map(
        _resolve_restore_arg_sharding, state_restore_args.restore_args
    )
  return jax.tree.map(_resolve_item_sharding, state_restore_args.item)


class ColocatedController:
  """Controller wrapper around the worker-side checkpoint backend."""

  def __init__(
      self,
      *,
      local_directory: epath.Path,
      global_mesh: jax.sharding.Mesh,
      options: Any,
      persistent_directory: epath.Path | None,
      handler_registry: handler_registration.CheckpointHandlerRegistry | None,
      checkpoint_manager_options_fn: Callable[
          ..., checkpoint_manager.CheckpointManagerOptions
      ],
  ) -> None:
    # pylint: disable=g-import-not-at-top,consider-using-from-import
    import orbax.checkpoint.experimental.emergency.multi_tier_checkpointing.sidecar_worker_checkpoint_manager as sidecar_worker_checkpoint_manager  # pytype: disable=import-error
    # pylint: enable=g-import-not-at-top,consider-using-from-import

    _validate_colocated_options(options)
    self._local_directory = local_directory
    self._enable_async_checkpointing = options.enable_async_checkpointing
    colocated_transport.install_pathways_colocated_serialization_patch()
    topology = pathways_topology.Topology.from_devices(
        tuple(global_mesh.devices.flat)
    )
    colocated_cpu_mesh = colocated_transport.colocated_cpu_mesh(global_mesh)
    colocated_cpu_mesh_device_ids = tuple(
        int(d.id) for d in colocated_cpu_mesh.devices.flat
    )
    distributed_to_cpu_device_ids = tuple(
        tuple(device_ids)
        for device_ids in topology.remap_distributed_device_ids(
            tuple(global_mesh.devices.flat),
            tuple(colocated_cpu_mesh.devices.flat),
        )
    )
    worker_manager_cls = (
        sidecar_worker_checkpoint_manager.WorkerCheckpointManager
    )
    self._worker_manager: Any = worker_manager_cls(
        local_directory=str(local_directory),
        mesh_shape=tuple(colocated_cpu_mesh.devices.shape),
        mesh_axis_names=tuple(global_mesh.axis_names),
        mesh_device_ids=colocated_cpu_mesh_device_ids,
        mesh_axis_types=tuple(global_mesh.axis_types),
        save_interval_steps=options.save_interval_steps,
        distributed_to_device_ids=distributed_to_cpu_device_ids,
        enable_async_checkpointing=options.enable_async_checkpointing,
        save_concurrent_gb=options.save_concurrent_gb,
        restore_concurrent_gb=options.restore_concurrent_gb,
    )
    # Worker-management RPCs should run once per worker VM. State arrays may
    # still use one colocated CPU per accelerator, so keep those ids separate.
    self._worker_cpu_devices = topology.worker_cpu_devices()
    state_cpu_devices = []
    seen_cpu_ids = set()
    for device in colocated_cpu_mesh.devices.flat:
      if device.id in seen_cpu_ids:
        continue
      seen_cpu_ids.add(device.id)
      state_cpu_devices.append(device)
    self._state_cpu_devices = tuple(state_cpu_devices)
    if not self._worker_cpu_devices:
      raise ValueError(
          'Pathways colocated checkpointing requires at least one colocated '
          'CPU device.'
      )
    self._colocated_cpu_ids = frozenset(
        d.id for d in self._state_cpu_devices
    )
    logging.info(
        'Pathways colocated MTC controller topology: workers=%s, '
        'mesh_shape=%s, global_tpu_ids=%s, colocated_cpu_ids=%s, '
        'worker_cpu_ids=%s, distributed_to_cpu_ids=%s.',
        topology.num_workers,
        tuple(colocated_cpu_mesh.devices.shape),
        colocated_utils.value_sample(
            device.id for device in global_mesh.devices.flat
        ),
        colocated_utils.value_sample(colocated_cpu_mesh_device_ids),
        colocated_utils.value_sample(
            device.id for device in self._worker_cpu_devices
        ),
        colocated_utils.nested_id_sample(distributed_to_cpu_device_ids),
    )
    self._dummy = dispatchers.get_dummy_input_array(self._worker_cpu_devices)
    self._specialize_worker_calls()

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
    self._worker_save_call_in_specs = None
    self._save_lifecycle = _ControllerSaveHandoffLifecycle(
        worker_wait_until_finished=self._worker_wait_until_finished,
        save_persistent_dataset=self._save_persistent_dataset,
    )

  @property
  def directory(self) -> epath.Path:
    """Returns the local checkpoint directory."""
    return self._local_directory

  @property
  def _pending_save(self) -> _PendingSave | None:
    return self._save_lifecycle.pending_save

  @_pending_save.setter
  def _pending_save(self, pending_save: _PendingSave | None) -> None:
    self._save_lifecycle.pending_save = pending_save

  def latest_step(self) -> int | None:
    """Returns the highest step present on every worker, or `None`."""
    attempt = 0
    deadline = time.time() + _LATEST_STEP_RETRY_TIMEOUT_SECS
    last_error = None
    while time.time() < deadline:
      attempt += 1
      try:
        result = self._worker_manager.all_steps(self._dummy)
        jax.block_until_ready(result)
        worker_step_arrays = colocated_utils.array_result_values(
            result, op_name='all_steps'
        )
        if not worker_step_arrays:
          return None

        worker_step_sets = []
        for steps in worker_step_arrays:
          worker_step_sets.append({
              int(step)
              for step in steps.reshape(-1)
              if int(step) != colocated_utils.NO_STEP_SENTINEL
          })

        common_steps = set.intersection(*worker_step_sets)
        if not common_steps:
          logging.vlog(
              1,
              'Workers reported no common checkpoint steps: %s',
              [sorted(steps) for steps in worker_step_sets],
          )
          return None
        return max(common_steps)
      except _RETRIABLE_COLOCATED_CALL_EXCEPTIONS as e:
        last_error = e
        logging.vlog(
            1,
            'latest_step transient failure on attempt=%s (%s), retrying...',
            attempt,
            e,
        )
        time.sleep(1)
    raise RuntimeError(
        'latest_step failed after retry budget'
        f' ({_LATEST_STEP_RETRY_TIMEOUT_SECS}s).'
    ) from last_error

  def should_save(self, step: int) -> bool:
    """Returns whether workers want to save `step`."""
    if step == colocated_utils.NO_STEP_SENTINEL:
      return False
    step_arr = _scalar_on_dummy(step, self._dummy, dtype=jnp.int32)
    result = self._worker_manager.should_save(step_arr)
    jax.block_until_ready(result)
    value = colocated_utils.require_unanimous_scalar_result(
        result, op_name='should_save'
    )
    return bool(value)

  def is_saving_in_progress(self) -> bool:
    """Returns whether any async save work is in flight."""
    self._save_lifecycle.drain(wait=False)
    if self._save_lifecycle.has_undrained_pending_save():
      return True
    worker_value = self._worker_is_saving_in_progress()
    persistent_value = (
        self._persistent_checkpoint_manager is not None
        and self._persistent_checkpoint_manager.is_saving_in_progress()
    )
    return bool(worker_value) or bool(persistent_value)

  def save(
      self,
      step: int,
      args: args_lib.Composite,
      *,
      force: bool = False,
  ) -> bool:
    """Saves state and returns whether a new save occurred.

    Args:
      step: The current training step.
      args: Replicator arguments for saving.
      force: If True, bypasses the normal save schedule.

    Returns:
      True if the step was dispatched for saving, False otherwise.
    """
    if step == colocated_utils.NO_STEP_SENTINEL:
      return False
    save_start = time.time()
    if not force:
      should_save_start = time.time()
      should_save = self.should_save(step)
      logging.vlog(
          1,
          'Pathways colocated MTC save step=%s: should_save=%s elapsed=%.3fs.',
          step,
          should_save,
          time.time() - should_save_start,
      )
      if not should_save:
        self._save_lifecycle.drain(wait=False)
        return False

    dataset_args = self._persistent_dataset_args(args)
    self._validate_persistent_dataset_args(dataset_args)

    pending_start = time.time()
    had_pending_save = self._save_lifecycle.has_pending_save()
    self._finish_pending_save()
    if had_pending_save:
      logging.info(
          'Pathways colocated MTC save step=%s: prior async save finalized '
          'elapsed=%.3fs.',
          step,
          time.time() - pending_start,
      )
    logging.info(
        'Pathways colocated MTC save step=%s force=%s: begin.',
        step,
        force,
    )
    prepare_start = time.time()
    state = self._prepare_state_for_save(step, args)
    logging.vlog(
        1,
        'Pathways colocated MTC save step=%s: prepared state elapsed=%.3fs.',
        step,
        time.time() - prepare_start,
    )
    step_arr = _step_arr_on_state_devices(step, state)
    force_arr = jax.device_put(
        jnp.array(force, dtype=jnp.bool_), step_arr.sharding
    )
    specialize_start = time.time()
    save_call = self._get_worker_save_call(step_arr, force_arr, state)
    logging.vlog(
        1,
        'Pathways colocated MTC save step=%s: worker save call ready '
        'elapsed=%.3fs.',
        step,
        time.time() - specialize_start,
    )
    dispatch_start = time.time()
    result = save_call(step_arr, force_arr, state)
    logging.vlog(
        1,
        'Pathways colocated MTC save step=%s: save_call returned '
        'elapsed=%.3fs.',
        step,
        time.time() - dispatch_start,
    )
    pending_save = _PendingSave(
        step=step,
        force=force,
        result=result,
        keepalive=(step_arr, force_arr, state),
        dataset_args=dataset_args,
        start_time=save_start,
    )
    self._start_pending_save_handoff_drain(pending_save)
    if not self._enable_async_checkpointing:
      logging.info(
          'Pathways colocated MTC save step=%s: async checkpointing disabled; '
          'waiting for worker save completion.',
          step,
      )
      return bool(self._finish_pending_save())
    logging.info(
        'Pathways colocated MTC save step=%s: dispatched async save '
        'total_elapsed=%.3fs.',
        step,
        time.time() - save_start,
    )
    return True

  def restore(
      self,
      step: int | None,
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
    if (
        state_restore_args.restore_args is None
        and state_restore_args.item is None
    ):
      raise NotImplementedError(
          'colocated restore requires PyTreeRestore.item or explicit '
          'restore_args.'
      )
    if (
        state_restore_args.item is not None
        and state_restore_args.restore_args is not None
    ):
      restore_treedef = jax.tree.structure(state_restore_args.restore_args)
      item_treedef = jax.tree.structure(state_restore_args.item)
      if restore_treedef != item_treedef:
        raise ValueError(
            'colocated restore requires restore_args and item to share the '
            'same pytree structure.'
        )
      _validate_restore_template_compatibility(
          item=state_restore_args.item,
          restore_args=state_restore_args.restore_args,
      )

    resolved_step = step
    if resolved_step is None:
      resolved_step = self.latest_step()
      if resolved_step is None:
        raise FileNotFoundError(f'No steps found in {self.directory}.')
    if resolved_step == colocated_utils.NO_STEP_SENTINEL:
      raise ValueError(
          'Pathways colocated MTC cannot restore step 0 because the MTC '
          'coordinator protocol reserves step 0 as "no checkpoint".'
      )

    target_shardings = _prepare_restore_target_shardings(state_restore_args)
    transport_target_shardings = _serialize_for_colocated_transport(
        target_shardings
    )
    cpu_target_shardings = jax.tree.map(
        colocated_transport.colocated_cpu_sharding,
        transport_target_shardings,
    )
    step_arr = _step_arr_on_state_devices(resolved_step, cpu_target_shardings)
    partial_restore_arr = jax.device_put(
        jnp.array(state_restore_args.partial_restore, dtype=jnp.bool_),
        step_arr.sharding,
    )
    result = _call_worker_restore_infer(
        self._worker_manager,
        state_cpu_devices=self._state_cpu_devices,
        step_arr=step_arr,
        partial_restore_arr=partial_restore_arr,
    )
    # Keep alive anchor to prevent aggressive garbage collection during async
    # restore.
    del step_arr, partial_restore_arr
    colocated_utils.assert_arrays_on_platform(
        result,
        expected_platform='cpu',
        tree_name='restored_state_from_workers',
    )
    colocated_utils.assert_arrays_on_allowed_cpu_ids(
        result,
        allowed_ids=self._colocated_cpu_ids,
        tree_name='restored_state_from_workers',
    )
    rebuild_template = (
        state_restore_args.item
        if state_restore_args.item is not None
        else state_restore_args.restore_args
    )
    state = self._rebuild_restored_state(result, rebuild_template)
    _validate_restored_shape_dtype(
        restored_state=state, template_state=rebuild_template
    )
    if (
        state_restore_args.restore_args is not None
        and state_restore_args.restore_args is not rebuild_template
    ):
      _validate_restored_shape_dtype(
          restored_state=state, template_state=state_restore_args.restore_args
      )
    final_target_shardings = (
        target_shardings
        if state_restore_args.item is not None
        else transport_target_shardings
    )
    if jax.tree.structure(state) != jax.tree.structure(final_target_shardings):
      raise ValueError(
          'colocated restore produced a pytree structure that does not match '
          'restore_args.'
      )
    target_specs = jax.tree.map(
        lambda leaf, sharding: jax.ShapeDtypeStruct(
            leaf.shape, leaf.dtype, sharding=sharding
        ),
        state,
        final_target_shardings,
    )
    final_transfer_start = time.time()
    logging.info(
        'Pathways colocated MTC restore step=%s: transferring restored state '
        'to final target shardings.',
        resolved_step,
    )
    state = colocated_transport.to_final_specs(state, target_specs)
    jax.block_until_ready(state)
    logging.info(
        'Pathways colocated MTC restore step=%s: final target shardings ready '
        'elapsed=%.3fs.',
        resolved_step,
        time.time() - final_transfer_start,
    )
    return self._finalize_restore_result(
        step=resolved_step,
        args=args,
        state=state,
        default_item_mode=default_item_mode,
    )

  def wait_until_finished(self) -> None:
    """Blocks until persistent and worker-side async save work finishes."""
    pending_save_exists = self._save_lifecycle.has_pending_save()
    ops = [self._finish_pending_save]
    if self._persistent_checkpoint_manager is not None:
      ops.append(self._persistent_checkpoint_manager.wait_until_finished)
    if not pending_save_exists:
      ops.append(self._worker_wait_until_finished)
    _run_lifecycle_ops(*ops)

  def check_for_errors(self) -> None:
    """Raises async checkpoint errors from persistent and worker managers."""
    ops = [self._check_pending_save_handoff_error]
    if self._persistent_checkpoint_manager is not None:
      ops.append(self._persistent_checkpoint_manager.check_for_errors)
    ops.append(self._worker_check_for_errors)
    _run_lifecycle_ops(*ops)

  def close(self) -> None:
    """Closes worker-side and persistent managers."""
    ops = [self._finish_pending_save]
    if self._persistent_checkpoint_manager is not None:
      ops.append(self._persistent_checkpoint_manager.close)
    ops.append(self._worker_close)
    _run_lifecycle_ops(*ops)

  def _worker_wait_until_finished(self) -> None:
    """Waits for worker-side checkpoint manager operations to finish."""
    result = self._worker_manager.wait_until_finished(self._dummy)
    jax.block_until_ready(result)

  def _worker_is_saving_in_progress(self) -> bool:
    """Returns whether worker-side checkpoint manager is saving."""
    result = self._worker_manager.is_saving_in_progress(self._dummy)
    jax.block_until_ready(result)
    values = colocated_utils.scalar_result_values(
        result, op_name='is_saving_in_progress'
    )
    non_bool_values = [value for value in values if not isinstance(value, bool)]
    if non_bool_values:
      raise TypeError(
          'is_saving_in_progress: expected worker results to be bool scalars, '
          f'got {[(type(value).__name__, value) for value in non_bool_values]}.'
      )
    # This is local worker status, not a global consensus decision: mixed
    # values simply mean some sidecars are still finalizing async save work.
    return any(values)

  def _worker_check_for_errors(self) -> None:
    result = self._worker_manager.check_for_errors(self._dummy)
    jax.block_until_ready(result)

  def _worker_close(self) -> None:
    result = self._worker_manager.close(self._dummy)
    jax.block_until_ready(result)

  def _specialize_worker_calls(self) -> None:
    """Specializes worker calls for colocated Python."""
    def _specialize(worker_call: Any, dtype: jnp.dtype) -> Any:
      return worker_call.specialize(
          out_specs_fn=lambda arg_spec: _scalar_spec_like(arg_spec, dtype),
          devices=self._worker_cpu_devices,
      )

    self._worker_manager.latest_step = _specialize(
        self._worker_manager.latest_step, jnp.int32
    )
    self._worker_manager.all_steps = self._worker_manager.all_steps.specialize(
        out_specs_fn=lambda arg_spec: jax.ShapeDtypeStruct(
            (colocated_utils.MAX_TRACKED_STEPS,),
            dtype=jnp.int32,
            sharding=colocated_utils.replicated_sharding_like(
                arg_spec.sharding
            ),
        ),
        devices=self._worker_cpu_devices,
    )
    self._worker_manager.should_save = _specialize(
        self._worker_manager.should_save, jnp.bool_
    )
    self._worker_manager.wait_until_finished = _specialize(
        self._worker_manager.wait_until_finished, jnp.bool_
    )
    self._worker_manager.check_for_errors = _specialize(
        self._worker_manager.check_for_errors, jnp.bool_
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
    if (
        self._worker_save_call is not None
        and self._worker_save_call_in_specs == in_specs
    ):
      return self._worker_save_call
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
        devices=self._state_cpu_devices,
    )
    self._worker_save_call_in_specs = in_specs
    return self._worker_save_call

  def _persistent_dataset_args(
      self,
      args: args_lib.Composite,
  ) -> args_lib.Composite | None:
    """Returns dataset-only args for persistent save, if present."""
    if _DATASET_ITEM_NAME not in args.keys():
      return None
    return args_lib.Composite(**{_DATASET_ITEM_NAME: args[_DATASET_ITEM_NAME]})

  def _save_persistent_dataset(
      self,
      step: int,
      dataset_args: args_lib.Composite | None,
      *,
      force: bool,
  ) -> None:
    """Saves persistent dataset if present in args."""
    if dataset_args is None:
      return
    self._validate_persistent_dataset_args(dataset_args)
    assert self._persistent_checkpoint_manager is not None
    self._persistent_checkpoint_manager.save(
        step,
        args=dataset_args,
        force=force,
    )

  def _validate_persistent_dataset_args(
      self,
      dataset_args: args_lib.Composite | None,
  ) -> None:
    """Validates persistent dataset save configuration before worker dispatch."""
    if dataset_args is None or self._persistent_checkpoint_manager is not None:
      return
    raise NotImplementedError(
        'colocated save does not support dataset state without a '
        'persistent_directory.'
    )

  def _start_pending_save_handoff_drain(
      self, pending_save: _PendingSave
  ) -> None:
    self._save_lifecycle.start_handoff_drain(pending_save)

  def _drain_pending_save(self, *, wait: bool = True) -> bool | None:
    return self._save_lifecycle.drain(wait=wait)

  def _check_pending_save_handoff_error(self) -> None:
    self._save_lifecycle.check_handoff_error()

  def _finish_pending_save(self) -> bool | None:
    return self._save_lifecycle.finish()

  def _prepare_state_for_save(
      self, step: int, args: args_lib.Composite
  ) -> PyTree:
    """Prepares state for colocated save, ensuring it is on CPU."""
    state_save_args = args[_STATE_ITEM_NAME]
    if not isinstance(state_save_args, args_lib.PyTreeSave):
      raise ValueError(
          'colocated save requires state args of type PyTreeSave, got '
          f'{type(state_save_args).__name__}.'
      )
    if logging.vlog_is_on(2):
      logging.vlog(
          2,
          'Pathways colocated MTC save step=%s: state before colocated '
          'transport: %s',
          step,
          _array_tree_summary(state_save_args.item),
      )
    serialize_start = time.time()
    state = _serialize_for_colocated_transport(state_save_args.item)
    logging.info(
        'Pathways colocated MTC save step=%s: serialized state structure '
        'elapsed=%.3fs.',
        step,
        time.time() - serialize_start,
    )
    transport_start = time.time()
    state = colocated_transport.to_colocated_python(state)
    logging.info(
        'Pathways colocated MTC save step=%s: to_colocated_python returned '
        'elapsed=%.3fs.',
        step,
        time.time() - transport_start,
    )
    validate_start = time.time()
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
    logging.info(
        'Pathways colocated MTC save step=%s: state validation elapsed=%.3fs.',
        step,
        time.time() - validate_start,
    )
    return state

  def _rebuild_restored_state(
      self,
      restored_state: PyTree,
      template_state: PyTree | None,
  ) -> PyTree:
    """Rebuilds restored state to match template structure."""
    if template_state is None:
      return restored_state

    def _try_rebuild(candidate: PyTree) -> PyTree | None:
      if jax.tree.structure(candidate) == jax.tree.structure(template_state):
        return candidate
      return _try_deserialize_from_colocated_transport(
          candidate, template_state
      )

    candidates = [restored_state]
    state_child = _state_mapping_child(restored_state)
    if state_child is not None:
      candidates.append(state_child)
    single_child = _single_mapping_child(restored_state)
    if single_child is not None and single_child is not state_child:
      candidates.append(single_child)

    for candidate in candidates:
      rebuilt = _try_rebuild(candidate)
      if rebuilt is not None:
        return rebuilt

    raise ValueError(
        'colocated restore produced a pytree structure that does not match the '
        f'caller template structure. Got: {jax.tree.structure(restored_state)} '
        f'vs Expected: {jax.tree.structure(template_state)}'
    )

  def _finalize_restore_result(
      self,
      *,
      step: int | None,
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
