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

"""Worker-side colocated checkpoint manager for Pathways SC."""
import collections
import itertools
import time
from typing import Any

from absl import logging
from etils import epath
import jax
from jax.experimental import colocated_python
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint import args as args_lib
from orbax.checkpoint._src.futures import signaling_client
from orbax.checkpoint._src.multihost import colocated_transport
from orbax.checkpoint.experimental.emergency.multi_tier_checkpointing import colocated_utils
from orbax.checkpoint.experimental.emergency.multi_tier_checkpointing import pathways_topology
from orbax.checkpoint.experimental.emergency.multi_tier_checkpointing import replicator_checkpoint_manager as rcm_lib


PyTree = Any
_STATE_ITEM_NAME = 'state'


def _check_unique_ids(name: str, ids: tuple[int, ...]) -> None:
  """Checks that all integer IDs in the given tuple are unique."""
  duplicates = {
      id_ for id_, count in collections.Counter(ids).items() if count > 1
  }
  if duplicates:
    raise ValueError(
        f'{name} must be unique. Duplicate ids: {sorted(duplicates)}.'
    )


def _array_on_sharding(
    value: np.ndarray,
    sharding: jax.sharding.Sharding,
) -> jax.Array:
  """Creates a JAX array with the given value on the specified sharding.

  Args:
    value: The numpy array containing the data.
    sharding: The sharding to use for the JAX array.

  Returns:
    A JAX array with the same shape and dtype as `value`, sharded according to
    `sharding`.
  """

  def data_callback(index):
    if index is None:
      return value
    return value[index]

  return jax.make_array_from_callback(
      value.shape,
      sharding,
      data_callback,
      dtype=value.dtype,
  )


class WorkerCheckpointManagerRaw:
  """Persistent checkpoint manager on each worker sidecar.

  State persists across calls via JAX's SINGLETON_OBJECT_STORE. The public
  worker methods stay array-friendly and reconstruct richer Orbax args locally.
  """

  def __init__(
      self,
      local_directory: str,
      mesh_shape: tuple[int, ...],
      mesh_axis_names: tuple[str, ...],
      save_interval_steps: int,
      mesh_device_ids: tuple[int, ...],
      mesh_axis_types: tuple[jax.sharding.AxisType, ...] | None = None,
      distributed_to_device_ids: tuple[tuple[int, ...], ...] | None = None,
      enable_async_checkpointing: bool = True,
      save_concurrent_gb: int | None = None,
      restore_concurrent_gb: int | None = None,
  ) -> None:
    colocated_transport.install_pathways_colocated_serialization_patch()
    signaling_client.mark_pathways_colocated_runtime_active()

    if len(mesh_device_ids) != np.prod(mesh_shape):
      raise ValueError(
          'mesh_device_ids must match mesh_shape size, got '
          f'{len(mesh_device_ids)} ids for mesh_shape={mesh_shape}.'
      )
    mesh_device_ids = tuple(int(device_id) for device_id in mesh_device_ids)
    _check_unique_ids('mesh_device_ids', mesh_device_ids)
    cpu_devices = colocated_transport.resolve_colocated_cpu_devices(
        mesh_device_ids
    )
    cpu_device_ids = tuple(int(device.id) for device in cpu_devices)
    _check_unique_ids('resolved colocated CPU device ids', cpu_device_ids)

    cpu_mesh = jax.sharding.Mesh(
        np.array(cpu_devices).reshape(mesh_shape),
        mesh_axis_names,
        axis_types=mesh_axis_types,
    )
    distributed_to_device_ids_fn = None
    local_distributed_to_device_ids = None
    if distributed_to_device_ids is not None:
      distributed_ids = sorted([
          int(device_id)
          for device_id in itertools.chain.from_iterable(
              distributed_to_device_ids
          )
      ])
      _check_unique_ids('distributed_to_device_ids', tuple(distributed_ids))
      local_distributed_to_device_ids = (
          pathways_topology.remap_nested_device_ids(
              distributed_to_device_ids,
              mesh_device_ids,
              cpu_device_ids,
              nested_device_ids_name='distributed_to_device_ids',
              source_device_ids_name='mesh_device_ids',
              target_device_ids_name='resolved colocated CPU device ids',
          )
      )
      local_distributed_to_device_ids = [
          list(device_ids) for device_ids in local_distributed_to_device_ids
      ]

      def get_distributed_to_device_ids() -> list[list[int]]:
        return local_distributed_to_device_ids

      distributed_to_device_ids_fn = get_distributed_to_device_ids

    self._rcm = rcm_lib.ReplicatorCheckpointManager(
        epath.Path(local_directory),
        options=rcm_lib.ReplicatorCheckpointManagerOptions(
            save_interval_steps=save_interval_steps,
            enable_async_checkpointing=enable_async_checkpointing,
            save_concurrent_gb=save_concurrent_gb,
            restore_concurrent_gb=restore_concurrent_gb,
        ),
        global_mesh=cpu_mesh,
        _is_sidecar=True,
        _distributed_to_device_ids_fn=distributed_to_device_ids_fn,
    )
    self._enable_async_checkpointing = enable_async_checkpointing
    self._save_concurrent_gb = save_concurrent_gb
    logging.info(
        'Pathways colocated MTC sidecar initialized: local_directory=%s, '
        'async_checkpointing=%s, save_concurrent_gb=%s, '
        'restore_concurrent_gb=%s, mesh_shape=%s, '
        'controller_to_local_cpu_ids=%s, distributed_to_local_cpu_ids=%s.',
        local_directory,
        enable_async_checkpointing,
        save_concurrent_gb,
        restore_concurrent_gb,
        mesh_shape,
        colocated_utils.value_sample(
            f'{controller_id}->{local_id}'
            for controller_id, local_id in zip(mesh_device_ids, cpu_device_ids)
        ),
        (
            colocated_utils.nested_id_sample(local_distributed_to_device_ids)
            if local_distributed_to_device_ids is not None
            else None
        ),
    )

  def save(
      self,
      step_array: jax.Array,
      force_array: jax.Array,
      state: PyTree,
  ) -> jax.Array:
    """Saves checkpoint on the worker, returning whether a save occurred.

    Args:
      step_array: The training step encoded as a scalar JAX array.
      force_array: Boolean encoded as array forcing the save.
      state: The PyTree payload.

    Returns:
      A scalar JAX array containing True if a step was saved.
    """
    save_start = time.time()
    logging.info(
        'Pathways colocated MTC sidecar save step=<unknown>: entered worker '
        'save function.'
    )
    step = int(np.asarray(step_array))
    logging.info(
        'Pathways colocated MTC sidecar save step=%s: resolved worker save '
        'step.',
        step,
    )
    if step == colocated_utils.NO_STEP_SENTINEL:
      return colocated_utils.make_scalar_on_like(
          False, step_array, dtype=jnp.bool_
      )
    force = bool(np.asarray(force_array))
    save_args = args_lib.Composite(
        state=args_lib.PyTreeSave(state),
    )
    local_save_start = time.time()
    saved = self._rcm.save(step, args=save_args, force=force)
    logging.info(
        'Pathways colocated MTC sidecar save step=%s: local RCM save returned '
        'saved=%s elapsed=%.3fs async_checkpointing=%s '
        'save_concurrent_gb=%s.',
        step,
        saved,
        time.time() - local_save_start,
        self._enable_async_checkpointing,
        self._save_concurrent_gb,
    )
    logging.info(
        'Pathways colocated MTC sidecar save step=%s force=%s: finished worker '
        'save function total_elapsed=%.3fs.',
        step,
        force,
        time.time() - save_start,
    )
    return colocated_utils.make_scalar_on_like(
        saved, step_array, dtype=jnp.bool_
    )

  def should_save(self, step_array: jax.Array) -> jax.Array:
    """Returns whether a checkpoint should be saved at `step_array`."""
    step = int(np.asarray(step_array))
    if step == colocated_utils.NO_STEP_SENTINEL:
      return colocated_utils.make_scalar_on_like(
          False, step_array, dtype=jnp.bool_
      )
    should_save = self._rcm.should_save(step)
    return colocated_utils.make_scalar_on_like(
        should_save, step_array, dtype=jnp.bool_
    )

  def restore_infer(
      self, step_array: jax.Array, partial_restore_array: jax.Array
  ) -> PyTree:
    """Restores state using worker-side inference.

    A negative step means "restore the latest local step".

    Args:
      step_array: The step to restore, as a scalar JAX array.
      partial_restore_array: Whether this is a partial restore.

    Returns:
      The restored state PyTree.
    """
    step = int(np.asarray(step_array))
    partial_restore = bool(np.asarray(partial_restore_array))
    if step == colocated_utils.NO_STEP_SENTINEL:
      raise ValueError(
          'Pathways colocated MTC cannot restore step 0 because the MTC '
          'coordinator protocol reserves step 0 as "no checkpoint".'
      )
    logging.vlog(
        1,
        'Pathways colocated MTC sidecar restore step=%s partial_restore=%s: '
        'entered worker restore function.',
        step,
        partial_restore,
    )
    result = self._rcm.restore(
        None if step < 0 else step,
        args_lib.Composite(
            state=args_lib.PyTreeRestore(partial_restore=partial_restore),
        ),
    )
    if isinstance(result, args_lib.Composite):
      result = result[_STATE_ITEM_NAME]
    return result

  def latest_step(self, dummy_array: jax.Array) -> jax.Array:
    """Returns latest_step_or_sentinel as a scalar int32."""
    step = self._rcm.latest_step()
    val = step if step is not None else colocated_utils.NO_STEP_SENTINEL
    return colocated_utils.make_scalar_on_like(
        val, dummy_array, dtype=jnp.int32
    )

  def all_steps(self, dummy_array: jax.Array) -> jax.Array:
    """Returns a fixed-size array of up to colocated_utils.MAX_TRACKED_STEPS local checkpoint steps."""
    local_steps = sorted(self._rcm.all_steps())
    # Keep only the latest MAX_TRACKED_STEPS steps if there are more.
    local_steps = local_steps[-colocated_utils.MAX_TRACKED_STEPS:]
    # Pad with NO_STEP_SENTINEL if fewer than MAX_TRACKED_STEPS.
    padded_steps = local_steps + [colocated_utils.NO_STEP_SENTINEL] * (
        colocated_utils.MAX_TRACKED_STEPS - len(local_steps)
    )
    return _array_on_sharding(
        np.asarray(padded_steps, dtype=np.int32),
        colocated_utils.replicated_sharding_like(dummy_array.sharding),
    )

  def is_saving_in_progress(self, dummy_array: jax.Array) -> jax.Array:
    """Returns whether the wrapped manager still has save work in flight."""
    result = self._rcm.is_saving_in_progress()
    return colocated_utils.make_scalar_on_like(
        result, dummy_array, dtype=jnp.bool_
    )

  def wait_until_finished(self, dummy_array: jax.Array) -> jax.Array:
    """Blocks until worker-side async save work completes."""
    wait_start = time.time()
    self._rcm.wait_until_finished()
    logging.info(
        'Pathways colocated MTC sidecar async save finalizer wait complete '
        'elapsed=%.3fs.',
        time.time() - wait_start,
    )
    return colocated_utils.make_scalar_on_like(
        True, dummy_array, dtype=jnp.bool_
    )

  def check_for_errors(self, dummy_array: jax.Array) -> jax.Array:
    """Raises async checkpoint errors from the wrapped manager."""
    self._rcm.check_for_errors()
    return colocated_utils.make_scalar_on_like(
        True, dummy_array, dtype=jnp.bool_
    )

  def close(self, dummy_array: jax.Array) -> jax.Array:
    """Closes the wrapped checkpoint manager."""
    self._rcm.close()
    return colocated_utils.make_scalar_on_like(
        True, dummy_array, dtype=jnp.bool_
    )


WorkerCheckpointManager = colocated_python.colocated_python_class(
    WorkerCheckpointManagerRaw
)
