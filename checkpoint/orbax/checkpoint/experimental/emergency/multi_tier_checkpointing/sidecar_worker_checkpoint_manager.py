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
from typing import Any

from etils import epath
import jax
from jax.experimental import colocated_python
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint import args as args_lib
from orbax.checkpoint._src.futures import signaling_client
from orbax.checkpoint._src.multihost import colocated_transport
from orbax.checkpoint.experimental.emergency.multi_tier_checkpointing import colocated_utils
from orbax.checkpoint.experimental.emergency.multi_tier_checkpointing import replicator_checkpoint_manager as rcm_lib


PyTree = Any
_STATE_ITEM_NAME = 'state'


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
      mesh_axis_types: tuple[jax.sharding.AxisType, ...] | None = None,
  ) -> None:
    colocated_transport.install_pathways_colocated_serialization_patch()
    signaling_client.mark_pathways_colocated_runtime_active()

    cpu_mesh = jax.sharding.Mesh(
        np.array(jax.devices()).reshape(mesh_shape),
        mesh_axis_names,
        axis_types=mesh_axis_types,
    )

    self._rcm = rcm_lib.ReplicatorCheckpointManager(
        epath.Path(local_directory),
        options=rcm_lib.ReplicatorCheckpointManagerOptions(
            save_interval_steps=save_interval_steps,
        ),
        global_mesh=cpu_mesh,
        _is_sidecar=True,
    )

  def save(
      self,
      step_array: jax.Array,
      force_array: jax.Array,
      state: PyTree,
  ) -> jax.Array:
    """Save checkpoint on the worker, returning whether a save occurred."""
    step = int(np.asarray(step_array))
    force = bool(np.asarray(force_array))
    save_args = args_lib.Composite(
        state=args_lib.PyTreeSave(state),
    )
    saved = self._rcm.save(step, args=save_args, force=force)
    return colocated_utils.make_scalar_on_like(
        saved, step_array, dtype=jnp.bool_
    )

  def should_save(self, step_array: jax.Array) -> jax.Array:
    """Returns whether a checkpoint should be saved at `step_array`."""
    step = int(np.asarray(step_array))
    should_save = self._rcm.should_save(step)
    return colocated_utils.make_scalar_on_like(
        should_save, step_array, dtype=jnp.bool_
    )

  def restore_infer(self, step_array: jax.Array) -> PyTree:
    """Restores state using worker-side inference.

    A negative step means "restore the latest local step".

    Args:
      step_array: The step to restore, as a scalar JAX array.

    Returns:
      The restored state PyTree.
    """
    step = int(np.asarray(step_array))
    result = self._rcm.restore(
        None if step < 0 else step,
        args_lib.Composite(
            state=args_lib.PyTreeRestore(),
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
    return jax.device_put(
        jnp.asarray(padded_steps, dtype=jnp.int32),
        dummy_array.sharding,
    )

  def is_saving_in_progress(self, dummy_array: jax.Array) -> jax.Array:
    """Returns whether the wrapped manager still has save work in flight."""
    result = self._rcm.is_saving_in_progress()
    return colocated_utils.make_scalar_on_like(
        result, dummy_array, dtype=jnp.bool_
    )

  def wait_until_finished(self, dummy_array: jax.Array) -> jax.Array:
    """Blocks until worker-side async save work completes."""
    self._rcm.wait_until_finished()
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
