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

from typing import Any, Optional

from etils import epath
import jax
from jax.experimental import colocated_python
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint import args as args_lib
from orbax.checkpoint._src.futures import signaling_client as _sc
from orbax.checkpoint._src.multihost import colocated_transport
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.multihost import pathways as pathways_utils
from orbax.checkpoint.experimental.emergency.multi_tier_checkpointing import (
    colocated_utils,
)


PyTree = Any
_STATE_ITEM_NAME = 'state'


class WorkerCheckpointManagerRaw:
  """Persistent checkpoint manager on each worker sidecar.

  Wraps a standard (multi-controller) ReplicatorCheckpointManager.
  State persists across calls via JAX's SINGLETON_OBJECT_STORE.
  """

  def __init__(
      self,
      local_directory: str,
      mesh_shape: tuple[int, ...],
      mesh_axis_names: tuple[str, ...],
      save_interval_steps: int,
      mesh_axis_types: Optional[tuple[jax.sharding.AxisType, ...]] = None,
  ):
    self._threadsafe_signaling_override_enabled = (
        not multihost.is_jax_distributed_client_initialized()
    )
    colocated_transport.install_pathways_colocated_serialization_patch()

    # Sidecars run with process_count > 1 but without jax.distributed
    # initialization. Opt in to thread-safe local signaling explicitly for
    # this process.
    if self._threadsafe_signaling_override_enabled:
      _sc.acquire_threadsafe_signaling_client_override()

    # Import lazily to avoid import cycle with replicator_checkpoint_manager.
    from orbax.checkpoint.experimental.emergency import replicator_checkpoint_manager as rcm_lib  # pylint: disable=g-import-not-at-top

    cpu_mesh = jax.sharding.Mesh(
        np.array(jax.devices()).reshape(mesh_shape),
        mesh_axis_names,
        axis_types=mesh_axis_types,
    )

    # Create standard (non-colocated) RCM.
    # Barriers use default behavior (processes=None path).
    self._rcm = rcm_lib.ReplicatorCheckpointManager(
        epath.Path(local_directory),
        options=rcm_lib.ReplicatorCheckpointManagerOptions(
            save_interval_steps=save_interval_steps,
        ),
        global_mesh=cpu_mesh,
        _distributed_to_device_ids_fn=(
            lambda: pathways_utils.compute_distributed_to_device_ids(
                jax.devices()
            )
        ),
        _is_sidecar=True,
    )

  def save(
      self,
      step_array: jax.Array,
      force_array: jax.Array,
      state: PyTree,
  ) -> jax.Array:
    """Save checkpoint. Returns bool scalar."""
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
    """Restores state using the standard worker-side inference path."""
    step = int(np.asarray(step_array))
    result = self._rcm.restore(
        step,
        args_lib.Composite(
            state=args_lib.PyTreeRestore(),
        ),
    )
    if isinstance(result, args_lib.Composite):
      result = result[_STATE_ITEM_NAME]
    return result

  def latest_step(self, dummy_array: jax.Array) -> jax.Array:
    """Returns latest_step_or_-1 as a scalar int32."""
    step = self._rcm.latest_step()
    val = step if step is not None else -1
    return colocated_utils.make_scalar_on_like(
        val, dummy_array, dtype=jnp.int32
    )

  def is_saving_in_progress(self, dummy_array: jax.Array) -> jax.Array:
    result = self._rcm.is_saving_in_progress()
    return colocated_utils.make_scalar_on_like(
        result, dummy_array, dtype=jnp.bool_
    )

  def wait_until_finished(self, dummy_array: jax.Array) -> jax.Array:
    self._rcm.wait_until_finished()
    return colocated_utils.make_scalar_on_like(
        True, dummy_array, dtype=jnp.bool_
    )

  def close(self, dummy_array: jax.Array) -> jax.Array:
    self._rcm.close()
    # Limit signaling override lifetime to this sidecar object.
    if getattr(self, '_threadsafe_signaling_override_enabled', False):
      _sc.release_threadsafe_signaling_client_override()
    return colocated_utils.make_scalar_on_like(
        True, dummy_array, dtype=jnp.bool_
    )


WorkerCheckpointManager = colocated_python.colocated_python_class(
    WorkerCheckpointManagerRaw
)
