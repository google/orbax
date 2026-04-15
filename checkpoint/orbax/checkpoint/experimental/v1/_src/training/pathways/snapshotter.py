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

"""Manages asynchronous backups of JAX array states to pinned host memory."""

import collections
from typing import Any

from etils import epath
import jax
from orbax.checkpoint.experimental.v1 import training
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types


class Snapshotter:
  """Manages asynchronous backups of JAX array states to pinned host memory."""

  _snapshots: collections.deque[tuple[tree_types.PyTree, int]]

  def __init__(self):
    self._snapshots = collections.deque(maxlen=2)

  def save_pytree(self, step: int, state: Any) -> None:
    """Move arrays onto CPU worker devices."""
    pinned_shardings = jax.tree.map(
        lambda x: x.sharding.with_memory_kind("pinned_host"), state
    )

    pinned_state = jax.device_put(state, pinned_shardings)

    self._snapshots.append((pinned_state, step))

  def load_pytree(self, abstract_state: Any) -> tree_types.PyTree:
    """Move arrays from workers onto TPU devices.

    Uses `abstract_state.sharding` to properly re-partition onto the new mesh.

    Args:
      abstract_state: An abstract representation of the state, used to provide
        the target shardings for the restored arrays on the TPU devices.

    Returns:
      The restored array state.

    Raises:
      RuntimeError: If no snapshots are available to restore from.
    """
    if not self._snapshots:
      raise RuntimeError("No snapshots available to restore from.")

    pinned_state, _ = self._snapshots[-1]

    # Re-shard on host to the target device mesh
    host_target_shardings = jax.tree.map(
        lambda x: x.sharding.with_memory_kind("pinned_host"), abstract_state
    )
    host_target_state = jax.device_put(pinned_state, host_target_shardings)

    # Move from host back to device (TPU) memory.
    restored_state = jax.device_put(
        host_target_state, jax.tree.map(lambda x: x.sharding, abstract_state)
    )
    jax.block_until_ready(restored_state)

    return restored_state

  @property
  def latest(self) -> training.CheckpointMetadata[None] | None:
    """Returns the training step of the most recently pinned backup."""
    if not self._snapshots:
      return None
    return training.CheckpointMetadata(
        step=self._snapshots[-1][1],
        path=epath.Path(),
        metadata=None,
    )
