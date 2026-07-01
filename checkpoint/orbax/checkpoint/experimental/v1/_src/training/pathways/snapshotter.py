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

import logging
import queue
import threading

from etils import epath
import jax
from orbax.checkpoint.experimental.v1 import training
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types
from pathwaysutils.experimental import concatenate_by_mesh_axis
from pathwaysutils.experimental import split_by_mesh_axis


def is_shardable_array(x: ...) -> bool:  # pyrefly: ignore[invalid-annotation]
  """Returns True if x is a concrete shardable array."""
  return isinstance(x, jax.Array)


class Snapshotter:
  """Manages asynchronous backups of JAX array states to pinned host memory."""

  def __init__(self, *, replica_axis_index: int = 0):
    self._latest_snapshot: tuple[tree_types.PyTree, int] | None = None
    self._lock = threading.Lock()
    self._queue = queue.Queue(maxsize=1)
    self.replica_axis_index = replica_axis_index

    self._worker_thread = threading.Thread(target=self._worker, daemon=True)
    self._worker_thread.start()

  def _worker(self):
    while True:
      pinned_state, step = self._queue.get()
      try:
        jax.block_until_ready(pinned_state)
        with self._lock:
          self._latest_snapshot = (pinned_state, step)
      finally:
        self._queue.task_done()

  def save(self, step: int, state: tree_types.PyTree) -> None:
    """Backs up JAX array states to pinned host memory, asynchronously.

    If previous snapshotting requests are still in progress, this request may
    be skipped.

    The saved state contains all replicas of user data. When restoring via
    `load_pytree`, snapshotter is able to reconstruct user data even if some
    replicas are unavailable, making it resilient to failures of some replicas.

    Args:
      step: The training step number associated with the state.
      state: The PyTree to be saved.
    """
    if self._queue.full():
      logging.warning("Snapshotter busy. Skipping snapshot for step %d", step)
      return

    pinned_state = jax.tree.map(
        lambda x: jax.device_put(x, x.sharding.with_memory_kind("pinned_host"))
        if is_shardable_array(x)
        else x,
        state,
    )

    self._queue.put((pinned_state, step))

  def load(
      self,
      abstract_state: tree_types.PyTree,
      *,
      reset_snapshot_state: bool = True,
  ) -> tree_types.PyTree:
    """Move arrays from workers onto TPU devices.

    Uses `abstract_state.sharding` to properly re-partition onto the new mesh.

    Args:
      abstract_state: An abstract representation of the state, used to provide
        the target shardings for the restored arrays on the TPU devices.
      reset_snapshot_state: If True, clears snapshot history and resets it to
        contain only the returned restored state (in host-pinned memory).

    Returns:
      The restored array state.

    Raises:
      RuntimeError: If no snapshots are available to restore from.
    """
    with self._lock:
      if self._latest_snapshot is None:
        raise RuntimeError("No snapshots available to restore from.")
      pinned_state, step = self._latest_snapshot

    def is_replica_active(arr):
      try:
        jax.block_until_ready(arr)
        return True
      except jax.errors.JaxRuntimeError as _:
        return False

    def get_active_pytree(x):
      mesh_axis_name = x.sharding.mesh.axis_names[self.replica_axis_index]
      all_replicas = split_by_mesh_axis.split_by_mesh_axis(
          x,
          mesh_axis_name,
      )

      active_replicas = [
          replica for replica in all_replicas if is_replica_active(replica)
      ]

      if not active_replicas:
        raise RuntimeError(
            "No active replicas found."
        )

      reconstructed_state = concatenate_by_mesh_axis.concatenate_by_mesh_axis(
          active_replicas,
          mesh_axis_name,
      )
      return reconstructed_state

    pinned_state = jax.tree.map(
        lambda x: get_active_pytree(x) if is_shardable_array(x) else x,
        pinned_state,
    )

    def _device_put_pinned(x, abs_x):
      if is_shardable_array(x):
        return jax.device_put(
            x, abs_x.sharding.with_memory_kind("pinned_host")
        )
      return x

    # Re-shard on host to the target device mesh
    host_target_state = jax.tree.map(
        _device_put_pinned,
        pinned_state,
        abstract_state,
    )

    def _device_put_to_device(x, abs_x):
      if is_shardable_array(x):
        return jax.device_put(x, abs_x.sharding.with_memory_kind(None))
      return x

    # Move from host back to device (TPU) memory.
    restored_state = jax.tree.map(
        _device_put_to_device,
        host_target_state,
        abstract_state,
    )
    jax.block_until_ready(restored_state)

    if reset_snapshot_state:
      with self._lock:
        self._latest_snapshot = (host_target_state, step)

    return restored_state

  @property
  def latest(self) -> training.CheckpointMetadata[None] | None:
    """Returns the training step of the most recently pinned backup."""
    with self._lock:
      if self._latest_snapshot is None:
        return None
      _, step = self._latest_snapshot
    return training.CheckpointMetadata(
        step=step,
        path=epath.Path(),
        metadata=None,
    )
