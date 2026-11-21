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

from collections.abc import Sequence
import contextlib
import logging
import queue
import threading
from typing import Any

from etils import epath
import jax
from orbax.checkpoint.experimental.v1._src.training.metadata import types as training_metadata_types
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types


def is_shardable_array(x: ...) -> bool:  # pyrefly: ignore[invalid-annotation]
  """Returns True if x is a concrete shardable array."""
  return isinstance(x, jax.Array)


def _is_prng_key(x: Any) -> bool:
  """Returns True if x has a JAX PRNGKeyArray dtype."""
  return (
      hasattr(x, "dtype")
      and hasattr(x, "shape")
      and jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key)
  )


def _unwrap_if_prng_key(x: Any) -> Any:
  """Extracts the underlying key data buffer if x is a PRNGKeyArray."""
  return jax.random.key_data(x) if _is_prng_key(x) else x


def _wrap_if_prng_key(x: Any, orig_x: Any) -> Any:
  """Wraps raw array x into a PRNGKeyArray if orig_x was a PRNGKeyArray."""
  if _is_prng_key(orig_x):
    return jax.random.wrap_key_data(x, dtype=orig_x.dtype)
  return x


def _is_replica_active(replica_tree: tree_types.PyTree) -> bool:
  """Returns True if `replica_tree`'s slice placement is reachable."""
  probe = next(
      filter(is_shardable_array, jax.tree.leaves(replica_tree)), None
  )
  if probe is None:
    return True
  try:
    jax.block_until_ready(_unwrap_if_prng_key(probe)[...])
    return True
  except (jax.errors.JaxRuntimeError, RuntimeError):
    return False


def _find_named_mesh(tree: tree_types.PyTree) -> jax.sharding.Mesh | None:
  """Returns the mesh of the first `NamedSharding` leaf in `tree`, or None."""
  for leaf in jax.tree.leaves(tree):
    sharding = getattr(leaf, "sharding", None)
    if isinstance(sharding, jax.sharding.NamedSharding) and isinstance(
        sharding.mesh, jax.sharding.Mesh
    ):
      return sharding.mesh
  return None


def _extract_arrays(tree: tree_types.PyTree) -> list[jax.Array]:
  """Returns unwrapped `jax.Array` leaves of `tree`."""
  return [
      _unwrap_if_prng_key(leaf)
      for leaf in jax.tree.leaves(tree)
      if is_shardable_array(leaf)
  ]


def _replace_arrays(
    tree: tree_types.PyTree, arrays: Sequence[jax.Array]
) -> tree_types.PyTree:
  """Returns `tree` with its `jax.Array` leaves replaced by `arrays`."""
  leaves, tree_def = jax.tree.flatten(tree)
  array_iter = iter(arrays)
  for i, leaf in enumerate(leaves):
    if is_shardable_array(leaf):
      leaves[i] = _wrap_if_prng_key(next(array_iter), leaf)
  return jax.tree.unflatten(tree_def, leaves)


def _target_sharding(leaf: jax.Array, target_leaf: Any) -> Any:
  """Returns the target `jax.sharding.Sharding` for `leaf`."""
  if isinstance(target_leaf, jax.sharding.Sharding):
    return target_leaf
  if (target_shape := tuple(target_leaf.shape)) != leaf.shape:
    raise RuntimeError(
        f"Restored replica shape mismatch: {target_shape} != {leaf.shape}."
    )
  return target_leaf.sharding


def _put_arrays(
    state: tree_types.PyTree, target: tree_types.PyTree | None = None
) -> tree_types.PyTree:
  """Moves array leaves of `state` onto `target` shardings via device memory."""
  leaves, tree_def = jax.tree.flatten(state)
  arrays = [
      _unwrap_if_prng_key(leaf) for leaf in leaves if is_shardable_array(leaf)
  ]
  if target is None:
    target_shardings = [
        array.sharding.with_memory_kind(None) for array in arrays
    ]
  else:
    targets, target_def = jax.tree.flatten(target)
    if tree_def != target_def:
      raise ValueError("Snapshot PyTree structure does not match target.")
    target_shardings = [
        _target_sharding(leaf, target_leaf)
        for leaf, target_leaf in zip(leaves, targets, strict=True)
        if is_shardable_array(leaf)
    ]
  # Stage cross-device transfers through device memory (b/574655289).
  stages = (
      [array.sharding.with_memory_kind(None) for array in arrays],
      [sharding.with_memory_kind(None) for sharding in target_shardings],
      target_shardings,
  )
  for stage_shardings in stages:
    to_move = [
        i
        for i, (array, sharding, target_sharding) in enumerate(
            zip(arrays, stage_shardings, target_shardings, strict=True)
        )
        if array.sharding not in (sharding, target_sharding)
    ]
    if to_move:
      moved = jax.device_put(
          [arrays[i] for i in to_move],
          [stage_shardings[i] for i in to_move],
      )
      for i, array in zip(to_move, moved, strict=True):
        arrays[i] = array
  return _replace_arrays(state, arrays)


def _replicate_single_device_leaves(
    tree: tree_types.PyTree,
) -> tree_types.PyTree:
  """Replicates single-device array leaves across the mesh of `tree`."""
  mesh = _find_named_mesh(tree)
  if mesh is None:
    return tree
  replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
  arrays = _extract_arrays(tree)
  shardings = [
      replicated
      if isinstance(array.sharding, jax.sharding.SingleDeviceSharding)
      else array.sharding
      for array in arrays
  ]
  return _replace_arrays(tree, _put_arrays(arrays, shardings))


def _get_pathwaysutils():
  """Imports and returns pathwaysutils experimental modules."""
  try:
    # pylint: disable=g-import-not-at-top
    from pathwaysutils.experimental import concatenate_by_mesh_axis  # pyrefly: ignore[missing-import]
    from pathwaysutils.experimental import split_by_mesh_axis  # pyrefly: ignore[missing-import]
  except ImportError as e:
    raise ImportError(
        "Snapshotter requires pathwaysutils. Please ensure pathwaysutils is"
        " installed or linked via"
        " //orbax/checkpoint/experimental/v1:pathways_support."
    ) from e
  return concatenate_by_mesh_axis, split_by_mesh_axis


class Snapshotter:
  """Manages asynchronous backups of JAX array states to pinned host memory."""

  def __init__(self, *, replica_axis_index: int = 0):
    """Initializes Snapshotter.

    Args:
      replica_axis_index: The index of the mesh axis representing data/model
        replicas. Typically corresponds to physical slices (connected by DCN).
    """
    self._concatenate_by_mesh_axis, self._split_by_mesh_axis = (
        _get_pathwaysutils()
    )
    self._latest_snapshot: tuple[list[tree_types.PyTree], int] | None = None
    self._lock = threading.Lock()
    self._queue = queue.Queue(maxsize=1)
    self._generation = 0
    self.replica_axis_index = replica_axis_index

    self._worker_thread = threading.Thread(target=self._worker, daemon=True)
    self._worker_thread.start()

  def _worker(self):
    """Processes background snapshot requests from the queue."""
    while True:
      pinned_state, step, generation = self._queue.get()
      try:
        with self._lock:
          if generation != self._generation:
            continue
        jax.block_until_ready(_extract_arrays(pinned_state))
      except (jax.errors.JaxRuntimeError, RuntimeError) as e:
        logging.exception("Failed to snapshot state at step %d: %s", step, e)
      else:
        with self._lock:
          if generation == self._generation:
            self._latest_snapshot = (pinned_state, step)
      finally:
        del pinned_state
        self._queue.task_done()

  def _to_snapshot_state(
      self, state: tree_types.PyTree
  ) -> list[tree_types.PyTree]:
    """Converts a PyTree into a list of replica PyTrees in pinned host memory.

    Args:
      state: The PyTree to be converted and split.

    Returns:
      A list of PyTrees where each PyTree represents a single replica of
      `state`. Non-array leaves (e.g., scalars, strings, metadata, Python
      objects) are preserved across all replica PyTrees.
    """
    replicated = _replicate_single_device_leaves(state)
    arrays = _extract_arrays(replicated)
    if not arrays:
      return [state]
    axis = ""
    replica_counts = set()
    for array in arrays:
      if isinstance(array.sharding, jax.sharding.NamedSharding):
        mesh = array.sharding.mesh
        axis = mesh.axis_names[self.replica_axis_index]
        replica_counts.add(mesh.shape[axis])
      else:
        replica_counts.add(1)
    if len(replica_counts) > 1:
      raise ValueError(
          "All JAX arrays must have the same number of replicas along the"
          f" replica axis, got {sorted(replica_counts)}."
      )
    pinned = jax.device_put(
        arrays,
        [array.sharding.with_memory_kind("pinned_host") for array in arrays],
        may_alias=False,
    )
    if replica_counts == {1}:
      return [_replace_arrays(replicated, pinned)]
    replicas = self._split_by_mesh_axis.split_by_mesh_axis(pinned, axis)
    return [_replace_arrays(replicated, replica) for replica in replicas]

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

    with self._lock:
      generation = self._generation
    pinned_state = self._to_snapshot_state(state)
    self._queue.put((pinned_state, step, generation))

  def load(
      self,
      abstract_state: tree_types.PyTree,
      *,
      reset_snapshot_state: bool | None = None,
  ) -> tree_types.PyTree:
    """Move arrays from workers onto TPU devices.

    Uses `abstract_state.sharding` to properly re-partition onto the new mesh.

    Args:
      abstract_state: An abstract representation of the state, used to provide
        the target shardings for the restored arrays on the TPU devices.
      reset_snapshot_state: If True, clears snapshot history and resets it to
        contain only the returned restored state (in host-pinned memory). If
        None, resets only when the mesh of `abstract_state` has more than one
        replica along the replica axis.

    Returns:
      The restored array state.

    Raises:
      RuntimeError: If no snapshots are available to restore from.
    """
    self.join()
    with self._lock:
      if self._latest_snapshot is None:
        raise RuntimeError("No snapshots available to restore from.")
      replica_trees, step = self._latest_snapshot

    active_trees = [tree for tree in replica_trees if _is_replica_active(tree)]
    if not active_trees:
      raise RuntimeError("No active replicas found.")

    # TODO(b/574655289): Pathways on Cloud workers crash on cross-slice
    # transfers of pinned_host buffers over RpcTransport ("Attempting to
    # decrease reference on a buffer with zero reference count") until their
    # images include the PJRT C API fix. Until then, copy each replica to device
    # memory on its own devices, then concatenate the copies and reshard them in
    # device memory. In either memory, concatenate copies rather than the
    # snapshot's replicas: concatenating donates its inputs, and the snapshot
    # must survive a failed load (e.g. a slice going down) for it to be retried.
    mesh = _find_named_mesh(abstract_state)
    device_state = self._restore_replicas(active_trees, mesh)
    restored_state = _put_arrays(device_state, abstract_state)
    jax.block_until_ready(_extract_arrays(restored_state))

    if reset_snapshot_state is None:
      reset_snapshot_state = bool(
          mesh and mesh.shape[mesh.axis_names[self.replica_axis_index]] > 1
      )
    if reset_snapshot_state:
      new_snapshot_state = self._to_snapshot_state(restored_state)
      jax.block_until_ready(_extract_arrays(new_snapshot_state))
      with self._lock:
        self._latest_snapshot = (new_snapshot_state, step)

    return restored_state

  def _restore_replicas(
      self,
      active_trees: Sequence[tree_types.PyTree],
      target_mesh: jax.sharding.Mesh | None,
  ) -> tree_types.PyTree:
    """Moves `active_trees` to device memory and concatenates along slices."""
    replica_mesh = _find_named_mesh(active_trees[0])
    if replica_mesh is None:
      return _put_arrays(active_trees[0])

    axis_index = self.replica_axis_index
    axis_name = replica_mesh.axis_names[axis_index]
    replicas = [_extract_arrays(tree) for tree in active_trees]
    can_match = (
        target_mesh is not None
        and target_mesh.axis_names == replica_mesh.axis_names
        and target_mesh.axis_types == replica_mesh.axis_types
        and target_mesh.devices.take([0], axis=axis_index).shape
        == replica_mesh.devices.shape
        and all(
            isinstance(array.sharding, jax.sharding.NamedSharding)
            and array.sharding.mesh == replica_mesh
            and not any(
                axis_name in jax.tree.leaves(dim)
                for dim in array.sharding.spec
            )
            for array in replicas[0]
        )
    )
    if target_mesh is None or not can_match:
      slices = _put_arrays(replicas)
    else:
      submeshes = [
          jax.sharding.Mesh(
              target_mesh.devices.take([i], axis=axis_index),
              target_mesh.axis_names,
              axis_types=target_mesh.axis_types,
          )
          for i in range(target_mesh.devices.shape[axis_index])
      ]
      pinned_by_mesh = {
          _find_named_mesh(replica): replica for replica in replicas
      }
      missing = [
          submesh for submesh in submeshes if submesh not in pinned_by_mesh
      ]
      if not missing:
        slices = _put_arrays([pinned_by_mesh[s] for s in submeshes])
      else:
        donors = _put_arrays(replicas)
        by_mesh = {donor[0].sharding.mesh: donor for donor in donors}
        specs = [array.sharding.spec for array in donors[0]]
        sources = [
            [donors[(k + j) % len(donors)][j] for j in range(len(specs))]
            for k in range(len(missing))
        ]
        target_shardings = [
            [jax.sharding.NamedSharding(submesh, spec) for spec in specs]
            for submesh in missing
        ]
        by_mesh.update(
            zip(missing, _put_arrays(sources, target_shardings), strict=True)
        )
        slices = [by_mesh[s] for s in submeshes]

    if len(slices) == 1:
      return _replace_arrays(active_trees[0], slices[0])
    concatenated = self._concatenate_by_mesh_axis.concatenate_by_mesh_axis(
        slices, axis_name
    )
    return _replace_arrays(active_trees[0], concatenated)

  def try_load(
      self,
      abstract_state: tree_types.PyTree,
      *,
      reset_snapshot_state: bool | None = None,
  ) -> tuple[int | None, tree_types.PyTree | None]:
    """Restores `(step, state)` from the latest snapshot, or `(None, None)`.

    Args:
      abstract_state: An abstract representation of the state, used to provide
        the target shardings for the restored arrays on the TPU devices.
      reset_snapshot_state: If True, resets the snapshot to the restored state.
        If None, resets only when the mesh of `abstract_state` has more than
        one replica along the replica axis.

    Returns:
      `(step, restored_state)`, or `(None, None)` if no snapshot is available
      or if `load` raises a non-JAX `RuntimeError` (clearing that snapshot).

    Raises:
      jax.errors.JaxRuntimeError: If a JAX or Pathways runtime error occurs.
      ValueError: If `abstract_state` does not match the snapshot structure.
    """
    self.join()
    with self._lock:
      if self._latest_snapshot is None:
        return None, None
      _, step = self._latest_snapshot
    try:
      return step, self.load(
          abstract_state, reset_snapshot_state=reset_snapshot_state
      )
    except jax.errors.JaxRuntimeError:
      raise
    except RuntimeError as e:
      logging.warning("Snapshot restore failed at step %d: %s", step, e)
      with self._lock:
        if (
            self._latest_snapshot is not None
            and self._latest_snapshot[1] == step
        ):
          self._latest_snapshot = None
      return None, None

  def join(self) -> None:
    """Blocks until all queued snapshot requests have been processed."""
    self._queue.join()

  def cancel_pending(self, timeout: float | None = 0.0) -> bool:
    """Cancels pending or in-flight snapshots, keeping the latest completed one.

    Args:
      timeout: Maximum seconds to wait for an in-flight snapshot to finish.
        Defaults to 0.0 seconds. If None, waits without a timeout.

    Returns:
      True if all queued requests finished within `timeout`, else False.
    """
    with self._lock:
      self._generation += 1
      with contextlib.suppress(queue.Empty):
        while True:
          self._queue.get_nowait()
          self._queue.task_done()
    with self._queue.all_tasks_done:
      return self._queue.all_tasks_done.wait_for(
          lambda: not self._queue.unfinished_tasks, timeout=timeout
      )

  @property
  def latest(self) -> training_metadata_types.CheckpointMetadata[None] | None:
    """Returns the training step of the most recently pinned backup."""
    with self._lock:
      if self._latest_snapshot is None:
        return None
      _, step = self._latest_snapshot
    return training_metadata_types.CheckpointMetadata(
        step=step,
        path=epath.Path(),
        metadata=None,
    )
