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

from collections.abc import Callable
import contextlib
import logging
import queue
import threading
from typing import Any

from etils import epath
import jax
from orbax.checkpoint.experimental.v1._src.training.metadata import types as training_metadata_types
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types

_DEFAULT_JOIN_TIMEOUT_SECS = 120.0


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
    return jax.random.wrap_key_data(x, impl=jax.random.key_impl(orig_x))
  return x


def _pin_host(array: Any) -> Any:
  """Copies the unwrapped JAX `array` into pinned host memory."""
  return jax.device_put(
      array, array.sharding.with_memory_kind("pinned_host"), may_alias=False
  )


def _has_sharding_without_mesh(leaf: Any) -> bool:
  """Returns True if `leaf` has a JAX `Sharding` without a `Mesh`."""
  sharding = getattr(leaf, "sharding", None)
  return isinstance(sharding, jax.sharding.Sharding) and not hasattr(
      sharding, "mesh"
  )


def _find_named_mesh(tree: Any) -> Any:
  """Returns the first `Mesh` with named axes in `tree`, or None."""
  for leaf in jax.tree.leaves(tree):
    mesh = getattr(getattr(leaf, "sharding", None), "mesh", None)
    if getattr(mesh, "axis_names", ()):
      return mesh
  return None


def _put_array(array: Any, target_sharding: Any) -> Any:
  """Moves `array` (or a PRNGKey) onto `target_sharding` via device memory."""
  unwrapped = _unwrap_if_prng_key(array)
  if unwrapped.sharding == target_sharding:
    return array
  if unwrapped.sharding.memory_kind == "pinned_host":
    unwrapped = jax.device_put(
        unwrapped, unwrapped.sharding.with_memory_kind(None)
    )
  for next_sharding in (
      target_sharding.with_memory_kind(None),
      target_sharding,
  ):
    if unwrapped.sharding != next_sharding:
      unwrapped = jax.device_put(unwrapped, next_sharding)
  return _wrap_if_prng_key(unwrapped, array)


def _replicate_single_device_leaves(tree: Any) -> Any:
  """Replicates single-device array or spec leaves across the tree's mesh."""
  has_single_device_leaf = any(
      _has_sharding_without_mesh(leaf) for leaf in jax.tree.leaves(tree)
  )
  if not has_single_device_leaf:
    return tree

  mesh = _find_named_mesh(tree)
  if mesh is None:
    return tree

  replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

  def _replicate(leaf: Any) -> Any:
    if not _has_sharding_without_mesh(leaf):
      return leaf
    if isinstance(leaf, jax.ShapeDtypeStruct):
      return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=replicated)
    return _put_array(leaf, replicated)

  return jax.tree.map(_replicate, tree)


def _restore_shardings(
    state: tree_types.PyTree,
    target: tree_types.PyTree,
    memory_kind: str | None = "keep",
) -> tree_types.PyTree:
  """Moves array leaves of `state` onto the shardings of `target`."""

  def _restore_leaf(leaf: Any, target_leaf: Any) -> Any:
    sharding = getattr(target_leaf, "sharding", None)
    if not is_shardable_array(leaf) or not isinstance(
        sharding, jax.sharding.Sharding
    ):
      return leaf
    shape = getattr(target_leaf, "shape", None)
    if shape is not None and tuple(shape) != leaf.shape:
      raise RuntimeError("Restored replica shape mismatch.")
    target_sharding: Any = sharding
    target_sharding = (
        target_sharding
        if memory_kind == "keep"
        else target_sharding.with_memory_kind(memory_kind)
    )
    return _put_array(leaf, target_sharding)

  return jax.tree.map(_restore_leaf, state, target)


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
    # The replicas of the latest snapshot, its step and its full-mesh state.
    self._latest_snapshot: (
        tuple[list[tree_types.PyTree], int, tree_types.PyTree] | None
    ) = None
    self._lock = threading.Lock()
    self._queue = queue.Queue(maxsize=1)
    self._generation = 0
    self.replica_axis_index = replica_axis_index

    self._worker_thread = threading.Thread(target=self._worker, daemon=True)
    self._worker_thread.start()

  def _worker(self):
    """Processes background snapshot requests from the queue."""
    while True:
      full_mesh_pinned_state, per_replica_pinned_trees, step, generation = (
          self._queue.get()
      )
      try:
        # `cancel_pending` ran after `save` queued this request, so discard it.
        if generation != self._generation:
          continue
        jax.block_until_ready(
            jax.tree.map(
                _unwrap_if_prng_key,
                (full_mesh_pinned_state, per_replica_pinned_trees),
            )
        )
      except (jax.errors.JaxRuntimeError, RuntimeError) as e:
        logging.exception("Failed to snapshot state at step %d: %s", step, e)
      else:
        with self._lock:
          if generation == self._generation:
            self._latest_snapshot = (
                per_replica_pinned_trees,
                step,
                full_mesh_pinned_state,
            )
      finally:
        del full_mesh_pinned_state, per_replica_pinned_trees
        self._queue.task_done()

  def _to_snapshot_state(
      self, state: tree_types.PyTree
  ) -> tuple[tree_types.PyTree, list[tree_types.PyTree]]:
    """Copies a PyTree to pinned host memory, whole and split into replicas.

    Args:
      state: The PyTree to be converted and split.

    Returns:
      A tuple of `state` with its arrays in pinned host memory, and a list of
      PyTrees where each PyTree represents a single replica of `state`. In all
      of these PyTrees, single-device arrays are replicated across the mesh, and
      non-array leaves (e.g., scalars, strings, metadata, Python objects) are
      preserved.
    """
    replicated = _replicate_single_device_leaves(state)
    leaves, tree_def = jax.tree_util.tree_flatten(replicated)
    full_mesh_pinned_leaves: list[Any] = []
    per_replica_pinned_leaves = []
    num_replicas = None

    for x in leaves:
      if not is_shardable_array(x):
        full_mesh_pinned_leaves.append(x)
        per_replica_pinned_leaves.append(x)
      else:
        data = _unwrap_if_prng_key(x)
        pinned = _pin_host(data)
        full_mesh_pinned_leaves.append(_wrap_if_prng_key(pinned, x))
        mesh = getattr(pinned.sharding, "mesh", None)
        replicas = [pinned]
        if mesh is not None:
          axis = mesh.axis_names[self.replica_axis_index]
          if mesh.shape[axis] > 1:
            replicas = self._split_by_mesh_axis.split_by_mesh_axis(pinned, axis)
        if num_replicas is None:
          num_replicas = len(replicas)
        elif len(replicas) != num_replicas:
          raise ValueError(
              "All JAX arrays must have the same number of replicas along the"
              f" replica axis, got {len(replicas)} and {num_replicas}."
          )
        reps = [_wrap_if_prng_key(rep, x) for rep in replicas]
        per_replica_pinned_leaves.append(reps)

    if num_replicas is None:
      return state, [state]

    per_replica_pinned_trees = []
    for r in range(num_replicas):
      r_leaves = [
          reps[r] if is_shardable_array(orig_x) else reps
          for reps, orig_x in zip(per_replica_pinned_leaves, leaves)
      ]
      per_replica_pinned_trees.append(
          jax.tree_util.tree_unflatten(tree_def, r_leaves)
      )

    full_mesh_pinned_state = jax.tree_util.tree_unflatten(
        tree_def, full_mesh_pinned_leaves
    )
    return full_mesh_pinned_state, per_replica_pinned_trees

  def _enqueue_snapshot(
      self, step: int, state: tree_types.PyTree
  ) -> tree_types.PyTree:
    """Queues the snapshot of `state` at `step` for the worker thread.

    Args:
      step: The training step number associated with the state.
      state: The PyTree to be saved.

    Returns:
      `state` with its arrays in pinned host memory, as the snapshot stores it.
    """
    pinned_state, per_replica_pinned_trees = self._to_snapshot_state(state)
    self._queue.put(
        (pinned_state, per_replica_pinned_trees, step, self._generation)
    )
    return pinned_state

  def save(
      self,
      step: int,
      state: tree_types.PyTree,
      *,
      wait: bool = False,
      checkpointer_wait_fn: Callable[[], Any] | None = None,
      checkpointer_save_fn: Callable[[tree_types.PyTree], Any] | None = None,
  ) -> Any:
    """Backs up JAX array states to pinned host memory.

    By default, the backup runs asynchronously. If previous snapshotting
    requests are still in progress, this request may be skipped.

    The saved state contains all replicas of user data. When restoring via
    `load_pytree`, snapshotter is able to reconstruct user data even if some
    replicas are unavailable, making it resilient to failures of some replicas.

    With `wait` or a checkpointer function, this method instead takes the
    snapshot of `step` before it returns, in the following order:

    1. Calls `checkpointer_wait_fn`, so that the previous checkpoint save
       releases its host arrays before the snapshot allocates new ones.
    2. Waits for the snapshotting requests in progress.
    3. Takes the snapshot.
    4. Calls `checkpointer_save_fn` with `state` backed by the host arrays of
       the snapshot, so that the checkpoint needs no second transfer from device
       memory. If the snapshot failed, passes `state` itself.

    Args:
      step: The training step number associated with the state.
      state: The PyTree to be saved.
      wait: Whether to take the snapshot of `step` before returning.
      checkpointer_wait_fn: Waits for the previous save of a checkpointer.
      checkpointer_save_fn: Saves the PyTree that it is given as a checkpoint.
        The arrays of that PyTree belong to the snapshot, so the function must
        not donate or delete them. Single-device arrays are replicated across
        the mesh.

    Returns:
      The result of `checkpointer_save_fn`, or None without that function.
    """
    if not (wait or checkpointer_wait_fn or checkpointer_save_fn):
      if self._queue.full():
        logging.warning("Snapshotter busy. Skipping snapshot for step %d", step)
        return None
      self._enqueue_snapshot(step, state)
      return None

    if checkpointer_wait_fn is not None:
      checkpointer_wait_fn()
    self.join()
    pinned_state = self._enqueue_snapshot(step, state)
    self.join()
    if checkpointer_save_fn is None:
      return None
    with self._lock:
      latest_snapshot = self._latest_snapshot
    if latest_snapshot is None or latest_snapshot[2] is not pinned_state:
      # The snapshot failed, so the checkpoint transfers `state` itself.
      pinned_state = state
    return checkpointer_save_fn(pinned_state)

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
      replica_trees, step, _ = self._latest_snapshot

    def is_replica_active(replica_tree):
      try:
        sliced_tree = jax.tree.map(
            lambda x: _unwrap_if_prng_key(x)[...]
            if is_shardable_array(x)
            else x,
            replica_tree,
        )
        jax.block_until_ready(sliced_tree)
        return True
      except (jax.errors.JaxRuntimeError, RuntimeError):
        return False

    active_trees = [
        tree for tree in replica_trees if is_replica_active(tree)
    ]
    if not active_trees:
      raise RuntimeError("No active replicas found.")

    def get_active_leaf(*args):
      replicas, abs_x = args[:-1], args[-1]
      if not is_shardable_array(replicas[0]):
        return replicas[0]
      if _has_sharding_without_mesh(abs_x):
        devices = abs_x.sharding.device_set
        matches = [
            replica
            for replica in replicas
            if devices <= replica.sharding.device_set
        ]
        return matches[0] if matches else replicas[0]
      unwrapped_replicas = [_unwrap_if_prng_key(rep) for rep in replicas]
      mesh_axis_name = abs_x.sharding.mesh.axis_names[self.replica_axis_index]
      reconstructed = self._concatenate_by_mesh_axis.concatenate_by_mesh_axis(
          unwrapped_replicas,
          mesh_axis_name,
      )
      return _wrap_if_prng_key(reconstructed, replicas[0])

    # TODO(b/574655289): Pathways on Cloud workers crash on cross-slice
    # transfers of pinned_host buffers over RpcTransport ("Attempting to
    # decrease reference on a buffer with zero reference count") until their
    # images include the PJRT C API fix. Until then, copy each replica to device
    # memory on its own devices, then concatenate the copies and reshard them in
    # device memory. In either memory, concatenate copies rather than the
    # snapshot's replicas: concatenating donates its inputs, and the snapshot
    # must survive a failed load (e.g. a slice going down) for it to be retried.
    device_trees = [
        _restore_shardings(tree, tree, None) for tree in active_trees
    ]
    device_state = jax.tree.map(get_active_leaf, *device_trees, abstract_state)
    restored_state = _restore_shardings(device_state, abstract_state)
    unwrapped_restored = jax.tree.map(_unwrap_if_prng_key, restored_state)
    jax.block_until_ready(unwrapped_restored)

    if (should_reset := reset_snapshot_state) is None:
      mesh = _find_named_mesh(abstract_state)
      should_reset = bool(
          mesh and mesh.shape[mesh.axis_names[self.replica_axis_index]] > 1
      )
    if should_reset:
      full_mesh_pinned_state, per_replica_pinned_trees = (
          self._to_snapshot_state(restored_state)
      )
      with self._lock:
        self._latest_snapshot = (
            per_replica_pinned_trees,
            step,
            full_mesh_pinned_state,
        )

    return restored_state

  def try_load(
      self,
      abstract_state: tree_types.PyTree,
      *,
      reset_snapshot_state: bool | None = None,
  ) -> tuple[int | None, tree_types.PyTree | None]:
    """Restores `(step, state)` from the latest snapshot, or `(None, None)`."""
    self.join()
    if (latest := self.latest) is None:
      return None, None
    try:
      return latest.step, self.load(
          abstract_state, reset_snapshot_state=reset_snapshot_state
      )
    except jax.errors.JaxRuntimeError:
      raise
    except RuntimeError as e:
      logging.warning("Snapshot restore failed at step %d: %s", latest.step, e)
      # Drop the unusable snapshot so that `latest` stops reporting it.
      with self._lock:
        self._latest_snapshot = None
      return None, None

  def join(self, timeout: float | None = _DEFAULT_JOIN_TIMEOUT_SECS) -> bool:
    """Blocks until all queued snapshot requests have been processed.

    A processed request may have failed, so check `latest` for the saved step.
    To take a blocking snapshot, call `save` with `wait=True`.

    Args:
      timeout: The maximum number of seconds to wait. Defaults to 120.0 seconds.
        If None, waits until all queued requests have been processed.

    Returns:
      True if all queued requests have been processed, or False if `timeout`
      elapsed first.
    """
    if timeout is None:
      self._queue.join()
      return True
    with self._queue.all_tasks_done:
      return self._queue.all_tasks_done.wait_for(
          lambda: not self._queue.unfinished_tasks,
          timeout=max(timeout, 0.0),
      )

  def cancel_pending(
      self, timeout: float | None = _DEFAULT_JOIN_TIMEOUT_SECS
  ) -> bool:
    """Cancels pending or in-flight snapshots, keeping the latest completed one.

    Args:
      timeout: Maximum seconds to wait for an in-flight snapshot to finish.
        Defaults to 120.0 seconds. If None, waits without a timeout.

    Returns:
      True if all queued requests finished within `timeout`, else False.
    """
    with self._lock, contextlib.suppress(queue.Empty):
      self._generation += 1
      while True:
        self._queue.get_nowait()
        self._queue.task_done()
    return self.join(timeout)

  @property
  def latest(self) -> training_metadata_types.CheckpointMetadata[None] | None:
    """Returns the training step of the most recently pinned backup."""
    with self._lock:
      if self._latest_snapshot is None:
        return None
      _, step, _ = self._latest_snapshot
    return training_metadata_types.CheckpointMetadata(
        step=step,
        path=epath.Path(),
        metadata=None,
    )
