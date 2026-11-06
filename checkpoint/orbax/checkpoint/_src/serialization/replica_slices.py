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

"""Handles replica slices of jax.Arrays and host transfers."""

import collections
import concurrent.futures
import dataclasses
import functools
import heapq
import itertools
import math
import threading
from typing import Optional, Sequence

from absl import logging
import jax
import numpy as np
from orbax.checkpoint._src.arrays import fragments
from orbax.checkpoint._src.arrays import numpy_utils
from orbax.checkpoint._src.arrays import types
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.sharding_utils import make_single_device_sharding


Shape = types.Shape
Index = types.Index
OptionalAxisAndShapeAndReplicaCount = tuple[
    int | None, Shape | None, int | None
]

HashableIndex = types.HashableIndex
HashableSlice = types.HashableSlice


@dataclasses.dataclass(frozen=True)
class SliceArgs:
  """Arguments for slicing a jax.Array.

  Intended to be passed to `ReplicaSlice` in order to select a slice of
  the `unsliced_data` array.
  """

  start_index: int
  limit_index: int
  axis: int


@dataclasses.dataclass(frozen=True)
class ReplicaSlice:
  """ReplicaSlice.

  ReplicaSlice represents the part of a jax.Shard that a replica is uniquely
  responsible for. A replica slice can be either on-device (backed by a slice of
  a single-sharding array) or on-host (backed by a numpy ndarray).

  With single-replica checkpointing the entirety of each jax.Shard is owned by
  exactly one replica. With replica-parallel checkpointing ownership of each
  jax.Shard is split evenly across replicas, hence each of the R replicas will
  be responsible for saving 1/R of each shard.

  `unsliced_data` refers to the corresponding jax.Shard's single-device array.
  The part of `unsliced_data` actually owned is given by `slice_args`.
  """

  index: Index
  unsliced_data: jax.Array | np.ndarray
  slice_args: Optional[SliceArgs]
  _is_on_host: Optional[bool] = None
  _owns_pinned_buffer: bool = False
  _host_slice_args: Optional[SliceArgs] = None

  def __post_init__(self):
    if self.is_on_host:
      assert not self.slice_args, f'On-host {self!r} must not have slice args'

  @property
  def is_on_host(self):
    if self._is_on_host is not None:
      return self._is_on_host
    return isinstance(self.unsliced_data, np.ndarray)

  def data(self):
    """Returns the sliced data for this replica.

    If `slice_args` is None, the entire `unsliced_data` is returned. Otherwise,
    a slice of `unsliced_data` is returned based on `slice_args`.
    """
    if self.slice_args is None:
      return self.unsliced_data
    else:
      # If a global mesh is set, slice_in_dim can fail with incompatible device
      # errors. To avoid this, we temporarily set a mesh constructed from
      # array's devices.
      mesh = jax.sharding.Mesh(
          np.array(list(self.unsliced_data.sharding.device_set)), ('data',)  # pyrefly: ignore[missing-attribute]
      )
      with jax.sharding.set_mesh(mesh):
        sliced_data = jax.lax.slice_in_dim(
            self.unsliced_data,
            start_index=self.slice_args.start_index,
            limit_index=self.slice_args.limit_index,
            axis=self.slice_args.axis,
        )
      return sliced_data


@dataclasses.dataclass(frozen=True)
class ReplicaSlices:
  """ReplicaSlices.

  ReplicaSlices groups all the sliced data of one jax.Array that a replica is
  uniquely responsible for. Slices are either all on-device or all on-host.
  """

  global_shape: Shape
  local_shape: Shape
  sharding: jax.sharding.Sharding
  dtype: np.dtype | jax.numpy.dtype
  is_on_host: bool
  replica_slices: list[ReplicaSlice]

  def __post_init__(self):
    if not all(
        rslice.is_on_host == self.is_on_host for rslice in self.replica_slices
    ):
      raise ValueError(f'Inconsistent is_on_host in {self!r}')

  @property
  def nbytes(self) -> int:
    """Returns the total number of bytes for all replica slices."""
    total_bytes = 0
    for rslice in self.replica_slices:
      if rslice.slice_args is None:
        # No slicing, use the full local_shape
        slice_nbytes = math.prod(self.local_shape) * self.dtype.itemsize
      else:
        # Replica-parallel, calculate bytes for the specific slice
        slice_shape = list(self.local_shape)
        axis = rslice.slice_args.axis
        slice_shape[axis] = (
            rslice.slice_args.limit_index - rslice.slice_args.start_index
        )
        slice_nbytes = math.prod(slice_shape) * self.dtype.itemsize
      total_bytes += slice_nbytes
    return total_bytes

  def to_fragments(self) -> fragments.ConcreteFragments:
    """Converts replica slices to fragments."""
    assert self.is_on_host
    if any(
        isinstance(rslice.unsliced_data, jax.Array)
        for rslice in self.replica_slices
    ):
      raise ValueError(
          'Cannot convert staged jax.Array slices to ConcreteFragments; call'
          ' `untile_pinned_slices_to_numpy` first.'
      )
    result = fragments.ConcreteFragments(
        shape=self.global_shape,
        dtype=self.dtype,
        fragments=[
            fragments.ConcreteFragment(
                index=numpy_utils.resolve_slice(  # pyrefly: ignore[bad-argument-type]
                    rslice.index, self.global_shape
                ),
                value=rslice.data(),
            )
            for rslice in self.replica_slices
        ],
    )
    if result.fragments:
      fragments.validate_fragments_can_be_stacked(result)
    if not result.is_degenerate():
      assert self.local_shape == result.fragments[0].shape
    return result


@functools.lru_cache(maxsize=4096)
def _sharding_num_replicas(
    sharding: jax.sharding.Sharding, global_shape: Shape
) -> int:
  """Get the number of unique replicas for a sharding/shape.

  Uses the devices_indices_map to get the mapping of devices to the slice of the
  global array. This gives us the domains of every shard, which may be
  non-unique. For any index (domain), we increment the count by one. When `n`
  devices have the same index, this results in the replica count for that index
  being `n`. We can assert that the number of replicas for each index should be
  the same.

  We can cache results because we typically expect `save` to be called
  repeatedly on the same model (with changing array values).
  The model shardings and shapes do not change during the course of a typical
  training run.

  Training typically occurs with stacked layers, so we expect the number of
  model parameters to be significantly less than the cache size. Checkpoints
  with unstacked layers may have thousands of parameters, but these are
  typically used for inference, so saving is less relevant.

  Args:
    sharding: Array sharding.
    global_shape: The global shape of the array.

  Returns:
    The number of unique replicas for the sharding/shape.
  """
  counts = collections.defaultdict(int)
  for index in sharding.devices_indices_map(global_shape).values():
    counts[numpy_utils.to_hashable_index(index, shape=global_shape)] += 1
  num_replicas = next(iter(counts.values()))
  assert all(count == num_replicas for count in counts.values())
  return num_replicas


def calculate_replica_parallel_axis_and_local_shape(
    arr: jax.Array, max_replicas_for_replica_parallel: Optional[int]
) -> OptionalAxisAndShapeAndReplicaCount:
  """Calculates a local shape for replica-parallel serialization."""
  shard_shape = arr.sharding.shard_shape(arr.shape)
  replica_count = _sharding_num_replicas(arr.sharding, arr.shape)
  if max_replicas_for_replica_parallel is not None:
    replica_count = min(replica_count, max_replicas_for_replica_parallel)
  if math.prod(shard_shape) == 0 or replica_count <= 1:
    return None, None, None
  try:
    axis = next(
        axis_index
        for axis_index, axis_size in enumerate(shard_shape)
        if axis_size % replica_count == 0
    )
  except StopIteration:
    return None, None, None
  local_shape = tuple(
      axis_size // (replica_count if axis_index == axis else 1)
      for axis_index, axis_size in enumerate(shard_shape)
  )
  return axis, local_shape, replica_count


def _is_safe_to_slice(arr: jax.Array) -> bool:
  """Returns whether the shards of `arr` can be sliced."""
  # Small pinned host arrays have layout requirements so slicing them might hang
  # this is a temporary workaround to avoid this. TODO: b/417243451
  if arr.sharding.memory_kind != 'pinned_host':
    return True
  shard_shape = arr.sharding.shard_shape(arr.shape)
  return (
      arr.ndim >= 2
      and math.prod(shard_shape) % 1024 == 0
      and shard_shape[-1] % 128 == 0
  )


def _get_replica_parallel_axis_and_local_shape(
    arr: jax.Array,
    min_slice_bytes_for_replica_parallel: Optional[int],
    max_replicas_for_replica_parallel: Optional[int],
) -> OptionalAxisAndShapeAndReplicaCount:
  """Returns how `arr` is split across replicas if it is saved replica-parallel.

  Args:
    arr: The array to save.
    min_slice_bytes_for_replica_parallel: See `get_replica_slices`.
    max_replicas_for_replica_parallel: See `get_replica_slices`.

  Returns:
    The axis along which each shard is split, the shape of each replica's slice
    of a shard, and the number of replicas that save a slice; or `None`s if
    `arr` is not saved replica-parallel.
  """
  # Check whether replica-parallel applies: we are dealing with non-empty
  # shards that are safe to slice, we have more than one replica, and some
  # dimension of the shards is evenly divisible across replicas.
  if not _is_safe_to_slice(arr):
    return None, None, None
  axis, local_shape, replica_count = (
      calculate_replica_parallel_axis_and_local_shape(
          arr, max_replicas_for_replica_parallel
      )
  )
  if axis is None or local_shape is None or replica_count is None:
    return None, None, None
  min_slice_bytes = min_slice_bytes_for_replica_parallel or 0
  if math.prod(local_shape) * arr.itemsize < min_slice_bytes:
    return None, None, None
  return axis, local_shape, replica_count


def get_replica_slices(
    arr: jax.Array,
    replica_id: Optional[int],
    use_replica_parallel: bool,
    min_slice_bytes_for_replica_parallel: Optional[int] = None,
    max_replicas_for_replica_parallel: Optional[int] = None,
) -> ReplicaSlices:
  """Returns the replica slices a given replica is responsible for.

  Does not transfer allocate or transfer any data.

  Args:
    arr: The jax.Array to get replica slices for.
    replica_id: The replica that saves `arr` whole if it is not saved
      replica-parallel: the configured replica_id, or the replica from
      `get_balanced_replica_ids` when balancing arrays across replicas. Ignored
      if `arr` is saved replica-parallel, in which case multiple replicas each
      save a slice of every shard. Omitting the replica id just picks the first
      addressable shard's replica id so that the process writes each of its
      addressable shards exactly once. (This is the desired behavior for local
      checkpointing.)
    use_replica_parallel: Whether to use replica-parallel serialization to allow
      arrays with replicated shards to be written cooperatively by different
      hosts.
    min_slice_bytes_for_replica_parallel: Minimum number of bytes per replica
      slice. Only uses replica-parallel when the amount of data written per
      replica is greater than or equal to this number.
    max_replicas_for_replica_parallel: Maximum number of replicas over which
      saving will be parallelized if use_replica_parallel is True.

  Returns:
    ReplicaSlices object.
  """
  Result = tuple[list[ReplicaSlice], Shape]
  shard0 = arr.addressable_shards[0]

  # single-replica: a single replica saves an entire shard.
  def pick_single_replica() -> Result:
    target_replica_id = shard0.replica_id if replica_id is None else replica_id
    rslices = [
        ReplicaSlice(
            index=shard.index,
            unsliced_data=shard.data,
            slice_args=None,
        )
        for shard in arr.addressable_shards
        if shard.replica_id == target_replica_id
    ]
    local_shape = shard0.data.shape
    return rslices, local_shape

  # replica-parallel: every replica saves part of a shard.
  # Logic based on axlearn:
  # https://github.com/apple/axlearn/blob/226d27ab7569668f2c38a35cf32d5dc5190ebdbb/axlearn/common/array_serialization.py#L75
  # TODO(gspschmid): Support replica-parallel in arrays without any evenly-
  # divisible dimension. The last replica would transfer a smaller slice.
  def maybe_pick_replica_parallel() -> Optional[Result]:
    if replica_id is None:
      raise ValueError(
          '`use_replica_parallel` is incompatible with local checkpointing'
      )

    axis, local_shape, replica_count = (
        _get_replica_parallel_axis_and_local_shape(
            arr,
            min_slice_bytes_for_replica_parallel,
            max_replicas_for_replica_parallel,
        )
    )
    if axis is None or local_shape is None or replica_count is None:
      return None

    rslices: list[ReplicaSlice] = []
    for shard in arr.addressable_shards:
      # Sanity check that all shards have the same shape.
      assert shard.data.shape == shard0.data.shape

      # Parallelize saving across only `replica_count` replicas.
      if shard.replica_id >= replica_count:
        continue

      size = local_shape[axis]
      slize = shard.index[axis]
      start = slize.start or 0
      assert slize.step is None
      assert slize.stop is None or slize.stop == start + shard.data.shape[axis]

      start_offset = shard.replica_id * size
      end_offset = start_offset + size
      new_slice = slice(start + start_offset, start + end_offset)

      rslices.append(
          ReplicaSlice(
              index=shard.index[:axis] + (new_slice,) + shard.index[axis + 1 :],
              unsliced_data=shard.data,
              slice_args=SliceArgs(start_offset, end_offset, axis),
          )
      )

    return rslices, local_shape

  if logging.vlog_is_on(1):
    logging.vlog(
        1,
        '[process=%d] get_replica_slices: replica_id=%s, shards=[%s]',
        multihost.process_index(),
        replica_id,  # note: may be None
        ', '.join([
            f'Shard(index={shard.index}, replica_id={shard.replica_id})'
            for shard in arr.addressable_shards
        ]),
    )

  # In order for all processes to agree on the right serialization metadata
  # we want to compute the correct local shape regardless of whether there
  # are any replica slices to save locally.
  candidate_slices = None
  if use_replica_parallel and _is_safe_to_slice(arr):
    candidate_slices = maybe_pick_replica_parallel()
  if candidate_slices is None:
    candidate_slices = pick_single_replica()
  rslices, local_shape = candidate_slices

  if multihost.process_index() == 0:
    # record sharded and replicated arrays metrics
    replica_count = _sharding_num_replicas(arr.sharding, arr.shape)
    if replica_count > 1:
      jax.monitoring.record_scalar(
          '/jax/orbax/write/replicated_array_gb',
          value=np.prod(arr.shape) * arr.dtype.itemsize / (1024**3),
      )
    else:
      jax.monitoring.record_scalar(
          '/jax/orbax/write/sharded_array_gb',
          value=np.prod(arr.shape) * arr.dtype.itemsize / (1024**3),
      )

  return ReplicaSlices(
      global_shape=arr.shape,
      local_shape=local_shape,
      sharding=arr.sharding,
      dtype=arr.dtype,
      is_on_host=False,
      replica_slices=rslices,
  )


def partition_into_k_groups(
    values: Sequence[float],
    k: int,
    initial_sums: Optional[Sequence[float]] = None,
) -> tuple[list[list[int]], list[float]]:
  """Partitions values into k subsets with nearly equal sums.

  Args:
    values: Non-negative numbers to partition.
    k: Number of subsets.
    initial_sums: Optional sums the k subsets start with. If given, subset `j`
      of the result is the one that starts with `initial_sums[j]`.

  Returns:
    k lists of indices into values, and the k sums of the subsets, including
    their initial sums.
  """
  if k < 1:
    raise ValueError(f'k must be at least 1, got {k}')
  if initial_sums is not None and len(initial_sums) != k:
    raise ValueError(f'Expected {k} initial sums, got {len(initial_sums)}')
  sums: list[float] = [0] * k if initial_sums is None else list(initial_sums)
  subsets = [[] for _ in range(k)]
  heap = [(s, j) for j, s in enumerate(sums)]
  heapq.heapify(heap)
  for i in sorted(range(len(values)), key=lambda i: (-values[i], i)):
    s, j = heap[0]
    s += values[i]
    heapq.heapreplace(heap, (s, j))
    sums[j] = s
    subsets[j].append(i)
  return subsets, sums


def get_balanced_replica_ids(
    arrays: Sequence[jax.Array],
    use_replica_parallel: bool,
    min_slice_bytes_for_replica_parallel: Optional[int] = None,
    max_replicas_for_replica_parallel: Optional[int] = None,
) -> list[int]:
  """Returns the replica that saves each array, balancing bytes per replica.

  Used instead of the configured `replica_id` when balancing arrays across
  replicas. Arrays saved replica-parallel are split across replicas as usual,
  and count towards the bytes of the replicas that save them. Every other
  replicated array is saved whole by one replica, chosen so that replicas (and,
  when arrays have different replica counts, groups of replicas such as slices)
  save nearly equal numbers of bytes, with ties broken toward lower replica
  indices.

  Args:
    arrays: The jax.Arrays to save, in the same order on every process.
    use_replica_parallel: See `get_replica_slices`.
    min_slice_bytes_for_replica_parallel: See `get_replica_slices`.
    max_replicas_for_replica_parallel: See `get_replica_slices`.

  Returns:
    For each array, the id of the replica that saves it whole. Unused for arrays
    saved replica-parallel.
  """
  replica_counts = [
      _sharding_num_replicas(arr.sharding, arr.shape) for arr in arrays
  ]
  replica_ids = [0] * len(arrays)
  initial_nbytes_by_k = {}
  unassigned_by_k = {}
  for i, (arr, num_replicas) in enumerate(zip(arrays, replica_counts)):
    if num_replicas <= 1:
      continue
    initial_nbytes = initial_nbytes_by_k.setdefault(
        num_replicas, [0] * num_replicas
    )
    unassigned = unassigned_by_k.setdefault(num_replicas, [])
    num_writers = None
    if use_replica_parallel:
      _, _, num_writers = _get_replica_parallel_axis_and_local_shape(
          arr,
          min_slice_bytes_for_replica_parallel,
          max_replicas_for_replica_parallel,
      )
    if num_writers is None:
      unassigned.append(i)
      continue
    for r in range(num_writers):
      initial_nbytes[r] += arr.nbytes // num_writers

  replica_nbytes = []
  for k in sorted(initial_nbytes_by_k, reverse=True):
    initial_nbytes = initial_nbytes_by_k[k]
    # Assumes replica axes are major in mesh device order, so replica `r` of a
    # `len(replica_nbytes)`-replica array maps to replica
    # `r * k // len(replica_nbytes)` of a `k`-replica array.
    for r, nbytes in enumerate(replica_nbytes):
      initial_nbytes[r * k // len(replica_nbytes)] += nbytes
    unassigned = unassigned_by_k[k]
    subsets, replica_nbytes = partition_into_k_groups(
        [arrays[i].nbytes for i in unassigned],
        k,
        initial_sums=initial_nbytes,
    )
    for r, subset in enumerate(subsets):
      for j in subset:
        replica_ids[unassigned[j]] = r
  return replica_ids


_DEFAULT_UNTILE_MAX_WORKERS = 8
_untile_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
_untile_executor_lock = threading.Lock()


def _get_untile_executor() -> concurrent.futures.ThreadPoolExecutor:
  """Returns a shared bounded pool for background host untiling."""
  global _untile_executor
  with _untile_executor_lock:
    if _untile_executor is None:
      _untile_executor = concurrent.futures.ThreadPoolExecutor(
          max_workers=_DEFAULT_UNTILE_MAX_WORKERS,
          thread_name_prefix='orbax_untile',
      )
    return _untile_executor


def _has_pinned_host_memory(device: jax.Device) -> bool:
  return any(m.kind == 'pinned_host' for m in device.addressable_memories())


def _is_pinned_host_array(arr: jax.Array) -> bool:
  return getattr(arr.sharding, 'memory_kind', None) == 'pinned_host'


def _slice_numpy(
    arr: np.ndarray, slice_args: Optional[SliceArgs]
) -> np.ndarray:
  """Applies `slice_args` to a host numpy array if present.

  Args:
    arr: Host numpy array to slice.
    slice_args: Optional slice bounds along a single axis.

  Returns:
    Original `arr` when `slice_args` is None, else an owned sliced `np.ndarray`.
  """
  if slice_args is None:
    return arr
  idx = (slice(None),) * slice_args.axis + (
      slice(slice_args.start_index, slice_args.limit_index),
  )
  return np.array(arr[idx], copy=True)


_SliceTransfer = tuple[ReplicaSlice, jax.Array, bool, Optional[SliceArgs]]


def _async_transfer_slice(
    rslice: ReplicaSlice,
    enable_pinned: bool,
) -> _SliceTransfer:
  """Starts async D2H transfer and tracks internal pinned-buffer ownership.

  Args:
    rslice: On-device replica slice to transfer.
    enable_pinned: Whether pinned_host transfer is enabled when supported.

  Returns:
    Tuple of `(rslice, transferred_array, owns_pinned, deferred_slice_args)`.
  """
  assert not rslice.is_on_host
  unsliced = rslice.unsliced_data
  if isinstance(unsliced, jax.Array) and _is_pinned_host_array(unsliced):
    return rslice, unsliced, False, rslice.slice_args
  data = rslice.data()
  assert isinstance(data, jax.Array)
  device = data.device
  assert isinstance(device, jax.Device)
  if enable_pinned and _has_pinned_host_memory(device):
    data = jax.device_put(
        data,
        make_single_device_sharding(device, memory_kind='pinned_host'),
    )
    return rslice, data, True, None
  data.copy_to_host_async()
  return rslice, data, False, None


def _finalize_async_untile_transfers(
    rslices_per_array: Sequence[ReplicaSlices],
    transfers_per_array: Sequence[Sequence[_SliceTransfer]],
) -> list[ReplicaSlices]:
  """Blocks on pinned_host DMA and stages slices for background untiling.

  Args:
    rslices_per_array: Original on-device `ReplicaSlices` per array.
    transfers_per_array: Per-array transfer tuples from `_async_transfer_slice`.

  Returns:
    Host-staged `ReplicaSlices` ready for `untile_pinned_slices_to_numpy`.
  """
  all_transfers = itertools.chain.from_iterable(transfers_per_array)
  pinned_transfers = [
      data for _, data, _, _ in all_transfers if _is_pinned_host_array(data)
  ]
  if pinned_transfers:
    jax.block_until_ready(pinned_transfers)
  return [
      dataclasses.replace(
          rslices,
          is_on_host=True,
          replica_slices=[
              dataclasses.replace(
                  rslice_on_device,
                  unsliced_data=(
                      data
                      if _is_pinned_host_array(data)
                      else _slice_numpy(np.asarray(data), deferred_slice)
                  ),
                  slice_args=None,
                  _is_on_host=True,
                  _owns_pinned_buffer=(
                      owns_pinned and _is_pinned_host_array(data)
                  ),
                  _host_slice_args=(
                      deferred_slice if _is_pinned_host_array(data) else None
                  ),
              )
              for (
                  rslice_on_device,
                  data,
                  owns_pinned,
                  deferred_slice,
              ) in transfers
          ],
      )
      for rslices, transfers in zip(rslices_per_array, transfers_per_array)
  ]


def transfer_arrays_to_host(
    arrays: Sequence[jax.Array],
    replica_ids: Sequence[Optional[int]],
    use_replica_parallel: bool,
    min_slice_bytes_for_replica_parallel: Optional[int] = None,
    max_replicas_for_replica_parallel: Optional[int] = None,
    *,
    enable_pinned_host_transfer: bool = False,
    async_host_untile: bool = False,
) -> Sequence[ReplicaSlices]:
  """Transfers arrays to host memory.

  Transfers jax.Arrays to host memory and returns all the fragments to be
  serialized by the given replica, along with local shape. Blocks until
  completion.

  Args:
    arrays: The jax.Arrays to transfer.
    replica_ids: For each array, the replica that saves it whole if it is not
      saved replica-parallel; see `get_replica_slices`.
    use_replica_parallel: Whether to use replica-parallel serialization to allow
      arrays with replicated shards to be written cooperatively by different
      hosts.
    min_slice_bytes_for_replica_parallel: Minimum number of bytes per replica
      slice. Only uses replica-parallel when the amount of data written per
      replica is greater than or equal to this number.
    max_replicas_for_replica_parallel: Maximum number of replicas over which to
      parallelize the writing of replicated shards if use_replica_parallel is
      True.
    enable_pinned_host_transfer: Whether to allow transfer to pinned host
      memory. Pinned memory is closely associated with a device and can speed up
      transfers.
    async_host_untile: If True, transfers device shards into pinned_host memory
      preserving raw device layout and defers numpy untiling to
      `untile_pinned_slices_to_numpy` on background threads. Falls back to host
      numpy conversion on backends without pinned_host memory.

  Returns:
    ReplicaSlices objects, in host memory.
  """
  enable_pinned = enable_pinned_host_transfer or async_host_untile
  logging.info(
      'Transferring arrays to host memory with options:'
      ' use_replica_parallel=%s, min_slice_bytes_for_replica_parallel=%s,'
      ' max_replicas_for_replica_parallel=%s, enable_pinned_host_transfer=%s,'
      ' async_host_untile=%s',
      use_replica_parallel,
      min_slice_bytes_for_replica_parallel,
      max_replicas_for_replica_parallel,
      enable_pinned,
      async_host_untile,
  )

  if len(replica_ids) != len(arrays):
    raise ValueError(
        f'Expected {len(arrays)} replica_ids, got {len(replica_ids)}'
    )
  rslices_per_array = [
      get_replica_slices(
          arr,
          arr_replica_id,
          use_replica_parallel,
          min_slice_bytes_for_replica_parallel,
          max_replicas_for_replica_parallel,
      )
      for arr, arr_replica_id in zip(arrays, replica_ids)
  ]
  transfers_per_array = [
      [
          _async_transfer_slice(rslice, enable_pinned)
          for rslice in rslices.replica_slices
      ]
      for rslices in rslices_per_array
  ]
  if async_host_untile:
    return _finalize_async_untile_transfers(
        rslices_per_array, transfers_per_array
    )
  return [
      dataclasses.replace(
          rslices,
          is_on_host=True,
          replica_slices=[
              dataclasses.replace(
                  rslice_on_device,
                  # Conversion to numpy arrays forces block_until_ready.
                  unsliced_data=_slice_numpy(np.asarray(data), deferred_slice),
                  slice_args=None,
              )
              for rslice_on_device, data, _, deferred_slice in transfers
          ],
      )
      for rslices, transfers in zip(rslices_per_array, transfers_per_array)
  ]


def release_pinned_slices(rslices_list: Sequence[ReplicaSlices]) -> None:
  """Deletes owned staged pinned_host buffers remaining in `rslices_list`.

  Args:
    rslices_list: Sequence of `ReplicaSlices` that may hold internally staged
      `pinned_host` `jax.Array` buffers.
  """
  for rslices in rslices_list:
    for rslice in rslices.replica_slices:
      data = rslice.unsliced_data
      if (
          rslice._owns_pinned_buffer  # pylint: disable=protected-access
          and isinstance(data, jax.Array)
          and not data.is_deleted()
      ):
        data.delete()


def _untile_slice(rslice: ReplicaSlice) -> ReplicaSlice:
  """Converts a staged pinned_host jax.Array slice into a host np.ndarray.

  Args:
    rslice: Host-staged `ReplicaSlice` backed by a `jax.Array` or `np.ndarray`.

  Returns:
    `ReplicaSlice` backed by an independent host `np.ndarray`.
  """
  data = rslice.unsliced_data
  if isinstance(data, np.ndarray):
    return rslice
  # Releases the Python GIL inside PJRT while untiling and deinterleaving lanes.
  host_array = _slice_numpy(
      np.asarray(data),
      rslice._host_slice_args,  # pylint: disable=protected-access
  )
  if rslice._owns_pinned_buffer:  # pylint: disable=protected-access
    if host_array.base is not None or not host_array.flags.owndata:
      host_array = np.array(host_array, copy=True)
    if not data.is_deleted():
      data.delete()
  return dataclasses.replace(
      rslice,
      unsliced_data=host_array,
      _is_on_host=None,
      _owns_pinned_buffer=False,
      _host_slice_args=None,
  )


def _run_untile_workers(
    flat_slices: Sequence[ReplicaSlice],
    max_workers: int | None,
) -> list[ReplicaSlice]:
  """Executes `_untile_slice` across `flat_slices` with bounded workers.

  Args:
    flat_slices: Flattened sequence of `ReplicaSlice` entries to untile.
    max_workers: Optional cap on worker threads.

  Returns:
    List of untiled `ReplicaSlice` entries in input order.
  """
  if len(flat_slices) == 1 or max_workers == 1:
    return [_untile_slice(rslice) for rslice in flat_slices]
  if max_workers is not None:
    workers = min(len(flat_slices), max_workers)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
      return list(pool.map(_untile_slice, flat_slices))
  executor = _get_untile_executor()
  futures = [executor.submit(_untile_slice, s) for s in flat_slices]
  try:
    return [f.result() for f in futures]
  except BaseException:
    for f in futures:
      f.cancel()
    concurrent.futures.wait(futures)
    raise


def untile_pinned_slices_to_numpy(
    rslices_list: Sequence[ReplicaSlices],
    max_workers: int | None = None,
) -> list[ReplicaSlices]:
  """Converts staged pinned_host ReplicaSlices to numpy arrays in parallel.

  Args:
    rslices_list: Sequence of `ReplicaSlices` staged via
      `transfer_arrays_to_host(..., async_host_untile=True)`.
    max_workers: Optional cap on worker threads. Defaults to a shared bounded
      pool (`_DEFAULT_UNTILE_MAX_WORKERS`) to parallelize per-shard untiling
      without unbounded transient host allocations.

  Returns:
    A list of `ReplicaSlices` backed by host `np.ndarray` buffers.

  Raises:
    ValueError: If `max_workers` is less than 1.
  """
  if max_workers is not None and max_workers < 1:
    raise ValueError(f'`max_workers` must be >= 1, got {max_workers}.')
  flat_slices = list(
      itertools.chain.from_iterable(
          rslices.replica_slices for rslices in rslices_list
      )
  )
  if not any(isinstance(s.unsliced_data, jax.Array) for s in flat_slices):
    return list(rslices_list)

  try:
    untiled_flat = _run_untile_workers(flat_slices, max_workers)
  except BaseException:
    release_pinned_slices(rslices_list)
    raise

  it = iter(untiled_flat)
  return [
      dataclasses.replace(
          rslices,
          replica_slices=[next(it) for _ in range(len(rslices.replica_slices))],
      )
      for rslices in rslices_list
  ]
