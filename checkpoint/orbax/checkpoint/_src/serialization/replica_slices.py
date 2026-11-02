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
import dataclasses
import functools
import heapq
import math
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

  def __post_init__(self):
    if self.is_on_host:
      assert not self.slice_args, f'On-host {self!r} must not have slice args'

  @property
  def is_on_host(self):
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


def get_local_shape(
    arr: jax.Array,
    *,
    use_replica_parallel: bool,
    min_slice_bytes_for_replica_parallel: Optional[int] = None,
    max_replicas_for_replica_parallel: Optional[int] = None,
) -> Shape:
  """Returns the per-replica write shape of `arr` without accessing shards."""
  if use_replica_parallel:
    _, local_shape, _ = _get_replica_parallel_axis_and_local_shape(
        arr,
        min_slice_bytes_for_replica_parallel,
        max_replicas_for_replica_parallel,
    )
    if local_shape is not None:
      return local_shape
  return arr.sharding.shard_shape(arr.shape)


@functools.lru_cache(maxsize=4096)
def _get_single_replica_sharding(
    sharding: jax.sharding.NamedSharding,
    global_shape: Shape,
    replica_id: int,
) -> Optional[jax.sharding.NamedSharding]:
  """Returns a sharding on the devices of `replica_id`, or `None`."""
  num_replicas = _sharding_num_replicas(sharding, global_shape)
  if num_replicas <= 1 or not (0 <= replica_id < num_replicas):
    return None

  devices_indices = sharding.devices_indices_map(global_shape)
  counts = collections.defaultdict(int)
  replica_devices = []
  for device, index in devices_indices.items():
    hashable_index = numpy_utils.to_hashable_index(index, shape=global_shape)
    if counts[hashable_index] == replica_id:
      replica_devices.append(device)
    counts[hashable_index] += 1

  used_mesh_axes = set()
  for spec_axis in sharding.spec:
    if (
        spec_axis is None
        or spec_axis is jax.sharding.PartitionSpec.UNCONSTRAINED
    ):
      continue
    if isinstance(spec_axis, tuple):
      used_mesh_axes.update(spec_axis)
    else:
      used_mesh_axes.add(spec_axis)

  mesh = sharding.mesh
  first_device = replica_devices[0]
  first_coords = [int(c[0]) for c in np.where(mesh.devices == first_device)]
  if len(first_coords) != mesh.devices.ndim:
    return None
  mesh_slice = tuple(
      slice(None)
      if axis_name in used_mesh_axes or axis_size == 1
      else slice(coord, coord + 1)
      for axis_name, axis_size, coord in zip(
          mesh.axis_names, mesh.devices.shape, first_coords
      )
  )
  sub_devices = mesh.devices[mesh_slice]
  if list(sub_devices.flatten()) != replica_devices:
    return None

  sub_mesh = jax.sharding.Mesh(
      sub_devices, mesh.axis_names, axis_types=mesh.axis_types
  )
  single_replica_sharding = sharding.update(mesh=sub_mesh)
  if list(
      single_replica_sharding.devices_indices_map(global_shape).items()
  ) != [(d, devices_indices[d]) for d in replica_devices]:
    return None
  return single_replica_sharding


def get_single_replica_sharding(
    arr: jax.Array,
    *,
    replica_id: Optional[int],
    use_replica_parallel: bool = False,
    min_slice_bytes_for_replica_parallel: Optional[int] = None,
    max_replicas_for_replica_parallel: Optional[int] = None,
) -> Optional[jax.sharding.NamedSharding]:
  """Returns a sharding restricting `arr` to `replica_id`, or `None`.

  Returns `None` if `arr` is not a device-resident `NamedSharding` array with
  multiple replicas, if `replica_id` is `None` or out of range, or if `arr` is
  saved replica-parallel across multiple replicas.

  Args:
    arr: The array to save.
    replica_id: The replica that saves `arr` whole.
    use_replica_parallel: See `get_replica_slices`.
    min_slice_bytes_for_replica_parallel: See `get_replica_slices`.
    max_replicas_for_replica_parallel: See `get_replica_slices`.
  """
  if (
      replica_id is None
      or not isinstance(arr.sharding, jax.sharding.NamedSharding)
      or arr.sharding.memory_kind not in (None, 'device')
  ):
    return None
  if use_replica_parallel:
    axis, _, _ = _get_replica_parallel_axis_and_local_shape(
        arr,
        min_slice_bytes_for_replica_parallel,
        max_replicas_for_replica_parallel,
    )
    if axis is not None:
      return None
  return _get_single_replica_sharding(arr.sharding, arr.shape, replica_id)


def select_single_replicas_on_device(
    arrays: Sequence[jax.Array],
    shardings: Sequence[Optional[jax.sharding.Sharding]],
) -> list[jax.Array]:
  """Restricts each array to the devices of its target single-replica sharding.

  Rewraps the existing per-device buffers for the selected replica into a
  `jax.Array` on that replica's sub-mesh without copying or allocating device
  memory. Arrays whose target sharding is `None` are returned unchanged.

  Args:
    arrays: The `jax.Array`s to restrict.
    shardings: For each array, the single-replica sharding from
      `get_single_replica_sharding`, or `None` to leave the array unchanged.

  Returns:
    The arrays, with each array that has a non-`None` sharding restricted to
    that sharding's devices.
  """
  selected = []
  for arr, sharding in zip(arrays, shardings, strict=True):
    if sharding is None:
      selected.append(arr)
    else:
      shards = [
          shard.data
          for shard in arr.addressable_shards
          if shard.device in sharding.device_set
      ]
      selected.append(
          jax.make_array_from_single_device_arrays(
              arr.shape, sharding, shards, dtype=arr.dtype
          )
      )
  return selected


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
  shard_shape = arr.sharding.shard_shape(arr.shape)

  # single-replica: a single replica saves an entire shard.
  def pick_single_replica() -> Result:
    if not arr.addressable_shards:
      return [], shard_shape
    target_replica_id = (
        arr.addressable_shards[0].replica_id
        if replica_id is None
        else replica_id
    )
    rslices = [
        ReplicaSlice(
            index=shard.index,
            unsliced_data=shard.data,
            slice_args=None,
        )
        for shard in arr.addressable_shards
        if shard.replica_id == target_replica_id
    ]
    return rslices, shard_shape

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
      assert shard.data.shape == shard_shape

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


def transfer_arrays_to_host(
    arrays: Sequence[jax.Array],
    replica_ids: Sequence[Optional[int]],
    use_replica_parallel: bool,
    min_slice_bytes_for_replica_parallel: Optional[int] = None,
    max_replicas_for_replica_parallel: Optional[int] = None,
    *,
    enable_pinned_host_transfer: bool = False,
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
      memory. Pinned memory is closely associated with a TPU device and can
      speed up transfers.

  Returns:
    ReplicaSlices objects, in host memory.
  """
  logging.info(
      'Transferring arrays to host memory with options:'
      ' use_replica_parallel=%s, min_slice_bytes_for_replica_parallel=%s,'
      ' max_replicas_for_replica_parallel=%s, enable_pinned_host_transfer=%s',
      use_replica_parallel,
      min_slice_bytes_for_replica_parallel,
      max_replicas_for_replica_parallel,
      enable_pinned_host_transfer,
  )

  def use_pinned_host_transfer(device: jax.Device):
    has_pinned_host = any(
        m.kind == 'pinned_host' for m in device.addressable_memories()
    )
    return enable_pinned_host_transfer and has_pinned_host

  def async_transfer_slice(
      rslice: ReplicaSlice,
  ) -> tuple[ReplicaSlice, jax.Array]:
    assert not rslice.is_on_host
    data = rslice.data()
    assert isinstance(data, jax.Array)
    device = data.device
    # Start the asynchronous device-to-host copy
    if use_pinned_host_transfer(device):  # pyrefly: ignore[bad-argument-type]
      # If available, transfer to pinned host memory
      data = jax.device_put(
          data,
          make_single_device_sharding(device, memory_kind='pinned_host'),
      )
    else:
      data.copy_to_host_async()
    return rslice, data

  # Gather the replica slices to be saved for each array.
  if len(replica_ids) != len(arrays):
    raise ValueError(
        f'Expected {len(arrays)} replica_ids, got {len(replica_ids)}'
    )
  for arr, r in zip(arrays, replica_ids):
    num_replicas = _sharding_num_replicas(arr.sharding, arr.shape)
    if r is not None and not 0 <= r < num_replicas:
      raise ValueError(
          f'replica_id {r} out of range [0, {num_replicas}) for array of'
          f' shape {arr.shape}'
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
  # Kick off transfers for all replica slices to be saved.
  transfers_per_array = [
      [async_transfer_slice(rslice) for rslice in rslices.replica_slices]
      for rslices in rslices_per_array
  ]
  # Wait for all the transferred data to be ready.
  return [
      dataclasses.replace(
          rslices,
          is_on_host=True,
          replica_slices=[
              dataclasses.replace(
                  rslice_on_device,
                  # Conversion to numpy arrays forces block_until_ready.
                  unsliced_data=np.asarray(data),
                  slice_args=None,
              )
              for rslice_on_device, data in transfers
          ],
      )
      for rslices, transfers in zip(rslices_per_array, transfers_per_array)
  ]
