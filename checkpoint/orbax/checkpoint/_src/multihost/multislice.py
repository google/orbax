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

"""Multislice utilities."""

import functools
import math
import os
from typing import Any, Optional, Set, Union

from absl import logging
import jax
from jax import numpy as jnp
import numpy as np
from orbax.checkpoint._src.multihost import multihost

PyTree = Any
ArrayOrAbstract = jax.Array | jax.ShapeDtypeStruct

# When using broadcasting from single replica to others, 3 copies of the data
# are stored in memory.
MEMORY_FACTOR = 3


def process_replica_id(
    process_index: int,
    global_mesh: jax.sharding.Mesh,
    *,
    replica_axis_index: int = 0,
) -> int:
  """Returns the replica id that the process_index belongs to."""

  for replica_id in range(
      replica_count(global_mesh, replica_axis_index=replica_axis_index)
  ):
    device_slice = replica_devices(
        global_mesh,
        replica_id=replica_id,
        replica_axis_index=replica_axis_index,
    )
    if process_index in multihost.unique_processes_from_devices(device_slice):
      return replica_id
  return -1


def _process_in_device_replica(
    process_index: int, device_slice: np.ndarray
) -> bool:
  return process_index in multihost.unique_processes_from_devices(device_slice)


def replica_devices(
    global_mesh: jax.sharding.Mesh,
    *,
    replica_id: int = 0,
    replica_axis_index: int = 0,
) -> np.ndarray:
  """Returns devices for the replica with the given ID."""
  return np.take(
      global_mesh.devices,
      replica_id,
      axis=replica_axis_index,
  )


def replica_count(
    global_mesh: jax.sharding.Mesh, *, replica_axis_index: int = 0
) -> int:
  """Number of slices implied by the mesh's replica dimension."""
  if len(global_mesh.shape_tuple) == 1:
    return 1
  return global_mesh.devices.shape[replica_axis_index]


def local_replica_devices(
    global_mesh: jax.sharding.Mesh, *, replica_axis_index: int = 0
) -> np.ndarray:
  """Get devices for the replica that the current process is in."""
  for replica_id in range(
      replica_count(global_mesh, replica_axis_index=replica_axis_index)
  ):
    if in_replica(
        multihost.process_index(),
        global_mesh,
        replica_id=replica_id,
        replica_axis_index=replica_axis_index,
    ):
      return replica_devices(
          global_mesh,
          replica_id=replica_id,
          replica_axis_index=replica_axis_index,
      )
  raise ValueError(
      f'process_index {multihost.process_index()} does not exist in provided'
      ' `global_mesh`'
  )


def primary_process_in_replica(
    global_mesh: jax.sharding.Mesh,
    *,
    replica_id: int = 0,
    replica_axis_index: int = 0,
) -> int:
  """Returns an arbitrary process in the requested slice to serve as primary."""
  device_replica = replica_devices(
      global_mesh,
      replica_axis_index=replica_axis_index,
      replica_id=replica_id,
  )
  processes = multihost.unique_processes_from_devices(device_replica)
  return next(iter(processes))


def in_replica(
    process_index: int,
    global_mesh: jax.sharding.Mesh,
    *,
    replica_id: int = 0,
    replica_axis_index: int = 0,
) -> bool:
  """Returns if the process belongs to the indicated slice ID."""
  return _process_in_device_replica(
      process_index,
      replica_devices(
          global_mesh,
          replica_id=replica_id,
          replica_axis_index=replica_axis_index,
      ),
  )


def get_device_memory() -> int:
  """Returns HBM capacity of the device on which the code is running(in bytes)."""
  device = jax.local_devices()[0]
  if device.platform == 'cpu':
    page_size = os.sysconf('SC_PAGE_SIZE')
    phys_pages = os.sysconf('SC_PHYS_PAGES')
    return int(page_size * phys_pages)

  if device.platform not in ('tpu', 'gpu'):
    raise ValueError('Only select TPU and GPU devices are supported.')

  return device.memory_stats()['bytes_limit']


def get_leaf_memory_per_device(arr: ArrayOrAbstract) -> int:
  """Returns the memory usage of a sharded array per device (in bytes)."""
  shard_shape = arr.sharding.shard_shape(arr.shape)
  return math.prod(shard_shape) * arr.dtype.itemsize


def tree_memory_per_device(
    tree: tuple[ArrayOrAbstract, ...] | ArrayOrAbstract,
) -> int:
  """Returns the memory usage of a PyTree on each device (in bytes)."""
  leaf_memory_per_device = jax.tree_util.tree_map(
      get_leaf_memory_per_device, tree
  )
  return jax.tree.reduce(lambda x, y: x + y, leaf_memory_per_device)


def get_available_memory(
    in_tree: tuple[ArrayOrAbstract, ...], scaling_factor: float
) -> int:
  """Returns estimated available memory for broadcasting (in bytes).

  After computing the available memory, we scale it by a factor of 0.75 to
  account for the fact that the actual memory usage could be different than the
  estimated memory usage. This will help us to avoid OOM errors for edge cases.

  Args:
    in_tree: pytree that occupies the memory.
    scaling_factor: indicates the frunction of the estimated available memory to
      be used when broadcustind data.
  """
  if scaling_factor > 1:
    raise ValueError('scaling_factorshould be less than 1.')
  total_device_memory = get_device_memory()
  used_device_memory = tree_memory_per_device(in_tree)
  available_memory = total_device_memory - used_device_memory
  return int(available_memory * scaling_factor / MEMORY_FACTOR)


def slice_count() -> int:
  """Returns the number of slices."""
  return (
      len(
          set(d.slice_index for d in jax.devices() if hasattr(d, 'slice_index'))
      )
      or 1
  )


def _get_slice_shape(
    index: tuple[slice, ...], global_shape: tuple[int, ...]
) -> tuple[int, ...]:
  """Calculates the shape of a slice from a global shape, assuming step is always 1."""
  return tuple(
      s.indices(global_shape[i])[1] - s.indices(global_shape[i])[0]
      for i, s in enumerate(index)
  )


def _globalized_abstract_array(
    leaf: ArrayOrAbstract,
    replica_axis_index: int,
    global_mesh: jax.sharding.Mesh,
) -> jax.ShapeDtypeStruct:
  """Returns the abstract array `_globalize_single_replica_arrays` produces."""
  num_replicas = global_mesh.devices.shape[replica_axis_index]
  replica_axis_name = global_mesh.axis_names[replica_axis_index]
  sharding = leaf.sharding
  if not isinstance(sharding, jax.sharding.NamedSharding):
    raise ValueError(
        'Must provide input arrays with NamedSharding. '
        f'Got {type(sharding)} instead.'
    )
  assert replica_axis_name not in sharding.spec, (
      f'Replica axis name {replica_axis_name} already exists in'
      f' sharding.spec {sharding.spec}'
  )
  global_spec = jax.sharding.PartitionSpec(
      replica_axis_name,
      *sharding.spec,
  )
  return jax.ShapeDtypeStruct(
      (num_replicas,) + tuple(leaf.shape),
      leaf.dtype,
      sharding=jax.sharding.NamedSharding(global_mesh, global_spec),
  )


def _single_device_scope(device: jax.Device):
  """Mesh context that pins eager ops to `device`, even under an active mesh."""
  mesh = jax.sharding.Mesh(np.array([device]), ('_single',))
  return jax.set_mesh(mesh) if hasattr(jax, 'set_mesh') else mesh


def _globalize_single_replica_arrays(
    inp: ArrayOrAbstract,
    replica_axis_index: int,
    global_mesh: jax.sharding.Mesh,
    is_source: bool,
) -> jax.Array:
  """Globalizes a single replica array.

  Adds a leading replica axis. Devices of the source replica hold the data;
  every other device holds zeros, so a sum over that axis broadcasts the
  data. Non-source hosts may pass a `jax.ShapeDtypeStruct`, since only its
  shape, dtype and sharding are used there.

  Args:
    inp: array sharded over the source replica, or its abstract equivalent.
    replica_axis_index: axis index along which the data is replicated.
    global_mesh: global mesh.
    is_source: whether this host belongs to the source replica.

  Returns:
    The array over `global_mesh` with the leading replica axis.
  """

  abstract = _globalized_abstract_array(inp, replica_axis_index, global_mesh)
  global_shape = abstract.shape
  global_sharding = abstract.sharding
  logging.vlog(
      1,
      'Globalizing array with local shape %s to Global shape: %s',
      inp.shape,
      global_shape,
  )

  source_device_map = {}

  if is_source:
    for s in inp.addressable_shards:
      with _single_device_scope(s.device):
        source_device_map[s.device] = jnp.expand_dims(s.data, axis=0)

  device_buffers = []
  for d, index in global_sharding.addressable_devices_indices_map(
      global_shape
  ).items():
    if d in source_device_map:
      device_buffers.append(source_device_map[d])
    else:
      # Use jax.numpy.zeros to allocate directly on device
      # to avoid Host RAM spike.
      slice_shape = _get_slice_shape(index, global_shape)  # pyrefly: ignore[bad-argument-type]
      with _single_device_scope(d):
        zero_data = jnp.zeros(slice_shape, dtype=inp.dtype, device=d)
      device_buffers.append(zero_data)

  logging.vlog(
      1,
      'Device buffers: %r',
      {d.device: d for d in device_buffers},
  )
  return jax.make_array_from_single_device_arrays(
      global_shape,
      global_sharding,
      device_buffers,
      dtype=inp.dtype,
  )


def _sum_over_replica_axis(
    tree: tuple[jax.Array, ...],
) -> tuple[jax.Array, ...]:
  # Module-level so JAX's jit cache, keyed on the function object, can hit
  # across chunks and across calls instead of recompiling each one.
  return jax.tree.map(functools.partial(jnp.sum, axis=0), tree)


def _merge_program(
    globalized_tree: tuple[ArrayOrAbstract, ...],
    global_mesh: jax.sharding.Mesh,
):
  """Jitted merge of globalized replicas back into single-replica shardings."""
  out_sharding = jax.tree.map(
      lambda x: jax.sharding.NamedSharding(
          global_mesh, jax.sharding.PartitionSpec(*x.sharding.spec[1:])
      ),
      globalized_tree,
  )
  return jax.jit(_sum_over_replica_axis, out_shardings=out_sharding)


def _chunk_by_memory(
    in_tree: tuple[ArrayOrAbstract, ...],
    memory_limit_bytes: int | None,
    memory_scaling_factor: float | None,
) -> list[tuple[int, int]]:
  """Splits leaf indices into contiguous ranges that fit the memory limit."""
  if memory_limit_bytes is None:
    memory_limit_bytes = get_available_memory(
        in_tree, memory_scaling_factor  # pyrefly: ignore[bad-argument-type]
    )
    logging.info('Using available memory of %d bytes.', memory_limit_bytes)
  tree_len = len(in_tree)
  chunks = []
  start = 0
  while start < tree_len:
    end = start
    if tree_memory_per_device(in_tree[start]) > memory_limit_bytes:
      logging.warning(
          'in_tree leaf size exceeds memory limit for broadcasting. '
          'Leaf size: %d bytes. Allowed memory limit: %d bytes. Proceeding.',
          tree_memory_per_device(in_tree[start]),
          memory_limit_bytes,
      )
      end += 1
    else:
      current_memory = 0
      while end < tree_len and (
          current_memory + tree_memory_per_device(in_tree[end])
          <= memory_limit_bytes
      ):
        current_memory += tree_memory_per_device(in_tree[end])
        end += 1
    chunks.append((start, end))
    start = end
  return chunks


def precompile_broadcast(
    in_tree: tuple[jax.ShapeDtypeStruct, ...],
    global_mesh: jax.sharding.Mesh,
    replica_axis_index: int,
    memory_limit_bytes: int | None = None,
    memory_scaling_factor: float | None = 0.75,
) -> None:
  """Compiles every merge program `broadcast_one_replica_to_all` will run.

  Takes the abstract leaves (shape, dtype and origin-replica sharding) so it
  can run before the arrays exist, e.g. on another thread while the source
  replica is still reading. It only warms JAX's compilation cache; the later
  broadcast call finds the programs there. If the arrays end up differing
  from these leaves, the broadcast compiles on the spot as before.

  Args:
    in_tree: abstract leaves with shardings corresponding to the origin
      replica.
    global_mesh: global mesh.
    replica_axis_index: axis index along which the data is replicated.
    memory_limit_bytes: memory limit for broadcasting in bytes.
    memory_scaling_factor: indicates the fraction of the estimated available
      memory to be used when broadcasting data.
  """
  for start, end in _chunk_by_memory(
      in_tree, memory_limit_bytes, memory_scaling_factor
  ):
    globalized = tuple(
        _globalized_abstract_array(x, replica_axis_index, global_mesh)
        for x in in_tree[start:end]
    )
    _merge_program(globalized, global_mesh).lower(globalized).compile()


def broadcast_one_replica_to_all(
    in_tree: tuple[ArrayOrAbstract, ...],
    global_mesh: jax.sharding.Mesh,
    replica_axis_index: int,
    is_source: bool,
    memory_limit_bytes: Optional[Union[int, None]] = None,
    memory_scaling_factor: Optional[float] = 0.75,
) -> tuple[tuple[jax.Array, ...], int]:
  """One replica reads the data and broadcasts to others.

  Args:
    in_tree: pytree to be broadcast. Shardings should correspond to the origin
      replica. Hosts outside it contribute only zeros, so they may pass
      `jax.ShapeDtypeStruct`s instead of arrays.
    global_mesh: global mesh.
    replica_axis_index: axis index along which the data is replicated.
    is_source: indicates if the current host is in origin replica.
    memory_limit_bytes: memory limit for broadcasting in bytes.
    memory_scaling_factor: indicates the fraction of the estimated available
      memory to be used when broadcasting data.

  Returns:
     Tuple containing:
      - pytree with broadcasted data
      - number of broadcasts performed.
  """
  chunks = _chunk_by_memory(in_tree, memory_limit_bytes, memory_scaling_factor)
  out_tree = []
  for start, end in chunks:
    subtree = tuple(in_tree[start:end])
    globalized_sharded_subtree = jax.tree.map(
        functools.partial(
            _globalize_single_replica_arrays,
            global_mesh=global_mesh,
            replica_axis_index=replica_axis_index,
            is_source=is_source,
        ),
        subtree,
    )
    # Delete immediately to conserve memory.
    for x in subtree:
      if isinstance(x, jax.Array):
        x.delete()
    out_subtree = _merge_program(globalized_sharded_subtree, global_mesh)(
        globalized_sharded_subtree
    )
    out_tree.extend(out_subtree)
    jax.block_until_ready(out_subtree)

  num_broadcasts = len(chunks)
  if is_source:
    logging.info('Total number of broadcasts: %d', num_broadcasts)
  return tuple(out_tree), num_broadcasts


def get_primary_replica_ids_and_pids(
    replica_axis_idx: int,
    mesh: jax.sharding.Mesh,
    primary_replica_id: int,
) -> tuple[Set[int], Set[int]]:
  """Returns the primary replica ids and process ids."""
  devices = replica_devices(
      mesh,
      replica_id=primary_replica_id,
      replica_axis_index=replica_axis_idx,
  ).flatten()
  ids = set([d.id for d in devices])
  pids = multihost.unique_processes_from_devices(devices)
  return ids, pids


def process_spans_multiple_replicas(
    global_mesh: jax.sharding.Mesh,
    *,
    replica_axis_index: int = 0,
) -> bool:
  """Checks if any JAX process controls devices across different replicas.

  Replicas are defined by slicing the `global_mesh` along the
  `replica_axis_index`. This function iterates through all unique JAX processes
  and, for each process, checks if the devices it manages belong to more than
  one replica group.

  Args:
    global_mesh: The global JAX mesh.
    replica_axis_index: The index of the axis in the mesh shape that
      differentiates the replicas.

  Returns:
    True if at least one process has devices in multiple replicas,
    False otherwise.
  """
  num_replicas = replica_count(
      global_mesh, replica_axis_index=replica_axis_index
  )
  all_processes = multihost.unique_processes_from_devices(
      global_mesh.devices.flatten()
  )

  for process_idx in all_processes:
    found_replica_ids = []
    for replica_id in range(num_replicas):
      devices_in_replica = replica_devices(
          global_mesh,
          replica_id=replica_id,
          replica_axis_index=replica_axis_index,
      )
      if process_idx in multihost.unique_processes_from_devices(
          devices_in_replica
      ):
        found_replica_ids.append(replica_id)

    if len(found_replica_ids) > 1:
      return True
  return False
