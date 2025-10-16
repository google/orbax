# Copyright 2025 The Orbax Authors.
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
from typing import Any, Optional, Set, Union

from absl import logging
import jax
from jax import numpy as jnp
import numpy as np
from orbax.checkpoint._src.multihost import multihost

PyTree = Any

# When using broadcasting from single replica to others, 3 copies of the data
# are stored in memory.
MEMORY_FACTOR = 3


def process_replica_id(
    process_index: int,
    global_mesh: jax.sharding.Mesh,
    *,
    replica_axis_index: int = 0,
) -> int:
  """Returns the slice id that the process_index belongs to."""
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
  """Get devices in the host-local slice."""
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


@functools.partial(jax.jit, static_argnums=(0, 1, 2))
def fake_zero_data(sharding, shape, dtype=jnp.float32) -> jax.Array:
  x = jnp.zeros(shape, dtype=dtype)
  return jax.lax.with_sharding_constraint(x, sharding)


def get_device_memory() -> int:
  """Returns HBM capacity of the device on which the code is running(in bytes)."""
  device = jax.devices()[0]
  if device.platform not in ('tpu', 'gpu'):
    raise ValueError('Only select TPU and GPU devices are supported.')
  hbm_memory = {
      'TPU v3': int(16e9),  # two cores per chip each with 16 GB HBM
      'TPU v4': int(32e9),  # one megacore per chip with 32 GB HBM
      'TPU v5 lite': int(16e9),  # one core per chip with 16 GB HBM
      'TPU v5': int(96e9),  # one megacore per chip with 96 GB HBM
      'TPU v6 lite': int(32e9),  # one core per chip with 32 GB HBM
      'NVIDIA H100 80GB HBM3': int(80e9),
      'NVIDIA H200': int(144e9),
      'NVIDIA B200': int(183e9),
  }
  memory = hbm_memory.get(device.device_kind, None)
  if memory is None:
    raise ValueError(
        f'get_device_memory is not supported for {device.device_kind}.'
    )
  return memory


def get_leaf_memory_per_device(arr: jax.Array) -> int:
  """Returns the memory usage of a sharded array per device (in bytes)."""
  shard = arr.addressable_shards[0]
  return shard.data.size * shard.data.itemsize


def tree_memory_per_device(tree: tuple[jax.Array, ...] | jax.Array) -> int:
  """Returns the memory usage of a PyTree on each device (in bytes)."""
  leaf_memory_per_device = jax.tree_util.tree_map(
      get_leaf_memory_per_device, tree
  )
  return jax.tree.reduce(lambda x, y: x + y, leaf_memory_per_device)


def get_available_memory(
    in_tree: tuple[jax.Array, ...], scaling_factor: float
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


def _globalize_single_replica_arrays(
    inp: jax.Array,
    replica_axis_index: int,
    global_mesh: jax.sharding.Mesh,
    is_source: bool,
) -> jax.Array:
  """Globalizes a single replica array."""

  num_replicas = global_mesh.devices.shape[replica_axis_index]
  replica_axis_name = global_mesh.axis_names[replica_axis_index]
  sharding = inp.sharding
  if not isinstance(sharding, jax.sharding.NamedSharding):
    raise ValueError(
        'Must provide input arrays with NamedSharding. '
        f'Got {type(sharding)} instead.'
    )
  local_replica_shape = inp.shape

  assert replica_axis_name not in sharding.spec, (
      f'Replica axis name {replica_axis_name} already exists in'
      f' sharding.spec {sharding.spec}'
  )
  global_shape = (num_replicas,) + local_replica_shape
  logging.vlog(
      1,
      'Globalizing array with local shape %s to Global shape: %s',
      local_replica_shape,
      global_shape,
  )
  global_spec = jax.sharding.PartitionSpec(
      replica_axis_name,
      *sharding.spec,
  )
  global_sharding = jax.sharding.NamedSharding(global_mesh, global_spec)

  source_device_map = {}

  @jax.jit
  def _expand_dims(x: jax.Array):
    return jnp.expand_dims(x, axis=0)

  inp = _expand_dims(inp)
  if is_source:
    for s in inp.addressable_shards:
      source_device_map[s.device] = s.data

  device_buffers = []
  for d, index in global_sharding.addressable_devices_indices_map(
      global_shape
  ).items():
    if d in source_device_map:
      device_buffers.append(source_device_map[d])
    else:
      zero_data = np.zeros(
          _get_slice_shape(index, global_shape), dtype=inp.dtype
      )
      device_buffers.append(jax.device_put(zero_data, d))

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


def _merge_globalized_replicas(
    globalized_tree: tuple[jax.Array, ...],
    global_mesh: jax.sharding.Mesh,
):
  """Merges globalized sharded replicas into a single replica."""
  out_sharding = jax.tree.map(
      lambda x: jax.sharding.NamedSharding(
          global_mesh, jax.sharding.PartitionSpec(*x.sharding.spec[1:])
      ),
      globalized_tree,
  )
  out_subtree = jax.jit(
      lambda tree: jax.tree.map(functools.partial(jnp.sum, axis=0), tree),
      out_shardings=out_sharding,
  )(globalized_tree)
  return out_subtree


def broadcast_one_replica_to_all(
    in_tree: tuple[jax.Array, ...],
    global_mesh: jax.sharding.Mesh,
    replica_axis_index: int,
    is_source: bool,
    memory_limit_bytes: Optional[Union[int, None]] = None,
    memory_scaling_factor: Optional[float] = 0.75,
) -> tuple[tuple[jax.Array, ...], int]:
  """One replica reads the data and broadcasts to others.

  Args:
    in_tree: pytree to be broadcast. Shardings should correspond to the origin
      replica.
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
  if memory_limit_bytes is None:
    memory_limit_bytes = get_available_memory(in_tree, memory_scaling_factor)
    logging.info('Using available memory of %d bytes.', memory_limit_bytes)

  tree_len = len(in_tree)
  start = 0
  out_tree = []
  num_broadcasts = 0
  while start < tree_len:
    subtree = []
    current_memory = 0
    end = start
    if tree_memory_per_device(in_tree[start]) > memory_limit_bytes:
      logging.warning(
          'in_tree leaf size exceeds memory limit for broadcasting. '
          'Leaf size: %d bytes. Allowed memory limit: %d bytes. Proceeding.',
          tree_memory_per_device(in_tree[start]),
          memory_limit_bytes,
      )
      subtree.append(in_tree[end])
      end += 1
    else:
      while end < tree_len and (
          current_memory + tree_memory_per_device(in_tree[end])
          <= memory_limit_bytes
      ):
        subtree.append(in_tree[end])
        current_memory += tree_memory_per_device(in_tree[end])
        end += 1
    subtree = tuple(subtree)
    num_broadcasts += 1
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
    jax.tree.map(lambda x: x.delete(), subtree)
    out_subtree = _merge_globalized_replicas(
        globalized_sharded_subtree, global_mesh
    )
    out_tree.extend(out_subtree)
    jax.block_until_ready(out_subtree)
    start = end

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
