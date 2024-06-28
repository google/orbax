# Copyright 2024 The Orbax Authors.
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

"""Multislice utils."""

import functools
from typing import Any, Optional, Tuple, Union

from absl import logging
import jax
from jax import numpy as jnp
import numpy as np
from orbax.checkpoint.multihost import utils

PyTree = Any

# When using broadcasting from single replica to others, 3 copies of the data
# are stored in memory.
MEMORY_FACTOR = 3


def process_slice_id(process_index: int, global_mesh: jax.sharding.Mesh) -> int:
  """Returns the slice id that the process_index belongs to."""
  for slice_id, device_slice in enumerate(global_mesh.devices):
    if process_index in _pid_in_slice(device_slice):
      return slice_id

  return -1


def _pid_in_slice(device_slice: np.ndarray) -> np.ndarray:
  pid = np.vectorize(
      lambda d: utils.runtime_to_distributed_process_id(d.process_index)
  )
  return pid(device_slice)


def in_slice(process_index: int, device_slice: np.ndarray) -> bool:
  return process_index in _pid_in_slice(device_slice)


def in_primary_slice(
    process_index: int, global_mesh: jax.sharding.Mesh
) -> bool:
  """Returns true if host is in primary slice (the first slice)."""
  primary_slice = global_mesh.devices[0]

  return in_slice(process_index, primary_slice)


def local_slice_devices(devices_array: np.ndarray) -> np.ndarray:
  for device_slice in devices_array:
    if in_slice(utils.process_index(), device_slice):
      return device_slice
  raise ValueError(
      f'process_index {utils.process_index()} does not exist in provided'
      ' `global_mesh`'
  )


def _sum(x, replica_axis_index):
  return jax.tree.map(functools.partial(jnp.sum, axis=replica_axis_index), x)


@functools.partial(jax.jit, static_argnums=0)
def fake_zero_data(sharding, x):
  x = jnp.zeros_like(x)
  return jax.lax.with_sharding_constraint(x, sharding)


def get_device_memory() -> int:
  """Returns HBM capacity of the device on which the code is running(in bytes)."""
  device = jax.devices()[0]
  if device.platform != 'tpu':
    raise ValueError('Only TPU devices are supported.')
  hbm_memory = {
      'TPU v3': int(16e9),  # two cores pre chip each with 16 GB HBM
      'TPU v4': int(32e9),  # one megacore per chip with 32 GB HBM
      'TPU v5 lite': int(16e9),  # one core per chip with 16 GB HBM
      'TPU v5p': int(96e9),  # one megacore per chip with 96 GB HBM
      'TPU trillium': int(32e9),  # one core per chip with 32 GB HBM
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


def tree_memory_per_device(tree: Tuple[PyTree, ...]) -> int:
  """Returns the memory usage of a PyTree on each device (in bytes)."""
  leaf_memory_per_device = jax.tree_util.tree_map(
      get_leaf_memory_per_device, tree
  )
  return jax.tree.reduce(lambda x, y: x + y, leaf_memory_per_device)


def get_available_memory(
    in_tree: Tuple[PyTree, ...], scaling_factor: float
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


def broadcast_one_replica_to_all(
    in_tree: Tuple[PyTree, ...],
    global_mesh: jax.sharding.Mesh,
    per_replica_shardings: Tuple[Optional[jax.sharding.NamedSharding], ...],
    replica_axis_index: int,
    is_source: bool,
    memory_limit_bytes: Optional[Union[int, None]] = None,
    memory_scaling_factor: Optional[float] = 0.75,
) -> Tuple[Tuple[PyTree, ...], int]:
  """One replica reads the data and broadcasts to others.

  Args:
    in_tree: pytree to be broadcasted.
    global_mesh: global mesh.
    per_replica_shardings: shardings for each replica.
    replica_axis_index: axis index along which the data is replicated.
    is_source: indicates if the current host is in primary replica.
    memory_limit_bytes: memory limit for broadcasting. in bytes.
    memory_scaling_factor: indicates the frunction of the estimated available
      memory to be used when broadcustind data.

  Returns:
     Tuple containing:
      - pytree with broadcasted data
      - number of broadcasts performed.
  """
  num_replicas = global_mesh.devices.shape[replica_axis_index]
  replica_axis_name = global_mesh.axis_names[replica_axis_index]

  if memory_limit_bytes is None:
    memory_limit_bytes = get_available_memory(in_tree, memory_scaling_factor)
    logging.info('Using available memory of %d bytes.', memory_limit_bytes)

  def pre_jit(x, per_replica_sharding):
    if is_source:
      inp = x
    else:
      inp = fake_zero_data(per_replica_sharding, x)
    inp = jnp.expand_dims(inp, axis=replica_axis_index)
    in_spec = jax.sharding.PartitionSpec(
        *x.sharding.spec[:replica_axis_index],
        replica_axis_name,
        *x.sharding.spec[replica_axis_index:],
    )
    global_shape = (
        x.shape[:replica_axis_index]
        + (num_replicas,)
        + x.shape[replica_axis_index:]
    )
    global_sharding = jax.sharding.NamedSharding(global_mesh, in_spec)
    return jax.make_array_from_single_device_arrays(
        global_shape, global_sharding, [s.data for s in inp.addressable_shards]
    )

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
    out_sharding = jax.tree.map(
        lambda x: jax.sharding.NamedSharding(
            global_mesh, jax.sharding.PartitionSpec(*x.sharding.spec)
        ),
        subtree,
    )
    in_tree_sharded = jax.tree.map(
        pre_jit, subtree, per_replica_shardings[start:end]
    )
    # Delete immediately to conserve memory.
    jax.tree.map(lambda x: x.delete(), subtree)
    out_subtree = jax.jit(
        functools.partial(_sum, replica_axis_index=replica_axis_index),
        out_shardings=out_sharding,
    )(in_tree_sharded)
    out_tree.extend(out_subtree)
    jax.block_until_ready(out_tree)
    start = end
  logging.info('Number of broadcasts: %d', num_broadcasts)
  return tuple(out_tree), num_broadcasts


def get_primary_replica_ids_and_pids(
    replica_axis_idx: int,
    mesh: jax.sharding.Mesh,
    primary_replica_id: int,
):
  """Returns the primary replica ids and process ids."""
  replica_devices = np.take(
      mesh.devices,
      primary_replica_id,
      axis=replica_axis_idx,
  ).flatten()
  pids = set([d.process_index for d in replica_devices])
  ids = set([d.id for d in replica_devices])
  return ids, pids
