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

"""Utils for constructing maximal shardings."""

from __future__ import annotations
import math
from typing import Any, Sequence
from absl import logging
import jax
import numpy as np

PyTree = Any


def _partition_axis_name(offset: int) -> str:
  return str(chr(ord('a') + offset))


def _construct_maximal_sharding(
    sds: jax.ShapeDtypeStruct,
    devices: Sequence[jax.Device] | None = None,
) -> jax.sharding.Sharding:
  """Constructs a sharding that partitions the array as much as possible."""
  devices = devices or jax.devices()
  device_count = len(devices)
  shape = sds.shape
  if not shape:
    return jax.sharding.NamedSharding(
        mesh=jax.sharding.Mesh(devices, ('a',)),
        spec=jax.sharding.PartitionSpec(),
    )
  if np.max(shape) < jax.device_count():
    # Array is small - no sharding needed.
    return jax.sharding.NamedSharding(
        mesh=jax.sharding.Mesh(devices, ('a',)),
        spec=jax.sharding.PartitionSpec(),
    )

  available_device_dim = device_count
  partition_axes = [None] * len(shape)
  mesh_shape = []
  mesh_axes = []

  current_partition_axis = 0
  # Max to min.
  for i in np.argsort(shape)[::-1]:
    assert available_device_dim > 0
    if available_device_dim == 1:
      break
    if shape[i] < available_device_dim:
      continue
    gcd = math.gcd(shape[i], available_device_dim)
    if gcd == 1:
      continue
    available_device_dim //= gcd
    mesh_shape.append(gcd)

    current_partition_axis_name = _partition_axis_name(current_partition_axis)
    partition_axes[i] = current_partition_axis_name
    mesh_axes.append(current_partition_axis_name)
    current_partition_axis += 1

  # Still have some partition dimension left over.
  if available_device_dim > 1:
    mesh_shape.append(available_device_dim)
    mesh_axes.append(_partition_axis_name(current_partition_axis))

  logging.info(
      'Constructed sharding for array with shape: %s, mesh_shape: %s,'
      ' mesh_axes: %s, partition_axes: %s',
      shape,
      mesh_shape,
      mesh_axes,
      partition_axes,
  )

  assert len(mesh_shape) == len(mesh_axes)
  assert len(partition_axes) == len(shape)
  mesh = jax.sharding.Mesh(
      np.asarray(devices).reshape(mesh_shape),
      tuple(mesh_axes),
  )
  pspec = jax.sharding.PartitionSpec(*partition_axes)
  return jax.sharding.NamedSharding(mesh=mesh, spec=pspec)


def construct_maximal_shardings(
    abstract_state: PyTree, devices: Sequence[jax.Device] | None = None
) -> PyTree:
  """Construct a sharding that partitions each array as much as possible.

  This method is subject to change and should not be considered stable.

  Args:
    abstract_state: PyTree of jax.ShapeDtypeStruct.
    devices: Devices to shard across. If None, uses all available devices.

  Returns:
    PyTree of jax.sharding.Sharding.
  """
  shardings = jax.tree.map(
      lambda x: _construct_maximal_sharding(x, devices=devices), abstract_state
  )

  total_size = 0

  def _calculate_sharding_hbm_consumption(
      sds: jax.ShapeDtypeStruct, sharding: jax.sharding.Sharding
  ):
    nonlocal total_size
    shard_shape = sharding.shard_shape(sds.shape)
    total_size += np.prod(shard_shape) * sds.dtype.itemsize

  jax.tree.map(_calculate_sharding_hbm_consumption, abstract_state, shardings)
  logging.info('Expected per-device HBM consumption: %s', total_size)
  return shardings
