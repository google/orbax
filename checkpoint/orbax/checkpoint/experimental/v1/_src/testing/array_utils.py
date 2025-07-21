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

"""Utilities for array creation for unit tests."""

from typing import Any, cast

import jax
import numpy as np
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.arrays import abstract_arrays
from orbax.checkpoint.experimental.v1._src.serialization import numpy_leaf_handler
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types


def create_sharded_array(
    arr: np.ndarray, sharding: jax.sharding.Sharding
) -> jax.Array:
  sharding = cast(jax.sharding.NamedSharding, sharding)
  if isinstance(arr, (int, float)):
    spec = jax.sharding.PartitionSpec()
  else:
    spec = sharding.spec
  return test_utils.create_sharded_array(arr, sharding.mesh, spec)


def create_numpy_pytree(*, add: int = 0, include_scalars: bool = True):
  pytree = test_utils.setup_pytree(add=add)
  if include_scalars:
    pytree.update({'x': 4.5, 'y': 3})
  return pytree, jax.tree.map(as_abstract_type, pytree)


def create_sharded_pytree(
    *, add: int = 0, reverse_devices: bool = False, include_scalars: bool = True
) -> tuple[tree_types.PyTree, tree_types.PyTree]:
  """Creates a sharded PyTree from `create_numpy_pytree`.

  Args:
    add: The value to add to leaf arrays.
    reverse_devices: Whether to reverse the devices in the mesh.
    include_scalars: Whether to include scalars in the pytree.

  Returns:
    A tuple of (pytree, abstract_pytree).
  """
  devices = jax.devices()
  num_devices = len(devices)
  devices = (
      np.asarray(list(reversed(devices)))
      if reverse_devices
      else np.asarray(devices)
  )

  mesh_2d = jax.sharding.Mesh(
      devices.reshape((2, num_devices // 2)), ('x', 'y')
  )
  mesh_axes_2d = jax.sharding.PartitionSpec('x', 'y')
  mesh_1d = jax.sharding.Mesh(devices, ('x',))
  mesh_axes_1d = jax.sharding.PartitionSpec(
      'x',
  )
  mesh_0d = jax.sharding.Mesh(devices, ('x',))
  mesh_axes_0d = jax.sharding.PartitionSpec(
      None,
  )

  shardings = {
      'a': jax.sharding.NamedSharding(mesh_0d, mesh_axes_0d),
      'b': jax.sharding.NamedSharding(mesh_1d, mesh_axes_1d),
      'c': {
          'a': jax.sharding.NamedSharding(mesh_2d, mesh_axes_2d),
          'e': jax.sharding.NamedSharding(mesh_2d, mesh_axes_2d),
      },
  }
  if include_scalars:
    scalar_axes = jax.sharding.PartitionSpec()
    shardings.update({
        'x': jax.sharding.NamedSharding(mesh_0d, scalar_axes),
        'y': jax.sharding.NamedSharding(mesh_0d, scalar_axes),
    })
  pytree, _ = create_numpy_pytree(add=add, include_scalars=include_scalars)
  pytree = jax.tree.map(create_sharded_array, pytree, shardings)
  return pytree, jax.tree.map(as_abstract_type, pytree)


def as_abstract_type(value) -> Any:
  """Converts a value to its abstract type."""

  if isinstance(value, jax.Array):
    return abstract_arrays.to_shape_dtype_struct(value)
  elif isinstance(value, np.ndarray):
    return numpy_leaf_handler.NumpyShapeDtype(value.shape, value.dtype)
  elif isinstance(value, int):
    return int
  elif isinstance(value, float):
    return float
  elif isinstance(value, np.number):
    return value.dtype.type
  elif isinstance(value, str):
    return str
  else:
    raise ValueError(f'Unsupported type: {type(value)}')
