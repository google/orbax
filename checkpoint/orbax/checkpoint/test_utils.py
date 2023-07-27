# Copyright 2023 The Orbax Authors.
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

"""Utils for orbax tests."""

import functools
from typing import List, Optional

from etils import epath
import jax
from jax import sharding
from jax.experimental import pjit
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint import pytree_checkpoint_handler
from orbax.checkpoint import utils


def save_fake_tmp_dir(
    directory: epath.Path,
    step: int,
    item: str,
    subdirs: Optional[List[str]] = None,
    step_prefix: Optional[str] = None,
) -> epath.Path:
  """Saves a directory with a tmp folder to simulate preemption."""
  subdirs = subdirs or []
  if not step_prefix:
    step_prefix = ''
  step_tmp_dir = utils.create_tmp_directory(
      directory / (step_prefix + str(step))
  )
  item_tmp_dir = utils.create_tmp_directory(step_tmp_dir / item)
  if jax.process_index() == 0:
    for sub in subdirs:
      (item_tmp_dir / sub).mkdir(parents=True)
  utils.sync_global_devices('save_fake_tmp_dir')
  return item_tmp_dir


def replicate_sharded_array(arr: jax.Array):
  """Returns the input array, but replicated across all devices."""
  mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ('x',))
  replicated_sharding = sharding.NamedSharding(
      mesh,
      jax.sharding.PartitionSpec(
          None,
      ),
  )
  return pjit.pjit(lambda x: x, out_shardings=replicated_sharding)(arr)


def apply_function(tree, function):
  """Applies the given function to every leaf in tree.

  Args:
    tree: a nested dict where every leaf is a sharded jax.Array.
    function: a function accepting an array and returning jax.Array.

  Returns:
    a transformed sharded array tree.
  """

  def f(arr):
    return pjit.pjit(function)(arr)

  return jax.tree_util.tree_map(f, tree)


def assert_array_equal(testclass, v_expected, v_actual):
  """Asserts that two arrays are equal."""
  testclass.assertIsInstance(v_actual, type(v_expected))
  if isinstance(v_expected, jax.Array):
    testclass.assertEqual(
        len(v_expected.addressable_shards), len(v_actual.addressable_shards)
    )
    for shard_expected, shard_actual in zip(
        v_expected.addressable_shards, v_actual.addressable_shards
    ):
      np.testing.assert_array_equal(shard_expected.data, shard_actual.data)
  elif isinstance(v_expected, (np.ndarray, jnp.ndarray)):
    np.testing.assert_array_equal(v_expected, v_actual)
  else:
    testclass.assertEqual(v_expected, v_actual)


def assert_tree_equal(testclass, expected, actual):
  """Asserts that two PyTrees are equal."""
  expected_flat = utils.to_flat_dict(expected)
  actual_flat = utils.to_flat_dict(actual)
  testclass.assertSameElements(expected_flat.keys(), actual_flat.keys())
  jax.tree_util.tree_map(
      functools.partial(assert_array_equal, testclass), expected, actual
  )


def setup_pytree(add: int = 0):
  """Creates a numpy PyTree for testing."""
  pytree = {
      'a': np.arange(8) * 1,
      'b': np.arange(16) * 2,
      'c': {
          'a': np.arange(8).reshape((2, 4)) * 3,
          'e': np.arange(16).reshape((4, 4)) * 4,
      }
  }
  pytree = jax.tree_util.tree_map(lambda x: x + add, pytree, is_leaf=is_leaf)
  return pytree


def setup_sharded_pytree():
  """Creates a PyTree of sharded arrays for testing."""
  devices = np.asarray(jax.devices())

  mesh_2d = jax.sharding.Mesh(
      devices.reshape((2, len(devices) // 2)), ('x', 'y')
  )
  mesh_axes_2d = jax.sharding.PartitionSpec('x', 'y')
  mesh_1d = jax.sharding.Mesh(devices, ('x',))
  mesh_axes_1d = jax.sharding.PartitionSpec('x',)
  mesh_0d = jax.sharding.Mesh(devices, ('x',))
  mesh_axes_0d = jax.sharding.PartitionSpec(None,)

  pytree = setup_pytree()
  mesh_tree = {
      'a': mesh_0d,
      'b': mesh_1d,
      'c': {
          'a': mesh_2d,
          'e': mesh_2d,
      }
  }
  axes_tree = {
      'a': mesh_axes_0d,
      'b': mesh_axes_1d,
      'c': {
          'a': mesh_axes_2d,
          'e': mesh_axes_2d,
      }
  }

  pytree = jax.tree_util.tree_map(
      create_sharded_array, pytree, mesh_tree, axes_tree, is_leaf=is_leaf)
  return pytree, mesh_tree, axes_tree


def is_leaf(x):
  return (
      isinstance(x, np.ndarray)
      or isinstance(x, jax.sharding.Mesh)
      or isinstance(x, jax.sharding.PartitionSpec)
      or isinstance(x, pytree_checkpoint_handler.ParamInfo)
  )


def create_sharded_array(arr, mesh, mesh_axes):
  """Create sharded jax.Array."""
  if isinstance(arr, (int, float)):
    arr = np.asarray(arr)
  return jax.make_array_from_callback(
      arr.shape, sharding.NamedSharding(mesh, mesh_axes), lambda idx: arr[idx]
  )
