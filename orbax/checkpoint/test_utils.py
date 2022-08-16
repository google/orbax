# Copyright 2022 The Orbax Authors.
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

from typing import List, Optional

from etils import epath
import flax.serialization
from flax.training.train_state import TrainState
import jax
from jax.experimental import multihost_utils
from jax.experimental import pjit
from jax.experimental.global_device_array import GlobalDeviceArray
from jax.experimental.maps import Mesh
import jax.numpy as jnp
import numpy as np
import optax
from orbax.checkpoint import pytree_checkpoint_handler
from orbax.checkpoint import transform_utils
from orbax.checkpoint import utils


def save_fake_tmp_dir(directory: epath.Path,
                      step: int,
                      item: str,
                      subdirs: Optional[List[str]] = None) -> epath.Path:
  """Saves a directory with a tmp folder to simulate preemption."""
  subdirs = subdirs or []
  suffix = ''
  if not utils.is_gcs_path(directory):
    suffix = '.orbax-checkpoint-tmp-1010101'
  path = directory / str(step) / (item + suffix)
  if jax.process_index() == 0:
    path.mkdir(parents=True)
    for sub in subdirs:
      (path / sub).mkdir(parents=True)
  multihost_utils.sync_global_devices('save_fake_tmp_dir')
  return path


def replicate_gda(gda: GlobalDeviceArray):
  """Returns the input gda, but replicated across all devices."""
  mesh_axes = pjit.PartitionSpec(None,)
  fn = pjit.pjit(
      lambda x: x,
      in_axis_resources=gda.mesh_axes,
      out_axis_resources=mesh_axes)
  with gda.mesh:
    result = fn(gda)
  return result


def apply_function(gda_tree, function):
  """Applies the given function to every leaf in gda_tree.

  Args:
    gda_tree: a nested dict where every leaf is a GDA
    function: a function accepting an array and returning an array.

  Returns:
    a transformed GDA tree.
  """

  def f(gda):
    pjitted = pjit.pjit(
        function,
        in_axis_resources=gda.mesh_axes,
        out_axis_resources=gda.mesh_axes)
    with gda.mesh:
      result = pjitted(gda)
    return result

  return jax.tree_util.tree_map(f, gda_tree)


def assert_tree_equal(testclass, expected, actual):
  """Asserts that two PyTrees are equal, whether they are GDA or np/jnp array."""

  def assert_array_equal(v_expected, v_actual):
    assert isinstance(
        v_actual, type(v_expected)
    ), f'Found incompatible types: {type(v_expected)}, {type(v_actual)}'
    if isinstance(v_expected, GlobalDeviceArray):
      testclass.assertIsInstance(v_actual, GlobalDeviceArray)
      testclass.assertEqual(
          len(v_expected.local_shards), len(v_actual.local_shards))
      for shard_expected, shard_actual in zip(v_expected.local_shards,
                                              v_actual.local_shards):
        np.testing.assert_array_equal(shard_expected.data, shard_actual.data)
    else:
      np.testing.assert_array_equal(v_expected, v_actual)

  jax.tree_util.tree_map(assert_array_equal, expected, actual, is_leaf=is_leaf)


def setup_pytree():
  """Creates a numpy PyTree for testing."""
  pytree = {
      'a': np.arange(8) * 1,
      'b': np.arange(16) * 2,
      'c': {
          'a': np.arange(8).reshape((2, 4)) * 3,
          'e': np.arange(16).reshape((4, 4)) * 4,
      }
  }
  return pytree


def setup_gda_pytree():
  """Creates a PyTree of sharded GDAs for testing."""
  devices = np.asarray(jax.devices())

  mesh_2d = Mesh(devices.reshape((2, len(devices) // 2)), ('x', 'y'))
  mesh_axes_2d = pjit.PartitionSpec('x', 'y')
  mesh_1d = Mesh(devices, ('x',))
  mesh_axes_1d = pjit.PartitionSpec('x',)
  mesh_0d = Mesh(devices, ('x',))
  mesh_axes_0d = pjit.PartitionSpec(None,)

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
      as_gda, pytree, mesh_tree, axes_tree, is_leaf=is_leaf)
  return pytree, mesh_tree, axes_tree


def is_leaf(x):
  return isinstance(x, np.ndarray) or isinstance(x, Mesh) or isinstance(
      x, pjit.PartitionSpec) or isinstance(x,
                                           pytree_checkpoint_handler.ParamInfo)


def as_gda(arr, mesh, mesh_axes):
  if isinstance(arr, (int, float)):
    arr = np.asarray(arr)
  return GlobalDeviceArray.from_callback(arr.shape, mesh, mesh_axes,
                                         lambda idx: arr[idx])


def init_flax_model(model):
  params = model.init(jax.random.PRNGKey(0), jnp.ones([8, 8]))
  tx = optax.adamw(learning_rate=0.001)
  state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
  return jax.tree_util.tree_map(np.asarray, state)


def _filter(mask_tree, *other_trees):
  """Filter optax EmptyState and MaskedNode out of PyTree."""
  if not isinstance(mask_tree, dict):
    return other_trees
  result_trees = [{} for _ in other_trees]
  for k, v in mask_tree.items():
    if not isinstance(v, (optax.EmptyState, optax.MaskedNode)):
      values = _preprocess_helper(v, *(t[k] for t in other_trees))
      for i, v1 in enumerate(values):
        if isinstance(v1, dict):
          if v1:
            result_trees[i][k] = v1
        else:
          result_trees[i][k] = v1
  return tuple(result_trees)


def _preprocess_helper(mask_tree, *other_trees):
  return jax.tree_util.tree_map(
      _filter, mask_tree, *other_trees, is_leaf=lambda x: isinstance(x, dict))


def preprocess_flax_pytree(mask_tree, *other_trees):
  mask_tree = flax.serialization.to_state_dict(mask_tree)
  other_trees = (flax.serialization.to_state_dict(t) for t in other_trees)
  r = _preprocess_helper(mask_tree, *other_trees)
  return r[0] if len(r) == 1 else r


def postprocess_flax_pytree(reference, processed):
  """Merge trees by taking values from processed or reference if key not present."""
  transforms = {}
  return transform_utils.apply_transformations(processed, transforms, reference)
