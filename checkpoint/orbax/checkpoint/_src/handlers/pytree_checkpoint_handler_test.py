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

"""Tests for PyTreeCheckpointHandler module."""

import contextlib
import copy
import functools
import json
from typing import Any, Optional
import unittest
from unittest import mock

from absl import flags
from absl.testing import parameterized
from etils import epath
import flax.training.train_state
import jax
from jax import numpy as jnp
from jax.experimental import pjit
import numpy as np
import optax
from orbax.checkpoint import msgpack_utils
from orbax.checkpoint import test_utils
from orbax.checkpoint import transform_utils
from orbax.checkpoint import utils
from orbax.checkpoint._src.handlers import base_pytree_checkpoint_handler
from orbax.checkpoint._src.handlers import pytree_checkpoint_handler
from orbax.checkpoint._src.metadata import array_metadata_store as array_metadata_store_lib
from orbax.checkpoint._src.metadata import empty_values
from orbax.checkpoint._src.metadata import sharding as sharding_metadata
from orbax.checkpoint._src.metadata import tree as tree_metadata
from orbax.checkpoint._src.metadata import value as value_metadata
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.serialization import type_handler_registry
from orbax.checkpoint._src.serialization import type_handlers
from orbax.checkpoint._src.testing import multiprocess_test
from orbax.checkpoint._src.tree import utils as tree_utils


PyTree = Any
ParamInfo = pytree_checkpoint_handler.ParamInfo
SaveArgs = pytree_checkpoint_handler.SaveArgs
RestoreArgs = pytree_checkpoint_handler.RestoreArgs
ArrayRestoreArgs = pytree_checkpoint_handler.ArrayRestoreArgs
PyTreeSaveArgs = pytree_checkpoint_handler.PyTreeSaveArgs
PyTreeRestoreArgs = pytree_checkpoint_handler.PyTreeRestoreArgs
Transform = transform_utils.Transform
RestoreTransform = transform_utils.RestoreTransform
PyTreeCheckpointHandler = test_utils.PyTreeCheckpointHandler
_SHARDING = '_sharding'
PYTREE_METADATA_FILE = pytree_checkpoint_handler.PYTREE_METADATA_FILE
PLACEHOLDER = base_pytree_checkpoint_handler.PLACEHOLDER


ARRAY_METADATA_STORE = array_metadata_store_lib.Store()


FLAGS = flags.FLAGS

jax.config.update('jax_enable_x64', True)


def _raise_file_not_found_error(*args, **kwargs):
  del args, kwargs
  raise FileNotFoundError()


# Not in common util because we need to eliminate OSS dependency on flax.
def init_flax_model(model):
  params = model.init(jax.random.PRNGKey(0), jnp.ones([8, 8]))
  tx = optax.adamw(learning_rate=0.001)
  state = flax.training.train_state.TrainState.create(
      apply_fn=model.apply, params=params, tx=tx
  )
  return jax.tree.map(np.asarray, state)


class PyTreeCheckpointHandlerTest(
    parameterized.TestCase,
    multiprocess_test.MultiProcessTest,
    unittest.IsolatedAsyncioTestCase,
):

  def setUp(self):
    super().setUp()

    pytree, mesh_tree, axes_tree = test_utils.setup_sharded_pytree()
    self.numpy_pytree = test_utils.setup_pytree()
    self.numpy_pytree.update({'x': 4.5, 'y': 3})
    self.empty_pytree = jax.tree.map(
        lambda x: object(), pytree, is_leaf=test_utils.is_leaf
    )
    self.pytree = pytree
    self.mesh_tree = mesh_tree
    self.axes_tree = axes_tree

    def _create_restore_args(arr, mesh, axes):
      return ArrayRestoreArgs(restore_type=type(arr), mesh=mesh, mesh_axes=axes)

    self.restore_args = jax.tree.map(
        _create_restore_args, pytree, mesh_tree, axes_tree
    )
    self.directory = epath.Path(
        self.create_tempdir(name='checkpointing_test').full_path
    )
    # TODO: b/365169723 - Add tests for support_rich_types=True.
    self.pytree_metadata_options = tree_metadata.PyTreeMetadataOptions(
        support_rich_types=False
    )

    # default to use_ocdbt=False, so we can test non-ocdbt handler first
    self.handler = self.enter_context(
        self.ocdbt_checkpoint_handler(
            use_ocdbt=False, array_metadata_store=ARRAY_METADATA_STORE
        )
    )
    test_utils.set_tensorstore_driver_for_test()

    test_utils.sync_global_processes(
        'PyTreeCheckpointHandlerTest:setup_complete'
    )

  def tearDown(self):
    test_utils.sync_global_processes(
        'PyTreeCheckpointHandlerTest:tests_complete'
    )
    super().tearDown()

  @contextlib.contextmanager
  def ocdbt_checkpoint_handler(
      self,  # pylint: disable=unused-argument
      use_ocdbt: bool,
      use_zarr3: bool = False,
      enable_pinned_host_transfer: bool | None = None,
      pytree_metadata_options: tree_metadata.PyTreeMetadataOptions = (
          tree_metadata.PYTREE_METADATA_OPTIONS
      ),
      array_metadata_store: array_metadata_store_lib.Store | None = (
          ARRAY_METADATA_STORE
      ),
      use_compression: bool = True,
  ):
    """Registers handlers with OCDBT support and resets when done."""
    handler_registry = copy.deepcopy(
        type_handler_registry.GLOBAL_TYPE_HANDLER_REGISTRY
    )
    handler_registry.get(jax.Array)._array_metadata_store = array_metadata_store

    handler = PyTreeCheckpointHandler(
        use_ocdbt=use_ocdbt,
        use_zarr3=use_zarr3,
        type_handler_registry=handler_registry,
        pytree_metadata_options=pytree_metadata_options,
        enable_pinned_host_transfer=enable_pinned_host_transfer,
        use_compression=use_compression,
    )
    try:
      yield handler
    finally:
      handler.close()

  def create_mixed_format_pytree(
      self,
      add: int = 0,
      strings: bool = False,
      key_name: str = 'new_key',
  ) -> PyTree:
    """Creates a PyTree with different leaf types for testing.

    Args:
      add: Adds the specified value to numeric leafs.
      strings: If true, adds string leaves to the tree.
      key_name: Name of the pytree leaf that can be modified.

    Returns:
      PyTree
    """
    pytree = dict(test_utils.setup_pytree(add=add))
    pytree[key_name] = self.pytree
    meshes = jax.tree.map(lambda x: None, pytree, is_leaf=test_utils.is_leaf)
    meshes[key_name] = dict(self.mesh_tree)
    mesh_axes = jax.tree.map(lambda x: None, pytree, is_leaf=test_utils.is_leaf)
    mesh_axes[key_name] = dict(self.axes_tree)
    if strings:
      pytree['foo'] = 'foo_val'
      pytree['bar'] = 'bar_val'
      meshes['foo'] = None
      meshes['bar'] = None
      mesh_axes['foo'] = None
      mesh_axes['bar'] = None

    def _save_args(arr):
      del arr
      return SaveArgs()

    def _restore_args(arr, mesh, axes):
      if isinstance(arr, jax.Array):
        return ArrayRestoreArgs(
            restore_type=type(arr), mesh=mesh, mesh_axes=axes
        )
      else:
        return RestoreArgs(restore_type=type(arr))

    save_args = jax.tree.map(_save_args, pytree, is_leaf=test_utils.is_leaf)
    restore_args = jax.tree.map(
        _restore_args, pytree, meshes, mesh_axes, is_leaf=test_utils.is_leaf
    )

    return pytree, save_args, restore_args

  def validate_save(
      self,
      path: epath.Path,
      expected: PyTree,
      checkpoint_handler: PyTreeCheckpointHandler,
      save_args: Optional[PyTree] = None,
      restore_args: Optional[PyTree] = None,
  ):
    """Validate save was performed correctly."""
    del save_args
    if restore_args is None:
      restore_args = jax.tree.map(lambda _: RestoreArgs(), expected)
    actual = checkpoint_handler.restore(
        path, args=PyTreeRestoreArgs(restore_args=restore_args)
    )
    test_utils.assert_tree_equal(self, expected, actual)

  def validate_restore(self, expected, actual):
    test_utils.assert_tree_equal(self, expected, actual)

  # TODO(b/301122724) Remove after b/301122724 is implemented.
  def should_validate_metadata(self) -> bool:
    return True

  def validate_metadata(
      self,
      *,
      expected_reference_metadata_tree: PyTree,
      actual_metadata: PyTree,
      pytree_metadata_options: tree_metadata.PyTreeMetadataOptions,
      save_args=None,
      array_metadata_store: array_metadata_store_lib.Store | None,
  ):
    """Validate metadata, provided the original tree that was saved."""
    del save_args
    expected_reference_metadata_tree = tree_metadata.serialize_tree(
        expected_reference_metadata_tree, pytree_metadata_options
    )

    def _metadata(value):
      if empty_values.is_supported_empty_value(value, pytree_metadata_options):
        return value
      if isinstance(value, np.ndarray):
        return value_metadata.ArrayMetadata(
            name='',
            directory=None,
            shape=value.shape,
            sharding=None,
            dtype=value.dtype,
            storage=value_metadata.StorageMetadata(
                chunk_shape=value.shape,
            ),
        )
      if isinstance(value, jax.Array):
        expected_chunk_shape = test_utils.get_expected_chunk_shape(value)
        return value_metadata.ArrayMetadata(
            name='',
            directory=None,
            shape=value.shape,
            sharding=sharding_metadata.from_jax_sharding(value.sharding),
            dtype=value.dtype,
            storage=value_metadata.StorageMetadata(
                chunk_shape=expected_chunk_shape,
                write_shape=(
                    expected_chunk_shape
                    if array_metadata_store is not None
                    else None
                ),
            ),
        )
      if isinstance(value, (float, int)):
        dtype = np.float64 if isinstance(value, float) else np.int64
        return value_metadata.ScalarMetadata(
            name='', directory=None, dtype=dtype
        )  # pytype: disable=wrong-arg-types  # jnp-type
      if isinstance(value, str):
        return value_metadata.StringMetadata(name='', directory=None)
      if isinstance(value, optax.EmptyState):
        return None
      raise ValueError(f'Unrecognized type: {type(value)}.')

    expected_metadata = jax.tree.map(
        _metadata,
        expected_reference_metadata_tree,
        is_leaf=tree_utils.is_empty_or_leaf,
    )
    test_utils.assert_tree_equal(self, expected_metadata, actual_metadata.tree)

  def test_get_param_names(self):
    param_names = pytree_checkpoint_handler.get_param_names(self.pytree)
    expected = {
        'a': 'a',
        'b': 'b',
        'c': {
            'a': 'c.a',
            'e': 'c.e',
        },
    }
    test_utils.assert_tree_equal(self, expected, param_names)

  def test_save_format(self):
    pytree = {'a': 0, 'c': {'d': np.arange(3), 'e': {'f': 5}}, 'g': 10}
    save_args = jax.tree.map(lambda x: SaveArgs(), pytree)
    self.handler.save(self.directory, args=PyTreeSaveArgs(pytree, save_args))
    fnames = ['a', 'c.d', 'c.e.f', 'g']
    paths = [self.directory / name for name in fnames]
    for p in paths:
      self.assertTrue(p.exists())
      self.assertTrue((p / '.zarray').exists())

  @parameterized.product(use_ocdbt=(True, False))
  def test_save_sharding(self, use_ocdbt: bool):
    if utils.is_pathways_backend():
      self.skipTest('Sharding metadata not present on Pathways.')
    with self.ocdbt_checkpoint_handler(use_ocdbt) as checkpoint_handler:
      pytree, save_args, restore_args = self.create_mixed_format_pytree(
          key_name='mlp/~/linear_0'
      )

      checkpoint_handler.save(
          self.directory, args=PyTreeSaveArgs(pytree, save_args)
      )

      self.validate_save(
          self.directory,
          pytree,
          checkpoint_handler,
          save_args=save_args,
          restore_args=restore_args,
      )

    path = self.directory

    self.assertTrue((path / _SHARDING).exists())
    with open(path / _SHARDING, 'r') as file:
      data = json.load(file)
      self.assertCountEqual(
          data.keys(),
          {
              'bWxwL34vbGluZWFyXzAuYQ==',  # mlp/~/linear_0.a
              'bWxwL34vbGluZWFyXzAuYg==',  # mlp/~/linear_0.b
              'bWxwL34vbGluZWFyXzAuYy5h',  # mlp/~/linear_0.c.a
              'bWxwL34vbGluZWFyXzAuYy5l',  # mlp/~/linear_0.c.e
          },
          None,
      )
      # mlp/~/linear_0.a
      self.assertEqual(
          sharding_metadata.NamedShardingMetadata.from_deserialized_dict(
              json.loads(data['bWxwL34vbGluZWFyXzAuYQ=='])
          ),
          sharding_metadata.NamedShardingMetadata.from_jax_sharding(
              pytree['mlp/~/linear_0']['a'].sharding
          ),
      )
      # mlp/~/linear_0.b
      self.assertEqual(
          sharding_metadata.NamedShardingMetadata.from_deserialized_dict(
              json.loads(data['bWxwL34vbGluZWFyXzAuYg=='])
          ),
          sharding_metadata.NamedShardingMetadata.from_jax_sharding(
              pytree['mlp/~/linear_0']['b'].sharding
          ),
      )
      # mlp/~/linear_0.c.a
      self.assertEqual(
          sharding_metadata.NamedShardingMetadata.from_deserialized_dict(
              json.loads(data['bWxwL34vbGluZWFyXzAuYy5h'])
          ),
          sharding_metadata.NamedShardingMetadata.from_jax_sharding(
              pytree['mlp/~/linear_0']['c']['a'].sharding
          ),
      )
      # mlp/~/linear_0.c.e
      self.assertEqual(
          sharding_metadata.NamedShardingMetadata.from_deserialized_dict(
              json.loads(data['bWxwL34vbGluZWFyXzAuYy5l'])
          ),
          sharding_metadata.NamedShardingMetadata.from_jax_sharding(
              pytree['mlp/~/linear_0']['c']['e'].sharding
          ),
      )

  @parameterized.product(
      use_ocdbt=(True, False),
      array_metadata_store=(None, ARRAY_METADATA_STORE),
  )
  def test_disable_write_sharding_file(
      self,
      use_ocdbt: bool,
      array_metadata_store: array_metadata_store_lib.Store | None,
  ):
    """Test case."""
    array_handler = type_handlers.ArrayHandler(
        enable_write_sharding_file=False,
        array_metadata_store=array_metadata_store,
    )
    ty = jax.Array
    fn = lambda ty: issubclass(ty, jax.Array)
    with test_utils.register_type_handler(ty, array_handler, fn):
      pytree, save_args, restore_args = self.create_mixed_format_pytree()
      with self.ocdbt_checkpoint_handler(
          use_ocdbt=use_ocdbt, array_metadata_store=array_metadata_store
      ) as checkpoint_handler:
        checkpoint_handler.save(
            self.directory, args=PyTreeSaveArgs(pytree, save_args)
        )
        self.validate_save(
            self.directory,
            pytree,
            checkpoint_handler,
            save_args=save_args,
            restore_args=restore_args,
        )
    self.assertFalse((self.directory / _SHARDING).exists())

  def test_sharding_variable_devices(self):
    if utils.is_pathways_backend():
      self.skipTest('Sharding metadata not present on Pathways.')
    mesh_axes = jax.sharding.PartitionSpec(
        'x',
    )
    devices_subset = []
    for idx in range(jax.process_count()):
      for d in jax.devices():
        if d.process_index == idx:
          devices_subset.append(d)
          break
    pytree = {
        'a': test_utils.create_sharded_array(
            np.arange(16),
            jax.sharding.Mesh(devices_subset, ('x',)),
            mesh_axes,
        ),
        'b': test_utils.create_sharded_array(
            np.arange(16), jax.sharding.Mesh(jax.devices(), ('x',)), mesh_axes
        ),
    }

    self.handler.save(self.directory, args=PyTreeSaveArgs(pytree))
    self.assertTrue((self.directory / _SHARDING).exists())
    a_sharding_metadata = sharding_metadata.NamedShardingMetadata(
        shape=np.array([2]),
        axis_names=['x'],
        partition_spec=('x',),
        axis_types=(jax.sharding.AxisType.Auto,),
        device_mesh=sharding_metadata.DeviceMetadataMesh.from_jax_mesh(
            jax.sharding.Mesh(devices_subset, ('x',))
        ),
    )
    b_sharding_metadata = sharding_metadata.NamedShardingMetadata(
        shape=np.array([8]),
        axis_names=['x'],
        partition_spec=('x',),
        axis_types=(jax.sharding.AxisType.Auto,),
        device_mesh=sharding_metadata.DeviceMetadataMesh.from_jax_mesh(
            jax.sharding.Mesh(jax.devices(), ('x',))
        ),
    )
    self.assertEqual(
        a_sharding_metadata,
        self.handler.metadata(self.directory)['a'].sharding,
    )
    self.assertEqual(
        b_sharding_metadata,
        self.handler.metadata(self.directory)['b'].sharding,
    )

  @parameterized.product(use_ocdbt=(True, False))
  def test_save_main(self, use_ocdbt: bool):
    with self.ocdbt_checkpoint_handler(use_ocdbt) as checkpoint_handler:
      checkpoint_handler.save(self.directory, args=PyTreeSaveArgs(self.pytree))
      self.validate_save(
          self.directory,
          self.pytree,
          checkpoint_handler,
          restore_args=self.restore_args,
      )
      self.assertEqual(
          type_handlers.is_ocdbt_checkpoint(self.directory), use_ocdbt
      )

  @parameterized.product(use_ocdbt=(True, False))
  def test_save_keys_with_slashes(self, use_ocdbt: bool):
    with self.ocdbt_checkpoint_handler(use_ocdbt) as checkpoint_handler:
      pytree = {
          'a': np.arange(2),
          'b/c': np.arange(4),
      }
      checkpoint_handler.save(self.directory, args=PyTreeSaveArgs(pytree))
      self.validate_save(
          self.directory,
          pytree,
          checkpoint_handler,
      )

  def test_save_non_sharded(self):

    def _save_args(arr):
      del arr
      return SaveArgs()

    save_args = jax.tree.map(
        _save_args, self.numpy_pytree, is_leaf=test_utils.is_leaf
    )
    restore_args = jax.tree.map(
        lambda arr: RestoreArgs(restore_type=type(arr)),
        self.numpy_pytree,
        is_leaf=test_utils.is_leaf,
    )

    self.handler.save(
        self.directory, args=PyTreeSaveArgs(self.numpy_pytree, save_args)
    )
    self.validate_save(
        self.directory,
        self.numpy_pytree,
        self.handler,
        save_args=save_args,
        restore_args=restore_args,
    )

  @parameterized.product(
      use_ocdbt=(True, False),
      array_metadata_store=(None, ARRAY_METADATA_STORE),
  )
  def test_save_mixed(
      self,
      use_ocdbt: bool,
      array_metadata_store: array_metadata_store_lib.Store | None,
  ):
    """Test case."""
    with self.ocdbt_checkpoint_handler(
        use_ocdbt, array_metadata_store=array_metadata_store
    ) as checkpoint_handler:
      pytree, save_args, restore_args = self.create_mixed_format_pytree(
          strings=True
      )
      checkpoint_handler.save(
          self.directory, args=PyTreeSaveArgs(pytree, save_args)
      )
      self.validate_save(
          self.directory,
          pytree,
          checkpoint_handler,
          save_args=save_args,
          restore_args=restore_args,
      )
      if use_ocdbt:
        self.assertContainsSubset(
            [
                '_strings.json',
                'ocdbt.process_0',
                'ocdbt.process_1',
            ],
            [f.name for f in self.directory.iterdir()],
        )
      else:
        self.assertIn(
            '_strings.json',
            [f.name for f in self.directory.iterdir()],
        )
      if self.should_validate_metadata():
        self.validate_metadata(
            expected_reference_metadata_tree=pytree,
            actual_metadata=checkpoint_handler.metadata(self.directory),
            pytree_metadata_options=self.pytree_metadata_options,
            save_args=save_args,
            array_metadata_store=array_metadata_store,
        )

  @parameterized.product(
      use_ocdbt=(True, False),
      array_metadata_store=(None, ARRAY_METADATA_STORE),
  )
  def test_save_strings(
      self,
      use_ocdbt: bool,
      array_metadata_store: array_metadata_store_lib.Store | None,
  ):
    """Test case."""
    if use_ocdbt and utils.is_pathways_backend():
      self.skipTest('Pathways + OCDBT not supported.')

    with self.ocdbt_checkpoint_handler(
        use_ocdbt, array_metadata_store=array_metadata_store
    ) as checkpoint_handler:
      pytree, _, restore_args = self.create_mixed_format_pytree(strings=True)

      save_args = jax.tree.map(
          lambda _: SaveArgs(), pytree, is_leaf=test_utils.is_leaf
      )
      checkpoint_handler.save(
          self.directory, args=PyTreeSaveArgs(pytree, save_args)
      )
      self.validate_save(
          self.directory,
          pytree,
          checkpoint_handler,
          save_args=save_args,
          restore_args=restore_args,
      )
      if self.should_validate_metadata():
        self.validate_metadata(
            expected_reference_metadata_tree=pytree,
            actual_metadata=checkpoint_handler.metadata(self.directory),
            pytree_metadata_options=self.pytree_metadata_options,
            save_args=save_args,
            array_metadata_store=array_metadata_store,
        )
      self.assertTrue((self.directory / '_strings.json').exists())
      with open(self.directory / '_strings.json') as file:
        data = json.load(file)
        self.assertCountEqual(
            data.keys(),
            {'foo', 'bar'},
            None,
        )
        self.assertEqual(data['foo'], 'foo_val')
        self.assertEqual(data['bar'], 'bar_val')

  def test_cast(self):
    pytree, save_args, restore_args = self.create_mixed_format_pytree()

    def set_dtype(args, dtype):
      args.dtype = dtype
      return args

    save_args = jax.tree.map(
        functools.partial(set_dtype, dtype=np.int16), save_args
    )
    self.handler.save(self.directory, args=PyTreeSaveArgs(pytree, save_args))

    def check_dtype(x, dtype):
      if not utils.is_scalar(x):
        self.assertEqual(x.dtype, dtype)

    # Restore without casting.
    restored = self.handler.restore(
        self.directory, args=PyTreeRestoreArgs(restore_args=restore_args)
    )
    jax.tree.map(lambda x: check_dtype(x, np.int16), restored)

    restore_args = jax.tree.map(
        functools.partial(set_dtype, dtype=np.uint32), restore_args
    )
    restored = self.handler.restore(
        self.directory, args=PyTreeRestoreArgs(restore_args=restore_args)
    )
    jax.tree.map(lambda x: check_dtype(x, np.uint32), restored)

  def test_cast_scalar(self):
    pytree = {'a': 5, 'b': 1.2}
    restore_args = {
        'a': RestoreArgs(
            restore_type=float
        ),  # pytype: disable=wrong-arg-types  # jnp-type
        'b': RestoreArgs(
            restore_type=int
        ),  # pytype: disable=wrong-arg-types  # jnp-type
    }

    self.handler.save(self.directory, args=PyTreeSaveArgs(pytree))
    restored = self.handler.restore(
        self.directory, args=PyTreeRestoreArgs(restore_args=restore_args)
    )
    self.assertIsInstance(restored['a'], float)
    self.assertIsInstance(restored['b'], int)

  def test_restore_type(self):
    pytree = {'a': 5, 'b': 6.1}
    restore_args = {
        'a': RestoreArgs(restore_type=np.ndarray),
        'b': RestoreArgs(restore_type=np.ndarray),
    }

    self.handler.save(self.directory, args=PyTreeSaveArgs(pytree))
    restored = self.handler.restore(
        self.directory, args=PyTreeRestoreArgs(restore_args=restore_args)
    )
    self.assertIsInstance(restored['a'], np.ndarray)
    self.assertIsInstance(restored['b'], np.ndarray)

  @parameterized.product(
      use_ocdbt=(True, False),
      use_zarr3=(True, False),
      array_metadata_store=(None, ARRAY_METADATA_STORE),
  )
  def test_restore(
      self,
      use_ocdbt: bool,
      use_zarr3: bool,
      array_metadata_store: array_metadata_store_lib.Store | None,
  ):
    """Test case."""
    with self.ocdbt_checkpoint_handler(
        use_ocdbt,
        use_zarr3=use_zarr3,
        array_metadata_store=array_metadata_store,
    ) as checkpoint_handler:
      checkpoint_handler.save(self.directory, args=PyTreeSaveArgs(self.pytree))
      restored = checkpoint_handler.restore(
          self.directory,
          args=PyTreeRestoreArgs(restore_args=self.restore_args),
      )
      self.validate_restore(self.pytree, restored)
      if self.should_validate_metadata():
        self.validate_metadata(
            expected_reference_metadata_tree=self.pytree,
            actual_metadata=checkpoint_handler.metadata(self.directory),
            pytree_metadata_options=self.pytree_metadata_options,
            array_metadata_store=array_metadata_store,
        )

  @parameterized.product(
      use_ocdbt=(True, False),
      use_zarr3=(False, True),
      array_metadata_store=(None, ARRAY_METADATA_STORE),
      use_compression=(True, False),
  )
  def test_compression(
      self,
      use_ocdbt: bool,
      use_zarr3: bool,
      array_metadata_store: array_metadata_store_lib.Store | None,
      use_compression: bool,
  ):
    """Test case for zarr2 compression."""
    with self.ocdbt_checkpoint_handler(
        use_ocdbt,
        use_zarr3=use_zarr3,
        array_metadata_store=array_metadata_store,
        use_compression=use_compression,
    ) as checkpoint_handler:
      checkpoint_handler.save(self.directory, args=PyTreeSaveArgs(self.pytree))
      restored = checkpoint_handler.restore(
          self.directory,
          args=PyTreeRestoreArgs(restore_args=self.restore_args),
      )
      self.validate_restore(self.pytree, restored)
      if self.should_validate_metadata():
        self.validate_metadata(
            expected_reference_metadata_tree=self.pytree,
            actual_metadata=checkpoint_handler.metadata(self.directory),
            pytree_metadata_options=self.pytree_metadata_options,
            array_metadata_store=array_metadata_store,
        )

      self.assertEqual(
          test_utils.is_compression_used(
              self.directory,
              'a',
              use_zarr3,
              use_ocdbt,
          ),
          use_compression,
      )

  @parameterized.product(use_ocdbt=(True, False))
  def test_restore_reverse_mesh(self, use_ocdbt: bool):
    if use_ocdbt and utils.is_pathways_backend():
      self.skipTest('Pathways + OCDBT not supported.')
    with self.ocdbt_checkpoint_handler(use_ocdbt) as checkpoint_handler:
      pytree, mesh_tree, axes_tree = test_utils.setup_sharded_pytree(
          reverse_devices=True
      )

      def _create_restore_args(arr, mesh, axes):
        return ArrayRestoreArgs(
            restore_type=type(arr), mesh=mesh, mesh_axes=axes
        )

      restore_args = jax.tree.map(
          _create_restore_args, pytree, mesh_tree, axes_tree
      )

      checkpoint_handler.save(self.directory, args=PyTreeSaveArgs(pytree))
      restored = checkpoint_handler.restore(
          self.directory, args=PyTreeRestoreArgs(restore_args=restore_args)
      )
      self.validate_restore(pytree, restored)

  def test_restore_with_sharding(self):
    if utils.is_pathways_backend():
      self.skipTest('Sharding metadata not present on Pathways.')

    jitted_pytree = jax.tree.map(
        jax.experimental.pjit.pjit(lambda x: x * 2), self.pytree
    )
    self.handler.save(self.directory, args=PyTreeSaveArgs(jitted_pytree))

    restore_args = jax.tree.map(
        lambda arr: ArrayRestoreArgs(sharding=arr.sharding), jitted_pytree
    )
    restored = self.handler.restore(
        self.directory, args=PyTreeRestoreArgs(restore_args=restore_args)
    )
    self.validate_restore(jitted_pytree, restored)

  def test_restore_with_sharding_metadata(self):
    if utils.is_pathways_backend():
      self.skipTest('Sharding metadata not present on Pathways.')

    jitted_pytree = jax.tree.map(
        jax.experimental.pjit.pjit(lambda x: x * 2), self.pytree
    )
    self.handler.save(self.directory, args=PyTreeSaveArgs(jitted_pytree))

    restore_args = jax.tree.map(
        lambda arr: ArrayRestoreArgs(
            sharding=sharding_metadata.from_jax_sharding(arr.sharding)
        ),
        jitted_pytree,
    )
    restored = self.handler.restore(
        self.directory, args=PyTreeRestoreArgs(restore_args=restore_args)
    )
    self.validate_restore(jitted_pytree, restored)

  def test_restore_with_sharding_without_sharding_arg(self):
    if utils.is_pathways_backend():
      self.skipTest('Sharding metadata not present on Pathways.')

    self.handler.save(self.directory, args=PyTreeSaveArgs(self.pytree))

    restore_args = jax.tree.map(lambda arr: ArrayRestoreArgs(), self.pytree)

    self.assertTrue((self.directory / _SHARDING).exists())
    restored_without_sharding_args = self.handler.restore(
        self.directory, args=PyTreeRestoreArgs(restore_args=restore_args)
    )
    self.validate_restore(self.pytree, restored_without_sharding_args)

    restored_without_restore_args = self.handler.restore(self.directory)
    self.validate_restore(self.pytree, restored_without_restore_args)

  def test_restore_different(self):
    for step in [0, 1]:
      directory = self.directory / str(step)
      if multihost.process_index() == 0:
        directory.mkdir()
      test_utils.sync_global_processes(
          'PyTreeCheckpointHandlerTest:test_restore_different_mkdir'
      )

      pytree, save_args, restore_args = self.create_mixed_format_pytree(
          add=step
      )
      self.handler.save(directory, args=PyTreeSaveArgs(pytree, save_args))

      restored = self.handler.restore(
          directory, args=PyTreeRestoreArgs(restore_args=restore_args)
      )
      self.validate_restore(pytree, restored)

  def test_restore_missing_checkpoint(self):
    directory = self.directory / 'nothing'
    with self.assertRaises(FileNotFoundError):
      self.handler.restore(directory)

  @parameterized.product(
      use_ocdbt=(True, False),
      array_metadata_store=(None, ARRAY_METADATA_STORE),
  )
  def test_flax_model(
      self,
      use_ocdbt: bool,
      array_metadata_store: array_metadata_store_lib.Store | None,
  ):
    """Test case."""

    @flax.struct.dataclass
    class Params(flax.struct.PyTreeNode):
      params: Any
      opt_state: Any

    @jax.jit
    def make_params():
      return Params(
          params=self.numpy_pytree,
          opt_state=(optax.EmptyState(), optax.EmptyState()),
      )

    params = make_params()
    empty_params = jax.eval_shape(make_params)
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ('devices',))
    mesh_axes = jax.sharding.PartitionSpec()
    params = jax.tree.map(
        lambda arr: test_utils.create_sharded_array(arr, mesh, mesh_axes),
        params,
    )
    restore_args = jax.tree.map(
        lambda _: ArrayRestoreArgs(mesh=mesh, mesh_axes=mesh_axes), params
    )

    save_args = jax.tree.map(lambda _: SaveArgs(), params)
    with self.ocdbt_checkpoint_handler(
        use_ocdbt, array_metadata_store=array_metadata_store
    ) as checkpoint_handler:
      checkpoint_handler.save(
          self.directory, args=PyTreeSaveArgs(params, save_args)
      )
      restored = checkpoint_handler.restore(
          self.directory,
          args=PyTreeRestoreArgs(item=empty_params, restore_args=restore_args),
      )
      self.validate_restore(params, restored)
      if self.should_validate_metadata():
        self.validate_metadata(
            expected_reference_metadata_tree=params,
            actual_metadata=checkpoint_handler.metadata(self.directory),
            pytree_metadata_options=self.pytree_metadata_options,
            save_args=save_args,
            array_metadata_store=array_metadata_store,
        )

  @parameterized.product(
      use_ocdbt=(
          True,
          False,
      ),
      data=(
          {},
          {'a': {}, 'b': 3},
          [1, {}, 2],
          None,
          {'a': None, 'b': 3},
          [1, None, 2],
          [],
          [1, [], 2],
          {'a': [], 'b': 3},
      ),
      save_args=(
          None,
          SaveArgs(),
      ),
      array_metadata_store=(None, ARRAY_METADATA_STORE),
  )
  def test_empty_data(
      self,
      use_ocdbt: bool,
      data: Any,
      save_args: SaveArgs | None,
      array_metadata_store: array_metadata_store_lib.Store | None,
  ):
    """Test case."""
    if save_args is None:
      save_args_tree = None
    else:
      save_args_tree = jax.tree.map(
          lambda _: save_args, data, is_leaf=tree_utils.is_empty_or_leaf
      )
    with self.ocdbt_checkpoint_handler(
        use_ocdbt, array_metadata_store=array_metadata_store
    ) as checkpoint_handler:
      if not data:
        with self.assertRaisesRegex(ValueError, 'Found empty item'):
          checkpoint_handler.save(
              self.directory,
              args=PyTreeSaveArgs(data, save_args=save_args_tree),
          )
        return

      checkpoint_handler.save(
          self.directory, args=PyTreeSaveArgs(data, save_args=save_args_tree)
      )
      restored = checkpoint_handler.restore(self.directory)
      self.assertEqual(restored, data)

      self.validate_metadata(
          expected_reference_metadata_tree=data,
          actual_metadata=checkpoint_handler.metadata(self.directory),
          pytree_metadata_options=self.pytree_metadata_options,
          save_args=save_args_tree,
          array_metadata_store=array_metadata_store,
      )

  @parameterized.product(
      use_ocdbt=(True, False),
      array_metadata_store=(None, ARRAY_METADATA_STORE),
  )
  def test_list(
      self,
      use_ocdbt: bool,
      array_metadata_store: array_metadata_store_lib.Store | None,
  ):
    """Test case."""
    item = [1, 2, 5, 6]
    with self.ocdbt_checkpoint_handler(
        use_ocdbt=use_ocdbt, array_metadata_store=array_metadata_store
    ) as checkpoint_handler:
      checkpoint_handler.save(self.directory, args=PyTreeSaveArgs(item))
      restore_args = jax.tree.map(lambda _: RestoreArgs(restore_type=int), item)
      restored = checkpoint_handler.restore(
          self.directory,
          args=PyTreeRestoreArgs(item=[0, 0, 0, 0], restore_args=restore_args),
      )
      self.assertListEqual(restored, item)
      self.validate_metadata(
          expected_reference_metadata_tree=[0, 0, 0, 0],
          actual_metadata=checkpoint_handler.metadata(self.directory),
          pytree_metadata_options=self.pytree_metadata_options,
          array_metadata_store=array_metadata_store,
      )

      restored = checkpoint_handler.restore(self.directory)
      self.assertListEqual(
          restored,
          [
              np.asarray([1]),
              np.asarray([2]),
              np.asarray([5]),
              np.asarray([6]),
          ],
      )

  def test_only_aggregation(self):
    tree = {
        'a': 1,
        'b': 2,
        'c': {
            'd': np.arange(3),
        },
    }
    msgpack = msgpack_utils.msgpack_serialize(tree)
    if multihost.process_index() == 0:
      (self.directory / 'checkpoint').write_bytes(msgpack)
    test_utils.sync_global_processes(
        'PyTreeCheckpointHandlerTest:write_flax_checkpoint'
    )

    restore_args = jax.tree.map(
        lambda arr: RestoreArgs(restore_type=type(arr)),
        tree,
        is_leaf=test_utils.is_leaf,
    )
    restored = self.handler.restore(
        self.directory, args=PyTreeRestoreArgs(restore_args=restore_args)
    )
    self.validate_restore(tree, restored)

  def test_transform(self):
    pytree = self.pytree
    pytree['int_key'] = 5
    pytree['float_key'] = 2.5
    self.handler.save(self.directory, args=PyTreeSaveArgs(pytree))

    def _pjitted_value_fn(x):
      return test_utils.apply_function([x], lambda y: y * 2 + 3)[0]

    replicated_sharding = jax.sharding.NamedSharding(
        jax.sharding.Mesh(jax.devices(), ('x',)),
        jax.sharding.PartitionSpec(
            None,
        ),
    )

    def _pjitted_add(key, tree, args):
      del key
      del args
      return pjit.pjit(lambda a, b: a + b, out_shardings=replicated_sharding)(
          tree['a'], tree['b']
      )

    def _split(key, tree, args):
      if key == 'split1':
        result = np.asarray(tree['float_key'] * 2)
        result = jax.make_array_from_callback(
            result.shape, args.sharding, lambda idx: result[idx]
        )
      else:
        self.assertEqual(args.restore_type, np.ndarray)
        result = np.asarray(tree['float_key'] * 4)
      return result

    reference_item = {
        'x': 0,
        'y': 0,
        'c': {
            'a': 0,
        },
        'z': 100,  # All values in this tree are unused except 'z'.
        'int_key': 0,
        'added': 0,
        'split1': 0,
        'split2': 0,
        'fallback': np.arange(4),
    }
    restore_args = {
        'x': self.restore_args['a'],
        'y': self.restore_args['c']['e'],
        'c': {
            'a': self.restore_args['c']['a'],
        },
        'z': RestoreArgs(restore_type=int),
        'int_key': RestoreArgs(restore_type=int),
        'split1': ArrayRestoreArgs(
            sharding=jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ('x',)),
                jax.sharding.PartitionSpec(None),
            )
        ),
        'split2': RestoreArgs(restore_type=np.ndarray),
        'added': RestoreArgs(restore_type=None),
        'fallback': RestoreArgs(restore_type=None),
    }
    expected = {
        'x': _pjitted_value_fn(pytree['a']),
        'y': pytree['c']['e'],
        'c': {
            'a': _pjitted_value_fn(pytree['c']['a']),
        },
        'z': 100,
        'int_key': 7,
        'added': test_utils.create_sharded_array(
            np.arange(8) + np.arange(8) * 2,
            replicated_sharding.mesh,
            replicated_sharding.spec,
        ),
        'split1': jax.make_array_from_callback(
            (),
            restore_args['split1'].sharding,
            lambda idx: np.asarray(5.0)[idx],
        ),
        'split2': np.asarray(10.0),
        'fallback': np.arange(4),
    }

    transforms = {
        'x': Transform(original_key='a', value_fn=_pjitted_value_fn),
        'y': Transform(original_key='c/e'),
        'c': {'a': Transform(value_fn=_pjitted_value_fn)},
        'int_key': Transform(value_fn=lambda x: x + 2),
        'added': RestoreTransform(
            multi_value_fn=_pjitted_add,
            multi_value_fn_input_args={
                'a': ArrayRestoreArgs(
                    sharding=replicated_sharding, strict=False
                ),
                'b': ArrayRestoreArgs(
                    sharding=replicated_sharding,
                    global_shape=(8,),
                    strict=False,
                ),
            },
        ),
        'split1': RestoreTransform(
            multi_value_fn=_split,
            multi_value_fn_input_args={
                'float_key': RestoreArgs(restore_type=float)
            },
        ),
        'split2': RestoreTransform(
            multi_value_fn=_split,
            multi_value_fn_input_args={
                'float_key': RestoreArgs(restore_type=float)
            },
        ),
        'fallback': Transform(use_fallback=True),
    }

    restored = self.handler.restore(
        self.directory,
        args=PyTreeRestoreArgs(
            item=reference_item,
            restore_args=restore_args,
            transforms=transforms,
        ),
    )
    self.validate_restore(expected, restored)

  @parameterized.product(
      use_ocdbt=(True, False),
      array_metadata_store=(None, ARRAY_METADATA_STORE),
  )
  def test_partial_restore(
      self,
      use_ocdbt: bool,
      array_metadata_store: array_metadata_store_lib.Store | None,
  ):
    """Test case."""
    checkpoint_handler = self.enter_context(
        self.ocdbt_checkpoint_handler(
            use_ocdbt=use_ocdbt, array_metadata_store=array_metadata_store
        )
    )
    checkpoint_handler.save(self.directory, args=PyTreeSaveArgs(self.pytree))

    reference_item = {
        'a': 0,
        'c': {
            'a': 0,
        },
    }
    restore_args = {
        'a': self.restore_args['a'],
        'c': {
            'a': self.restore_args['c']['a'],
        },
    }
    expected = {
        'a': self.pytree['a'],
        'c': {
            'a': self.pytree['c']['a'],
        },
    }
    transforms = {}

    # Ensure that no more parameters are being restored than the ones that are
    # strictly needed.
    with mock.patch.object(
        type_handlers.ArrayHandler, 'deserialize', autospec=True
    ) as mock_deserialize:
      checkpoint_handler.restore(
          self.directory,
          args=PyTreeRestoreArgs(
              item=reference_item,
              restore_args=restore_args,
              transforms=transforms,
          ),
      )
      mock_deserialize.assert_called_once()
      mock_args, _ = mock_deserialize.call_args
      _, infos, args = mock_args
      self.assertLen(infos, 2)
      self.assertLen(args, 2)

    restored = checkpoint_handler.restore(
        self.directory,
        args=PyTreeRestoreArgs(
            item=reference_item,
            restore_args=restore_args,
            transforms=transforms,
        ),
    )
    self.validate_restore(expected, restored)
    self.validate_metadata(
        expected_reference_metadata_tree=self.pytree,
        actual_metadata=checkpoint_handler.metadata(self.directory),
        pytree_metadata_options=self.pytree_metadata_options,
        array_metadata_store=array_metadata_store,
    )


if __name__ == '__main__':
  multiprocess_test.main()
