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

import dataclasses

from absl.testing import absltest
from etils import epath
import jax
import numpy as np
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.metadata import value as value_metadata
import orbax.checkpoint.experimental.v1 as ocp
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.context import options as options_lib
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout
from orbax.checkpoint.experimental.v1._src.metadata import types as metadata_types
from orbax.checkpoint.experimental.v1._src.serialization import numpy_leaf_handler
from orbax.checkpoint.experimental.v1._src.testing import array_utils as array_test_utils
from orbax.checkpoint.experimental.v1._src.testing import handler_utils
import safetensors.numpy


Foo = handler_utils.Foo
Bar = handler_utils.Bar
AbstractFoo = handler_utils.AbstractFoo
AbstractBar = handler_utils.AbstractBar
InvalidLayoutError = ocp.errors.InvalidLayoutError
PYTREE_CHECKPOINTABLE_KEY = checkpoint_layout.PYTREE_CHECKPOINTABLE_KEY


class PyTreeMetadataTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(self.create_tempdir().full_path) / 'ckpt'
    self.pytree, self.abstract_pytree = array_test_utils.create_numpy_pytree()
    ocp.save_pytree(self.directory, self.pytree)

  def _create_value_metadata(self, value):
    if isinstance(value, np.ndarray):
      storage_metadata = value_metadata.StorageMetadata(
          chunk_shape=value.shape,
          write_shape=None,
      )
      return numpy_leaf_handler.NumpyMetadata(
          shape=value.shape,
          dtype=value.dtype,
          storage_metadata=storage_metadata,
      )
    elif isinstance(value, (int, np.integer)):
      return 0
    elif isinstance(value, (float, np.floating)):
      return 0.0
    else:
      raise TypeError(f'Unsupported type: {type(value)}')

  def test_invalid_path(self):
    with self.assertRaises(InvalidLayoutError):
      ocp.pytree_metadata(self.directory.parent)
    with self.assertRaises(InvalidLayoutError):
      ocp.pytree_metadata(self.directory.parent / 'foo')

  def test_pytree_metadata_default_checkpointable_name(self):
    expected_pytree_metadata = jax.tree.map(
        self._create_value_metadata, self.pytree
    )

    metadata = ocp.pytree_metadata(self.directory)
    self.assertIsInstance(metadata, metadata_types.CheckpointMetadata)
    self.assertEqual(expected_pytree_metadata, metadata.metadata)

  def test_pytree_metadata_custom_checkpointable_name(self):
    self.directory.rmtree()

    custom_name = 'custom_pytree'
    ocp.save_checkpointables(self.directory, {custom_name: self.pytree})

    expected_pytree_metadata = jax.tree.map(
        self._create_value_metadata, self.pytree
    )

    metadata = ocp.pytree_metadata(
        self.directory, checkpointable_name=custom_name
    )
    self.assertIsInstance(metadata, metadata_types.CheckpointMetadata)
    self.assertEqual(expected_pytree_metadata, metadata.metadata)

  def test_pytree_metadata_checkpointable_name_none(self):
    expected_pytree_metadata = jax.tree.map(
        self._create_value_metadata, self.pytree
    )

    pytree_dir = self.directory / PYTREE_CHECKPOINTABLE_KEY
    metadata = ocp.pytree_metadata(pytree_dir, checkpointable_name=None)
    self.assertIsInstance(metadata, metadata_types.CheckpointMetadata)
    self.assertEqual(expected_pytree_metadata, metadata.metadata)

  def test_load_with_metadata(self):
    metadata = ocp.pytree_metadata(self.directory)

    def _set_numpy_cast_type(x):
      if isinstance(x, np.ndarray):
        return x.astype(np.int16)
      elif isinstance(x, numpy_leaf_handler.NumpyMetadata) and not isinstance(
          x, type
      ):
        return dataclasses.replace(x, dtype=np.dtype('int16'))
      elif x == np.int64:
        return int
      elif x == np.float64:
        return float
      else:
        return x

    metadata = metadata_types.CheckpointMetadata[
        metadata_types.PyTreeMetadata
    ].from_metadata(jax.tree.map(_set_numpy_cast_type, metadata.metadata))
    expected_pytree = jax.tree.map(_set_numpy_cast_type, self.pytree)

    with self.subTest('pytree_metadata'):
      loaded_pytree = ocp.load_pytree(self.directory, metadata.metadata)

      test_utils.assert_tree_equal(self, expected_pytree, loaded_pytree)
    with self.subTest('full_metadata'):
      loaded_pytree = ocp.load_pytree(self.directory, metadata)
      test_utils.assert_tree_equal(self, expected_pytree, loaded_pytree)

  def test_pytree_metadata_safetensors(self):
    st_path = epath.Path(self.create_tempdir().full_path) / 'model.safetensors'
    tensor_data = {
        'x': np.array([[1.0, 2.0]], dtype=np.float32),
        'y': np.array([1, 2, 3], dtype=np.int64),
    }
    st_custom_meta = {'framework': 'test', 'version': '1.0'}
    safetensors.numpy.save_file(tensor_data, st_path, metadata=st_custom_meta)

    expected_metadata_tree = {
        'x': jax.ShapeDtypeStruct(shape=(1, 2), dtype=np.float32),
        'y': jax.ShapeDtypeStruct(shape=(3,), dtype=np.int64),
    }

    with context_lib.Context(
        checkpoint_layout=options_lib.CheckpointLayout.SAFETENSORS
    ):
      ckpt_metadata = ocp.pytree_metadata(st_path)

    self.assertIsInstance(ckpt_metadata, metadata_types.CheckpointMetadata)
    self.assertEqual(
        ckpt_metadata.metadata.keys(), expected_metadata_tree.keys()
    )
    for key, expected_sds in expected_metadata_tree.items():
      actual_sds = ckpt_metadata.metadata[key]
      self.assertEqual(actual_sds.shape, expected_sds.shape)
      self.assertEqual(actual_sds.dtype, expected_sds.dtype)
    self.assertIsNone(ckpt_metadata.init_timestamp_nsecs)
    self.assertEqual(ckpt_metadata.custom_metadata, st_custom_meta)
    self.assertIsNotNone(ckpt_metadata.commit_timestamp_nsecs)

    # Test invalid path
    with self.assertRaises(ocp.errors.InvalidLayoutError):
      with context_lib.Context(
          checkpoint_layout=options_lib.CheckpointLayout.SAFETENSORS
      ):
        ocp.pytree_metadata(self.directory)


  def test_pytree_metadata_with_incompatible_item(self):
    self.directory.rmtree()
    # Save a valid PyTree to 'state'
    ocp.save_checkpointables(self.directory, {'state': self.pytree})

    # Create dummy files in datasets to simulate a non-pytree item
    (self.directory / 'datasets').mkdir()
    (self.directory / 'datasets' / 'data.txt').write_text('some data')

    metadata = ocp.pytree_metadata(self.directory, checkpointable_name='state')
    self.assertIsInstance(metadata, metadata_types.CheckpointMetadata)
    self.assertIsInstance(metadata.metadata, dict)
    self.assertSetEqual(
        {'a', 'b', 'c', 'x', 'y'}, set(metadata.metadata.keys())
    )


class CheckpointablesMetadataTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(self.create_tempdir().full_path) / 'ckpt'
    checkpointables_options = (
        options_lib.CheckpointablesOptions.create_with_handlers(
            handler_utils.FooHandler,
            handler_utils.BarHandler,
        )
    )
    self.enter_context(
        context_lib.Context(checkpointables_options=checkpointables_options)
    )
    checkpointables = {
        'foo': Foo(1, 'foo'),
        'bar': Bar(2, 'bar'),
    }
    ocp.save_checkpointables(self.directory, checkpointables)

  def test_checkpointables_metadata(self):
    metadata = ocp.checkpointables_metadata(self.directory)
    self.assertIsInstance(metadata, metadata_types.CheckpointMetadata)
    self.assertIsInstance(metadata.metadata, dict)
    self.assertIsInstance(metadata.metadata['foo'], AbstractFoo)
    self.assertIsInstance(metadata.metadata['bar'], AbstractBar)

  def test_load_with_metadata(self):
    path1 = epath.Path(self.create_tempdir().full_path) / 'ckpt1'
    path2 = epath.Path(self.create_tempdir().full_path) / 'ckpt2'

    ocp.save_checkpointables(path1, {'foo': Foo(1, 'foo')})
    ocp.save_checkpointables(path2, {'bar': Bar(3, 'bar')})

    metadata = ocp.checkpointables_metadata(path1)
    self.assertSameElements(metadata.metadata.keys(), ['foo'])
    loaded = ocp.load_checkpointables(
        path2,
        metadata_types.CheckpointMetadata[dict[str, AbstractFoo]].from_metadata(
            {'bar': AbstractFoo()}
        ),
    )
    self.assertSameElements(loaded.keys(), ['bar'])
    self.assertEqual(Foo(3, 'bar'), loaded['bar'])

  def test_checkpointables_metadata_safetensors(self):
    st_path = epath.Path(self.create_tempdir().full_path) / 'model.safetensors'
    tensor_data = {
        'item1': np.array([1.0], dtype=np.float32),
        'item2': np.array([1], dtype=np.int32),
    }
    st_custom_meta = {'framework': 'test', 'version': '1.0'}
    safetensors.numpy.save_file(tensor_data, st_path, metadata=st_custom_meta)

    expected_st_metadata = {
        'item1': jax.ShapeDtypeStruct(shape=(1,), dtype=np.float32),
        'item2': jax.ShapeDtypeStruct(shape=(1,), dtype=np.int32),
    }

    with context_lib.Context(
        checkpoint_layout=options_lib.CheckpointLayout.SAFETENSORS
    ):
      ckpt_metadata = ocp.checkpointables_metadata(st_path)

    self.assertIsInstance(ckpt_metadata, metadata_types.CheckpointMetadata)
    self.assertIn(PYTREE_CHECKPOINTABLE_KEY, ckpt_metadata.metadata)

    st_pytree_metadata = ckpt_metadata.metadata[PYTREE_CHECKPOINTABLE_KEY]
    self.assertEqual(st_pytree_metadata.keys(), expected_st_metadata.keys())
    for key, expected_sds in expected_st_metadata.items():
      actual_sds = st_pytree_metadata[key]
      self.assertEqual(actual_sds.shape, expected_sds.shape)
      self.assertEqual(actual_sds.dtype, expected_sds.dtype)

    self.assertIsNone(ckpt_metadata.init_timestamp_nsecs)
    self.assertEqual(ckpt_metadata.custom_metadata, st_custom_meta)
    self.assertIsNotNone(ckpt_metadata.commit_timestamp_nsecs)

    # Test invalid path
    with self.assertRaises(ocp.errors.InvalidLayoutError):
      with context_lib.Context(
          checkpoint_layout=options_lib.CheckpointLayout.SAFETENSORS
      ):
        ocp.checkpointables_metadata(self.directory)


  def test_checkpointables_metadata_with_incompatible_item(self):
    self.directory.rmtree()
    # Save a valid PyTree to 'state'
    ocp.save_checkpointables(
        self.directory, {'state': {'a': 1, 'b': 2, 'c': {'d': 3}}}
    )

    # Create dummy files in datasets to simulate a non-pytree item
    (self.directory / 'datasets').mkdir()
    (self.directory / 'datasets' / 'data.txt').write_text('some data')

    metadata = ocp.checkpointables_metadata(self.directory)
    self.assertIsInstance(metadata, metadata_types.CheckpointMetadata)
    self.assertIsInstance(metadata.metadata, dict)
    self.assertSetEqual({'state', 'datasets'}, set(metadata.metadata.keys()))
    self.assertSetEqual({'a', 'b', 'c'}, set(metadata.metadata['state'].keys()))
    self.assertIsNone(metadata.metadata['datasets'])


if __name__ == '__main__':
  absltest.main()
