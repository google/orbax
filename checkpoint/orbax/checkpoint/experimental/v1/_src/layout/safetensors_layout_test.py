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

import unittest
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import jax
import numpy as np
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout
from orbax.checkpoint.experimental.v1._src.layout import safetensors_layout
from orbax.checkpoint.experimental.v1._src.metadata import types as metadata_types
from orbax.checkpoint.experimental.v1._src.path import types
from orbax.checkpoint.experimental.v1._src.saving import saving
import safetensors.numpy

SafetensorsLayout = safetensors_layout.SafetensorsLayout
np_save_file = safetensors.numpy.save_file
InvalidLayoutError = checkpoint_layout.InvalidLayoutError


class SafetensorsLayoutTest(
    unittest.IsolatedAsyncioTestCase, parameterized.TestCase
):

  def setUp(self):
    super().setUp()
    self.test_dir = self.create_tempdir()
    self.orbax_path = epath.Path(self.test_dir.full_path) / 'test_checkpoint'
    self.safetensors_path = (
        epath.Path(self.test_dir.full_path) / 'test_checkpoint.safetensors'
    )

    # Create a mock SafeTensors and Orbax checkpoint
    self.object_to_save = {
        'a': np.array(3 * [1, 2, 3], dtype=np.int32),
        'b': np.array([0, 1, 0.2], dtype=np.float32),
    }
    self.custom_metadata = {'framework': 'JAX', 'version': '1.0'}
    np_save_file(
        self.object_to_save,
        self.safetensors_path,
        metadata=self.custom_metadata,
    )
    saving.save_pytree(self.orbax_path, self.object_to_save)

  async def test_valid_safetensors_checkpoint(self):
    layout = SafetensorsLayout()
    await layout.validate(self.safetensors_path)

  async def test_invalid_safetensors_checkpoint_orbax(self):
    layout = SafetensorsLayout()
    with self.assertRaises(InvalidLayoutError):
      await layout.validate(self.orbax_path / '0')

  async def test_validate_fails_wrong_suffix(self):
    wrong_suffix_path = (
        epath.Path(self.test_dir.full_path) / 'test_checkpoint.txt'
    )
    layout = SafetensorsLayout()
    with self.assertRaises(InvalidLayoutError):
      await layout.validate(wrong_suffix_path)

  @parameterized.product(
      dtype=[
          np.int8,
          np.int32,
          np.int64,
          np.float16,
          np.float32,
          np.float64,
          np.bool_,
          jax.numpy.bfloat16,
      ]
  )
  async def test_load_safetensors_checkpoint(self, dtype: np.dtype):
    """Tests loading a SafeTensors checkpoint with various dtypes."""
    test_path = (
        epath.Path(self.test_dir.full_path)
        / f'test_{dtype.__name__}.safetensors'
    )
    if dtype == np.bool_:
      arr = np.array([True, False, True, False])
    else:
      arr = np.arange(8, dtype=dtype)

    obj_to_save = {'x': arr, 'y': np.array([1, 2, 3], dtype=np.int32)}
    np_save_file(obj_to_save, test_path)

    # Load the checkpoint
    layout = SafetensorsLayout()
    restore_fn = await layout.load(test_path)
    restored_checkpointables = await restore_fn
    pytree = restored_checkpointables['pytree']

    # Verify restored data
    # TODO(b/430651483)
    if np.issubdtype(dtype, np.floating):
      np.testing.assert_allclose(pytree['x'], obj_to_save['x'], strict=True)
    else:
      np.testing.assert_array_equal(pytree['x'], obj_to_save['x'], strict=True)
    np.testing.assert_array_equal(pytree['y'], obj_to_save['y'])

  async def test_load_fails_with_incomplete_dtypes(self):
    incomplete_dtypes = {
        'F32': np.float32,
        'BOOL': np.bool_,
        # Intentionally missing I32: int32 for testing, which is used in the
        # test checkpoint.
    }
    layout = SafetensorsLayout()
    with self.assertRaises(ValueError):
      with mock.patch.object(
          safetensors_layout,
          '_get_dtypes',
          return_value=incomplete_dtypes,
          spec=True,
      ):
        awaitable_fn = await layout.load(self.safetensors_path)
        _ = await awaitable_fn

  async def test_metadata(self):
    layout = SafetensorsLayout()
    metadata = await layout.metadata(self.safetensors_path)
    self.assertIsInstance(metadata, metadata_types.CheckpointMetadata)
    self.assertEqual(
        metadata.metadata,
        {
            checkpoint_layout.PYTREE_CHECKPOINTABLE_KEY: {
                'b': jax.ShapeDtypeStruct(shape=(3,), dtype=np.float32),
                'a': jax.ShapeDtypeStruct(shape=(9,), dtype=np.int32),
            }
        },
    )
    self.assertEqual(metadata.custom_metadata, self.custom_metadata)
    self.assertIsInstance(metadata.commit_timestamp_nsecs, int)
    self.assertGreater(metadata.commit_timestamp_nsecs, 0)

  async def test_save_raises_not_implemented(self):
    layout = SafetensorsLayout()
    mock_path = mock.Mock(spec=types.PathAwaitingCreation)
    with self.assertRaises(NotImplementedError):
      await layout.save(mock_path, checkpointables={})


class SafetensorsLayoutDirectoryTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  def setUp(self):
    super().setUp()
    self.test_dir = self.create_tempdir()
    self.checkpoint_dir = epath.Path(self.test_dir.full_path) / 'checkpoint_dir'
    self.checkpoint_dir.mkdir()

    self.file1 = self.checkpoint_dir / 'part1.safetensors'
    self.file2 = self.checkpoint_dir / 'part2.safetensors'

    self.data1 = {
        'a': np.array([1, 2], dtype=np.int32),
        'c': np.array([5, 6], dtype=np.int32),
    }
    self.data2 = {
        'b': np.array([3.0, 4.0], dtype=np.float32),
        'd': np.array([7.0, 8.0], dtype=np.float32),
    }

    np_save_file(self.data1, self.file1)
    np_save_file(self.data2, self.file2)

  async def test_validate_directory(self):
    layout = SafetensorsLayout()
    await layout.validate(self.checkpoint_dir)

  async def test_validate_directory_fails_empty(self):
    empty_dir = epath.Path(self.test_dir.full_path) / 'empty'
    empty_dir.mkdir()
    layout = SafetensorsLayout()
    with self.assertRaisesRegex(InvalidLayoutError, 'does not contain any'):
      await layout.validate(empty_dir)

  async def test_load_directory(self):
    layout = SafetensorsLayout()
    restore_fn = await layout.load(self.checkpoint_dir)
    res = await restore_fn
    pytree = res[checkpoint_layout.PYTREE_CHECKPOINTABLE_KEY]
    np.testing.assert_array_equal(pytree['a'], self.data1['a'])
    np.testing.assert_array_equal(pytree['b'], self.data2['b'])
    np.testing.assert_array_equal(pytree['c'], self.data1['c'])
    np.testing.assert_array_equal(pytree['d'], self.data2['d'])

  async def test_metadata_directory(self):
    layout = SafetensorsLayout()
    metadata = await layout.metadata(self.checkpoint_dir)
    pytree_meta = metadata.metadata[checkpoint_layout.PYTREE_CHECKPOINTABLE_KEY]
    self.assertIn('a', pytree_meta)
    self.assertIn('b', pytree_meta)
    self.assertIn('c', pytree_meta)
    self.assertIn('d', pytree_meta)
    self.assertEqual(pytree_meta['a'].shape, (2,))
    self.assertEqual(pytree_meta['a'].dtype, np.int32)
    self.assertEqual(pytree_meta['b'].shape, (2,))
    self.assertEqual(pytree_meta['b'].dtype, np.float32)
    self.assertEqual(pytree_meta['c'].shape, (2,))
    self.assertEqual(pytree_meta['c'].dtype, np.int32)
    self.assertEqual(pytree_meta['d'].shape, (2,))
    self.assertEqual(pytree_meta['d'].dtype, np.float32)

  async def test_load_directory_abstract_tree_all_keys(self):
    layout = SafetensorsLayout()
    tree = {
        'a': jax.ShapeDtypeStruct(shape=(2,), dtype=np.int32),
        'b': jax.ShapeDtypeStruct(shape=(2,), dtype=np.float32),
        'c': jax.ShapeDtypeStruct(shape=(2,), dtype=np.int32),
        'd': jax.ShapeDtypeStruct(shape=(2,), dtype=np.float32),
    }
    restore_fn = await layout.load(
        self.checkpoint_dir,
        abstract_checkpointables={
            checkpoint_layout.PYTREE_CHECKPOINTABLE_KEY: tree
        },
    )
    res = await restore_fn
    pytree = res[checkpoint_layout.PYTREE_CHECKPOINTABLE_KEY]
    self.assertLen(pytree, 4)
    np.testing.assert_array_equal(pytree['a'], self.data1['a'])
    np.testing.assert_array_equal(pytree['b'], self.data2['b'])
    np.testing.assert_array_equal(pytree['c'], self.data1['c'])
    np.testing.assert_array_equal(pytree['d'], self.data2['d'])

  async def test_load_directory_abstract_tree_subset_one_file(self):
    layout = SafetensorsLayout()
    tree = {
        'a': jax.ShapeDtypeStruct(shape=(2,), dtype=np.int32),
        'c': jax.ShapeDtypeStruct(shape=(2,), dtype=np.int32),
    }
    restore_fn = await layout.load(
        self.checkpoint_dir,
        abstract_checkpointables={
            checkpoint_layout.PYTREE_CHECKPOINTABLE_KEY: tree
        },
    )
    res = await restore_fn
    pytree = res[checkpoint_layout.PYTREE_CHECKPOINTABLE_KEY]
    self.assertLen(pytree, 2)
    self.assertIn('a', pytree)
    self.assertIn('c', pytree)
    np.testing.assert_array_equal(pytree['a'], self.data1['a'])
    np.testing.assert_array_equal(pytree['c'], self.data1['c'])

  async def test_load_directory_abstract_tree_subset_many_files(self):
    layout = SafetensorsLayout()
    tree = {
        'a': jax.ShapeDtypeStruct(shape=(2,), dtype=np.int32),
        'b': jax.ShapeDtypeStruct(shape=(2,), dtype=np.float32),
    }
    restore_fn = await layout.load(
        self.checkpoint_dir,
        abstract_checkpointables={
            checkpoint_layout.PYTREE_CHECKPOINTABLE_KEY: tree
        },
    )
    res = await restore_fn
    pytree = res[checkpoint_layout.PYTREE_CHECKPOINTABLE_KEY]
    self.assertLen(pytree, 2)
    self.assertIn('a', pytree)
    self.assertIn('b', pytree)
    np.testing.assert_array_equal(pytree['a'], self.data1['a'])
    np.testing.assert_array_equal(pytree['b'], self.data2['b'])

  async def test_load_directory_abstract_tree_key_not_found(self):
    layout = SafetensorsLayout()
    tree = {
        'e': jax.ShapeDtypeStruct(shape=(2,), dtype=np.int32),
    }
    restore_fn = await layout.load(
        self.checkpoint_dir,
        abstract_checkpointables={
            checkpoint_layout.PYTREE_CHECKPOINTABLE_KEY: tree
        },
    )
    with self.assertRaisesRegex(KeyError, "Tensor 'e' not found"):
      await restore_fn


if __name__ == '__main__':
  absltest.main()
