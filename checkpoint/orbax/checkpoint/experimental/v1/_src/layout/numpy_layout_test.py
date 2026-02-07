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

import unittest
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import jax
import numpy as np
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout
from orbax.checkpoint.experimental.v1._src.layout import numpy_layout
from orbax.checkpoint.experimental.v1._src.metadata import types as metadata_types
from orbax.checkpoint.experimental.v1._src.path import types


NumpyLayout = numpy_layout.NumpyLayout
InvalidLayoutError = checkpoint_layout.InvalidLayoutError


class NumpyLayoutTest(unittest.IsolatedAsyncioTestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = self.create_tempdir()
    self.numpy_path = (
        epath.Path(self.test_dir.full_path) / 'test_checkpoint.npz'
    )

    self.object_to_save = {
        'a': np.array(3 * [1, 2, 3], dtype=np.int32),
        'b': np.array([0, 1, 0.2], dtype=np.float32),
    }
    np.savez(self.numpy_path, **self.object_to_save)

  async def test_valid_numpy_checkpoint(self):
    layout = NumpyLayout()
    await layout.validate(self.numpy_path)

  async def test_validate_fails_not_file(self):
    layout = NumpyLayout()
    with self.assertRaisesRegex(InvalidLayoutError, 'Path is not a file'):
      await layout.validate(epath.Path(self.test_dir.full_path))

  async def test_validate_fails_wrong_suffix(self):
    wrong_suffix_path = (
        epath.Path(self.test_dir.full_path) / 'test_checkpoint.txt'
    )
    wrong_suffix_path.touch()
    layout = NumpyLayout()
    with self.assertRaisesRegex(InvalidLayoutError, 'must have a .npz suffix'):
      await layout.validate(wrong_suffix_path)

  async def test_validate_fails_not_zip(self):
    bad_zip_path = epath.Path(self.test_dir.full_path) / 'bad_zip.npz'
    bad_zip_path.write_text('this is not a zip file')
    layout = NumpyLayout()
    with self.assertRaisesRegex(InvalidLayoutError, 'is not a valid ZIP file'):
      await layout.validate(bad_zip_path)

  @parameterized.product(
      dtype=[
          np.int8,
          np.int32,
          np.int64,
          np.float16,
          np.float32,
          np.float64,
          np.bool_,
      ]
  )
  async def test_load_numpy_checkpoint(self, dtype: np.dtype):
    """Tests loading a NumPy checkpoint with various dtypes."""
    test_path = (
        epath.Path(self.test_dir.full_path) / f'test_{dtype.__name__}.npz'
    )
    if dtype == np.bool_:
      arr = np.array([True, False, True, False])
    else:
      arr = np.arange(8, dtype=dtype)

    obj_to_save = {'x': arr, 'y': np.array([1, 2, 3], dtype=np.int32)}
    np.savez(test_path, **obj_to_save)

    # Load the checkpoint
    layout = NumpyLayout()
    restore_fn = await layout.load(test_path)
    restored_checkpointables = await restore_fn
    pytree = restored_checkpointables['pytree']

    # Verify restored data
    if np.issubdtype(dtype, np.floating):
      np.testing.assert_allclose(pytree['x'], obj_to_save['x'])
    else:
      np.testing.assert_array_equal(pytree['x'], obj_to_save['x'])
    np.testing.assert_array_equal(pytree['y'], obj_to_save['y'])

  async def test_metadata(self):
    layout = NumpyLayout()
    metadata = await layout.metadata(self.numpy_path)
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
    self.assertIsInstance(metadata.commit_timestamp_nsecs, int)
    self.assertGreater(metadata.commit_timestamp_nsecs, 0)

  async def test_save_raises_not_implemented(self):
    layout = NumpyLayout()
    mock_path = mock.Mock(spec=types.PathAwaitingCreation)
    with self.assertRaises(NotImplementedError):
      await layout.save(mock_path, checkpointables={})


if __name__ == '__main__':
  absltest.main()
