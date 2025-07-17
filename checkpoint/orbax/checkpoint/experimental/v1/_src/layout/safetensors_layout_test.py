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
import numpy as np
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout
from orbax.checkpoint.experimental.v1._src.layout import safetensors_layout
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
    self.layout = SafetensorsLayout()

    # Create a mock SafeTensors and Orbax checkpoint
    self.object_to_save = {
        'a': np.array(3 * [1, 2, 3], dtype=np.int32),
        'b': np.array([0, 1, 0.2], dtype=np.float32),
    }
    np_save_file(self.object_to_save, self.safetensors_path)
    saving.save_pytree(self.orbax_path, self.object_to_save)

  def test_valid_safetensors_checkpoint(self):
    self.layout.validate(path=self.safetensors_path)

  def test_invalid_safetensors_checkpoint_orbax(self):
    with self.assertRaises(InvalidLayoutError):
      self.layout.validate(path=self.orbax_path / '0')

  def test_validate_fails_not_file(self):
    with self.assertRaises(InvalidLayoutError):
      self.layout.validate(path=epath.Path(self.test_dir.full_path))

  def test_validate_fails_wrong_suffix(self):
    wrong_suffix_path = (
        epath.Path(self.test_dir.full_path) / 'test_checkpoint.txt'
    )
    with self.assertRaises(InvalidLayoutError):
      self.layout.validate(path=wrong_suffix_path)

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
    restore_fn = await self.layout.load(directory=test_path)
    restored_checkpointables = await restore_fn

    # Verify restored data
    # TODO(b/430651483)
    if np.issubdtype(dtype, np.floating):
      np.testing.assert_allclose(
          restored_checkpointables['x'], obj_to_save['x']
      )
    else:
      np.testing.assert_array_equal(
          restored_checkpointables['x'], obj_to_save['x']
      )
    np.testing.assert_array_equal(
        restored_checkpointables['y'], obj_to_save['y']
    )

  async def test_load_fails_with_incomplete_dtypes(self):
    incomplete_dtypes = {
        'F32': np.float32,
        'BOOL': np.bool_,
        # Intentionally missing I32: int32 for testing, which is used in the
        # test checkpoint.
    }
    with self.assertRaises(ValueError):
      with mock.patch.object(
          safetensors_layout,
          '_get_dtypes',
          return_value=incomplete_dtypes,
          spec=True,
      ):
        awaitable_fn = await self.layout.load(directory=self.safetensors_path)
        _ = await awaitable_fn


if __name__ == '__main__':
  absltest.main()
