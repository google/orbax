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
from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import numpy as np
from orbax.checkpoint import args
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.metadata import value as value_metadata
from orbax.checkpoint._src.path import async_path
from orbax.checkpoint.checkpoint_manager import CheckpointManager
from orbax.checkpoint.experimental.v1._src.handlers import composite_handler
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout
from orbax.checkpoint.experimental.v1._src.layout import orbax_layout
from orbax.checkpoint.experimental.v1._src.metadata import types as metadata_types
from orbax.checkpoint.experimental.v1._src.saving import saving
from orbax.checkpoint.experimental.v1._src.serialization import numpy_leaf_handler
from orbax.checkpoint.experimental.v1._src.testing import array_utils as array_test_utils
import safetensors.numpy


np_save_file = safetensors.numpy.save_file
OrbaxLayout = orbax_layout.OrbaxLayout
InvalidLayoutError = orbax_layout.InvalidLayoutError


async def _unlink_indicator(path: epath.Path):
  await async_path.unlink(
      path / orbax_layout.ORBAX_CHECKPOINT_INDICATOR_FILE,
      missing_ok=True,
  )


async def _unlink_pytree_metadata(path: epath.Path):
  await async_path.unlink(path / '_METADATA', missing_ok=True)
  for subdir in await async_path.iterdir(path):
    if not await async_path.is_dir(subdir):
      continue
    await async_path.unlink(subdir / '_METADATA', missing_ok=True)


async def _unlink_checkpoint_metadata(path: epath.Path):
  await async_path.unlink(path / '_CHECKPOINT_METADATA', missing_ok=True)


class OrbaxLayoutTest(unittest.IsolatedAsyncioTestCase, parameterized.TestCase):

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
    np_save_file(self.object_to_save, self.safetensors_path)
    saving.save_pytree(
        self.orbax_path / '0',
        self.object_to_save,
        custom_metadata=self.custom_metadata,
    )

  async def test_valid_orbax_checkpoint(self):
    layout = OrbaxLayout()
    await layout.validate(self.orbax_path / '0')

  async def test_invalid_orbax_checkpoint(self):
    layout = OrbaxLayout()
    with self.assertRaises(InvalidLayoutError):
      await layout.validate(self.safetensors_path)

  async def test_validate_fails_not_directory(self):
    layout = OrbaxLayout()
    with self.assertRaises(InvalidLayoutError):
      await layout.validate(self.orbax_path / '1')

  async def test_validate_no_indicator_file(self):
    layout = OrbaxLayout()
    indicator_path = (
        self.orbax_path
        / '0'
        / composite_handler.ORBAX_CHECKPOINT_INDICATOR_FILE
    )
    indicator_path.rmtree()  # Remove the indicator file
    with self.assertRaises(InvalidLayoutError):
      await layout.validate(self.orbax_path / '0')

  async def test_validate_no_metadata_file(self):
    layout = OrbaxLayout()
    metadata_path = self.orbax_path / '0' / '_CHECKPOINT_METADATA'
    self.assertTrue(metadata_path.exists())
    metadata_path.rmtree()  # Remove the metadata file
    with self.assertRaises(InvalidLayoutError):
      await layout.validate(self.orbax_path / '0')

  async def test_validate_no_indicator_or_metadata_files(self):
    layout = OrbaxLayout()
    indicator_path = (
        self.orbax_path
        / '0'
        / composite_handler.ORBAX_CHECKPOINT_INDICATOR_FILE
    )
    indicator_path.rmtree()  # Remove the indicator file
    metadata_path = self.orbax_path / '0' / '_CHECKPOINT_METADATA'
    metadata_path.rmtree()
    pytree_metadata_path = self.orbax_path / '0' / 'pytree' / '_METADATA'
    pytree_metadata_path.rmtree()
    with self.assertRaises(InvalidLayoutError):
      await layout.validate(self.orbax_path / '0')

  async def test_validate_fails_tmp_directory(self):
    # This simulates a temporary directory created by Orbax (should fail)
    test_utils.save_fake_tmp_dir(self.orbax_path, 0, 'test_checkpoint.tmp')
    path = epath.Path(self.test_dir.full_path) / 'test_checkpoint.tmp'
    layout = OrbaxLayout()
    with self.assertRaises(InvalidLayoutError):
      await layout.validate(path)

  async def test_load_orbax_checkpoint(self):
    layout = OrbaxLayout()
    restored_checkpointables_await = await layout.load_checkpointables(
        self.orbax_path / '0'
    )
    restored_checkpointables = await restored_checkpointables_await
    test_utils.assert_tree_equal(
        self, restored_checkpointables['pytree'], self.object_to_save
    )

  async def test_metadata(self):
    """Tests the metadata() method."""
    layout = OrbaxLayout()
    result_metadata = await layout.metadata(self.orbax_path / '0')

    self.assertIsInstance(result_metadata, metadata_types.CheckpointMetadata)

    expected_structs = {
        checkpoint_layout.PYTREE_CHECKPOINTABLE_KEY: {
            'a': numpy_leaf_handler.NumpyMetadata(
                shape=(9,),
                dtype=np.dtype(np.int32),
                storage_metadata=value_metadata.StorageMetadata(
                    chunk_shape=(9,), write_shape=None
                ),
            ),
            'b': numpy_leaf_handler.NumpyMetadata(
                shape=(3,),
                dtype=np.dtype(np.float32),
                storage_metadata=value_metadata.StorageMetadata(
                    chunk_shape=(3,), write_shape=None
                ),
            ),
        }
    }
    self.assertEqual(result_metadata.metadata, expected_structs)
    self.assertEqual(result_metadata.custom_metadata, self.custom_metadata)
    self.assertIsInstance(result_metadata.init_timestamp_nsecs, int)
    self.assertGreater(result_metadata.init_timestamp_nsecs, 0)
    self.assertIsInstance(result_metadata.commit_timestamp_nsecs, int)
    self.assertGreater(result_metadata.commit_timestamp_nsecs, 0)


class V1ValidationTest(
    unittest.IsolatedAsyncioTestCase, parameterized.TestCase
):

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(self.create_tempdir().full_path) / 'ckpt'
    self.pytree, _ = array_test_utils.create_numpy_pytree()
    saving.save_pytree(self.directory, self.pytree)

  async def test_nonexistent_path(self):
    with self.assertRaises(FileNotFoundError):
      await OrbaxLayout()._validate(self.directory / 'foo')

  async def test_not_a_directory(self):
    await async_path.write_text(self.directory / 'foo', 'foo')
    with self.assertRaises(NotADirectoryError):
      await OrbaxLayout()._validate(self.directory / 'foo')

  async def test_missing_indicator_file(self):
    await _unlink_indicator(self.directory)
    with self.assertRaises(FileNotFoundError):
      await OrbaxLayout()._validate(self.directory)

  async def test_deleted_pytree(self):
    (self.directory / 'pytree').rmtree()

    await OrbaxLayout()._validate(self.directory)
    with self.assertRaises(FileNotFoundError):
      await OrbaxLayout()._validate_pytree(self.directory, 'pytree')

  async def test_missing_checkpointable_matching_name(self):
    with self.assertRaises(FileNotFoundError):
      await OrbaxLayout()._validate_pytree(self.directory, 'foo')

  async def test_no_pytree_metadata(self):
    await _unlink_pytree_metadata(self.directory / 'pytree')

    with self.assertRaises(FileNotFoundError):
      await OrbaxLayout()._validate_pytree(self.directory, 'pytree')


class IsOrbaxV1CheckpointTest(parameterized.TestCase):

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
    np_save_file(self.object_to_save, self.safetensors_path)
    saving.save_pytree(
        self.orbax_path / '0',
        self.object_to_save,
        custom_metadata=self.custom_metadata,
    )
    self.composite_dir = (
        epath.Path(self.test_dir.full_path) / 'composite_checkpoint'
    )
    mngr = CheckpointManager(self.composite_dir)
    mngr.save(
        0,
        args=args.Composite(state=args.StandardSave({'a': 1, 'b': 2})),
    )
    mngr.wait_until_finished()

  def test_is_v0_orbax_checkpoint(self):
    self.assertTrue(orbax_layout.is_orbax_v1_checkpoint(self.orbax_path / '0'))
    self.assertFalse(orbax_layout.is_orbax_v1_checkpoint(self.safetensors_path))
    self.assertFalse(
        orbax_layout.is_orbax_v1_checkpoint(self.composite_dir / '0')
    )


if __name__ == '__main__':
  absltest.main()
