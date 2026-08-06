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

from typing import Any, cast
import unittest
from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import numpy as np
from orbax.checkpoint import args
from orbax.checkpoint import checkpoint_manager
from orbax.checkpoint import options as v0_options_lib
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.metadata import value as value_metadata
from orbax.checkpoint._src.path import async_path
from orbax.checkpoint._src.path import atomicity
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.context import options as v1_options_lib
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout
from orbax.checkpoint.experimental.v1._src.layout import orbax_layout
from orbax.checkpoint.experimental.v1._src.metadata import types as metadata_types
from orbax.checkpoint.experimental.v1._src.saving import saving
from orbax.checkpoint.experimental.v1._src.serialization import numpy_leaf_handler
from orbax.checkpoint.experimental.v1._src.testing import array_utils as array_test_utils
import safetensors.numpy


STATE_CHECKPOINTABLE_KEY = checkpoint_layout.STATE_CHECKPOINTABLE_KEY
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
    self.test_dir = self.create_tempdir(name='orbax_layout_test')
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
    saving.save(
        self.orbax_path / '0',
        self.object_to_save,  # pyrefly: ignore[bad-argument-type]
        custom_metadata=self.custom_metadata,  # pyrefly: ignore[bad-argument-type]
    )

  async def test_valid_orbax_checkpoint(self):
    layout = OrbaxLayout()
    await layout.validate_checkpointables(self.orbax_path / '0')

  async def test_invalid_orbax_checkpoint(self):
    layout = OrbaxLayout()
    with self.assertRaises(InvalidLayoutError):
      await layout.validate_checkpointables(self.safetensors_path)

  async def test_validate_fails_not_directory(self):
    layout = OrbaxLayout()
    with self.assertRaises(InvalidLayoutError):
      await layout.validate_checkpointables(self.orbax_path / '1')

  async def test_validate_no_indicator_file(self):
    layout = OrbaxLayout()
    indicator_path = (
        self.orbax_path / '0' / orbax_layout.ORBAX_CHECKPOINT_INDICATOR_FILE
    )
    indicator_path.rmtree()  # Remove the indicator file
    with self.assertRaises(InvalidLayoutError):
      await layout.validate_checkpointables(self.orbax_path / '0')

  async def test_validate_no_metadata_file(self):
    layout = OrbaxLayout()
    metadata_path = self.orbax_path / '0' / '_CHECKPOINT_METADATA'
    self.assertTrue(metadata_path.exists())
    metadata_path.rmtree()  # Remove the metadata file
    with self.assertRaises(InvalidLayoutError):
      await layout.validate_checkpointables(self.orbax_path / '0')

  async def test_validate_no_indicator_or_metadata_files(self):
    layout = OrbaxLayout()
    indicator_path = (
        self.orbax_path / '0' / orbax_layout.ORBAX_CHECKPOINT_INDICATOR_FILE
    )
    indicator_path.rmtree()  # Remove the indicator file
    metadata_path = self.orbax_path / '0' / '_CHECKPOINT_METADATA'
    metadata_path.rmtree()
    pytree_metadata_path = (
        self.orbax_path / '0' / STATE_CHECKPOINTABLE_KEY / '_METADATA'
    )
    pytree_metadata_path.rmtree()
    with self.assertRaises(InvalidLayoutError):
      await layout.validate_checkpointables(self.orbax_path / '0')

  async def test_validate_fails_tmp_directory(self):
    # This simulates a temporary directory created by Orbax (should fail)
    test_utils.save_fake_tmp_dir(self.orbax_path, 0, 'test_checkpoint.tmp')
    path = epath.Path(self.test_dir.full_path) / 'test_checkpoint.tmp'
    layout = OrbaxLayout()
    with self.assertRaises(InvalidLayoutError):
      await layout.validate_checkpointables(path)

  async def test_load_orbax_checkpoint(self):
    layout = OrbaxLayout()
    restored_checkpointables_await = await layout.load_checkpointables(
        self.orbax_path / '0'
    )
    restored_checkpointables = await restored_checkpointables_await
    test_utils.assert_tree_equal(
        self,
        restored_checkpointables[STATE_CHECKPOINTABLE_KEY],
        self.object_to_save,
    )

  async def test_save_with_custom_name(self):
    custom_path = epath.Path(self.test_dir.full_path) / 'custom_checkpoint'
    saving.save(
        custom_path,
        self.object_to_save,  # pyrefly: ignore[bad-argument-type]
        checkpointable_name='my_custom_name',
    )
    self.assertTrue((custom_path / 'my_custom_name').exists())
    layout = OrbaxLayout()
    restored_checkpointables = await (
        await layout.load_checkpointables(custom_path)
    )
    test_utils.assert_tree_equal(
        self, restored_checkpointables['my_custom_name'], self.object_to_save
    )

  async def test_v1_save_and_load_with_custom_atomicity_options(self):
    path = epath.Path(self.test_dir.full_path) / 'atomicity_test_ckpt'
    atomicity_opts = v1_options_lib.AtomicityOptions(
        mode=v0_options_lib.AtomicityMode.COMMIT_FILE
    )

    with context_lib.Context(atomicity_options=atomicity_opts):
      saving.save(path, cast(Any, self.object_to_save))
      layout = OrbaxLayout()
      restored = await (await layout.load_checkpointables(path))

      test_utils.assert_tree_equal(
          self, restored[STATE_CHECKPOINTABLE_KEY], self.object_to_save
      )

  async def test_v1_validate_checkpointables_with_legacy_atomic_rename_fallback_enabled(
      self,
  ):
    path = epath.Path(self.test_dir.full_path) / 'fallback_enabled_test_ckpt'
    atomicity_opts_save = v1_options_lib.AtomicityOptions(
        mode=v0_options_lib.AtomicityMode.ATOMIC_RENAME
    )
    with context_lib.Context(atomicity_options=atomicity_opts_save):
      saving.save(path, cast(Any, self.object_to_save))

    for commit_file in (
        path / atomicity.COMMIT_SUCCESS_FILE,
        path / STATE_CHECKPOINTABLE_KEY / atomicity.COMMIT_SUCCESS_FILE,
    ):
      if commit_file.exists():
        commit_file.unlink()

    atomicity_opts = v1_options_lib.AtomicityOptions(
        mode=v0_options_lib.AtomicityMode.COMMIT_FILE,
        allow_legacy_atomic_rename=True,
    )

    with context_lib.Context(atomicity_options=atomicity_opts):
      layout = OrbaxLayout()
      await layout.validate_checkpointables(path)

  async def test_v1_validate_checkpointables_with_legacy_atomic_rename_fallback_disabled(
      self,
  ):
    path = epath.Path(self.test_dir.full_path) / 'fallback_disabled_test_ckpt'
    atomicity_opts_save = v1_options_lib.AtomicityOptions(
        mode=v0_options_lib.AtomicityMode.ATOMIC_RENAME
    )
    with context_lib.Context(atomicity_options=atomicity_opts_save):
      saving.save(path, cast(Any, self.object_to_save))

    for commit_file in (
        path / atomicity.COMMIT_SUCCESS_FILE,
        path / STATE_CHECKPOINTABLE_KEY / atomicity.COMMIT_SUCCESS_FILE,
    ):
      if commit_file.exists():
        commit_file.unlink()

    atomicity_opts = v1_options_lib.AtomicityOptions(
        mode=v0_options_lib.AtomicityMode.COMMIT_FILE,
        allow_legacy_atomic_rename=False,
    )

    with context_lib.Context(atomicity_options=atomicity_opts):
      layout = OrbaxLayout()
      with self.assertRaises((ValueError, InvalidLayoutError)):
        await layout.validate_checkpointables(path)

  async def test_checkpointables_metadata(self):
    """Tests the metadata() method."""
    layout = OrbaxLayout()
    result_metadata = await layout.checkpointables_metadata(
        self.orbax_path / '0'
    )

    self.assertIsInstance(result_metadata, metadata_types.CheckpointMetadata)

    expected_structs = {
        STATE_CHECKPOINTABLE_KEY: {
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

  async def test_metadata(self):
    """Tests the metadata() method for a single checkpointable."""
    layout = OrbaxLayout()
    result_metadata = await layout.metadata(
        self.orbax_path / '0', STATE_CHECKPOINTABLE_KEY
    )

    self.assertIsInstance(result_metadata, metadata_types.CheckpointMetadata)

    expected_structs = {
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
    self.directory = (
        epath.Path(self.create_tempdir(name='v1_validation_test').full_path)
        / 'ckpt'
    )
    self.pytree, _ = array_test_utils.create_numpy_pytree()
    saving.save(self.directory, self.pytree)

  async def test_nonexistent_path(self):
    with self.assertRaises(FileNotFoundError):
      await OrbaxLayout()._validate_checkpointables(self.directory / 'foo')

  async def test_not_a_directory(self):
    await async_path.write_text(self.directory / 'foo', 'foo')
    with self.assertRaises(NotADirectoryError):
      await OrbaxLayout()._validate_checkpointables(self.directory / 'foo')

  async def test_missing_indicator_file(self):
    await _unlink_indicator(self.directory)
    with self.assertRaises(FileNotFoundError):
      await OrbaxLayout()._validate_checkpointables(self.directory)

  async def test_deleted_pytree(self):
    (self.directory / STATE_CHECKPOINTABLE_KEY).rmtree()

    await OrbaxLayout()._validate_checkpointables(self.directory)
    with self.assertRaises(FileNotFoundError):
      await OrbaxLayout()._validate(self.directory, STATE_CHECKPOINTABLE_KEY)

  async def test_missing_checkpointable_matching_name(self):
    with self.assertRaises(FileNotFoundError):
      await OrbaxLayout()._validate(self.directory, 'foo')

  async def test_no_pytree_metadata(self):
    await _unlink_pytree_metadata(self.directory / STATE_CHECKPOINTABLE_KEY)

    with self.assertRaises(FileNotFoundError):
      await OrbaxLayout()._validate(self.directory, STATE_CHECKPOINTABLE_KEY)


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
    saving.save(
        self.orbax_path / '0',
        self.object_to_save,  # pyrefly: ignore[bad-argument-type]
        custom_metadata=self.custom_metadata,  # pyrefly: ignore[bad-argument-type]
    )
    self.composite_dir = (
        epath.Path(self.test_dir.full_path) / 'composite_checkpoint'
    )
    mngr = checkpoint_manager.CheckpointManager(self.composite_dir)
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
