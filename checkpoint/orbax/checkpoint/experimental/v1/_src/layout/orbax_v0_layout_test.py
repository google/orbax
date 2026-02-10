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
from orbax.checkpoint._src.checkpointers import checkpointer
from orbax.checkpoint._src.checkpointers import standard_checkpointer
from orbax.checkpoint._src.handlers import composite_checkpoint_handler
from orbax.checkpoint._src.handlers import standard_checkpoint_handler
from orbax.checkpoint._src.path import async_path
from orbax.checkpoint.checkpoint_manager import CheckpointManager
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout
from orbax.checkpoint.experimental.v1._src.layout import orbax_v0_layout
from orbax.checkpoint.experimental.v1._src.testing import array_utils as array_test_utils
import safetensors.numpy


InvalidLayoutError = checkpoint_layout.InvalidLayoutError
np_save_file = safetensors.numpy.save_file


class OrbaxV0LayoutTest(
    unittest.IsolatedAsyncioTestCase, parameterized.TestCase
):

  def setUp(self):
    super().setUp()
    self.test_dir = self.create_tempdir('test_dir')
    self.orbax_path = epath.Path(self.test_dir.full_path) / 'test_checkpoint'
    self.safetensors_path = (
        epath.Path(self.test_dir.full_path) / 'test_checkpoint.safetensors'
    )

    # Create a mock SafeTensors and Orbax V0 checkpoint
    self.object_to_save = {
        'a': np.array(3 * [1, 2, 3], dtype=np.int32),
        'b': np.array([0, 1, 0.2], dtype=np.float32),
    }
    self.custom_metadata = {'framework': 'JAX', 'version': '1.0'}
    np_save_file(self.object_to_save, self.safetensors_path)

    # Save V0 checkpoint
    ckptr = checkpointer.Checkpointer(
        composite_checkpoint_handler.CompositeCheckpointHandler()
    )
    ckptr.save(
        self.orbax_path / '0',
        composite_checkpoint_handler.CompositeArgs(
            state=standard_checkpoint_handler.StandardSaveArgs(
                self.object_to_save
            )
        ),
    )

  async def test_valid_v0_checkpoint(self):
    layout = orbax_v0_layout.OrbaxV0Layout()
    await layout.validate(self.orbax_path / '0')

  async def test_invalid_v0_checkpoint(self):
    layout = orbax_v0_layout.OrbaxV0Layout()
    with self.assertRaises(InvalidLayoutError):
      await layout.validate(self.safetensors_path)

  async def test_validate_fails_not_directory(self):
    layout = orbax_v0_layout.OrbaxV0Layout()
    with self.assertRaises(InvalidLayoutError):
      await layout.validate(self.orbax_path / '1')

  async def test_validate_no_metadata_file(self):
    layout = orbax_v0_layout.OrbaxV0Layout()
    metadata_path = self.orbax_path / '0' / '_CHECKPOINT_METADATA'
    self.assertTrue(metadata_path.exists())
    metadata_path.rmtree()
    await layout.validate(self.orbax_path / '0')

  async def test_validate_no_metadata_files_fails(self):
    layout = orbax_v0_layout.OrbaxV0Layout()
    metadata_path = self.orbax_path / '0' / '_CHECKPOINT_METADATA'
    metadata_path.rmtree()
    pytree_metadata_path = self.orbax_path / '0' / 'state' / '_METADATA'
    pytree_metadata_path.rmtree()
    with self.assertRaises(InvalidLayoutError):
      await layout.validate(self.orbax_path / '0')

  async def test_load_v0_checkpoint(self):
    layout = orbax_v0_layout.OrbaxV0Layout()
    restored_checkpointables_await = await layout.load_checkpointables(
        self.orbax_path / '0'
    )
    restored_checkpointables = await restored_checkpointables_await
    test_utils.assert_tree_equal(
        self, restored_checkpointables['state'], self.object_to_save
    )

  async def test_metadata(self):
    with self.subTest('PyTree checkpoint with _CHECKPOINT_METADATA in path'):
      layout = orbax_v0_layout.OrbaxV0Layout()
      metadata = await layout.metadata(self.orbax_path / '0')
      self.assertIsNotNone(metadata.init_timestamp_nsecs)

    with self.subTest('PyTree Checkpoint with no _CHECKPOINT_METADATA'):
      await async_path.unlink(self.orbax_path / '0' / '_CHECKPOINT_METADATA')
      layout = orbax_v0_layout.OrbaxV0Layout()
      metadata = await layout.metadata(self.orbax_path / '0' / 'state')
      self.assertIsNotNone(metadata)
      self.assertIsNone(metadata.init_timestamp_nsecs)


class V0ValidationTest(
    unittest.IsolatedAsyncioTestCase, parameterized.TestCase
):

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(self.create_tempdir().full_path) / 'ckpt'
    self.v0_pytree_directory = (
        epath.Path(self.create_tempdir().full_path) / 'v0_pytree'
    )
    self.pytree, _ = array_test_utils.create_numpy_pytree()
    # Save a checkpoint with a checkpointable name, `state`.
    ckptr = checkpointer.Checkpointer(
        composite_checkpoint_handler.CompositeCheckpointHandler()
    )
    ckptr.save(
        self.directory,
        composite_checkpoint_handler.CompositeArgs(
            state=standard_checkpoint_handler.StandardSaveArgs(self.pytree),
            params=standard_checkpoint_handler.StandardSaveArgs(self.pytree),
        ),
    )

    with standard_checkpointer.StandardCheckpointer() as v0_checkpointer:
      v0_checkpointer.save(self.v0_pytree_directory, self.pytree)

  async def test_nonexistent_path(self):
    with self.assertRaises(FileNotFoundError):
      await orbax_v0_layout.OrbaxV0Layout()._validate(self.directory / 'foo')

  async def test_not_a_directory(self):
    await async_path.write_text(self.directory / 'foo', 'foo')
    with self.assertRaises(NotADirectoryError):
      await orbax_v0_layout.OrbaxV0Layout()._validate(self.directory / 'foo')

  async def test_no_checkpoint_metadata(self):
    await async_path.unlink(
        self.directory / '_CHECKPOINT_METADATA',
        missing_ok=True,
    )

    # Passes since it contains a pytree checkpointable with _METADATA file.
    await orbax_v0_layout.OrbaxV0Layout()._validate(self.directory)
    await orbax_v0_layout.OrbaxV0Layout()._validate_pytree(
        self.directory, 'state'
    )

  async def test_deleted_pytree(self):
    (self.directory / 'state').rmtree()

    # Passes since it contains checkpoint metadata.
    await orbax_v0_layout.OrbaxV0Layout()._validate(self.directory)
    with self.assertRaises(FileNotFoundError):
      await orbax_v0_layout.OrbaxV0Layout()._validate_pytree(
          self.directory, 'state'
      )

  async def test_missing_checkpointable_matching_name(self):
    with self.assertRaises(FileNotFoundError):
      await orbax_v0_layout.OrbaxV0Layout()._validate_pytree(
          self.directory, 'foo'
      )

  async def test_no_pytree_metadata(self):
    await async_path.unlink(
        self.directory / 'state' / '_METADATA',
        missing_ok=True,
    )

    # Passes because it contains checkpoint metadata.
    await orbax_v0_layout.OrbaxV0Layout()._validate(self.directory)
    with self.assertRaises(FileNotFoundError):
      # Fails because it does not contain pytree metadata.
      await orbax_v0_layout.OrbaxV0Layout()._validate_pytree(
          self.directory, 'state'
      )

  async def test_valid_pytree(self):
    await orbax_v0_layout.OrbaxV0Layout()._validate_pytree(
        self.directory, 'state'
    )

  async def test_load_pytree(self):
    layout = orbax_v0_layout.OrbaxV0Layout()
    loaded = await (
        await layout.load_pytree(self.directory, 'state', self.pytree)
    )
    test_utils.assert_tree_equal(self, self.pytree, loaded['state'])

  async def test_load_pytree_no_checkpoint_metadata(self):
    await async_path.unlink(self.directory / '_CHECKPOINT_METADATA')
    layout = orbax_v0_layout.OrbaxV0Layout()

    with self.assertRaises(FileNotFoundError):
      await layout.load_pytree(self.directory, 'state', self.pytree)

  async def test_load_pytree_v0_checkpoint(self):
    layout = orbax_v0_layout.OrbaxV0Layout()
    loaded = await (
        await layout.load_pytree(self.v0_pytree_directory, None, self.pytree)
    )
    test_utils.assert_tree_equal(self, self.pytree, loaded)

  async def test_v0_pytree_no_checkpoint_metadata(self):
    await async_path.unlink(
        self.v0_pytree_directory / '_CHECKPOINT_METADATA',
        missing_ok=True,
    )
    layout = orbax_v0_layout.OrbaxV0Layout()

    await orbax_v0_layout.OrbaxV0Layout()._validate(self.v0_pytree_directory)

    loaded = await (
        await layout.load_pytree(self.v0_pytree_directory, None, self.pytree)
    )
    # Passes because we still have the pytree metadata.
    test_utils.assert_tree_equal(self, self.pytree, loaded)

  async def test_v0_pytree_no_pytree_metadata(self):
    await async_path.unlink(
        self.v0_pytree_directory / '_METADATA',
        missing_ok=True,
    )
    layout = orbax_v0_layout.OrbaxV0Layout()
    with self.assertRaises(FileNotFoundError):
      await layout._validate_pytree(self.v0_pytree_directory, None)


class IsOrbaxV0CheckpointTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = self.create_tempdir('test_dir')
    self.orbax_path = epath.Path(self.test_dir.full_path) / 'test_checkpoint'
    self.safetensors_path = (
        epath.Path(self.test_dir.full_path) / 'test_checkpoint.safetensors'
    )

    # Create a mock SafeTensors and Orbax V0 checkpoint
    self.object_to_save = {
        'a': np.array(3 * [1, 2, 3], dtype=np.int32),
        'b': np.array([0, 1, 0.2], dtype=np.float32),
    }
    self.custom_metadata = {'framework': 'JAX', 'version': '1.0'}
    np_save_file(self.object_to_save, self.safetensors_path)

    # Save V0 checkpoint
    ckptr = checkpointer.Checkpointer(
        composite_checkpoint_handler.CompositeCheckpointHandler()
    )
    ckptr.save(
        self.orbax_path / '0',
        composite_checkpoint_handler.CompositeArgs(
            state=standard_checkpoint_handler.StandardSaveArgs(
                self.object_to_save
            )
        ),
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
    self.assertTrue(
        orbax_v0_layout.is_orbax_v0_checkpoint(self.orbax_path / '0')
    )
    self.assertFalse(
        orbax_v0_layout.is_orbax_v0_checkpoint(self.safetensors_path)
    )
    self.assertTrue(
        orbax_v0_layout.is_orbax_v0_checkpoint(self.composite_dir / '0')
    )


if __name__ == '__main__':
  absltest.main()
