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
from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import numpy as np
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.checkpointers import checkpointer
from orbax.checkpoint._src.handlers import composite_checkpoint_handler
from orbax.checkpoint._src.handlers import standard_checkpoint_handler
from orbax.checkpoint._src.path import async_path
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout
from orbax.checkpoint.experimental.v1._src.layout import orbax_v0_layout
from orbax.checkpoint.experimental.v1._src.metadata import types as metadata_types
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
    layout = orbax_v0_layout.OrbaxV0Layout(self.orbax_path / '0')
    await layout.validate()

  async def test_invalid_v0_checkpoint(self):
    layout = orbax_v0_layout.OrbaxV0Layout(self.safetensors_path)
    with self.assertRaises(InvalidLayoutError):
      await layout.validate()

  async def test_validate_fails_not_directory(self):
    layout = orbax_v0_layout.OrbaxV0Layout(self.orbax_path / '1')
    with self.assertRaises(InvalidLayoutError):
      await layout.validate()

  async def test_validate_no_metadata_file(self):
    layout = orbax_v0_layout.OrbaxV0Layout(self.orbax_path / '0')
    metadata_path = self.orbax_path / '0' / '_CHECKPOINT_METADATA'
    self.assertTrue(metadata_path.exists())
    metadata_path.rmtree()
    await layout.validate()

  async def test_validate_no_metadata_files_fails(self):
    layout = orbax_v0_layout.OrbaxV0Layout(self.orbax_path / '0')
    metadata_path = self.orbax_path / '0' / '_CHECKPOINT_METADATA'
    metadata_path.rmtree()
    pytree_metadata_path = self.orbax_path / '0' / 'state' / '_METADATA'
    pytree_metadata_path.rmtree()
    with self.assertRaises(InvalidLayoutError):
      await layout.validate()

  async def test_load_v0_checkpoint(self):
    layout = orbax_v0_layout.OrbaxV0Layout(self.orbax_path / '0')
    restored_checkpointables_await = await layout.load_checkpointables()
    restored_checkpointables = await restored_checkpointables_await
    test_utils.assert_tree_equal(
        self, restored_checkpointables['state'], self.object_to_save
    )

  async def test_metadata(self):
    with self.subTest('PyTree checkpoint with _CHECKPOINT_METADATA in parent'):
      layout = orbax_v0_layout.OrbaxV0Layout(self.orbax_path / '0' / 'state')
      metadata = await layout.metadata()
      self.assertIsNotNone(metadata.init_timestamp_nsecs)

    with self.subTest('PyTree checkpoint with _CHECKPOINT_METADATA in path'):
      # Copy metadata to the PyTree directory.
      content = await async_path.read_bytes(
          self.orbax_path / '0' / '_CHECKPOINT_METADATA'
      )
      await async_path.write_bytes(
          self.orbax_path / '0' / 'state' / '_CHECKPOINT_METADATA', content
      )
      layout = orbax_v0_layout.OrbaxV0Layout(self.orbax_path / '0' / 'state')
      metadata = await layout.metadata()
      self.assertIsNotNone(metadata.init_timestamp_nsecs)

    with self.subTest('PyTree Checkpoint with no _CHECKPOINT_METADATA'):
      await async_path.unlink(
          self.orbax_path / '0' / 'state' / '_CHECKPOINT_METADATA'
      )
      await async_path.unlink(self.orbax_path / '0' / '_CHECKPOINT_METADATA')
      layout = orbax_v0_layout.OrbaxV0Layout(self.orbax_path / '0' / 'state')
      metadata = await layout.metadata()
      self.assertIsNone(metadata.init_timestamp_nsecs)

    with self.subTest('Non-PyTree restore'):
      layout = orbax_v0_layout.OrbaxV0Layout(self.orbax_path / '0')
      metadata = await layout.metadata()
      self.assertIsInstance(metadata, metadata_types.CheckpointMetadata)


class V0InternalValidationTest(
    unittest.IsolatedAsyncioTestCase, parameterized.TestCase
):

  def setUp(self):
    super().setUp()
    self.directory = (
        epath.Path(self.create_tempdir('test_directory').full_path) / 'ckpt'
    )
    self.pytree, _ = array_test_utils.create_numpy_pytree()

    ckptr = checkpointer.Checkpointer(
        composite_checkpoint_handler.CompositeCheckpointHandler()
    )

    ckptr.save(
        self.directory,
        composite_checkpoint_handler.CompositeArgs(
            state=standard_checkpoint_handler.StandardSaveArgs(self.pytree)
        ),
    )

  async def test_nonexistent_path(self):
    with self.assertRaises(FileNotFoundError):
      await orbax_v0_layout.OrbaxV0Layout(self.directory / 'foo')._validate()

  async def test_not_a_directory(self):
    await async_path.write_text(self.directory / 'foo', 'foo')
    with self.assertRaises(NotADirectoryError):
      await orbax_v0_layout.OrbaxV0Layout(self.directory / 'foo')._validate()

  @parameterized.product(checkpointable_name=['state', None])
  async def test_no_checkpoint_metadata(self, checkpointable_name: str | None):
    directory = (
        self.directory / checkpointable_name
        if checkpointable_name is not None
        else self.directory
    )
    await async_path.unlink(directory / '_CHECKPOINT_METADATA', missing_ok=True)

    # V0Layout should validate successfully even without _CHECKPOINT_METADATA
    await orbax_v0_layout.OrbaxV0Layout(directory)._validate()
    if checkpointable_name is None:
      await orbax_v0_layout.OrbaxV0Layout(directory)._validate_pytree('state')
    else:
      await orbax_v0_layout.OrbaxV0Layout(directory)._validate_pytree(None)

  async def test_deleted_pytree(self):
    directory = self.directory
    (directory / 'state').rmtree()

    await orbax_v0_layout.OrbaxV0Layout(directory)._validate()
    with self.assertRaises(FileNotFoundError):
      await orbax_v0_layout.OrbaxV0Layout(directory)._validate_pytree('state')

  async def test_missing_checkpointable_matching_name(self):
    with self.assertRaises(FileNotFoundError):
      await orbax_v0_layout.OrbaxV0Layout(self.directory)._validate_pytree(
          'foo'
      )

  @parameterized.product(checkpointable_name=['state', None])
  async def test_no_pytree_metadata(self, checkpointable_name: str | None):
    directory = (
        self.directory / checkpointable_name
        if checkpointable_name is not None
        else self.directory
    )
    await async_path.unlink(self.directory / 'state' / '_METADATA')

    if checkpointable_name is None:
      await orbax_v0_layout.OrbaxV0Layout(directory)._validate()
      with self.assertRaises(FileNotFoundError):
        await orbax_v0_layout.OrbaxV0Layout(directory)._validate_pytree('state')
    else:
      with self.assertRaises(InvalidLayoutError):
        await orbax_v0_layout.OrbaxV0Layout(directory)._validate()
      with self.assertRaises(FileNotFoundError):
        await orbax_v0_layout.OrbaxV0Layout(directory)._validate_pytree(None)

  @parameterized.product(checkpointable_name=['state', None])
  async def test_valid_pytree(self, checkpointable_name: str | None):
    directory = (
        self.directory / checkpointable_name
        if checkpointable_name is not None
        else self.directory
    )
    if checkpointable_name is None:
      await orbax_v0_layout.OrbaxV0Layout(directory)._validate_pytree('state')
    else:
      await orbax_v0_layout.OrbaxV0Layout(directory)._validate_pytree(None)


if __name__ == '__main__':
  absltest.main()
