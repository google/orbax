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

"""Base class for partial saving tests."""

from __future__ import annotations

import asyncio
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import numpy as np
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.path.snapshot import snapshot
from orbax.checkpoint._src.tree import structure_utils as tree_structure_utils
from orbax.checkpoint.experimental.v1._src.handlers import pytree_handler
from orbax.checkpoint.experimental.v1._src.loading import loading
from orbax.checkpoint.experimental.v1._src.metadata import loading as metadata_loading
from orbax.checkpoint.experimental.v1._src.partial import path as partial_path_lib
from orbax.checkpoint.experimental.v1._src.partial import saving
from orbax.checkpoint.experimental.v1._src.testing import array_utils as array_test_utils
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types

create_numpy_pytree = array_test_utils.create_numpy_pytree
PyTree = tree_types.PyTree


class PartialSavingTest(parameterized.TestCase):
  """Base class for partial saving tests."""

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(
        self.create_tempdir(name='partial_saving_test').full_path
    )

    # Set up a new event loop for each test
    self.loop = asyncio.new_event_loop()
    asyncio.set_event_loop(self.loop)

    self.pytree, self.abstract_pytree = create_numpy_pytree()
    self.numpy_pytree, self.abstract_numpy_pytree = create_numpy_pytree()

  def tearDown(self):
    super().tearDown()
    self.loop.close()

  @parameterized.parameters(True, False)
  def test_finalize_conforming(self, finalize_partial_path: bool):
    """Tests that finalize() conforms a partial path."""
    final_path = self.directory / 'test_finalize_conforming'
    partial_path = partial_path_lib.add_partial_save_suffix(final_path)
    path_to_finalize = partial_path if finalize_partial_path else final_path

    partial_path.mkdir(parents=True)
    self.assertTrue(partial_path.exists())
    self.assertFalse(final_path.exists())

    saving.finalize(path_to_finalize)

    self.assertFalse(partial_path.exists())
    self.assertTrue(final_path.exists())

  @parameterized.parameters(True, False)
  def test_finalize_partial_missing(self, finalize_partial_path: bool):
    """Tests that finalize() raises an error if the partial path is missing."""
    final_path = self.directory / 'test_finalize_partial_missing'
    partial_path = partial_path_lib.add_partial_save_suffix(final_path)
    path_to_finalize = partial_path if finalize_partial_path else final_path

    self.assertFalse(partial_path.exists())
    self.assertFalse(final_path.exists())

    with self.assertRaises(FileNotFoundError):
      saving.finalize(path_to_finalize)

  @parameterized.parameters(True, False)
  def test_finalize_final_exists(self, finalize_partial_path: bool):
    """Tests finalize() error when final path exists."""
    final_path = self.directory / 'test_finalize_final_exists'
    partial_path = partial_path_lib.add_partial_save_suffix(final_path)
    path_to_finalize = partial_path if finalize_partial_path else final_path

    partial_path.mkdir(parents=True)
    final_path.mkdir(parents=True)
    self.assertTrue(partial_path.exists())
    self.assertTrue(final_path.exists())

    with self.assertRaises(FileExistsError):
      saving.finalize(path_to_finalize)

  def test_finalize_rename_os_error(self):
    """Tests that finalize() raises an error if rename() raises an error."""
    final_path = self.directory / 'test_finalize_rename_os_error'
    partial_path = partial_path_lib.add_partial_save_suffix(final_path)
    path_to_finalize = final_path

    partial_path.mkdir(parents=True)
    self.assertTrue(partial_path.exists())
    self.assertFalse(final_path.exists())

    async def mock_rename(src, dst):
      del src, dst
      raise OSError('Test error.')

    with mock.patch(
        'orbax.checkpoint.experimental.v1._src.partial.saving.async_path.rename',
        new=mock_rename,
    ):
      with self.assertRaises(OSError):
        saving.finalize(path_to_finalize)

    self.assertTrue(partial_path.exists())
    self.assertFalse(final_path.exists())

  @parameterized.product(
      pytrees=[
          ({'a': 1}, {'b': 2}),
          ({'a': 1}, {'b': {'bb': 2}}),
          ({'a': {'aa': 1}}, {'b': 2}),
          ({'a': {'aa': 1}}, {'b': {'bb': 2}}),
          ({'a': 1}, {'b': {'bb': 2}, 'c': 3}),
          ({'a': 1}, {'b': {'bb': 2}, 'c': {'cc': 3}}),
          ({'a': {'aa': 1}}, {'b': 2, 'c': 3}),
          ({'a': {'aa': 1}}, {'b': {'bb': 2}, 'c': {'cc': 3}}),
          (
              {
                  'params': {
                      'layer0': np.arange(8),
                      'layer1': np.ones(4),
                  },
                  'step': 10000,
              },
              {
                  'metrics': {
                      'accuracy': 0.98,
                      'loss': 0.05,
                  }
              },
          ),
      ],
  )
  def test_simple_save_pytree_addition(self, pytrees: tuple[PyTree, PyTree]):
    """Tests that save() adds pytrees correctly."""
    final_path = self.directory / 'save_pytree'
    partial_path = partial_path_lib.add_partial_save_suffix(final_path)
    pytree_1, pytree_2 = pytrees

    # First save.
    saving.save(final_path, pytree_1)
    self.assertTrue(partial_path.exists())
    self.assertFalse(final_path.exists())

    # Second save.
    saving.save(final_path, pytree_2)
    self.assertTrue(partial_path.exists())
    self.assertFalse(final_path.exists())

    # Finalize.
    saving.finalize(final_path)
    self.assertFalse(partial_path.exists())
    self.assertTrue(final_path.exists())

    restored_pytree = loading.load(final_path)
    expected_pytree = tree_structure_utils.merge_trees(pytree_1, pytree_2)
    test_utils.assert_tree_equal(self, restored_pytree, expected_pytree)

  def test_save_pytree_addition_with_nested_keys(self):
    """Tests that save() adds pytrees with nested keys correctly."""
    final_path = self.directory / 'save_pytree'
    partial_path = partial_path_lib.add_partial_save_suffix(final_path)

    # First save.
    saving.save(final_path, self.pytree)
    self.assertTrue(partial_path.exists())
    self.assertFalse(final_path.exists())

    new_node_pytree = {
        'new_node': self.pytree['a'],
        'c': {'new_nested_node': self.pytree['b']},
    }

    # Second save.
    saving.save(final_path, new_node_pytree)
    self.assertTrue(partial_path.exists())
    self.assertFalse(final_path.exists())

    # Finalize.
    saving.finalize(final_path)
    self.assertFalse(partial_path.exists())
    self.assertTrue(final_path.exists())

    restored_pytree = loading.load(final_path)
    expected_pytree = tree_structure_utils.merge_trees(
        self.pytree, new_node_pytree
    )
    test_utils.assert_tree_equal(self, restored_pytree, expected_pytree)

  def test_save_pytree_replacement_raises_error(self):
    """Tests that save() raises an error if replacement occurs."""
    final_path = self.directory / 'save_pytree_replacement'

    # First save.
    saving.save(final_path, self.pytree)

    # Second save, with replacement.
    replacement_pytree = {
        'a': self.pytree['b'],  # replacement
        'new_node': self.pytree['a'],  # addition
    }

    with self.assertRaisesRegex(
        pytree_handler.PartialSaveReplacementError,
        'Partial saving currently does not support REPLACEMENT.',
    ):
      saving.save(final_path, replacement_pytree)

  @parameterized.parameters(True, False)
  def test_save_pytree_subtree_replacement_raises_error(
      self, first_save_leaf_is_subtree: bool
  ):
    """Tests save() error on subtree replacement."""
    final_path = self.directory / 'save_pytree_subtree_replacement'

    if first_save_leaf_is_subtree:
      first_save_pytree = {'a': {'b': 1}}
      second_save_pytree = {'a': 2}
    else:
      first_save_pytree = {'a': 2}
      second_save_pytree = {'a': {'b': 1}}

    # First save.
    saving.save(final_path, first_save_pytree)

    # Second save, with subtree replacement.
    with self.assertRaisesRegex(
        pytree_handler.PartialSaveReplacementError,
        'Partial saving currently does not support REPLACEMENT.',
    ):
      saving.save(final_path, second_save_pytree)

  def test_partial_save_disk_usage_not_copied(self):
    """Tests that partial saves don't recursively copy existing data."""
    final_path = self.directory / 'disk_usage_test'
    partial_path = partial_path_lib.add_partial_save_suffix(final_path)

    # Use a large enough array so metadata overhead is negligible.
    arr_size_bytes = 512 * 1024
    arr1 = np.ones(arr_size_bytes // 4, dtype=np.float32)
    saving.save(final_path, {'arr1': arr1})

    with self.subTest('first_save_pending_dir'):
      pending_dirs = asyncio.run(snapshot.list_pending_dirs(partial_path))
      self.assertLen(pending_dirs, 1)

    def get_dir_size(path: epath.Path) -> int:
      total_size = 0
      for f in path.iterdir():
        if f.is_file():
          total_size += f.stat().length
        elif f.is_dir():
          total_size += get_dir_size(f)
      return total_size

    base_size = get_dir_size(pending_dirs[0])

    arr2 = np.ones(arr_size_bytes // 4, dtype=np.float32)
    saving.save(final_path, {'arr2': arr2})
    pending_dirs = asyncio.run(snapshot.list_pending_dirs(partial_path))

    with self.subTest('second_save_pending_dirs'):
      self.assertLen(pending_dirs, 2)

    with self.subTest('disk_usage_not_copied'):
      # If it was recursively copying (like IN_PLACE), the second pending dir
      # would contain both arr1 and arr2, roughly doubling in size.
      # With SnapshotType.EMPTY, each pending dir only contains the new data.
      for p_dir in pending_dirs:
        dir_size = get_dir_size(p_dir)
        # Assert size is strictly less than 1.5x the first save's size.
        self.assertLess(dir_size, base_size * 1.5)

    with self.subTest('restored_content'):
      # Finalize to ensure the checkpoint is valid
      saving.finalize(final_path)
      restored = loading.load(final_path)
      self.assertIn('arr1', restored)
      self.assertIn('arr2', restored)

  def test_empty_initial_save(self):
    """Tests that save() raises an error if the initial save is empty."""
    final_path = self.directory / 'empty_initial_save'

    # First save - empty.
    with self.assertRaisesRegex(ValueError, 'Found empty item.'):
      saving.save(final_path, {})

  @parameterized.named_parameters(
      ('none_then_meta', None, {'meta1': 'val1'}, {'meta1': 'val1'}),
      ('meta_then_none', {'meta1': 'val1'}, None, {'meta1': 'val1'}),
      (
          'meta1_then_meta2',
          {'meta1': 'val1'},
          {'meta2': 'val2'},
          {'meta1': 'val1', 'meta2': 'val2'},
      ),
  )
  def test_custom_metadata(self, meta1, meta2, expected_meta):
    """Tests that save() saves custom metadata correctly."""
    final_path = self.directory / 'custom_metadata'

    saving.save(final_path, {'a': 1}, custom_metadata=meta1)
    saving.save(final_path, {'b': 2}, custom_metadata=meta2)
    saving.finalize(final_path)

    restored_metadata = metadata_loading.checkpointables_metadata(final_path)
    self.assertEqual(restored_metadata.custom_metadata, expected_meta)

  def test_save_no_changes(self):
    """Tests that save() raises an error if no changes occur."""
    final_path = self.directory / 'save_no_changes'
    saving.save(final_path, {'a': 1})
    with self.assertRaisesRegex(
        pytree_handler.PartialSaveReplacementError,
        'Partial saving currently does not support REPLACEMENT.',
    ):
      saving.save(final_path, {'a': 1})  # Save the same tree again

  def test_finalize_file_collision(self):
    """Tests that finalize() raises FileExistsError on file collision."""
    final_path = self.directory / 'test_finalize_file_collision'
    partial_path = partial_path_lib.add_partial_save_suffix(final_path)

    saving.save(final_path, {'a': 1})
    saving.save(final_path, {'b': 2})

    pending_dirs = asyncio.run(snapshot.list_pending_dirs(partial_path))
    self.assertLen(pending_dirs, 2)

    for p_dir in pending_dirs:
      (p_dir / 'colliding_file.txt').write_text('collision')

    with self.assertRaisesRegex(
        FileExistsError,
        'File collision on colliding_file.txt during finalize. Overwriting '
        'destination file is not allowed.',
    ):
      saving.finalize(final_path)

  def test_save_after_finalize(self):
    """Tests save() error if the final path already exists."""
    final_path = self.directory / 'save_after_finalize'
    saving.save(final_path, {'a': 1})
    saving.finalize(final_path)
    self.assertTrue(final_path.exists())

    with self.assertRaises(FileExistsError):
      saving.save(final_path, {'b': 2})

if __name__ == '__main__':
  absltest.main()
