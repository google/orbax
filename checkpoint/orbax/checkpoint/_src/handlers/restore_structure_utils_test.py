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

from absl.testing import absltest
import numpy as np
from orbax.checkpoint._src.handlers import restore_structure_utils
from orbax.checkpoint._src.metadata import tree as tree_metadata
from orbax.checkpoint._src.serialization import type_handlers

PLACEHOLDER = type_handlers.PLACEHOLDER
_OPTIONS = tree_metadata.PYTREE_METADATA_OPTIONS


def _entry() -> tree_metadata.ValueMetadataEntry:
  return tree_metadata.ValueMetadataEntry(value_type='np.ndarray')


def _align(item, value_metadata_tree, restore_args=None, **kwargs):
  return restore_structure_utils.validate_and_align_restore_structures(
      item=item,
      serialized_item=tree_metadata.serialize_tree(item, _OPTIONS),
      value_metadata_tree=value_metadata_tree,
      restore_args=restore_args,
      partial_restore=kwargs.get('partial_restore', False),
      support_rich_types=_OPTIONS.support_rich_types,
  )


class ValidateAndAlignRestoreStructuresTest(absltest.TestCase):

  def test_no_item_restores_metadata_tree(self):
    metadata = {'a': _entry(), 'b': {'c': _entry()}}

    item, aligned_metadata, restore_args = _align(None, metadata)

    self.assertIs(item, metadata)
    self.assertIs(aligned_metadata, metadata)
    self.assertIsNone(restore_args)

  def test_matching_structures(self):
    item = {'a': np.zeros(2), 'b': {'c': np.zeros(3)}}
    metadata = {'a': _entry(), 'b': {'c': _entry()}}

    aligned_item, aligned_metadata, _ = _align(item, metadata)

    self.assertIs(aligned_item, item)
    self.assertEqual(aligned_metadata, metadata)

  def test_mismatched_structures_raise(self):
    item = {'a': np.zeros(2)}
    metadata = {'a': _entry(), 'b': _entry()}

    with self.assertRaisesRegex(ValueError, 'partial_restore=True'):
      _align(item, metadata)

  def test_placeholder_for_saved_leaf(self):
    item = {'a': np.zeros(2), 'b': PLACEHOLDER}
    metadata = {'a': _entry(), 'b': _entry()}

    _, aligned_metadata, _ = _align(item, metadata)

    self.assertEqual(aligned_metadata['a'], _entry())
    self.assertIs(aligned_metadata['b'], PLACEHOLDER)

  def test_placeholder_for_unsaved_leaf(self):
    item = {'a': np.zeros(2), 'b': PLACEHOLDER}
    metadata = {'a': _entry()}

    _, aligned_metadata, _ = _align(item, metadata)

    self.assertEqual(aligned_metadata, {'a': _entry(), 'b': PLACEHOLDER})

  def test_placeholder_with_unexpected_leaf_raises(self):
    # Scalar leaves, matching the handler tests: `format_tree_diff` cannot
    # format array leaves.
    item = {'a': 0, 'b': PLACEHOLDER, 'c': 0}
    metadata = {'a': _entry(), 'b': _entry()}

    with self.assertRaisesRegex(ValueError, 'do not match'):
      _align(item, metadata)

  def test_partial_restore_trims_metadata_and_restore_args(self):
    item = {'a': np.zeros(2)}
    metadata = {'a': _entry(), 'b': {'c': _entry()}}
    restore_args = {'a': type_handlers.RestoreArgs()}

    aligned_item, aligned_metadata, aligned_restore_args = _align(
        item, metadata, restore_args, partial_restore=True
    )

    self.assertIs(aligned_item, item)
    self.assertEqual(aligned_metadata, {'a': _entry()})
    self.assertEqual(aligned_restore_args, restore_args)

  def test_partial_restore_allows_unsaved_leaves(self):
    item = {'a': np.zeros(2), 'z': np.zeros(2)}
    metadata = {'a': _entry(), 'b': _entry()}

    aligned_item, aligned_metadata, _ = _align(
        item, metadata, partial_restore=True
    )

    self.assertIs(aligned_item, item)
    self.assertEqual(aligned_metadata['a'], _entry())
    self.assertNotIn('b', aligned_metadata)

if __name__ == '__main__':
  absltest.main()
