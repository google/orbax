# Copyright 2024 The Orbax Authors.
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

from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
from orbax.checkpoint._src.metadata import tree as tree_metadata_lib
from orbax.checkpoint._src.serialization import type_handlers
from orbax.checkpoint._src.serialization import types
from orbax.checkpoint._src.testing import test_tree_utils
from orbax.checkpoint._src.tree import utils as tree_utils
from orbax.checkpoint._src.metadata import tree as tree_metadata_lib


def _to_param_infos(
    tree: Any,
    pytree_metadata_options: tree_metadata_lib.PyTreeMetadataOptions,
):
  return jax.tree.map(
      # Other properties are not relevant.
      lambda x: types.ParamInfo(
          value_typestr=types.get_param_typestr(
              x,
              type_handlers.GLOBAL_TYPE_HANDLER_REGISTRY,
              pytree_metadata_options,
          )
      ),
      tree,
      is_leaf=tree_utils.is_empty_or_leaf,
  )


class InternalTreeMetadataEntryTest(parameterized.TestCase):

  @parameterized.product(
      test_pytree=test_tree_utils.TEST_PYTREES,
      pytree_metadata_options=[
          tree_metadata_lib.PyTreeMetadataOptions(support_rich_types=False),
          tree_metadata_lib.PyTreeMetadataOptions(support_rich_types=True),
      ],
  )
  def test_as_nested_tree(
      self,
      test_pytree: test_tree_utils.TestPyTree,
      pytree_metadata_options: tree_metadata_lib.PyTreeMetadataOptions,
  ):
    tree = test_pytree.provide_tree()
    original_internal_tree_metadata = (
        tree_metadata_lib.InternalTreeMetadata.build(
            param_infos=_to_param_infos(tree, pytree_metadata_options),
            pytree_metadata_options=pytree_metadata_options,
        )
    )
    json_object = original_internal_tree_metadata.to_json()
    restored_internal_tree_metadata = (
        tree_metadata_lib.InternalTreeMetadata.from_json(
            json_object, pytree_metadata_options
        )
    )

    if pytree_metadata_options.support_rich_types:
      expected_tree_metadata = (
          test_pytree.expected_nested_tree_metadata_with_rich_types
      )
    else:
      expected_tree_metadata = test_pytree.expected_nested_tree_metadata
    restored_tree_metadata = restored_internal_tree_metadata.as_nested_tree()
    chex.assert_trees_all_equal(restored_tree_metadata, expected_tree_metadata)

  @parameterized.product(
      test_pytree=test_tree_utils.TEST_PYTREES,
      pytree_metadata_options_switch=[
          (
              tree_metadata_lib.PyTreeMetadataOptions(support_rich_types=False),
              tree_metadata_lib.PyTreeMetadataOptions(support_rich_types=True),
          ),
          (
              tree_metadata_lib.PyTreeMetadataOptions(support_rich_types=True),
              tree_metadata_lib.PyTreeMetadataOptions(support_rich_types=False),
          ),
      ],
  )
  def test_switching_between_support_rich_types(
      self,
      test_pytree: test_tree_utils.TestPyTree,
      pytree_metadata_options_switch: tuple[
          tree_metadata_lib.PyTreeMetadataOptions,
          tree_metadata_lib.PyTreeMetadataOptions,
      ],
  ):
    write_pytree_metadata_options, read_pytree_metadata_options = (
        pytree_metadata_options_switch
    )
    if (
        write_pytree_metadata_options.support_rich_types
        == read_pytree_metadata_options.support_rich_types
    ):
      self.fail(
          'This test is only meant for scenarios when support_rich_types will'
          ' be different between write and read.'
      )
    else:
      expected_tree_metadata = test_pytree.expected_nested_tree_metadata
    tree = test_pytree.provide_tree()

    original_internal_tree_metadata = (
        tree_metadata_lib.InternalTreeMetadata.build(
            param_infos=_to_param_infos(tree, write_pytree_metadata_options),
            pytree_metadata_options=write_pytree_metadata_options,
        )
    )
    json_object = original_internal_tree_metadata.to_json()
    restored_internal_tree_metadata = (
        tree_metadata_lib.InternalTreeMetadata.from_json(
            json_object, read_pytree_metadata_options
        ))

    restored_tree_metadata = restored_internal_tree_metadata.as_nested_tree()
    chex.assert_trees_all_equal(restored_tree_metadata, expected_tree_metadata)


class TreeMetadataTest(parameterized.TestCase):

  def _check_tree_property(
      self, expected_tree: Any, metadata: tree_metadata_lib.TreeMetadata
  ):
    if isinstance(expected_tree, dict):
      self.assertDictEqual(metadata.tree, expected_tree)
    elif isinstance(expected_tree, list):
      self.assertListEqual(metadata.tree, expected_tree)
    elif isinstance(expected_tree, tuple):
      self.assertTupleEqual(metadata.tree, expected_tree)
    else:
      raise ValueError(f'Unsupported tree type: {type(expected_tree)}')

  @parameterized.parameters(({'a': 1, 'b': 2},), ([1, 2],), ((1, 2),))
  def test_properties(self, tree):
    custom = {'foo': 1}
    metadata = tree_metadata_lib.TreeMetadata(tree=tree, custom=custom)
    self.assertDictEqual(metadata.custom, custom)
    self._check_tree_property(tree, metadata)


  @parameterized.parameters(({'a': 1, 'b': 2},), ([1, 2],), ((1, 2),))
  def test_tree_map(self, tree):
    custom = {'foo': 1}
    metadata = tree_metadata_lib.TreeMetadata(tree=tree, custom=custom)
    metadata = jax.tree.map(lambda x: x + 1, metadata)
    self.assertDictEqual(metadata.custom, custom)
    self._check_tree_property(jax.tree.map(lambda x: x + 1, tree), metadata)

  @parameterized.parameters(({'a': 1, 'b': 2},), ([1, 2],), ((1, 2),))
  def test_multiple_tree_map(self, tree):
    metadata = tree_metadata_lib.TreeMetadata(tree=tree)
    with self.assertRaises(ValueError):
      _ = jax.tree.map(lambda x, y: x + y, metadata, tree)

  @parameterized.parameters(({'a': 1, 'b': 2},), ([1, 2],), ((1, 2),))
  def test_accessors(self, tree):
    metadata = tree_metadata_lib.TreeMetadata(tree=tree)
    self.assertLen(metadata, 2)
    if isinstance(tree, dict):
      self.assertIn('a', metadata)
      self.assertIn('b', metadata)
      self.assertNotIn('c', metadata)
      self.assertEqual(metadata['a'], 1)
      self.assertEqual(metadata['b'], 2)
      self.assertEqual(metadata.get('a'), 1)
      self.assertEqual(metadata.get('b'), 2)
      self.assertIsNone(metadata.get('c'))
      with self.assertRaises(KeyError):
        _ = metadata['c']
    else:
      self.assertNotIn(0, metadata)
      self.assertIn(1, metadata)
      self.assertIn(2, metadata)
      self.assertEqual(metadata[0], 1)
      self.assertEqual(metadata[1], 2)
      self.assertEqual(metadata[-1], 2)
      self.assertEqual(metadata.get(0), 1)
      self.assertEqual(metadata.get(1), 2)
      self.assertEqual(metadata.get(-1), 2)
      self.assertIsNone(metadata.get(2))
      with self.assertRaises(IndexError):
        _ = metadata[2]

  @parameterized.parameters(({'a': 1, 'b': 2},), ([1, 2],), ((1, 2),))
  def test_tree_flatten(self, tree):
    metadata = tree_metadata_lib.TreeMetadata(tree=tree)
    flat, treedef = jax.tree.flatten(metadata)
    self.assertSequenceEqual(flat, [1, 2])
    unflat = jax.tree.unflatten(treedef, flat)
    self.assertIsInstance(unflat, tree_metadata_lib.TreeMetadata)
    self._check_tree_property(tree, unflat)

  @parameterized.parameters(({'a': 1, 'c': {'b': 2}},), ([1, 2],), ((1, 2),))
  def test_with_path(self, tree):
    metadata = tree_metadata_lib.TreeMetadata(tree=tree)
    metadata = jax.tree_util.tree_map_with_path(lambda _, x: x + 1, metadata)
    self._check_tree_property(
        jax.tree_util.tree_map_with_path(lambda _, x: x + 1, tree), metadata
    )

    flat_with_keys, treedef = jax.tree_util.tree_flatten_with_path(metadata)
    keys, values = zip(*flat_with_keys)
    expected_keys = (
        list(tree.keys())
        if isinstance(tree, dict)
        else [str(i) for i in range(len(tree))]
    )
    self.assertSequenceEqual(
        expected_keys, [tree_utils.tuple_path_from_keypath(k)[0] for k in keys]
    )
    self.assertSequenceEqual(values, [2, 3])

    flat, _ = jax.tree.flatten(metadata)
    unflat = jax.tree.unflatten(treedef, flat)
    self.assertIsInstance(unflat, tree_metadata_lib.TreeMetadata)
    self._check_tree_property(jax.tree.map(lambda x: x + 1, tree), unflat)


if __name__ == '__main__':
  absltest.main()
