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

from typing import Any, NamedTuple

from absl.testing import absltest
from absl.testing import parameterized
import chex
from etils import epath
import jax
import numpy as np
from orbax.checkpoint._src.metadata import tree as tree_metadata_lib
from orbax.checkpoint._src.serialization import type_handler_registry
from orbax.checkpoint._src.serialization import types
from orbax.checkpoint._src.testing import test_tree_utils
from orbax.checkpoint._src.tree import utils as tree_utils


InternalTreeMetadata = tree_metadata_lib.InternalTreeMetadata
TreeMetadata = tree_metadata_lib.TreeMetadata
_TreeMetadataImpl = tree_metadata_lib._TreeMetadataImpl


def _to_param_infos(
    tree: Any,
    pytree_metadata_options: (
        tree_metadata_lib.PyTreeMetadataOptions | None
    ) = None,
):
  pytree_metadata_options = pytree_metadata_options or (
      tree_metadata_lib.PYTREE_METADATA_OPTIONS
  )
  return jax.tree.map(
      # Other properties are not relevant.
      lambda x: types.ParamInfo(
          name='',
          parent_dir=epath.Path(''),
          value_typestr=type_handler_registry.get_param_typestr(
              x,
              type_handler_registry.GLOBAL_TYPE_HANDLER_REGISTRY,
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
    original_internal_tree_metadata = InternalTreeMetadata.build(
        param_infos=_to_param_infos(tree, pytree_metadata_options),
        pytree_metadata_options=pytree_metadata_options,
    )
    json_object = original_internal_tree_metadata.to_json()
    restored_internal_tree_metadata = InternalTreeMetadata.from_json(
        json_object, pytree_metadata_options
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
      support_rich_types=(True, False),
  )
  def test_integer_dict_keys(self, support_rich_types):
    tree = {0: 'a', 1: 'b'}
    pytree_metadata_options = tree_metadata_lib.PyTreeMetadataOptions(
        support_rich_types=support_rich_types
    )
    original_internal_tree_metadata = InternalTreeMetadata.build(
        param_infos=_to_param_infos(
            tree, pytree_metadata_options=pytree_metadata_options
        ),
        pytree_metadata_options=pytree_metadata_options,
    )
    json_object = original_internal_tree_metadata.to_json()
    restored_internal_tree_metadata = InternalTreeMetadata.from_json(
        json_object, pytree_metadata_options=pytree_metadata_options
    )
    restored_tree_metadata = restored_internal_tree_metadata.as_nested_tree()
    self.assertSequenceEqual([0, 1], list(restored_tree_metadata.keys()))

  @parameterized.product(
      support_rich_types=(True, False),
  )
  def test_missing_key_python_type(self, support_rich_types):
    # JSON representation of a tree {'1': 1} with missing key_python_type using
    # both data-rich and non data-rich format.
    json_object = {
        'tree_metadata': {
            "('1',)": {
                'key_metadata': ({'key': '1', 'key_type': 2},),
                'value_metadata': {
                    'value_type': 'scalar',
                    'skip_deserialize': False,
                },
            }
        },
        'use_zarr3': False,
        'store_array_data_equal_to_fill_value': False,
        'custom_metadata': None,
    }
    if support_rich_types:
      json_object['value_metadata_tree'] = (
          '{"1": {"value": {"category": "custom",'
          ' "clazz": "ValueMetadataEntry", "data": {"value_type": "scalar",'
          ' "skip_deserialize": false}}}}'
      )
    pytree_metadata_options = tree_metadata_lib.PyTreeMetadataOptions(
        support_rich_types=support_rich_types
    )
    restored_internal_tree_metadata = InternalTreeMetadata.from_json(
        json_object,
        pytree_metadata_options=pytree_metadata_options,
    )
    restored_tree_metadata = restored_internal_tree_metadata.as_nested_tree()
    # Should fall back to string keys when key_python_type is missing.
    self.assertIn('1', restored_tree_metadata)
    self.assertSequenceEqual(['1'], list(restored_tree_metadata.keys()))

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

    original_internal_tree_metadata = InternalTreeMetadata.build(
        param_infos=_to_param_infos(tree, write_pytree_metadata_options),
        pytree_metadata_options=write_pytree_metadata_options,
    )
    json_object = original_internal_tree_metadata.to_json()
    restored_internal_tree_metadata = InternalTreeMetadata.from_json(
        json_object, read_pytree_metadata_options
    )

    restored_tree_metadata = restored_internal_tree_metadata.as_nested_tree()
    chex.assert_trees_all_equal(restored_tree_metadata, expected_tree_metadata)

  @parameterized.parameters(
      (test_tree_utils.MyDataClass(),),
      ([test_tree_utils.MyDataClass()],),
      ({'a': test_tree_utils.MyFlax()},),
  )
  def test_invalid_custom_metadata(self, custom_metadata):
    tree = {'scalar_param': 1}
    with self.assertRaisesRegex(TypeError, 'Failed to encode'):
      InternalTreeMetadata.build(
          param_infos=_to_param_infos(tree), custom_metadata=custom_metadata
      )

  @parameterized.parameters(
      ({'a': 1, 'b': [{'c': 2}, 1]},),
      ([1, [{'c': 2}, 1]],),
  )
  def test_custom_metadata(self, custom_metadata):
    tree = {'scalar_param': 1}
    internal_tree_metadata = InternalTreeMetadata.build(
        param_infos=_to_param_infos(tree), custom_metadata=custom_metadata
    )
    self.assertEqual(internal_tree_metadata.custom_metadata, custom_metadata)

  def test_merge(self):
    tree1 = {'a': 1, 'b': {'c': 2}}
    custom1 = {'x': 10, 'y': {'z': 20}}
    meta1 = InternalTreeMetadata.build(
        param_infos=_to_param_infos(tree1), custom_metadata=custom1
    )

    tree2 = {'b': {'d': 99}, 'e': 4}
    custom2 = {'y': {'w': 30}, 'u': 40}
    meta2 = InternalTreeMetadata.build(
        param_infos=_to_param_infos(tree2), custom_metadata=custom2
    )

    merged_meta = meta1.merge(meta2, overwrite=False)

    # Check custom_metadata is merged.
    # The behavior of merge_trees is to merge dicts recursively, but to not
    # overwrite existing leaf values.
    expected_custom_metadata = {'x': 10, 'y': {'z': 20, 'w': 30}, 'u': 40}
    self.assertEqual(merged_meta.custom_metadata, expected_custom_metadata)

    # Check tree_metadata_entries are merged with overwrite=False.
    expected_merged_tree = {'a': 1, 'b': {'c': 2, 'd': 99}, 'e': 4}
    # Build expected metadata to compare against.
    expected_meta = InternalTreeMetadata.build(
        param_infos=_to_param_infos(expected_merged_tree)
    )
    expected_nested_tree = expected_meta.as_nested_tree()

    merged_tree_metadata = merged_meta.as_nested_tree()
    chex.assert_trees_all_equal(merged_tree_metadata, expected_nested_tree)

  def test_merge_overwrite_true(self):
    tree1 = {'a': 1, 'b': np.int32(2)}
    custom1 = {'x': 10, 'y': 20}
    meta1 = InternalTreeMetadata.build(
        param_infos=_to_param_infos(tree1), custom_metadata=custom1
    )

    tree2 = {'b': np.int64(3), 'c': 4}
    custom2 = {'y': 30, 'z': 40}
    meta2 = InternalTreeMetadata.build(
        param_infos=_to_param_infos(tree2), custom_metadata=custom2
    )

    merged_meta = meta1.merge(meta2, overwrite=True)

    # Check custom_metadata is merged, with tree2 overwriting tree1.
    expected_custom_metadata = {'x': 10, 'y': 30, 'z': 40}
    self.assertEqual(merged_meta.custom_metadata, expected_custom_metadata)

    # Check tree_metadata_entries are merged, with tree2 overwriting tree1.
    # The value of 'b' from tree1 is replaced by tree2's.
    expected_merged_tree = {'a': 1, 'b': np.int64(3), 'c': 4}
    expected_meta = InternalTreeMetadata.build(
        param_infos=_to_param_infos(expected_merged_tree)
    )
    expected_nested_tree = expected_meta.as_nested_tree()

    merged_tree_metadata = merged_meta.as_nested_tree()
    chex.assert_trees_all_equal(merged_tree_metadata, expected_nested_tree)

  def test_merge_overwrite_false_raises_error(self):
    tree1 = {'a': 1, 'b': np.int32(2)}
    meta1 = InternalTreeMetadata.build(param_infos=_to_param_infos(tree1))
    tree2 = {'b': np.int64(3), 'c': 4}
    meta2 = InternalTreeMetadata.build(param_infos=_to_param_infos(tree2))

    with self.assertRaisesRegex(ValueError, 'exists in both metadata trees'):
      meta1.merge(meta2, overwrite=False)

  @parameterized.product(overwrite=[True, False])
  def test_merge_with_rich_types(self, overwrite: bool):
    pytree_metadata_options = tree_metadata_lib.PyTreeMetadataOptions(
        support_rich_types=True
    )
    tree1 = {'a': 1, 'b': test_tree_utils.IntegerNamedTuple(x=1, y=2)}
    meta1 = InternalTreeMetadata.build(
        param_infos=_to_param_infos(tree1, pytree_metadata_options),
        pytree_metadata_options=pytree_metadata_options,
    )
    tree2 = {'b': test_tree_utils.IntegerNamedTuple(x=3, y=4), 'c': 5}
    meta2 = InternalTreeMetadata.build(
        param_infos=_to_param_infos(tree2, pytree_metadata_options),
        pytree_metadata_options=pytree_metadata_options,
    )

    if not overwrite:
      with self.assertRaises(ValueError):
        meta1.merge(meta2, overwrite=overwrite)
      return

    merged_meta = meta1.merge(meta2, overwrite=True)

    self.assertIsNotNone(merged_meta.value_metadata_tree)

    expected_merged_tree = {
        'a': 1,
        'b': test_tree_utils.IntegerNamedTuple(x=3, y=4),
        'c': 5,
    }
    expected_meta = InternalTreeMetadata.build(
        param_infos=_to_param_infos(
            expected_merged_tree, pytree_metadata_options
        ),
        pytree_metadata_options=pytree_metadata_options,
    )
    # This is the important check for value_metadata_tree
    chex.assert_trees_all_equal(
        merged_meta.value_metadata_tree, expected_meta.value_metadata_tree
    )

    # This check is also good to have.
    merged_tree_metadata = merged_meta.as_nested_tree()
    expected_nested_tree = expected_meta.as_nested_tree()
    chex.assert_trees_all_equal(merged_tree_metadata, expected_nested_tree)


class NestedNamedTuple(NamedTuple):
  a: int
  b: int
  c: dict[str, int]


class TreeMetadataTest(parameterized.TestCase):

  def _check_tree_property(self, expected_tree: Any, metadata: TreeMetadata):
    if tree_utils.isinstance_of_namedtuple(expected_tree):
      self.assertTrue(tree_utils.isinstance_of_namedtuple(metadata.tree))
    elif isinstance(expected_tree, dict):
      self.assertDictEqual(metadata.tree, expected_tree)
    elif isinstance(expected_tree, list):
      self.assertListEqual(metadata.tree, expected_tree)
    elif isinstance(expected_tree, tuple):
      self.assertTupleEqual(metadata.tree, expected_tree)
    else:
      raise ValueError(f'Unsupported tree type: {type(expected_tree)}')

  @parameterized.parameters(({'a': 1, 'b': 2},), ([1, 2],), ((1, 2),))
  def test_properties(self, tree):
    custom_metadata = {'foo': 1}
    metadata = _TreeMetadataImpl(tree=tree, custom_metadata=custom_metadata)
    self.assertDictEqual(metadata.custom_metadata, custom_metadata)
    self._check_tree_property(tree, metadata)

  @parameterized.parameters(
      (1,),
      (test_tree_utils.MyDataClass(),),
      (test_tree_utils.MyFlax(),),
  )
  def test_invalid_tree_type(self, tree):
    with self.assertRaises(ValueError):
      _TreeMetadataImpl(tree=tree)


  @parameterized.parameters(({'a': 1, 'b': 2},), ([1, 2],), ((1, 2),))
  def test_multiple_tree_map(self, tree):
    metadata = _TreeMetadataImpl(tree=tree)
    with self.assertRaises(ValueError):
      _ = jax.tree.map(lambda x, y: x + y, metadata, tree)

  def test_dict_accessors(self):
    tree = {'a': 1, 'b': {'c': 2}}
    metadata = _TreeMetadataImpl(tree=tree)
    self.assertLen(metadata, 2)
    self.assertIn('a', metadata)
    self.assertIn('b', metadata)
    self.assertNotIn('c', metadata)
    self.assertEqual(metadata['a'], 1)
    self.assertEqual(metadata['b'], {'c': 2})
    self.assertEqual(metadata['b']['c'], 2)
    self.assertEqual(metadata.get('a'), 1)
    self.assertEqual(metadata.get('b'), {'c': 2})
    self.assertEqual(metadata.get('b').get('c'), 2)
    self.assertIsNone(metadata.get('c'))
    with self.assertRaises(KeyError):
      _ = metadata['c']

  @parameterized.parameters(([1, 2, [3]],), ((1, 2, (3,)),))
  def test_sequence_accessors(self, tree):
    metadata = _TreeMetadataImpl(tree=tree)
    self.assertLen(metadata, 3)
    self.assertNotIn(0, metadata)
    self.assertIn(1, metadata)
    self.assertIn(2, metadata)
    self.assertNotIn(3, metadata)
    self.assertEqual(metadata[0], 1)
    self.assertEqual(metadata[1], 2)
    self.assertLen(metadata[2], 1)
    self.assertEqual(metadata[2][0], 3)
    self.assertEqual(metadata.get(0), 1)
    self.assertEqual(metadata.get(1), 2)
    self.assertEqual(metadata.get(2)[0], 3)
    self.assertIsNone(metadata.get(3))
    with self.assertRaises(IndexError):
      _ = metadata[3]

  @parameterized.parameters(
      ({'a': 1, 'b': 2, 'c': {'d': [{'x': 3, 'y': 4}]}},),
      ([1, 2, [3, 4]],),
      ((1, 2, (3, 4)),),
      (NestedNamedTuple(a=1, b=2, c={'d': [3, 4]}),),
      ({},),
      ([],),
      ({'a': []},),
      ({'a': None},),
      ({'a': {}},),
      ([{}, {}],),
      (tuple([]),),
      (test_tree_utils.EmptyNamedTuple(),),
  )
  def test_tree_map(self, tree):
    custom_metadata = {'foo': 1}
    metadata = _TreeMetadataImpl(tree=tree, custom_metadata=custom_metadata)
    metadata = jax.tree.map(lambda x: x + 1, metadata)
    self.assertDictEqual(metadata.custom_metadata, custom_metadata)
    self._check_tree_property(jax.tree.map(lambda x: x + 1, tree), metadata)

  @parameterized.parameters(
      ({'a': 1, 'b': 2, 'c': {'d': [{'x': 3, 'y': 4}]}},),
      ([1, 2, [3, 4]],),
      ((1, 2, (3, 4)),),
      (NestedNamedTuple(a=1, b=2, c={'d': [3, 4]}),),
  )
  def test_tree_flatten(self, tree):
    metadata = _TreeMetadataImpl(tree=tree)
    flat, treedef = jax.tree.flatten(metadata)
    self.assertSequenceEqual(flat, [1, 2, 3, 4])
    unflat = jax.tree.unflatten(treedef, flat)
    self.assertIsInstance(unflat, TreeMetadata)
    self._check_tree_property(tree, unflat)

  @parameterized.parameters(
      ({'a': 1, 'b': 2, 'c': {'d': [{'x': 3, 'y': 4}]}},),
      ([1, 2, [3, 4]],),
      ((1, 2, (3, 4)),),
      (NestedNamedTuple(a=1, b=2, c={'d': [3, 4]}),),
  )
  def test_with_path(self, tree):
    metadata = _TreeMetadataImpl(tree=tree)
    metadata = jax.tree_util.tree_map_with_path(lambda _, x: x + 1, metadata)
    self._check_tree_property(
        jax.tree_util.tree_map_with_path(lambda _, x: x + 1, tree), metadata
    )

    flat_with_keys, treedef = jax.tree_util.tree_flatten_with_path(metadata)
    keys, values = zip(*flat_with_keys)
    if isinstance(tree, dict) or tree_utils.isinstance_of_namedtuple(tree):
      expected_keys = ['a', 'b', 'c', 'c']
    else:
      expected_keys = ['0', '1', '2', '2']
    self.assertSequenceEqual(
        expected_keys, [tree_utils.tuple_path_from_keypath(k)[0] for k in keys]
    )
    self.assertSequenceEqual(values, [2, 3, 4, 5])

    flat, _ = jax.tree.flatten(metadata)
    unflat = jax.tree.unflatten(treedef, flat)
    self.assertIsInstance(unflat, TreeMetadata)
    self._check_tree_property(jax.tree.map(lambda x: x + 1, tree), unflat)

  @parameterized.parameters(
      ({},),
      ([],),
      ({'a': []},),
      ({'a': None},),
      ({'a': {}},),
      ([{}, {}],),
      (tuple([]),),
      (test_tree_utils.EmptyNamedTuple(),),
  )
  def test_flatten_empty_trees(self, tree):
    metadata = _TreeMetadataImpl(tree=tree)
    flat, treedef = jax.tree.flatten(metadata)
    self.assertEmpty(flat)
    unflat = jax.tree.unflatten(treedef, flat)
    self.assertIsInstance(unflat, TreeMetadata)
    self._check_tree_property(tree, unflat)

  @parameterized.parameters(
      ({},),
      ([],),
      ({'a': []},),
      ({'a': None},),
      ({'a': {}},),
      ([{}, {}],),
      (tuple([]),),
      (test_tree_utils.EmptyNamedTuple(),),
  )
  def test_with_path_empty_trees(self, tree):
    metadata = _TreeMetadataImpl(tree=tree)
    metadata = jax.tree_util.tree_map_with_path(lambda _, x: x + 1, metadata)
    self._check_tree_property(
        jax.tree_util.tree_map_with_path(lambda _, x: x + 1, tree), metadata
    )
    flat_with_keys, _ = jax.tree_util.tree_flatten_with_path(metadata)
    self.assertEmpty(flat_with_keys)


if __name__ == '__main__':
  absltest.main()
