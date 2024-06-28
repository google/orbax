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

"""Test for utils.py."""

from typing import Mapping, Sequence

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import flax
import jax
import optax
from orbax.checkpoint import test_utils
from orbax.checkpoint import tree as tree_utils


class UtilsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(
        self.create_tempdir(name='checkpointing_test').full_path
    )

  @parameterized.parameters(
      ({'a': 1, 'b': {'c': {}, 'd': 2}}, {('a',): 1, ('b', 'd'): 2}),
      ({'x': ['foo', 'bar']}, {('x', '0'): 'foo', ('x', '1'): 'bar'}),
  )
  def test_to_flat_dict(self, tree, expected):
    self.assertDictEqual(expected, tree_utils.to_flat_dict(tree))

  @parameterized.parameters(
      ({'a': 1, 'b': {'d': 2}}, {('a',): 1, ('b', 'd'): 2}),
      ({'x': ['foo', 'bar']}, {('x', '0'): 'foo', ('x', '1'): 'bar'}),
      ({'a': 1, 'b': 2}, {('b',): 2, ('a',): 1}),
  )
  def test_from_flat_dict(self, expected, flat_dict):
    empty = jax.tree.map(lambda _: 0, expected)
    self.assertDictEqual(
        expected, tree_utils.from_flat_dict(flat_dict, target=empty)
    )

  @parameterized.parameters(
      ({'a': 1, 'b': {'d': 2}}, {('a',): 1, ('b', 'd'): 2}),
      ({'a': 1, 'b': 2}, {('b',): 2, ('a',): 1}),
  )
  def test_from_flat_dict_without_target(self, expected, flat_dict):
    self.assertDictEqual(expected, tree_utils.from_flat_dict(flat_dict))

  @parameterized.parameters(
      ({'a': 1, 'b': {'d': 2}}, {'a': 1, 'b/d': 2}),
      ({'a': 1, 'b': 2}, {'b': 2, 'a': 1}),
      ({'a': {'b': {'c': 1}}}, {'a/b/c': 1}),
  )
  def test_from_flat_dict_with_sep(self, expected, flat_dict):
    self.assertDictEqual(
        expected, tree_utils.from_flat_dict(flat_dict, sep='/')
    )

  def test_serialize(self):
    tree = {'a': 1, 'b': {'c': {'d': 2}}, 'e': [1, {'x': 5, 'y': 7}, [9, 10]]}
    serialized = tree_utils.serialize_tree(tree, keep_empty_nodes=True)
    self.assertDictEqual(tree, serialized)
    deserialized = tree_utils.deserialize_tree(
        serialized, target=tree, keep_empty_nodes=True
    )
    test_utils.assert_tree_equal(self, tree, deserialized)

  def test_serialize_empty(self):
    serialized = tree_utils.serialize_tree({}, keep_empty_nodes=True)
    test_utils.assert_tree_equal(self, {}, serialized)
    deserialized = tree_utils.deserialize_tree(
        serialized, target={}, keep_empty_nodes=True
    )
    test_utils.assert_tree_equal(self, {}, deserialized)

  def test_serialize_empty_no_keep_empty(self):
    with self.assertRaises(ValueError):
      tree_utils.serialize_tree({}, keep_empty_nodes=False)

  def test_serialize_single_element(self):
    tree = {}
    serialized = tree_utils.serialize_tree(12345, keep_empty_nodes=True)
    test_utils.assert_tree_equal(self, 12345, serialized)
    deserialized = tree_utils.deserialize_tree(
        serialized, target=tree, keep_empty_nodes=True
    )
    test_utils.assert_tree_equal(self, 12345, deserialized)

  def test_serialize_list(self):
    tree = [1, {'a': 2}, [3, 4]]
    serialized = tree_utils.serialize_tree(tree, keep_empty_nodes=True)
    self.assertListEqual(tree, serialized)
    deserialized = tree_utils.deserialize_tree(
        serialized, target=tree, keep_empty_nodes=True
    )
    test_utils.assert_tree_equal(self, tree, deserialized)

  def test_serialize_filters_empty(self):
    tree = {'a': 1, 'b': None, 'c': {}, 'd': [], 'e': optax.EmptyState()}
    serialized = tree_utils.serialize_tree(tree, keep_empty_nodes=False)
    self.assertDictEqual({'a': 1}, serialized)
    deserialized = tree_utils.deserialize_tree(
        serialized, target=tree, keep_empty_nodes=False
    )
    test_utils.assert_tree_equal(self, tree, deserialized)

  def test_serialize_class(self):
    @flax.struct.dataclass
    class Foo(flax.struct.PyTreeNode):
      a: int
      b: Mapping[str, str]
      c: Sequence[optax.EmptyState]
      d: Sequence[Mapping[str, str]]

    foo = Foo(
        1,
        {'a': 'b', 'c': 'd'},
        [optax.EmptyState(), optax.EmptyState()],
        [{}, {'x': 'y'}, None],
    )
    serialized = tree_utils.serialize_tree(foo, keep_empty_nodes=True)
    expected = {
        'a': 1,
        'b': {'a': 'b', 'c': 'd'},
        'c': [optax.EmptyState(), optax.EmptyState()],
        'd': [{}, {'x': 'y'}, None],
    }
    self.assertDictEqual(expected, serialized)
    deserialized = tree_utils.deserialize_tree(
        serialized, target=foo, keep_empty_nodes=False
    )
    test_utils.assert_tree_equal(self, foo, deserialized)

  def test_serialize_nested_class(self):
    @flax.struct.dataclass
    class Foo(flax.struct.PyTreeNode):
      a: int

    nested = {
        'x': Foo(a=1),
        'y': {'z': Foo(a=2)},
    }
    serialized = tree_utils.serialize_tree(nested, keep_empty_nodes=True)
    expected = {
        'x': dict(a=1),
        'y': {'z': dict(a=2)},
    }
    self.assertDictEqual(expected, serialized)

  @parameterized.parameters(
      (1, True),
      (dict(), True),
      ({}, True),
      ({'a': {}}, False),
      ([], True),
      ([[]], False),
      (None, True),
      ((1, 2), False),
  )
  def test_is_empty_or_leaf(self, value, expected):
    self.assertEqual(expected, tree_utils.is_empty_or_leaf(value))


if __name__ == '__main__':
  absltest.main()
