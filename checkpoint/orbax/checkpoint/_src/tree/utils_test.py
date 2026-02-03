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

"""Test for utils module."""

from typing import Any, Mapping, Sequence

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import flax
import jax
import jax.tree_util as jtu
import numpy as np
import optax
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.testing import test_tree_utils
from orbax.checkpoint._src.tree import utils as tree_utils


PyTree = tree_utils.PyTree


@flax.struct.dataclass
class FlaxRecord:
  alpha: Any
  beta: Any


# TODO: b/365169723 - Add tests: PyTreeMetadataOptions.support_rich_types=True.
class SerializeTreeTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(
        self.create_tempdir(name='serialize_tree_test').full_path
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

  @parameterized.parameters(
      (1, True, False),
      (np.zeros(1), True, False),
      (dict(), True, True),
      ({}, True, True),
      ({'a': {}}, False, False),
      ([], True, True),
      ([[]], False, False),
      ([tuple()], False, False),
      ([dict()], False, False),
      ([{}], False, False),
      ([1], False, False),
      (tuple(), True, True),
      ((tuple(),), False, False),
      (([],), False, False),
      (({},), False, False),
      ((dict(),), False, False),
      ((1,), False, False),
      (None, True, True),
      ((1, 2), False, False),
      (test_tree_utils.EmptyNamedTuple(), True, True),
      (optax.EmptyState(), True, True),
      (test_tree_utils.MuNu(mu=None, nu=None), False, False),
      (test_tree_utils.MyEmptyClass(), True, False),
      (test_tree_utils.MyClass(), True, False),
      (test_tree_utils.MyClass(a=None, b=None), True, False),
      (test_tree_utils.MyClass(a=None, b=np.zeros(1)), True, False),
      (test_tree_utils.MyEmptyChex(), True, True),
      (test_tree_utils.MyChex(), False, False),
      (
          test_tree_utils.MyChex(my_jax_array=None, my_np_array=None),
          False,
          False,
      ),
      (
          test_tree_utils.MyChex(my_jax_array=None, my_np_array=np.zeros(1)),
          False,
          False,
      ),
      (test_tree_utils.MyEmptyFlax(), True, True),
      (test_tree_utils.MyFlax(), False, False),
      (
          test_tree_utils.MyFlax(
              my_jax_array=None, my_nested_mapping=None, my_sequence=None
          ),
          False,
          False,
      ),
      (test_tree_utils.MyFlax(my_nested_mapping={'a': 1}), False, False),
      (test_tree_utils.MyEmptyDataClass(), True, False),
      (test_tree_utils.MyDataClass(), True, False),
      (
          test_tree_utils.MyDataClass(
              my_jax_array=None,
              my_np_array=None,
              my_empty_dataclass=None,
              my_chex=None,
          ),
          True,
          False,
      ),
  )
  def test_is_empty_or_leaf(
      self,
      value: Any,
      expected_is_empty_or_leaf: bool,
      expected_is_empty_node: bool,
  ):
    self.assertEqual(
        expected_is_empty_or_leaf, tree_utils.is_empty_or_leaf(value)
    )
    self.assertEqual(expected_is_empty_node, tree_utils.is_empty_node(value))

  @parameterized.parameters(
      (test_tree_utils.EmptyNamedTuple(),),
      (test_tree_utils.IntegerNamedTuple(None, None),),
      (test_tree_utils.IntegerNamedTuple(1, 2),),
  )
  def test_named_tuple_type_detection(self, nt):
    self.assertTrue(tree_utils.isinstance_of_namedtuple(nt))
    self.assertTrue(tree_utils.issubclass_of_namedtuple(type(nt)))

  @parameterized.parameters(
      ((1, 2),),
      ({'a': 1, 'b': 2},),
      ([1, 2],),
  )
  def test_non_named_tuple_type_detection(self, nt):
    self.assertFalse(tree_utils.isinstance_of_namedtuple(nt))
    self.assertFalse(tree_utils.issubclass_of_namedtuple(type(nt)))


class LookUpPyTreeKeyTest(parameterized.TestCase):

  def test_retrieves_element_of_internal_node(self):
    self.assertEqual(
        tree_utils.look_up_pytree_key(
            [[0], [10], [20]],
            jtu.SequenceKey(1),
        ),
        [10],
    )
    self.assertEqual(
        tree_utils.look_up_pytree_key(
            {'a': [0], 'b': [100], 'c': [200]},
            jtu.DictKey('b'),
        ),
        [100],
    )
    self.assertEqual(
        tree_utils.look_up_pytree_key(
            FlaxRecord(alpha=[0, 10, 20], beta={'a': 100, 'b': 200}),
            jtu.FlattenedIndexKey(1),
        ),
        {'a': 100, 'b': 200},
    )
    self.assertEqual(
        tree_utils.look_up_pytree_key(
            FlaxRecord(alpha=[0, 10, 20], beta={'a': 100, 'b': 200}),
            jtu.GetAttrKey('beta'),
        ),
        {'a': 100, 'b': 200},
    )

  def test_raises_key_error_if_key_is_not_present(self):
    with self.subTest('sequence'):
      with self.assertRaises(KeyError):
        tree_utils.look_up_pytree_key(
            [[0], [10], [20]],
            jtu.SequenceKey(4),
        )
    with self.subTest('dict'):
      with self.assertRaises(KeyError):
        tree_utils.look_up_pytree_key(
            {'a': [0], 'b': [100], 'c': [200]},
            jtu.DictKey('d'),
        )
    with self.subTest('flattened_index'):
      with self.assertRaises(KeyError):
        tree_utils.look_up_pytree_key(
            FlaxRecord(alpha=[0, 10, 20], beta={'a': 100, 'b': 200}),
            jtu.FlattenedIndexKey(2),
        )
    with self.subTest('attr'):
      with self.assertRaises(KeyError):
        tree_utils.look_up_pytree_key(
            FlaxRecord(alpha=[0, 10, 20], beta={'a': 100, 'b': 200}),
            jtu.GetAttrKey('gamma'),
        )


class PyTreePathAsTreePathTest(parameterized.TestCase):

  def test_converts_pytree_path_to_tree_path(self):
    structure = {
        'a': [0, 10],
        'b': FlaxRecord(alpha=[65], beta=(66,)),
    }
    paths_and_values, _ = jtu.tree_flatten_with_path(structure)
    legacy_paths = [
        tree_utils.pytree_path_as_tree_path(p) for p, _ in paths_and_values
    ]

    self.assertSameStructure(
        [
            ('a', 0),
            ('a', 1),
            ('b', 'alpha', 0),
            ('b', 'beta', 0),
        ],
        legacy_paths,
    )


class SelectByPyTreePathTest(parameterized.TestCase):

  def test_select_dict(self):
    tree = {'a': 1, 'b': 2}
    self.assertEqual(
        tree_utils.select_by_pytree_path(tree, (jtu.DictKey('a'),)), 1
    )
    self.assertEqual(
        tree_utils.select_by_pytree_path(tree, (jtu.DictKey('b'),)), 2
    )

  def test_select_sequence(self):
    tree = [10, 20]
    self.assertEqual(
        tree_utils.select_by_pytree_path(tree, (jtu.SequenceKey(0),)), 10
    )
    self.assertEqual(
        tree_utils.select_by_pytree_path(tree, (jtu.SequenceKey(1),)), 20
    )

  def test_select_named_tuple(self):
    tree = test_tree_utils.IntegerNamedTuple(1, 2)
    self.assertEqual(
        tree_utils.select_by_pytree_path(tree, (jtu.GetAttrKey('x'),)), 1
    )
    self.assertEqual(
        tree_utils.select_by_pytree_path(tree, (jtu.GetAttrKey('y'),)), 2
    )

  def test_select_object(self):
    tree = FlaxRecord(alpha=1, beta=2)
    self.assertEqual(
        tree_utils.select_by_pytree_path(tree, (jtu.GetAttrKey('alpha'),)), 1
    )
    self.assertEqual(
        tree_utils.select_by_pytree_path(tree, (jtu.GetAttrKey('beta'),)), 2
    )

  def test_select_nested(self):
    tree = {'a': FlaxRecord(alpha={'x': 10}, beta=20)}
    self.assertEqual(
        tree_utils.select_by_pytree_path(
            tree, (jtu.DictKey('a'), jtu.GetAttrKey('alpha'), jtu.DictKey('x'))
        ),
        10,
    )
    self.assertEqual(
        tree_utils.select_by_pytree_path(
            tree, (jtu.DictKey('a'), jtu.GetAttrKey('beta'))
        ),
        20,
    )

  def test_select_empty_path(self):
    tree = {'a': 1}
    self.assertEqual(tree_utils.select_by_pytree_path(tree, ()), tree)

  def test_select_dict_raises_error(self):
    tree = {'a': 1}
    with self.assertRaisesRegex(ValueError, 'Path .* does not exist'):
      tree_utils.select_by_pytree_path(tree, (jtu.DictKey('b'),))

  def test_select_sequence_raises_error(self):
    tree = [10, 20]
    with self.assertRaisesRegex(ValueError, 'Path .* does not exist'):
      tree_utils.select_by_pytree_path(tree, (jtu.SequenceKey(2),))

  def test_select_named_tuple_raises_error(self):
    tree = test_tree_utils.IntegerNamedTuple(1, 2)
    with self.assertRaisesRegex(ValueError, 'Path .* does not exist'):
      tree_utils.select_by_pytree_path(tree, (jtu.GetAttrKey('z'),))

  def test_select_object_raises_error(self):
    tree = FlaxRecord(alpha=1, beta=2)
    with self.assertRaisesRegex(ValueError, 'Path .* does not exist'):
      tree_utils.select_by_pytree_path(tree, (jtu.GetAttrKey('gamma'),))


class SelectByTreePathTest(parameterized.TestCase):

  def test_select_dict(self):
    tree = {'a': 1, 'b': 2}
    self.assertEqual(tree_utils.select_by_tree_path(tree, ('a',)), 1)
    self.assertEqual(tree_utils.select_by_tree_path(tree, ('b',)), 2)

  def test_select_sequence(self):
    tree = [10, 20]
    self.assertEqual(tree_utils.select_by_tree_path(tree, (0,)), 10)
    self.assertEqual(tree_utils.select_by_tree_path(tree, (1,)), 20)

  def test_select_named_tuple(self):
    tree = test_tree_utils.IntegerNamedTuple(1, 2)
    self.assertEqual(tree_utils.select_by_tree_path(tree, ('x',)), 1)
    self.assertEqual(tree_utils.select_by_tree_path(tree, ('y',)), 2)

  def test_select_object(self):
    tree = FlaxRecord(alpha=1, beta=2)
    self.assertEqual(tree_utils.select_by_tree_path(tree, ('alpha',)), 1)
    self.assertEqual(tree_utils.select_by_tree_path(tree, ('beta',)), 2)

  def test_select_nested(self):
    tree = {'a': FlaxRecord(alpha={'x': 10}, beta=20)}
    self.assertEqual(
        tree_utils.select_by_tree_path(tree, ('a', 'alpha', 'x')), 10
    )
    self.assertEqual(tree_utils.select_by_tree_path(tree, ('a', 'beta')), 20)

  def test_select_empty_path(self):
    tree = {'a': 1}
    self.assertEqual(tree_utils.select_by_tree_path(tree, ()), tree)

  def test_select_dict_raises_error(self):
    tree = {'a': 1}
    with self.assertRaisesRegex(ValueError, 'Path .* does not exist'):
      tree_utils.select_by_tree_path(tree, ('b',))

  def test_select_sequence_raises_error(self):
    tree = [10, 20]
    with self.assertRaisesRegex(ValueError, 'Path .* does not exist'):
      tree_utils.select_by_tree_path(tree, (2,))

  def test_select_named_tuple_raises_error(self):
    tree = test_tree_utils.IntegerNamedTuple(1, 2)
    with self.assertRaisesRegex(ValueError, 'Path .* does not exist'):
      tree_utils.select_by_tree_path(tree, ('z',))

  def test_select_object_raises_error(self):
    tree = FlaxRecord(alpha=1, beta=2)
    with self.assertRaisesRegex(ValueError, 'Path .* does not exist'):
      tree_utils.select_by_tree_path(tree, ('gamma',))


if __name__ == '__main__':
  absltest.main()
