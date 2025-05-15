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

"""Test for tree structure utils module."""

from typing import Any, NamedTuple

from absl.testing import absltest
from absl.testing import parameterized
import flax
import jax
import numpy as np
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.tree import structure_utils


Diff = structure_utils.Diff
PyTree = structure_utils.PyTree


class Record(NamedTuple):
  first: Any
  second: Any


def _as_string_equality(a, b) -> bool:
  if a is None or b is None:
    return a == b
  return str(a) == str(b)


class TreeDifferenceTest(parameterized.TestCase):

  @parameterized.parameters(
      (
          {'a': 100, 'b': [200, {'c': 300}]},
          {'a': 1, 'b': [2, {'c': 3}]},
          {'a': 1, 'b': [2, {'c': 3}]},
      ),
      (
          {'a': 100, 'b': [200, {'c': 300}]},
          {'a': 1, 'b': [2, {'c': 3, 'd': 4}], 'e': 5},
          {'a': 1, 'b': [2, {'c': 3}]},
      ),
      (
          {'a': [100, 200], 'b': {'c': 300}},
          {'a': (1, 2), 'b': {'c': 3, 'd': 4}},
          {'a': [1, 2], 'b': {'c': 3}},
      ),
      (
          {},
          {'a': 1, 'b': 2},
          {},
      ),
      (
          {},
          {},
          {},
      ),
  )
  def test_tree_trim_valid(
      self, template: PyTree, structure: PyTree, expected: PyTree
  ):
    self.assertDictEqual(
        expected, structure_utils.tree_trim(template, structure)
    )

  def test_doc_example(self):
    first = {'x': [1, 2, 3], 'y': 'same', 'z': {'p': {}, 'q': 4, 'r': 6}}
    second = {'x': [1, 3], 'y': 'same', 'z': {'p': {}, 'q': 5, 's': 6}}
    actual = structure_utils.tree_difference(first, second)
    expected = {
        'x': Diff(lhs=(list, 3), rhs=(list, 2)),
        'z': {
            'r': Diff(lhs=6, rhs=None),
            's': Diff(lhs=None, rhs=6),
        },
    }
    self.assertEqual(actual, expected)

  def test_uses_leaves_equal_predicate(self):
    first = {'x': [1, 2, 3], 'y': 'same', 'z': {'p': {}, 'q': 4, 'r': 6}}
    second = {'x': [1, 2, 3], 'y': 'same', 'z': {'p': {}, 'q': 4, 's': 6}}
    leaf_equality_fn = lambda x, y: x == y

    actual = structure_utils.tree_difference(
        first, second, leaves_equal=leaf_equality_fn
    )
    expected = {'z': {'r': Diff(lhs=6, rhs=None), 's': Diff(lhs=None, rhs=6)}}
    self.assertEqual(actual, expected)

  @parameterized.parameters([
      ((),),
      ([],),
      ({},),
      ((1, 2, 3),),  # tuples of same elements.
      (
          (1, 2, 3),
          (2, 3, 4),
      ),  # tuples of same length using type equality.
      ([1, 2, 3],),
      (
          [1, 2, 3],
          [2, 3, 4],
      ),  # lists of same length using type equality.
      ({'one': 1, 'two': 2, 'three': 3},),
      (Record(first=1, second=2),),
  ])
  def test_match_returns_none(self, lhs, rhs=None):
    if rhs is None:
      rhs = lhs
    self.assertIsNone(structure_utils.tree_difference(lhs, rhs))

  @parameterized.parameters([
      ('abc', 'abc'),
      (123, 123),
      (123, '123'),
      ('123', 123),
  ])
  def test_match_returns_none_custom_equality_fn(self, lhs, rhs):
    self.assertIsNone(
        structure_utils.tree_difference(
            lhs, rhs, leaves_equal=_as_string_equality
        )
    )

  @parameterized.parameters([('abc', 'bcd'), (123, 234), (123, '123')])
  def test_match_returns_diff_custom_equality_fn(self, lhs, rhs):
    self.assertEqual(
        structure_utils.tree_difference(
            lhs, rhs, leaves_equal=lambda x, y: x == y
        ),
        Diff(lhs=lhs, rhs=rhs),
    )

  def test_named_tuples_are_compared_fieldwise(self):
    self.assertEqual(
        structure_utils.tree_difference(
            Record(first=3, second=4), Record(first=5, second='four')
        ),
        Record(first=None, second=Diff(lhs=4, rhs='four')),
    )

  def test_named_tuples_are_compared_fieldwise_custom_equality_fn(self):
    self.assertEqual(
        structure_utils.tree_difference(
            Record(first='3', second=4),
            Record(first=3, second='four'),
            leaves_equal=_as_string_equality,
        ),
        Record(first=None, second=Diff(lhs=4, rhs='four')),
    )

  @parameterized.product(
      sequence_type=[tuple, list], eq_fn=[None, lambda x, y: x == y]
  )
  def test_length_difference_is_reported_as_pair_of_type_and_length(
      self, sequence_type, eq_fn
  ):
    s = sequence_type
    self.assertEqual(
        structure_utils.tree_difference(
            s([1, 2, 3]), s(['one', 'two']), leaves_equal=eq_fn
        ),
        Diff(lhs=(s, 3), rhs=(s, 2)),
    )

  @parameterized.parameters(tuple, list)
  def test_element_difference_is_none_at_matching_indices(self, sequence_type):
    s = sequence_type
    self.assertEqual(
        structure_utils.tree_difference(s([1, 2, 3]), s([1, 'two', 2])),
        s([None, Diff(lhs=2, rhs='two'), None]),
    )

  @parameterized.parameters(tuple, list)
  def test_element_difference_is_none_at_matching_indices_custom_equality_fn(
      self, sequence_type
  ):
    s = sequence_type
    leaf_equality_fn = _as_string_equality

    self.assertEqual(
        structure_utils.tree_difference(
            s([1, 2, 3]), s([1, 'two', '3']), leaves_equal=leaf_equality_fn
        ),
        s([None, Diff(lhs=2, rhs='two'), None]),
    )

  def test_mapping_difference_only_contains_mismatched_elements(self):
    self.assertEqual(
        structure_utils.tree_difference(
            {'one': 1, 'two': 2, 'three': 3},
            {'one': 1, 'two': 'two', 'three': 3},
        ),
        {'two': Diff(lhs=2, rhs='two')},
    )

  def test_difference_with_custom_pytree_node(self):
    @flax.struct.dataclass
    class NewModel(flax.struct.PyTreeNode):
      x: Any
      y: Any

    self.assertEqual(
        structure_utils.tree_difference(
            NewModel(
                x=jax.ShapeDtypeStruct(shape=(16,), dtype=np.int64),
                y=Ellipsis,
            ),
            NewModel(
                x=jax.ShapeDtypeStruct(shape=(16,), dtype=np.int64),
                y=jax.ShapeDtypeStruct(shape=(16,), dtype=np.float64),
            ),
        ),
        NewModel(
            x=None,
            y=Diff(
                lhs=Ellipsis,
                rhs=jax.ShapeDtypeStruct(shape=(16,), dtype=np.float64),
            ),
        ),
    )

  def test_uses_is_leaf_to_recognize_leaves(self):

    def custom_is_leaf(x: PyTree) -> bool:
      return isinstance(x, int) or isinstance(x, list)

    first = {
        'x': 0,
        'y': [0],
        'z': {'p': 0, 'q': [0, 1]},
        'w': [0, [1, 2], [3], 4],
    }
    second = {
        'x': 1,
        'y': [1, 2],
        'z': {'p': 1, 'q': [2]},
        'w': [5, [6], [7], 8],
    }

    self.assertEqual(
        structure_utils.tree_difference(first, second),
        {
            'y': Diff(lhs=(list, 1), rhs=(list, 2)),
            'z': {
                'q': Diff(lhs=(list, 2), rhs=(list, 1)),
            },
            'w': [None, Diff(lhs=(list, 2), rhs=(list, 1)), None, None],
        },
    )

    self.assertIsNone(
        structure_utils.tree_difference(first, second, is_leaf=custom_is_leaf)
    )


class TreeTrimTest(parameterized.TestCase):

  @parameterized.parameters(
      (
          {'a': 100, 'b': [200, {'c': 300}]},
          {'a': 1, 'b': [2, {'c': 3}]},
          {'a': 1, 'b': [2, {'c': 3}]},
      ),
      (
          {'a': 100, 'b': [200, {'c': 300}]},
          {'a': 1, 'b': [2, {'c': 3, 'd': 4}], 'e': 5},
          {'a': 1, 'b': [2, {'c': 3}]},
      ),
      (
          {'a': [100, 200], 'b': {'c': 300}},
          {'a': (1, 2), 'b': {'c': 3, 'd': 4}},
          {'a': [1, 2], 'b': {'c': 3}},
      ),
      (
          {},
          {'a': 1, 'b': 2},
          {},
      ),
      (
          {},
          {},
          {},
      ),
  )
  def test_tree_trim_valid(
      self, template: PyTree, structure: PyTree, expected: PyTree
  ):
    self.assertDictEqual(
        expected, structure_utils.tree_trim(template, structure)
    )

  @parameterized.parameters(
      int,
      float,
      complex,
      str,
      bytes,
      bool,
      np.int32,
      np.float32,
  )
  def test_tree_trim_leaves(self, val_type: Any):
    val = val_type()
    self.assertEqual(val, structure_utils.tree_trim(val, val))

  def test_tree_trim_extra_path_in_structure(self):
    structure = {'a': 1, 'b': [2, {'c': 3}]}

    template = {'a': 100, 'b': [200, {'c': 300, 'd': 400}], 'e': 500}
    with self.assertRaisesRegex(ValueError, 'Path "b.1.d" exists in template'):
      structure_utils.tree_trim(template, structure)

    template = {'a': 100, 'b': [200, {'c': 300}], 'e': 500}
    with self.assertRaisesRegex(ValueError, 'Path "e" exists in template'):
      structure_utils.tree_trim(template, structure)

  def test_tree_trim_empty_template(self):
    structure = {}
    template = {'a': 1, 'b': 2}
    with self.assertRaisesRegex(ValueError, 'Path "a" exists in template'):
      structure_utils.tree_trim(template, structure)

  def test_tree_trim_mismatch_structure(self):
    structure = 10
    template = {'a': 20}
    with self.assertRaisesRegex(ValueError, 'Path "a" exists in template'):
      structure_utils.tree_trim(template, structure)

  def test_tree_trim_mismatch_template(self):
    structure = {'a': 10}
    template = 20
    with self.assertRaisesRegex(ValueError, 'Path "" exists in template'):
      structure_utils.tree_trim(template, structure)

  def test_tree_trim_v2_custom_node(self):
    @flax.struct.dataclass
    class SimpleData:
      x: int
      y: str

    structure = {'data': SimpleData(x=1, y='hello'), 'other': 10}
    template = {'data': SimpleData(x=100, y='world')}
    expected = {'data': SimpleData(x=1, y='hello')}

    result = structure_utils.tree_trim(template, structure)
    test_utils.assert_tree_equal(self, expected, result)


if __name__ == '__main__':
  absltest.main()
