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

from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import chex
import flax
import jax
import jax.numpy as jnp
from orbax.checkpoint._src.tree import parts_of

PartsOf = parts_of.PartsOf

PH = parts_of.PLACEHOLDER
X = object()


@chex.dataclass(frozen=True)
class MyState:
  a: Any
  b: Any


@flax.struct.dataclass
class FlaxState:
  w: Any
  b: Any


class PartsOfTest(parameterized.TestCase):

  def test_full_value_returns_contained_tree(self):
    t = MyState(a=(1, 2), b=[3, 4])
    self.assertSameStructure(
        t,
        PartsOf(t, t).full_structure,
    )

  @parameterized.named_parameters([
      ('chex', MyState(a=(X, X), b=[X, X]), MyState(a=(1, 2), b=[PH, PH])),
      ('flax', FlaxState(w=(X, X), b=[X, X]), FlaxState(w=(1, 2), b=[PH, PH])),
  ])
  def test_full_value_raises_if_contained_tree_is_not_complete(
      self,
      template: Any,
      t: Any,
  ):
    with self.assertRaisesRegex(
        parts_of.TreeNotCompleteError, 'Tree is not complete'
    ):
      _ = PartsOf(template, t).full_structure

  @parameterized.named_parameters([('no_jit', False), ('jit', True)])
  def test_overlay_overlays_one_tree_on_another(self, jit=False):
    template = MyState(a=(X, X), b=[X, X])
    t0 = PartsOf(template, MyState(a=(1, 2), b=[PH, PH]))
    t1 = PartsOf(template, MyState(a=(11, PH), b=[33, PH]))

    overlay = jax.jit(PartsOf.overlay) if jit else PartsOf.overlay

    t = overlay(t0, t1)

    leaf = jnp.asarray if jit else lambda x: x
    expected = MyState(a=(leaf(11), leaf(2)), b=[leaf(33), PH])

    self.assertSameStructure(
        expected,
        t.unsafe_structure,
    )

  def test_overlay_in_strict_mode(self):
    template = MyState(a=(X, X), b=[X, X, X])

    with self.subTest('successful_union_of_disjoint_trees'):
      t0 = PartsOf(template, MyState(a=(PH, 2), b=[33, PH, PH]))
      t1 = PartsOf(template, MyState(a=(11, PH), b=[PH, PH, 44]))
      t = t0.overlay(t1, strict=True)
      self.assertSameStructure(
          MyState(a=(11, 2), b=[33, PH, 44]),
          t.unsafe_structure,
      )
    with self.subTest('raises_if_trees_are_not_disjoint'):
      t0 = PartsOf(template, MyState(a=(PH, 2), b=[33, PH, PH]))
      t1 = PartsOf(template, MyState(a=(11, PH), b=[33, PH, 44]))
      with self.assertRaisesRegex(
          ValueError,
          r"keys {\('b', '0'\)} are present in both PartsOf structures",
      ):
        _ = t0.overlay(t1, strict=True)

  def test_subtract_masks_one_tree_with_another(self):
    template = MyState(a=(X, X), b=[X, X])
    t0 = PartsOf(template, MyState(a=(1, 2), b=[PH, PH]))
    t1 = PartsOf(template, MyState(a=(11, PH), b=[33, PH]))
    t = t0 - t1
    self.assertSameStructure(
        MyState(a=(PH, 2), b=[PH, PH]),
        t.unsafe_structure,
    )

  def test_intersect_one_tree_with_another(self):
    template = MyState(a=(X, X), b=[X, X, X])
    t0 = PartsOf(template, MyState(a=(1, 2), b=[22, PH, PH]))
    t1 = PartsOf(template, MyState(a=(11, PH), b=[33, PH, 44]))
    t = t0.intersect(t1)
    self.assertSameStructure(
        MyState(a=(1, PH), b=[22, PH, PH]),
        t.unsafe_structure,
    )

  def test_from_full_structure_creates_natural_parts_of_object(self):
    value = MyState(a=(11, 12), b=[13, 14])
    t = PartsOf.from_full_structure(value)
    self.assertSameStructure(
        value,
        t.full_structure,
    )

  @parameterized.product(
      method=(PartsOf.from_full_structure, PartsOf.from_structure)
  )
  def test_from_full_structure_accepts_partial_structure(self, method):
    value = MyState(a=(11, PH), b=[PH, 14])
    t = method(value)
    self.assertSameStructure(
        value,
        t.unsafe_structure,
    )

  def test_like_creates_parts_of_t_with_same_template(self):
    template = MyState(a=(X, X), b=[X, X])

    t0 = PartsOf(template, MyState(a=(1, 2), b=[PH, PH]))
    t1 = PartsOf.like(t0, MyState(a=(PH, PH), b=[30, 40]))
    self.assertSameStructure(
        t0._get_template(),
        t1._get_template(),
    )

  def test_is_empty(self):
    template = MyState(a=(X, X), b=[X, X])
    t = PartsOf(template, MyState(a=(1, 2), b=[PH, PH]))
    self.assertFalse(t.is_empty())
    t = PartsOf(template, MyState(a=(PH, PH), b=[PH, PH]))
    self.assertTrue(t.is_empty())

  def test_like_converts_none_to_placeholder(self):
    template = MyState(a=(X, None), b=[X, X])
    t = PartsOf(template, MyState(a=(1, None), b=[3, 4]))
    t_with_none = PartsOf.like(t, MyState(a=(12, None), b=[None, None]))
    self.assertSameStructure(
        t_with_none.unsafe_structure,
        MyState(a=(12, None), b=[PH, PH]),
    )

  def test_filter_values(self):
    template = MyState(a=(X, X), b=[X, X])
    t = PartsOf(template, MyState(a=(1, 2), b=[33, PH]))

    self.assertSameStructure(
        parts_of.filter_values(t, {('a', '0'), ('b', '0')}).unsafe_structure,
        MyState(a=(1, PH), b=[33, PH]),
    )
    self.assertSameStructure(
        parts_of.filter_values(t, {('b', '0')}).unsafe_structure,
        MyState(a=(PH, PH), b=[33, PH]),
    )

  def test_treats_none_as_always_empty_internal_node(self):
    t = PartsOf({'a': None}, {'a': None})
    self.assertSameStructure({'a': None}, t.unsafe_structure)
    self.assertSameStructure({'a': None}, t.full_structure)

    t = PartsOf({'a': None}, ())
    self.assertSameStructure({'a': None}, t.unsafe_structure)
    self.assertSameStructure({'a': None}, t.full_structure)

    t = PartsOf({'a': None}, None)
    self.assertSameStructure({'a': None}, t.unsafe_structure)
    self.assertSameStructure({'a': None}, t.full_structure)

    with self.assertRaisesRegex(
        parts_of.Error, r'not part of the full tree structure'):
      PartsOf({'a': None}, {'a': 1})

  def test_unflatten_raises_if_placeholder_is_encountered(self):
    template = MyState(a=(X, X), b=[X, X])
    t = PartsOf(template, MyState(a=(1, 2), b=[PH, PH]))
    leaves, treedef = jax.tree_util.tree_flatten(t)
    leaves = [PH for _ in leaves]
    with self.assertRaisesRegex(
        parts_of.UnexpectedPlaceholderError,
        'PartsOf tree unflattening discovered a placeholder',
    ):
      _ = jax.tree.unflatten(treedef, leaves)

  def test_structure_hash_match_with_different_values(self):
    template = MyState(a=(X, X), b=[X, X])
    t1 = PartsOf(template, MyState(a=(1, 2), b=[PH, PH]))
    t2 = PartsOf(template, MyState(a=(2, 1), b=[PH, PH]))
    self.assertEqual(t1.structure_hash, t2.structure_hash)

  def test_structure_hash_mismatch_with_different_placeholders(self):
    template = MyState(a=(X, X), b=[X, X])
    t1 = PartsOf(template, MyState(a=(1, 2), b=[PH, PH]))
    t2 = PartsOf(template, MyState(a=(PH, PH), b=[1, 1]))
    self.assertNotEqual(t1.structure_hash, t2.structure_hash)


class MapLeavesTest(parameterized.TestCase):

  def test_maps_with_placeholders_over_trees_with_compatible_templates(self):
    template = MyState(a=(X, X), b=[X, X])

    t0 = PartsOf(template, MyState(a=(1, 2), b=[PH, PH]))
    t1 = PartsOf(template, MyState(a=(10, PH), b=[30, PH]))

    f = lambda a, b: a + b if PH not in (a, b) else PH

    t = parts_of.map_leaves(f, t0, t1)
    self.assertSameStructure(
        MyState(a=(11, PH), b=[PH, PH]),
        t.unsafe_structure,
    )

  def test_maps_with_path(self):
    template = MyState(a=(X, X), b=[X, X])

    t0 = PartsOf(template, MyState(a=(1, 2), b=[15, PH]))
    t1 = PartsOf(template, MyState(a=(10, PH), b=[30, PH]))

    f = lambda path, a, b: a + b if PH not in (a, b) and path[0] == 'a' else PH

    t = parts_of.map_leaves_with_path(f, t0, t1)
    self.assertSameStructure(
        MyState(a=(11, PH), b=[PH, PH]),
        t.unsafe_structure,
    )

  def test_raises_if_f_returns_non_leaf_value(self):
    template = MyState(a=(X, X), b=[X, X])

    t0 = PartsOf(template, MyState(a=(1, 2), b=[PH, PH]))
    t1 = PartsOf(template, MyState(a=(10, PH), b=[30, PH]))

    f = lambda *ab: ab

    with self.assertRaisesRegex(parts_of.Error, r'returned non-leaf value'):
      parts_of.map_leaves(f, t0, t1)

  def test_raises_if_f_returns_none(self):
    template = MyState(a=(X, X), b=[X, X])

    t0 = PartsOf(template, MyState(a=(1, 2), b=[PH, PH]))
    t1 = PartsOf(template, MyState(a=(10, PH), b=[30, PH]))

    f = lambda a, b: a + b if PH not in (a, b) else None

    with self.assertRaisesRegex(parts_of.Error, r'returned None'):
      parts_of.map_leaves(f, t0, t1)

  def test_raises_if_structures_are_incompatible(self):
    template = MyState(a=(X, X), b=[X, X])
    template2 = MyState(a=X, b=[X, X])

    t0 = PartsOf(template, MyState(a=(1, 2), b=[PH, PH]))
    t1 = PartsOf(template, MyState(a=(10, PH), b=[30, PH]))
    t2 = PartsOf(template2, MyState(a=1, b=[30, PH]))

    f = lambda a, b, c: PH
    with self.assertRaisesRegex(parts_of.Error, r'Mismatched templates'):
      parts_of.map_leaves(f, t0, t1, t2)


class ValueKeyFromPathTest(parameterized.TestCase):

  def test_value_key_from_path(self):
    template = MyState(a=(X, X), b={'c': X, 'd': X})

    t = PartsOf(template, MyState(a=(1, PH), b={'c': PH, 'd': 33}))

    flat_t, _ = jax.tree_util.tree_flatten_with_path(t)
    for path, value in flat_t:
      self.assertIs(t._present[parts_of.value_key_from_path(path)], value)


class SelectTest(parameterized.TestCase):

  def test_select_extracts_nested_dict(self):
    template = MyState(a=(X, X), b={'c': {'d': X}})
    t = PartsOf(template, MyState(a=(1, PH), b={'c': {'d': 2}}))

    path = ('b', 'c')
    select_template = {'d': X}
    result = t.select(path, select_template)

    self.assertEqual(result.full_structure, {'d': 2})

  def test_select_with_placeholders(self):
    template = MyState(a=(X, X), b={'c': {'d': X}})
    t = PartsOf(template, MyState(a=(1, 1), b={'c': {'d': PH}}))

    path = ('b', 'c')
    select_template = {'d': X}
    result = t.select(path, select_template)

    self.assertEqual(result.unsafe_structure, {'d': PH})
    with self.assertRaises(parts_of.TreeNotCompleteError):
      _ = result.full_structure

  def test_select_missing_key_returns_empty(self):
    template = {'a': {'b': X}}
    t = PartsOf(template, {'a': {'b': 1}})

    result = t.select(('nonexistent',), {'c': X})

    self.assertTrue(result.is_empty())


class PartitionTest(parameterized.TestCase):

  def test_partition(self):
    template = MyState(a=(X, X), b={'c': X, 'd': X, 'e': X})
    t = PartsOf(template, MyState(a=(1, 2), b={'c': 4, 'd': 3, 'e': PH}))
    classifier = lambda x: x % 2 == 0
    self.assertSameStructure(
        parts_of.partition(classifier, t),
        {
            False: PartsOf(
                template, MyState(a=(1, PH), b={'c': PH, 'd': 3, 'e': PH})
            ),
            True: PartsOf(
                template, MyState(a=(PH, 2), b={'c': 4, 'd': PH, 'e': PH})
            ),
        },
    )


if __name__ == '__main__':
  absltest.main()
