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

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import chex
from orbax.experimental.model.tf2obm import tree_util

Tree = tree_util.Tree


@dataclasses.dataclass
class NonJaxDataclass:
  x: int
  y: int


@chex.dataclass
class ChexDataclass:
  x: int
  y: int


class TreeUtilTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._tree1: Tree[int] = (
        1,
        [],
        (),
        None,
        {},
        (2, 3, ()),
        {"4": 4, "5": 5},
        [6, None, 7],
        (8, ({}, 9, {"empty": [()]}, 10)),
    )
    self._flat1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    self._tree2: Tree[int] = (
        10,
        [],
        (),
        None,
        {},
        (20, 30, ()),
        {"4": 40, "5": 50},
        [60, None, 70],
        (80, ({}, 90, {"empty": [()]}, 100)),
    )
    self._map_fn = lambda x: x * 10
    self._flat2 = [self._map_fn(x) for x in self._flat1]

    self._tree_dc: Tree[int | NonJaxDataclass] = (
        1,
        NonJaxDataclass(x=2, y=3),
        [4, NonJaxDataclass(x=5, y=6)],
        {"7": NonJaxDataclass(x=8, y=9)},
        None,
    )
    self._tree_chex_dc: Tree[int | ChexDataclass] = (
        1,
        ChexDataclass(x=2, y=3),
        [4, ChexDataclass(x=5, y=6)],
        {"7": ChexDataclass(x=8, y=9)},
        None,
    )
    self._flat_dc = [1, 2, 3, 4, 5, 6, 8, 9]
    self._tree_chex_dc_mapped: Tree[int | ChexDataclass] = (
        10,
        ChexDataclass(x=20, y=30),
        [40, ChexDataclass(x=50, y=60)],
        {"7": ChexDataclass(x=80, y=90)},
        None,
    )
    self._flat_dc_mapped = [self._map_fn(x) for x in self._flat_dc]
    self._flat_dc_as_leaf = [
        1,
        NonJaxDataclass(x=2, y=3),
        4,
        NonJaxDataclass(x=5, y=6),
        NonJaxDataclass(x=8, y=9),
    ]
    self._tree_dc_mapped = (
        10,
        NonJaxDataclass(x=20, y=30),
        [40, NonJaxDataclass(x=50, y=60)],
        {"7": NonJaxDataclass(x=80, y=90)},
        None,
    )
    self._flat_dc_as_leaf_mapped = [
        10,
        NonJaxDataclass(x=20, y=30),
        40,
        NonJaxDataclass(x=50, y=60),
        NonJaxDataclass(x=80, y=90),
    ]

  @parameterized.named_parameters(
      ("basic", "_tree1", "_flat1"),
      ("chex_dataclass", "_tree_chex_dc", "_flat_dc"),
      ("non_jax_dataclass", "_tree_dc", "_flat_dc_as_leaf"),
  )
  def test_flatten(self, tree_attr, expected_flat_attr):
    tree = getattr(self, tree_attr)
    expected_flat = getattr(self, expected_flat_attr)
    flat = tree_util.flatten(tree)
    self.assertEqual(flat, expected_flat)

  @parameterized.named_parameters(
      ("basic", "_tree1", "_flat2", "_tree2"),
      (
          "chex_dataclass",
          "_tree_chex_dc",
          "_flat_dc_mapped",
          "_tree_chex_dc_mapped",
      ),
      (
          "non_jax_dataclass",
          "_tree_dc",
          "_flat_dc_as_leaf_mapped",
          "_tree_dc_mapped",
      ),
  )
  def test_unflatten(self, structure_attr, flat_attr, expected_tree_attr):
    structure = getattr(self, structure_attr)
    flat = getattr(self, flat_attr)
    expected_tree = getattr(self, expected_tree_attr)
    tree = tree_util.unflatten(structure, flat)
    self.assertEqual(tree, expected_tree)

  def test_unflatten_with_too_many_leaves(self):
    leaves = tree_util.flatten(self._tree1)
    leaves.append(42)  # Add an extra leaf.
    with self.assertRaisesRegex(
        ValueError, "After unflattening, there are still leaves left."
    ):
      tree_util.unflatten(self._tree1, leaves)

  @parameterized.named_parameters(
      ("basic", "_tree1", "_tree2", None),
      (
          "chex_dataclass",
          "_tree_chex_dc",
          "_tree_chex_dc_mapped",
          None,
      ),
      (
          "non_jax_dataclass",
          "_tree_dc",
          None,
          (TypeError, "unsupported operand"),
      ),
  )
  def test_tree_map(self, tree_attr, expected_tree_attr, raises):
    tree = getattr(self, tree_attr)
    if raises:
      error_type, error_regex = raises
      with self.assertRaisesRegex(error_type, error_regex):
        tree_util.tree_map(self._map_fn, tree)
    else:
      expected_tree = getattr(self, expected_tree_attr)
      result_tree = tree_util.tree_map(self._map_fn, tree)
      self.assertEqual(result_tree, expected_tree)

  @parameterized.named_parameters(
      ("leaf", 1),
      ("list", [1, 2, 3]),
      ("tuple", (1, 2, 3)),
      ("dict", {"a": 1, "b": 2}),
      ("nested", [1, (2, {"c": 3}), None]),
      ("none", None),
  )
  def test_assert_tree_success(self, tree):
    tree_util.assert_tree(lambda x: self.assertIsInstance(x, int), tree)

  @parameterized.named_parameters(
      ("leaf", "a"),
      ("list", [1, "a", 3]),
      ("tuple", (1, 2, "a")),
      ("dict", {"a": 1, "b": "a"}),
      ("nested", [1, (2, {"c": "a"}), None]),
  )
  def test_assert_tree_failure(self, tree):
    with self.assertRaises(AssertionError):
      tree_util.assert_tree(lambda x: self.assertIsInstance(x, int), tree)


if __name__ == "__main__":
  absltest.main()
