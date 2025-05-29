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

from orbax.experimental.model.core.python import tree_util
from orbax.experimental.model.core.python.tree_util import Tree

from absl.testing import absltest


class TreeUtilTest(absltest.TestCase):

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

  def test_flatten(self):
    flat: list[int] = tree_util.flatten(self._tree1)
    self.assertEqual(flat, self._flat1)

  def test_unflatten(self):
    tree = tree_util.unflatten(self._tree1, self._flat2)
    self.assertEqual(tree, self._tree2)

  def test_tree_map(self):
    tree = tree_util.tree_map(self._map_fn, self._tree1)
    self.assertEqual(tree, self._tree2)

  def test_assert_tree(self):
    err_msg = "Expected int, got type: "

    def assert_int(x: Any) -> None:
      if not isinstance(x, int):
        raise ValueError(f"{err_msg}{type(x)}")

    tree_util.assert_tree(assert_int, self._tree1)
    wrong_tree = (1, {"k": (3, [2, "a"])})
    self.assertRaisesRegex(
        ValueError,
        err_msg,
        lambda: tree_util.assert_tree(assert_int, wrong_tree),
    )


if __name__ == "__main__":
  absltest.main()
