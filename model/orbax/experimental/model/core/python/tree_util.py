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

"""Generic tree (similar to JAX's PyTree) (i.e. nested structures)."""

from typing import Any, Callable
from typing import TypeVar

from jax import tree_util as jax_tree_util


T = TypeVar('T')
# TODO(b/427248531): Use the `type Tree[T] = T | list[Tree[T]] | ...`
# syntax once Python 3.12 is fully available in .
Tree = T | list['Tree'] | tuple['Tree', ...] | dict[str, 'Tree'] | None


def assert_tree(assert_leaf: Callable[[Any], None], tree: Any) -> None:
  """Checks that an `Any` object is a valid `Tree`.

  If `assert_leaf` checks the type of the leaf, the caller can safely infer
  the type parameter of the `Tree` after `assert_tree` passes. For example,
  if `assert_leaf` is `lambda x: assert isinstance(x, str)`, after `assert_tree`
  passes, one can safely claim that `obj` is a `Tree[str]`.

  Args:
    assert_leaf: A function that checks that a leaf is valid. When traversing
      `tree` as a tree, any non-list/tuple/dict/None node will be passed to
      `assert_leaf`.
    tree: The tree where all leaves are to be checked by `assert_leaf`.
  """
  leaves = jax_tree_util.tree_leaves(tree)
  for x in leaves:
    assert_leaf(x)
