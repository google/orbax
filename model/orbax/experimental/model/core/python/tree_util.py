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

from typing import Any, Callable, Dict, Iterable, Iterator, List, Sequence, Tuple
from typing import TypeVar

from jax import tree_util as jax_tree_util


T = TypeVar('T')
U = TypeVar('U')
# TODO(b/427248531): Use the `type Tree[T] = T | List[Tree[T]] | ...`
# syntax once Python 3.12 is fully available in .
Tree = T | List['Tree'] | Tuple['Tree', ...] | Dict[str, 'Tree'] | None


def _tuple_or_list_constructor(a: Tuple[Any, ...] | List[Any]):
  if isinstance(a, tuple):
    return tuple
  else:
    return list


# TODO(b/456768655): This function may not be needed. It is being kept for now,
# and can be deleted if it is confirmed to be unnecessary.
def tree_map(f: Callable[[T], U], tree: Tree[T]) -> Tree[U]:
  """Recursively applies `f` to all leaves of `Tree`."""
  if isinstance(tree, (tuple, list)):
    return _tuple_or_list_constructor(tree)(tree_map(f, x) for x in tree)
  elif isinstance(tree, dict):
    tree: Dict[str, Tree[T]]
    return {k: tree_map(f, v) for k, v in tree.items()}
  elif tree is None:
    return None
  else:
    return f(tree)


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


def flatten_lists(lists: Sequence[List[T]]) -> List[T]:
  """Flattens a sequence of lists into a single list."""
  return sum(lists, [])


# TODO(b/456768655): This function may not be needed. It is being kept for now,
# and can be deleted if it is confirmed to be unnecessary.
def flatten(tree: Tree[T]) -> List[T]:
  """Recursively flattens leaves of `Tree` into a list."""
  if isinstance(tree, (tuple, list)):
    return flatten_lists(list(flatten(x) for x in tree))
  elif isinstance(tree, dict):
    tree: Dict[str, Tree[T]]
    # Sorts by key order (as opposed to insertion order like `OrderedDict`).
    return flatten_lists(list(flatten(v) for _, v in sorted(tree.items())))
  elif tree is None:
    return []
  else:
    return [tree]


def _unflatten_iter(tree: Tree[Any], it: Iterator[T]) -> Tree[T]:
  """Unflattens a `Tree` from an iterator of leaves."""
  if isinstance(tree, (tuple, list)):
    elems = []
    for x in tree:
      elems.append(_unflatten_iter(x, it))
    return _tuple_or_list_constructor(tree)(elems)
  elif isinstance(tree, dict):
    tree: Dict[str, Any]
    pairs = []
    # Sorts by key order (as opposed to insertion order like `OrderedDict`).
    for k, v in sorted(tree.items()):
      pairs.append((k, _unflatten_iter(v, it)))
    return dict(pairs)
  elif tree is None:
    return None
  else:
    return next(it)


# TODO(b/456768655): This function may not be needed. It is being kept for now,
# and can be deleted if it is confirmed to be unnecessary.
def unflatten(tree: Tree[Any], leaves: Iterable[T]) -> Tree[T]:
  """Unflattens a `Tree` from a sequence of leaves.

  Implies that the leaves have been previously flattened with `flatten` using
  the same tree pattern.

  Args:
    tree: The target tree pattern.
    leaves: A sequence of leaves.

  Returns:
    The result tree matching the original tree pattern but with leaf values
    from the provided sequence of leaves.

  Raises:
    ValueError: If the number of leaves is not equal to the number of nodes in
      the tree.
  """

  it = iter(leaves)
  result = _unflatten_iter(tree, it)

  try:
    next(it)
  except StopIteration:
    return result
  raise ValueError('After unflattening, there are still leaves left')
