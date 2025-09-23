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

from types import NoneType  # pylint: disable=g-importing-member
from typing import Any, Callable, Dict, Iterable, Iterator, List, Sequence, Tuple
from typing import TypeVar


# TODO(wangpeng): Use the `type Tree[T] = T | List[Tree[T]] | ...` syntax once
#   Python 3.12 is fully available in .
T = TypeVar("T")
Tree = T | List["Tree"] | Tuple["Tree", ...] | NoneType | Dict[str, "Tree"]


def tuple_or_list_constructor(a: Tuple[Any, ...] | List[Any]):
  if isinstance(a, tuple):
    return tuple
  else:
    return list


T1 = TypeVar("T1")
T2 = TypeVar("T2")


def tree_map(f: Callable[[T1], T2], tree: Tree[T1]) -> Tree[T2]:
  if isinstance(tree, (tuple, list)):
    return tuple_or_list_constructor(tree)(tree_map(f, x) for x in tree)
  elif isinstance(tree, dict):
    tree: Dict[str, Tree[T1]]
    return {k: tree_map(f, v) for k, v in tree.items()}
  elif tree is None:
    return None
  else:
    return f(tree)


def assert_tree(assert_leaf: Callable[[Any], None], obj: Any) -> None:
  """Checks that an `Any` object is a valid `Tree`.

  If `assert_leaf` checks the type of the leaf, the caller can safely infer
  the type parameter of the `Tree` after `assert_tree` passes. For example,
  if `assert_leaf` is `lambda x: assert isinstance(x, str)`, after `assert_tree`
  passes, one can safely claim that `obj` is a `Tree[str]`.

  Args:
    assert_leaf: a function that checks that a leaf is valid. When traversing
      `obj` as a tree, any non-list/tuple/dict/None node will be passed to
      `assert_leaf`.
    obj: the object to check.
  """
  if isinstance(obj, (tuple, list)):
    for x in obj:
      assert_tree(assert_leaf, x)
  elif isinstance(obj, dict):
    obj: Dict[Any, Any]
    for v in obj.values():
      assert_tree(assert_leaf, v)
  elif obj is None:
    pass
  else:
    assert_leaf(obj)


T7 = TypeVar("T7")


def flatten_lists(lists: Sequence[List[T7]]) -> List[T7]:
  return sum(lists, [])


T4 = TypeVar("T4")


def flatten(tree: Tree[T4]) -> List[T4]:
  if isinstance(tree, (tuple, list)):
    return flatten_lists(list(flatten(x) for x in tree))
  elif isinstance(tree, dict):
    tree: Dict[str, Tree[T4]]
    # Sorts by key order (as opposed to insertion order like `OrderedDict`).
    return flatten_lists(list(flatten(v) for _, v in sorted(tree.items())))
  elif tree is None:
    return []
  else:
    return [tree]


T5 = TypeVar("T5")


def unflatten_with_iterator(tree: Tree[Any], it: Iterator[T5]) -> Tree[T5]:
  """Unflattens a sequence from an iterator.

  After unflattening, the iterator is allowed to have some left-over elements.

  Args:
    tree: the target tree pattern.
    it: an iterator.

  Returns:
    The result tree.
  """
  if isinstance(tree, (tuple, list)):
    elems = []
    for x in tree:
      elems.append(unflatten_with_iterator(x, it))
    return tuple_or_list_constructor(tree)(elems)
  elif isinstance(tree, dict):
    tree: Dict[str, Any]
    pairs = []
    # Sorts by key order (as opposed to insertion order like `OrderedDict`).
    for k, v in sorted(tree.items()):
      pairs.append((k, unflatten_with_iterator(v, it)))
    return dict(pairs)
  elif tree is None:
    return None
  else:
    return next(it)


T6 = TypeVar("T6")


def unflatten(tree: Tree[Any], leaves: Iterable[T6]) -> Tree[T6]:
  it = iter(leaves)
  result = unflatten_with_iterator(tree, it)
  try:
    next(it)
  except StopIteration:
    return result
  raise ValueError("After unflattening, there are still leaves left.")


def prune_tree(tree: Tree[Any], wanted_node_type: Any) -> Tree[Any]:
  """Prunes the tree by removing unwanted leaves."""
  if isinstance(tree, (tuple, list)):
    return tuple_or_list_constructor(tree)(
        prune_tree(x, wanted_node_type) for x in tree
    )
  elif isinstance(tree, dict):
    tree: Dict[str, Tree[Any]]
    return {k: prune_tree(v, wanted_node_type) for k, v in tree.items()}
  elif isinstance(tree, wanted_node_type):
    return tree
  else:
    # If the node type is None or not wanted, we return None.
    return None
