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

"""Generic tree functions.

Partial subtitute for orbax.experimental.model.core.tree_util that supports
dataclasses registered with JAX.
"""

import dataclasses
from typing import Any, Callable, Dict, Iterable, Iterator, List, Sequence, Tuple, TypeVar

from jax import tree_util
from orbax.experimental.model import core as obm

Tree = obm.tree_util.Tree


def _is_registered_dataclass(obj: Any) -> bool:
  """Checks if an object is a dataclass instance registered with JAX.

  It performs a tree flattening operation to observe if JAX considers the
  dataclass instance as an internal node or a leaf. If it's an internal node,
  it means the dataclass is registered with jax.tree_util.

  Args:
    obj: The object to check.

  Returns:
    True if obj is a dataclass instance registered with jax.tree_util.
  """
  if not dataclasses.is_dataclass(obj) or isinstance(obj, type):
    return False
  # If a dataclass is not registered with JAX, it's treated as a leaf node,
  # in which case tree_leaves(obj) would return [obj]. If it is registered,
  # it's treated as an internal node, and tree_leaves will traverse its fields.
  leaves = tree_util.tree_leaves(obj)
  return not (len(leaves) == 1 and leaves[0] is obj)


def tuple_or_list_constructor(a: Tuple[Any, ...] | List[Any]):
  if isinstance(a, tuple):
    return tuple
  else:
    return list


T1 = TypeVar("T1")
T2 = TypeVar("T2")


def tree_map(f: Callable[[T1], T2], tree: Tree[T1]) -> Tree[T2]:
  """Maps a function over a tree."""
  if isinstance(tree, (tuple, list)):
    return tuple_or_list_constructor(tree)(tree_map(f, x) for x in tree)
  elif isinstance(tree, dict):
    tree: Dict[str, Tree[T1]]
    return {k: tree_map(f, v) for k, v in tree.items()}
  elif _is_registered_dataclass(tree):
    fields = dataclasses.fields(tree)
    kwargs = {
        field.name: tree_map(f, getattr(tree, field.name)) for field in fields
    }
    return type(tree)(**kwargs)
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
    assert_leaf: a function that checks that a leaf is valid. When traversing
      `tree` as a tree, any non-list/tuple/dict/None node will be passed to
      `assert_leaf`.
    tree: the object to check.
  """
  leaves = tree_util.tree_leaves(tree)
  for x in leaves:
    assert_leaf(x)


T7 = TypeVar("T7")


def flatten_lists(lists: Sequence[List[T7]]) -> List[T7]:
  return sum(lists, [])


T4 = TypeVar("T4")


def flatten(tree: Tree[T4]) -> List[T4]:
  """Flattens a tree to a list."""
  if isinstance(tree, (tuple, list)):
    return flatten_lists(list(flatten(x) for x in tree))
  elif isinstance(tree, dict):
    tree: Dict[str, Tree[T4]]
    # Sorts by key order (as opposed to insertion order like `OrderedDict`).
    return flatten_lists(list(flatten(v) for _, v in sorted(tree.items())))
  elif _is_registered_dataclass(tree):
    return flatten_lists([
        flatten(getattr(tree, field.name)) for field in dataclasses.fields(tree)
    ])
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
  elif _is_registered_dataclass(tree):
    kwargs = {
        field.name: unflatten_with_iterator(getattr(tree, field.name), it)
        for field in dataclasses.fields(tree)
    }
    return type(tree)(**kwargs)
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
