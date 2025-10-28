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

from jax import tree_util as jax_tree_util
from orbax.experimental.model import core as obm

T = TypeVar('T')
U = TypeVar('U')
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
  leaves = jax_tree_util.tree_leaves(obj)
  return not (len(leaves) == 1 and leaves[0] is obj)


def _tuple_or_list_constructor(a: Tuple[Any, ...] | List[Any]):
  if isinstance(a, tuple):
    return tuple
  else:
    return list


def tree_map(f: Callable[[T], U], tree: Tree[T]) -> Tree[U]:
  """Recursively applies `f` to all leaves of `Tree`."""
  if isinstance(tree, (tuple, list)):
    return _tuple_or_list_constructor(tree)(tree_map(f, x) for x in tree)
  elif isinstance(tree, dict):
    tree: Dict[str, Tree[T]]
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
  leaves = jax_tree_util.tree_leaves(tree)
  for x in leaves:
    assert_leaf(x)


def flatten_lists(lists: Sequence[List[T]]) -> List[T]:
  return sum(lists, [])


def flatten(tree: Tree[T]) -> List[T]:
  """Recursively flattens leaves of `Tree` into a list."""
  if isinstance(tree, (tuple, list)):
    return flatten_lists(list(flatten(x) for x in tree))
  elif isinstance(tree, dict):
    tree: Dict[str, Tree[T]]
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
  elif _is_registered_dataclass(tree):
    kwargs = {
        field.name: _unflatten_iter(getattr(tree, field.name), it)
        for field in dataclasses.fields(tree)
    }
    return type(tree)(**kwargs)
  elif tree is None:
    return None
  else:
    return next(it)


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
