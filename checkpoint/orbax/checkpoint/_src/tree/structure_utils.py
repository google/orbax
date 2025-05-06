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

"""Higher-level utilities for working with tree structures."""

from typing import Any, Callable, Generic, Mapping, NamedTuple, Protocol, TypeVar

import jax
from orbax.checkpoint._src.tree import utils

T = TypeVar('T')
PyTree = Any
PyTreeOf = PyTree | T


class Diff(NamedTuple):
  lhs: Any
  rhs: Any


def tree_difference(
    a: PyTree,
    b: PyTree,
    *,
    is_leaf: Callable[[PyTree], bool] | None = None,
    leaves_equal: Callable[[PyTree, PyTree], bool] | None = None,
):
  """Recursively compute differences in tree structure.

  Lists, tuples, named tuples, mappings and custom PyTree nodes are treated
  as internal nodes and compared element-wise. Other types are treated as
  leaf nodes and compared by type or using a custom equality function, if
  one is provided.

  All leaves of a subtree must be equal according to the equality function to
  avoid being reported as differences.

  The result is a tree that is structurally a prefix of both the inputs.
  Differences in equality of leaf nodes, sequence length or mapping key set at
  corresponding nodes of the two inputs are summarized as `Diff` instances at
  the resulting node in the output.

  Map values that do not differ are omitted from the output; sequence elements
  that do not differ are represented as None.

  For example:
  ```
  first  = {'x': [1, 2, 3], 'y': 'same', 'z': {'p': {}, 'q': 4, 'r': 6}}
  second = {'x': [1, 3],    'y': 'same', 'z': {'p': {}, 'q': 5, 's': 6}}
  tree_difference(first, second) == {
      'x': Diff(lhs=(list, 3), rhs=(list, 2)),
      'z': {'r': Diff(lhs=6, rhs=None), 's': Diff(lhs=None, rhs=6)},
  }
  ```

  Args:
    a: The first object to compare.
    b: The second object to compare.
    is_leaf: Optional function to determine if a node is a leaf. If None,
      defaults to `jax.tree_util.all_leaves`.
    leaves_equal: Optional equality function to apply between corresponding
      leaves of the two data structures. If None, leaf nodes are considered
      equal if they are of the same type.

  Returns:
    A tree summarizing the structural differences between the arguments, or
    `None` if there are no such differences.
  """
  is_leaf = is_leaf or (lambda t: jax.tree_util.all_leaves([t]))
  leaves_equal = leaves_equal or (lambda a, b: type(a) is type(b))

  if is_leaf(a) and is_leaf(b):
    # `a` and `b` are leaf nodes; compare using equality function.
    if leaves_equal(a, b):
      return None
    else:
      return Diff(lhs=a, rhs=b)

  if type(a) is not type(b):
    # Can't check if either node is nested if the node types are different.
    return Diff(lhs=type(a), rhs=type(b))

  if utils.isinstance_of_namedtuple(a):
    diffs = type(a)(*(
        tree_difference(aa, bb, is_leaf=is_leaf, leaves_equal=leaves_equal)
        for aa, bb in zip(a, b)
    ))
    return None if all(d is None for d in diffs) else diffs
  elif isinstance(a, (list, tuple)):
    if len(a) != len(b):
      return Diff(lhs=(type(a), len(a)), rhs=(type(b), len(b)))
    diffs = type(a)(
        tree_difference(aa, bb, is_leaf=is_leaf, leaves_equal=leaves_equal)
        for aa, bb in zip(a, b)
    )
    return None if all(d is None for d in diffs) else diffs
  elif isinstance(a, Mapping):
    diffs = {
        k: diff
        for k, diff in (
            (
                k,
                tree_difference(
                    a[k], b[k], is_leaf=is_leaf, leaves_equal=leaves_equal
                ),
            )
            for k in a
            if k in b
        )
        if diff is not None
    }
    diffs.update((k, Diff(lhs=v, rhs=None)) for k, v in a.items() if k not in b)
    diffs.update((k, Diff(lhs=None, rhs=v)) for k, v in b.items() if k not in a)

    return diffs or None
  else:
    # Custom PyTree nodes. This is limited to the case where `a` and `b` have
    # the same list of children, because `unflatten` requires a `tree_def` that
    # has come from one of them or the other. It's not clear that there's any
    # way to remove that limitation.
    a_keys_and_children, a_tree_def = utils.tree_flatten_with_path_one_level(a)
    b_keys_and_children, _ = utils.tree_flatten_with_path_one_level(b)
    a = {k: v for k, v in a_keys_and_children}
    b = {k: v for k, v in b_keys_and_children}
    diffs = {
        k: diff
        for k, diff in (
            (
                k,
                tree_difference(
                    a[k], b[k], is_leaf=is_leaf, leaves_equal=leaves_equal
                ),
            )
            for k in a
            if k in b
        )
        if diff is not None
    }

    diffs.update((k, Diff(lhs=v, rhs=None)) for k, v in a.items() if k not in b)
    diffs.update((k, Diff(lhs=None, rhs=v)) for k, v in b.items() if k not in a)
    if diffs:
      return jax.tree.unflatten(
          a_tree_def, [diffs.get(k) for k, _ in a_keys_and_children]
      )

    return None


class TrimmedStructureCallback(Protocol, Generic[T]):

  def __call__(
      self,
      path: tuple[str | int, ...],
      structure: PyTreeOf[T],
  ) -> None:
    ...


# TODO: b/407092826 - Substitute for full PartsOf version later.
def tree_trim(
    template: PyTreeOf[Any],
    structure: PyTreeOf[T],
    *,
    trimmed_structure_callback: TrimmedStructureCallback[T] | None = None,
    strict: bool = True,
) -> PyTreeOf[T]:
  """Removes nodes in `structure` so that its shape matches that of `template`.

  Args:
    template: The tree whose shape is to be matched.
    structure: The tree to be trimmed.
    trimmed_structure_callback: If present, will be called with the path to, and
      value of, any node that is removed from `structure`.
    strict: Require every element of `template` to be matched by an element of
      `structure`.

  Returns:
    A PyTree with the same structure as `template`, but with values from
    `structure`.
  """
  del trimmed_structure_callback  # Unused.
  if not strict:
    raise NotImplementedError('Non-strict mode is not implemented yet.')

  structure_flat = utils.to_flat_dict(structure, sep=None)

  def _get_value_from_structure(path: utils.PyTreePath, template_leaf: Any):
    """Maps template path/leaf to structure value."""
    del template_leaf  # Unused, we only care about the template tree structure.
    tuple_key = utils.tuple_path_from_keypath(path)
    if tuple_key in structure_flat:
      return structure_flat[tuple_key]
    else:
      path_str = '.'.join(tuple_key)
      raise ValueError(
          f'Path "{path_str}" exists in template but not found in structure '
          'PyTree.'
      )

  return jax.tree_util.tree_map_with_path(_get_value_from_structure, template)
