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

"""Higher-level utilities for working with tree structures."""

from collections import abc
from collections.abc import Iterable
import functools
import operator
from typing import Any, Callable, Generic, Literal, NamedTuple, Protocol, Type, TypeVar, overload

import jax
from orbax.checkpoint._src.tree import parts_of
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
  elif isinstance(a, abc.Mapping):
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


@overload
def tree_trim(
    template: PyTreeOf[Any],
    structure: PyTreeOf[T],
    *,
    trimmed_structure_callback: TrimmedStructureCallback[T] | None = None,
    strict: Literal[False],
) -> parts_of.PartsOf[PyTreeOf[T]]:
  ...


@overload
def tree_trim(
    template: PyTreeOf[Any],
    structure: PyTreeOf[T],
    *,
    trimmed_structure_callback: TrimmedStructureCallback[T] | None = None,
    strict: Literal[True] = True,
) -> PyTreeOf[T]:
  ...


def tree_trim(
    template: PyTreeOf[Any],
    structure: PyTreeOf[T],
    *,
    trimmed_structure_callback: TrimmedStructureCallback[T] | None = None,
    strict: bool = True,
) -> PyTreeOf[T] | parts_of.PartsOf[PyTreeOf[T]]:
  """Removes nodes in `structure` so that its shape matches that of `template`.

  Only dictionary entries are trimmed; sequences are unchanged and the length
  of a sequence node in `structure` must match that of the corresponding node
  in `template`.

  If `not strict`, any subtree of a mapping or named tuple node of `template`
  that is missing from the corresponding node of `structure` will be replaced
  with an appropriately-shaped subtree full of `...` placeholders (Ellipsis)
  instead of causing an error. In this mode, the tree structure of the result
  is guaranteed to match the tree structure of `template`.

  Args:
    template: The tree whose shape is to be matched.
    structure: The tree to be trimmed.
    trimmed_structure_callback: If present, will be called with the path to, and
      value of, any node that is removed from `structure`.
    strict: Require every element of `template` to be matched by an element of
      `structure`.

  Returns:
    A subset of `structure` that has the same shape as `template`.

  Raises:
    TypeError: If the type of a node in `structure` does not match the
      type of the corresponding node in `template`.
    ValueError: If keys in a dictionary node in `template` are not present
      in the corresponding node in `structure`, or if the length of a sequence
      node in `structure` does not match the length of the corresponding
      sequence node in `template`, or if an internal node that isn't a
      sequence or dictionary is encountered.
  """
  result = _tree_trim_impl(
      template,
      structure,
      trimmed_structure_callback=trimmed_structure_callback,
      strict=strict,
  )
  return result.full_structure if strict else result


def _tree_trim(
    path: tuple[str | int, ...],
    template: PyTreeOf[Any],
    structure: PyTreeOf[T],
    trimmed_structure_callback: TrimmedStructureCallback[T] | None = None,
    strict: bool = True,
) -> PyTreeOf[T]:
  match template:
    # This wants to be `case abc.Mapping()` but http://b/283787842.
    case mapping if isinstance(mapping, abc.Mapping):
      if utils.isinstance_of_namedtuple(structure):
        structure_dict = structure._asdict()  # pytype:disable=attribute-error
      elif isinstance(structure, abc.Mapping):
        structure_dict = structure
      elif structure is None:
        structure_dict = {}
      else:
        raise TypeError(
            f'Type mismatch at key path {path}: template has type'
            f' {type(template)}, but structure has type {type(structure)}.'
        )

      keep_items = []
      drop_items = []
      placeholder_items = []

      if missing := [k for k in template if k not in structure_dict]:
        if strict:
          raise ValueError(
              f'Missing {len(missing)} keys in structure path {path}, '
              f'including: {missing[:10]}'
          )
        else:
          # Fill the result with placeholders
          placeholder_items.extend(
              (k, jax.tree.map(lambda x: ..., template[k])) for k in missing
          )

      for k, n in structure_dict.items():
        (keep_items if k in template else drop_items).append((k, n))

      if trimmed_structure_callback:
        for k, n in drop_items:
          trimmed_structure_callback((*path, k), n)

      keep_dict = {
          k: _tree_trim(
              (*path, k), template[k], v, trimmed_structure_callback, strict
          )
          for k, v in keep_items
      }
      return type(template)((*keep_dict.items(), *placeholder_items))  # pytype:disable=wrong-arg-count
    case named_tuple if utils.isinstance_of_namedtuple(named_tuple):
      if structure is None:
        structure = ()
      if isinstance(structure, abc.Mapping):
        children_dict = _tree_trim(
            path,
            named_tuple._asdict(),
            structure,
            trimmed_structure_callback,
            strict,
        )
        return type(template)(**children_dict)
      elif isinstance(structure, abc.Sequence):
        children_sequence = _tree_trim(
            path,
            tuple(named_tuple),
            structure,
            trimmed_structure_callback,
            strict,
        )
        return type(template)(*children_sequence)
      else:
        raise TypeError(
            f'Type mismatch at key path {path}: template has type'
            f' {type(template)}, but structure has type {type(structure)}.'
        )
    # This wants to be `case abc.Sequence()` but http://b/283787842.
    case sequence if isinstance(sequence, abc.Sequence) and not isinstance(
        sequence, str
    ):
      if structure is None:
        structure = ()
      elif not isinstance(structure, abc.Sequence):
        raise TypeError(
            f'Type mismatch at key path {path}: template has type'
            f' {type(template)}, but structure has type {type(structure)}'
        )
      if len(structure) != len(template):
        raise ValueError(
            f'Length mismatch at key path {path}: template has length'
            f' {len(template)}, but structure has length {len(structure)}'
        )
      elements = (
          _tree_trim((*path, i), t, s, trimmed_structure_callback, strict)
          for i, (t, s) in enumerate(zip(template, structure))
      )
      return type(template)(elements)  # pytype:disable=wrong-arg-count
    case n if n is not None and utils.is_jax_internal_node(n):
      s_children_dict = utils._internal_node_as_dict(structure)  # pylint: disable=protected-access

      t_keys_and_children, t_tree_def = utils.tree_flatten_with_path_one_level(
          template
      )
      t_children_dict = {
          jax.tree_util.keystr(k): v for k, v in t_keys_and_children
      }

      # Note: unlike other cases, this does not treat the children
      # individually. Instead we have effectively cast the structure and
      # the template to mappings and will deal with them in their entirety
      # by reusing the mapping case.
      children_dict = _tree_trim(
          path,
          t_children_dict,
          s_children_dict,
          trimmed_structure_callback,
          strict,
      )
      # Now cast back to the result type.
      children = [
          children_dict[jax.tree_util.keystr(k)] for k, _ in t_keys_and_children
      ]
      return jax.tree_util.tree_unflatten(t_tree_def, children)
    case None:
      # None is special: it's the only type of template tree node that can
      # match both leaves and internal nodes of the structure to be trimmed.
      if utils.is_leaf_node(structure):
        if trimmed_structure_callback:
          trimmed_structure_callback(path, structure)
        return None
      else:
        # Make sure any callback is called appropriately on all elements of
        # `structure`.
        _tree_trim(path, {}, structure, trimmed_structure_callback, strict)
      return None
    case v if utils.is_leaf_node(v):
      if not utils.is_leaf_node_or_none(structure):
        raise TypeError(
            f'Type mismatch at key path {path}: template has type'
            f' {type(template)}, but structure has type {type(structure)}.'
        )
      return structure
    case _:
      raise TypeError(
          f'Unknown type at key path {path}: structure has type'
          f' {type(structure)}.'
      )


def _tree_trim_impl(
    template: PyTreeOf[Any],
    structure: PyTreeOf[T],
    *,
    trimmed_structure_callback: TrimmedStructureCallback[T] | None = None,
    strict: bool = True,
) -> parts_of.PartsOf[PyTreeOf[T]]:
  """Implementation of `tree_trim()` that always returns a `PartsOf`."""
  # To avoid a self-referential recursion, we create a partial that captures
  # the `trimmed_structure_callback` and `strict` arguments instead of doing an
  # implicit closure.
  tree_trim_fn = functools.partial(
      _tree_trim,
      trimmed_structure_callback=trimmed_structure_callback,
      strict=strict,
  )
  return parts_of.PartsOf(template, tree_trim_fn((), template, structure))


def _check_for_common_keys(
    trees: Iterable[PyTree],
    is_leaf: Callable[[Any], bool],
) -> None:
  """Checks `trees` for common keys and raises a `ValueError` if found."""
  seen_keys = set()
  for tree in trees:
    flat_tree = utils.to_flat_dict(tree, is_leaf=is_leaf)
    common_keys = seen_keys.intersection(flat_tree.keys())
    if common_keys:
      raise ValueError(
          'Found common key paths when overwrite is False: '
          f'{sorted(list(common_keys))}'
      )
    seen_keys.update(flat_tree.keys())


def _recursive_merge(
    t1: Any,
    t2: Any,
    overwrite: bool,
    is_leaf: Callable[[Any], bool],
) -> Any:
  """Recursively merges t1 into t2 with structure-aware logic."""
  if type(t1) is not type(t2) and not overwrite:
    raise ValueError(f'Types do not match: {type(t1)} and {type(t2)}')

  if is_leaf(t1) or is_leaf(t2):
    return t1 if t1 is not None else t2

  node_type = type(t1)

  if isinstance(t1, abc.Mapping) or utils.isinstance_of_namedtuple(t1):
    t1_dict = t1._asdict() if utils.isinstance_of_namedtuple(t1) else t1
    t2_dict = t2._asdict() if utils.isinstance_of_namedtuple(t2) else t2
    merged = dict(t2_dict)
    for k, v1 in t1_dict.items():
      if k in t2_dict:
        merged[k] = _recursive_merge(v1, t2_dict[k], overwrite, is_leaf)
      else:
        merged[k] = v1
    try:
      if utils.isinstance_of_namedtuple(t1):
        return node_type(**merged)
      return node_type(merged)
    except (TypeError, ValueError):
      return merged

  if isinstance(t1, abc.Sequence) and not isinstance(t1, str):
    if len(t1) != len(t2):
      raise ValueError(
          f'Sequence lengths do not match: {len(t1)} and {len(t2)}'
      )
    merged = [
        _recursive_merge(e1, e2, overwrite, is_leaf) for e1, e2 in zip(t1, t2)
    ]
    try:
      return node_type(merged)
    except (TypeError, ValueError):
      return merged

  if utils.is_jax_internal_node(t1):
    t1_keys_and_children, t1_treedef = utils.tree_flatten_with_path_one_level(
        t1
    )
    t1_children_dict = {
        jax.tree_util.keystr(k): v for k, v in t1_keys_and_children
    }
    t2_children_dict = utils._internal_node_as_dict(t2)  # pylint: disable=protected-access

    merged_children_dict = _recursive_merge(
        t1_children_dict, t2_children_dict, overwrite, is_leaf
    )

    children = [
        merged_children_dict[jax.tree_util.keystr(k)]
        for k, _ in t1_keys_and_children
    ]
    return jax.tree_util.tree_unflatten(t1_treedef, children)

  return t1


def merge_trees(
    *trees: PyTree,
    overwrite: bool = False,
    is_leaf: Callable[[Any], bool] | None = None,
) -> PyTree:
  """Merges trees into a single tree using a comprehensive recursive strategy.

  This implementation handles standard Python containers (dicts, lists, tuples),
  named tuples, and custom JAX PyTree nodes, mirroring the robustness of
  utilities like `tree_trim`.

  - Mappings (dict, etc.) are merged by key.
  - Sequences (list, tuple) are merged element-wise if they have the same
    length; otherwise, a ValueError is raised.
  - Dataclasses (dataclass, etc.) are merged by field name, where non-None
    values overwrite None values.
  - If `overwrite` is False, a ValueError is raised for mismatched types.

  Args:
    *trees: The trees to merge.
    overwrite: If True, later values from `trees` will overwrite earlier values
      where leaf paths conflict. If False, a ValueError is raised for
      conflicting leaves.
    is_leaf: Optional function to determine if a node is a leaf. Defaults to
      `jax.tree_util.all_leaves`.

  Returns:
    A new PyTree representing the merged content of `trees`.

  Raises:
    ValueError: If `overwrite` is False and there are common leaf key paths
      between `trees`. Or if `overwrite` is False and the types of nodes do not
      match.
  """
  is_leaf_fn = is_leaf or utils.is_leaf_node_or_none
  trees = list(trees)
  if not trees:
    return {}

  if not overwrite:
    _check_for_common_keys(trees, is_leaf_fn)

  return functools.reduce(
      lambda acc, t: _recursive_merge(t, acc, overwrite, is_leaf_fn),
      trees,
  )


def format_tree_diff(
    diff: PyTree,
    path_prefix: str = '',
    source_label: str = 'Source',
    target_label: str = 'Target',
) -> str | None:
  """Format a tree difference structure into a readable multi-line string.

  Args:
    diff: object representing the difference between two PyTrees
    path_prefix: Current path prefix for nested structures
    source_label: Label for the source value
    target_label: Label for the target value

  Returns:
    A formatted string showing the differences in a multi-line structure.
  """
  source_label = f'    - {source_label}:'
  target_label = f'    - {target_label}:'
  missing_symbol = 'MISSING'

  lines = []

  # Leaf nodes
  if isinstance(diff, Diff):
    if path_prefix:
      lines.append(f'{path_prefix}:')
    else:
      lines.append('Mismatch:')

    def _format_value(value):
      return missing_symbol if value in (None, parts_of.PLACEHOLDER) else value

    lines.append(f'{source_label} {_format_value(diff.lhs)}')
    lines.append(f'{target_label} {_format_value(diff.rhs)}')
    return '\n'.join(lines)

  # Nested nodes
  children, _ = utils.tree_flatten_with_path_one_level(diff)
  for path, value in children:
    key = path[0]
    if value is not None:
      if isinstance(key, jax.tree_util.SequenceKey):
        new_path = f'{path_prefix}[{key.idx}]'
      elif isinstance(key, jax.tree_util.DictKey):
        new_path = f'{path_prefix}.{key.key}' if path_prefix else str(key.key)
      elif isinstance(key, jax.tree_util.GetAttrKey):
        new_path = f'{path_prefix}.{key.name}' if path_prefix else str(key.name)
      else:
        raise ValueError(f'Unsupported key type: {type(key)}')

      formatted = format_tree_diff(value, new_path)
      if formatted:
        lines.append(formatted)
  return '\n\n'.join(lines)


class TreeStructureError(ValueError):
  pass


_ErrorType = TypeVar('_ErrorType', bound=Exception)


def build_mismatched_tree_structure_error(
    source_tree: PyTreeOf[Any],
    target_tree: PyTreeOf[Any],
    log_message: str,
    exception_cls: Type[_ErrorType] = TreeStructureError,
) -> Type[_ErrorType]:
  """Builds a TreeStructureError pointing to where exactly two trees differ."""
  if isinstance(source_tree, parts_of.PartsOf):
    source_tree = source_tree.unsafe_structure
  if isinstance(target_tree, parts_of.PartsOf):
    target_tree = target_tree.unsafe_structure

  diff = tree_difference(
      source_tree,
      target_tree,
      leaves_equal=operator.eq,
  )

  if diff is None:
    return exception_cls(f'{log_message}. But no diff was found.')

  formatted_diff = format_tree_diff(diff)
  return exception_cls(f'{log_message}.\n\n{formatted_diff}')
