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

"""`PartsOf[T]` holds a data structure `T` whose leaves may be missing.

`Optional[T]` is equivalent to `T | None` and represents either an entire `T`
or no value. `PartsOf[T]` lies somewhere between those two extremes.

Not to be confused with `jax_tree_utils.Partial`, which does partial function
application.
"""

import collections
import dataclasses
from typing import Any, Callable, Generic, Mapping, TypeVar

from absl import logging
from jax import tree_util as jax_tree_util
from orbax.checkpoint._src.tree import utils as tree_utils

T = TypeVar('T')

# Represents a structure that is PyTree-compatible with a `T` but whose leaves
# may be placeholders.
PyTreeWithPlaceholdersLike = Any | T


# The value used to represent a missing leaf. This specific value is useful
# because:
#   1. JAX does not treat it as an internal node (which it does with `None`);
#   2. There is only one of its type, so `is`-based comparison works.
PLACEHOLDER = ...
Placeholder = type(PLACEHOLDER)


class Error(ValueError):
  pass


class TreeNotCompleteError(Error):
  pass


class UnexpectedPlaceholderError(Error):
  pass


def _check_templates_match(t0: Any, t1: Any) -> None:
  if t0 != t1:
    logging.warning(
        'Mismatched templates: \nCurrent Template - t0= %s \nOther Template -'
        ' t1= %s ',
        t0,
        t1,
    )
    raise Error('Mismatched templates')


@dataclasses.dataclass(init=False)
class PartsOf(Generic[T]):
  """Holds a data structure `T` whose leaves may be missing.

  `T` must be understandable as a JAX PyTree.

  The only operations that may safely be performed on a `PartsOf[T]` are:
  - Element-wise combination with another `PartsOf[T]`;
  - Checked extraction of a complete structure.

  For example:
  ```
  def fetch_all() -> MyStructure:
    some_parts: PartsOf[MyStructure] = fetch_from_one_place()
    other_parts: PartsOf[MyStructure] = fetch_from_another_place()
    return some_parts.overlay(other_parts).full_structure
  ```

  A `PartsOf[T]` can be introduced by constructing it with a template `T` and
  a value that supplies none, some or all of the leaves of T. Every path to a
  leaf in the value must correspond to a path to a leaf in the template.
  (This is not what is usually meant by the term 'subtree' and perhaps the
  term 'recursive subset' would best describe the relationship between the
  value and the template.)

  For example:
  ```
  @dataclasses.dataclass
  class MyStructure:
    x: int
    y: dict[str, int]

  template = MyStructure(x=0, y={'one': 1, 'two': 2})
  value = {'y': {'two': 4}}

  partial_structure = PartsOf(template, value)
  ```

  In the case where the value supplies no leaves at all,
  `PartsOf.empty(template)` is an available shorthand.
  """

  _template: jax_tree_util.PyTreeDef
  _present: dict[tuple[Any, ...], Any]

  def __init__(
      self,
      template: T,
      value: PyTreeWithPlaceholdersLike[T],
  ):
    unused_leaves, self._template = jax_tree_util.tree_flatten(template)
    self._present = {
        k: v
        for k, v in tree_utils.to_flat_dict(value).items()
        if v is not PLACEHOLDER
    }

    template_paths = tree_utils.to_flat_dict(template)
    bogus_paths = {*self._present} - {*template_paths}
    if bogus_paths:
      raise Error(
          'The following paths in input value are not part of the full tree'
          f' structure: {bogus_paths}.\nGot value paths {self._present}\nand'
          f' template paths {template_paths}'
      )

  @classmethod
  def empty(
      cls,
      template: T,
  ) -> 'PartsOf[T]':
    return cls(template, ())

  @classmethod
  def from_structure(
      cls,
      structure: T
  ) -> 'PartsOf[T]':
    """Makes a PartsOf from a structure using it as both template and value.

    Args:
      structure: The structure to use as both template and value. May contain
        placeholders for missing leaves.

    Returns:
      A PartsOf representing the given structure.
    """
    return cls(structure, structure)

  # Alias for backwards compatibility.
  from_full_structure = from_structure

  @classmethod
  def like(
      cls,
      existing: 'PartsOf[Any]',
      value: PyTreeWithPlaceholdersLike[T],
  ) -> 'PartsOf[T]':
    return cls(
        existing._get_template(),  # pylint:disable=protected-access
        value,
    )

  @classmethod
  def _with_present_values(
      cls,
      template: jax_tree_util.PyTreeDef,
      present_values: dict[tuple[Any, ...], Any],
  ):
    result = cls.empty(())
    result._template = template
    result._present = present_values
    return result

  def _get_template(self):
    return self._template.unflatten([PLACEHOLDER] * self._template.num_leaves)

  def is_empty(self) -> bool:
    """Returns True if the PartsOf is empty."""
    return not self._present

  @property
  def structure_hash(self) -> int:
    """Returns a hash of the structure."""
    return hash((self._template, tuple(self._present.keys())))

  @property
  def full_structure(self) -> T:
    """Extracts a full PyTree with no missing leaves, or raises Error."""
    try:
      return tree_utils.from_flat_dict(self._present, self._get_template())
    except KeyError as e:
      logging.warning(
          'Detected mismatch between values and template.\n'
          'Value keys: %s\n'
          'Template: %s',
          self._present.keys(),
          self._get_template(),
      )
      raise TreeNotCompleteError(f'Tree is not complete: {e}') from e

  @property
  def unsafe_structure(self) -> PyTreeWithPlaceholdersLike[T]:
    """Extracts a `T` that may have missing leaves. Avoid this if possible."""
    leaves = {
        **tree_utils.to_flat_dict(self._get_template()),
        **self._present,
    }
    return tree_utils.from_flat_dict(leaves, self._get_template())

  def overlay(
      self,
      other: 'PartsOf[T]',
      *,
      strict: bool = False,
      verbose: bool = False,
  ) -> 'PartsOf[T]':
    """Overlay another partial tree on this one.

    Args:
      other: The partial tree to overlay. Must have the same template as this
        one.
      strict: If True, the overlay operation will fail if sets of values are
        not disjoint in this tree and `other`.
      verbose: Whether to log the keys that are being overwritten from `other`.

    Returns:
      A new `PartsOf[T]` where a set of values has been updated with values from
      `other`.
      If a value for some tree path is present in both trees:
        - if `strict` is False: the value from `other` is used (overwrites the
          value from this tree).
        - if `strict` is True: an error is raised.

      Example:
        ```
        t = {'a': 1, 'b': 2, 'c': ...}
        t_other = {'a': ..., 'b': 13, 'c': 4}

        t.overlay(t_other)  # Succeeds, returns {'a': 1, 'b': 13, 'c': 4}.
        t.overlay(t_other, strict=True)  # Raises (can't overwrite 'b').
        ```
    """
    _check_templates_match(self._template, other._template)  # pylint:disable=protected-access
    common_keys = self._present.keys() & other._present.keys()  # pylint:disable=protected-access
    if strict:
      if common_keys:
        raise ValueError(
            f'Strict overlay failed: keys {common_keys} are present in both'
            ' PartsOf structures.'
        )
    elif verbose:
      if common_keys:
        logging.info('Overwriting %d keys:', len(common_keys))
        for key in sorted(list(common_keys)):
          logging.info('  %s', key)
      else:
        logging.info('No keys to overwrite.')
    new_leaves = self._present.copy()
    new_leaves.update(other._present)  # pylint:disable=protected-access
    return PartsOf._with_present_values(self._template, new_leaves)

  def intersect(self, other: 'PartsOf[T]') -> 'PartsOf[T]':
    """Removes all leaves that are not present in `other`."""
    _check_templates_match(self._template, other._template)  # pylint:disable=protected-access
    new_leaves = {k: v for k, v in self._present.items() if k in other._present}  # pylint:disable=protected-access
    return PartsOf._with_present_values(self._template, new_leaves)

  def __sub__(self, other: 'PartsOf[T]') -> 'PartsOf[T]':
    """Removes all leaves that are present in `other`."""
    _check_templates_match(self._template, other._template)
    new_leaves = {
        k: v for k, v in self._present.items() if k not in other._present
    }
    return PartsOf._with_present_values(self._template, new_leaves)

  def tree_flatten(self) -> tuple[tuple[T, ...], jax_tree_util.PyTreeDef]:
    """JAX pytree flatten implementation for PartsOf."""
    return (self._present,), self._template

  @classmethod
  def tree_unflatten(
      cls,
      aux_data: jax_tree_util.PyTreeDef,
      children: tuple[dict[tuple[Any, ...], Any]],  # pylint:disable=g-one-element-tuple
  ) -> 'PartsOf[T]':
    """JAX pytree unflatten implementation for PartsOf."""
    template = aux_data
    (present,) = children
    for p, v in present.items():
      if v is PLACEHOLDER:
        raise UnexpectedPlaceholderError(
            'PartsOf tree unflattening discovered a placeholder value at path'
            f' {p}, which is not supported. If you need to map over a PartsOf'
            ' tree *returning placeholder values*, please use'
            ' `parts_of.map_leaves` or `parts_of.map_leaves_with_path` instead'
            ' of the standard JAX tree mapping functions.'
        )
    return PartsOf._with_present_values(template, present)


jax_tree_util.register_pytree_node_class(PartsOf)


def map_leaves(
    f: Callable[..., Any | Placeholder],
    *xs: PartsOf[T],
) -> PartsOf[T]:
  """Applies a function to all leaves (present or not).

  PartsOf-compatible equivalent of `jax.tree.map`.

  Args:
    f: The function to apply to each leaf of the structure. It must accept as
        many arguments as given here, any number of which may be `PLACEHOLDER`.
        It must return an appropriate non-`None` leaf value, or `PLACEHOLDER`.
    *xs: One or more partially known structures.

  Returns:
    A partially known structure.
  """
  def _wrapped_f(path, *leaves):
    del path
    return f(*leaves)

  return map_leaves_with_path(_wrapped_f, *xs)


def map_leaves_with_path(
    f: Callable[..., Any | Placeholder],
    *xs: PartsOf[T],
) -> PartsOf[T]:
  """Applies a function to all leaves (present or not), with value path.

  PartsOf-compatible equivalent of `jax_tree_util.tree_map_with_path`.

  Args:
    f: The function to apply to each leaf of the structure. It must accept tree
      path + as many arguments as *xs given here, any number of which may be
      `PLACEHOLDER`. It must return an appropriate non-`None` leaf value, or
      `PLACEHOLDER`.
    *xs: One or more partially known structures.

  Returns:
    A partially known structure.
  """
  x0, *xs = xs
  for x in xs:
    _check_templates_match(x0._template, x._template)  # pylint:disable=protected-access
  def f_(path, _):
    path = tree_utils.tuple_path_from_keypath(path)
    leaves = [x._present.get(path, PLACEHOLDER) for x in (x0, *xs)]  # pylint:disable=protected-access
    result = f(path, *leaves)
    if result is None:
      raise Error(f'At {path}, {f} returned None.')
    if not jax_tree_util.all_leaves([result]):
      raise Error(f'At {path}, {f} returned non-leaf value {result}.')
    return result
  t0 = x0._get_template()  # pylint:disable=protected-access
  return PartsOf(t0, jax_tree_util.tree_map_with_path(f_, t0))


def filter_values(
    existing: 'PartsOf[T]',
    value_keys: set[tuple[Any, ...]],
) -> 'PartsOf[T]':
  """Makes a new PartsOf from existing with only the values with given keys."""
  # pylint:disable=protected-access
  result = PartsOf.empty(existing._get_template())
  result._present = {
      k: v for k, v in existing._present.items() if k in value_keys
  }
  # pylint:enable=protected-access
  return result


def value_key_from_path(path: tuple[Any, ...]) -> tuple[Any, ...]:
  """Converts a PartsOf JAX pytree path into a key for the corresponding value.

  Args:
    path: a JAX tree path into a PartsOf object (i.e. obtained via one of
      `jax_tree_util.*` methods on a PartsOf pytree).

  Returns:
    A key for the corresponding value in PartsOf template's flattened dict
    (i.e. `_present`).
  """
  assert len(path) == 2, (
      f'Too many elements in a PartsOf tree path: {path}, expected 2.'
  )
  assert isinstance(
      path[1], jax_tree_util.DictKey
  ), f'Expected DictKey, found: {path[1]}'
  return path[1].key


C = TypeVar('C')


def partition(
    classifier: Callable[[T], C],
    structure: PartsOf[T],
) -> Mapping[C, PartsOf[T]]:
  """Partitions a PartsOf[T] based on a leaf classifier function.

  Args:
    classifier: A function that assigns a class label to each leaf of the
      structure.
    structure: The structure to partition.

  Returns:
    A mapping from classification result to a PartsOf[T] containing only the
    values that were classified as such.

  Example:
    ```
    template = MyDataclass(a=(X, X), b={'c': X, 'd': X})
    t = PartsOf(template, MyDataclass(a=(1, 2), b={'c': 4, 'd': 3}))
    # Split the structure into one with even and another with odd values.
    even_odd = partition(lambda x: x % 2, t)

    assert even_odd[0].unsafe_structure == (
        MyDataclass(a=(..., 2), b={'c': 4, 'd': ...}
    )
    assert even_odd[1].unsafe_structure == (
        MyDataclass(a=(1, ...), b={'c': ...,' d': 3}
    )
    ```
  """
  value_paths_by_class = collections.defaultdict(set)
  for k, v in structure._present.items():  # pylint:disable=protected-access
    c = classifier(v)
    value_paths_by_class[c].add(k)
  return {
      c: filter_values(structure, value_paths)
      for c, value_paths in value_paths_by_class.items()
  }
