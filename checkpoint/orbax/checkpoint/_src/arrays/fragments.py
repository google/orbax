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

"""Management of fragments of arrays.

A fragment is a lot like a shard but its shape is not constrained by any
relationship to a mesh of devices, or to other fragments.
"""

import dataclasses
from typing import Optional, Sequence, Union

import jax
import numpy as np
from orbax.checkpoint._src.arrays import numpy_utils as np_utils

Shape = np_utils.Shape
Index = np_utils.Index


def _ndarray_from_index(idx: Index) -> np.ndarray:
  if idx:
    return np.stack([np_utils.int_tuple_from_slice(s) for s in idx])
  else:
    return np.empty([0, 3], dtype=int)


def _index_from_ndarray(a: np.ndarray) -> Index:
  return tuple(slice(*xs) for xs in a)


@dataclasses.dataclass(frozen=True, init=False)
class Fragment:
  """One of a collection of slices into the same (abstract or concrete) array.

  Fields:
    np_index: A (start, stop, step) triple for each dimension, that locates
      the fragment within the global shape. A sequence of `slice` objects can be
      used to express this as well (see the `index` parameter to `__init__`) but
      this form is more convenient because it can't contain `None`.
    value: The data for this fragment. If this is `None`, the fragment is
      abstract.
  """
  np_index: np.ndarray  # shape=[{rank}, 3], dtype=int
  value: Optional[np.ndarray] = None

  def __init__(
      self,
      *,
      index: Optional[Index] = None,
      np_index: Optional[np.ndarray] = None,
      value: Optional[np.ndarray] = None,
  ):
    if value is not None and not isinstance(value, np.ndarray):
      raise TypeError(
          f'Fragment value must be an np.ndarray (or None), not {type(value)}.'
      )

    if index is not None and np_index is not None:
      raise ValueError('Cannot specify both index and np_index.')
    if index is None and np_index is None:
      raise ValueError('Must specify either index or np_index.')
    if index is not None:
      if not isinstance(index, tuple):
        raise TypeError(
            f'Fragment index must be a tuple of slices, got {type(index)}.'
        )
      np_index = _ndarray_from_index(index)
    elif not isinstance(np_index, np.ndarray):
      raise TypeError(
          f'Fragment np_index must be an np.ndarray , got {type(np_index)}.'
      )

    object.__setattr__(self, 'value', value)
    object.__setattr__(self, 'np_index', np_index)

  @property
  def index(self) -> Index:
    """A tuple of slices, locating this fragment in the global array."""
    return _index_from_ndarray(self.np_index)

  @property
  def start(self) -> np.ndarray:  # shape=[{rank}]
    return self.np_index[:, 0]

  @property
  def stop(self) -> np.ndarray:  # shape=[{rank}]
    return self.np_index[:, 1]

  @property
  def step(self) -> np.ndarray:  # shape=[{rank}]
    return self.np_index[:, 2]

  @property
  def shape(self) -> Shape:
    return tuple(np_utils.slice_shape(self.index))

  @property
  def size(self) -> int:
    return np.prod(self.shape)

  def __eq__(self, other: 'Fragment'):
    if not isinstance(other, Fragment):
      return False
    if not np.array_equal(self.np_index, other.np_index):
      return False
    self_value = self.value
    other_value = other.value
    if self_value is None:
      return other_value is None
    else:
      return (
          other_value is not None and
          self_value.dtype == other_value.dtype and
          np.array_equal(self_value, other_value))

  def __repr__(self):
    maybe_value_str = ', value=...' if self.value is not None else ''
    return (
        f'Fragment(index={np_utils.pretty_nd_slice(self.index)}'
        f'{maybe_value_str})'
    )

  def __array__(self) -> np.ndarray:
    if self.value is None:
      raise ValueError(
          f"Can\'t convert abstract fragment to array: {self!r}.'"
      )
    return self.value  # pytype: disable=bad-return-type

  def is_degenerate(self) -> bool:
    """Whether the index has any slices of length zero."""
    return (self.stop == self.start).any()

  @property
  def nbytes(self) -> int:
    if self.value is None:
      raise ValueError(
          'Can\'t compute nbytes of abstract fragment. Use nbytes_astype().')
    return self.value.nbytes

  def nbytes_astype(self, dtype: np.dtype) -> int:
    return np.prod([dtype.itemsize, *self.shape])

  def offset_by(self, delta: np.ndarray) -> 'Fragment':
    out_idx = self.np_index.copy()
    out_idx[:, :2] += np.expand_dims(delta, axis=1)
    return Fragment(np_index=out_idx, value=self.value)


@dataclasses.dataclass(frozen=True)
class Fragments:
  """An abstract or concrete collection of fragments.

  A `Fragments` is a lot like a `jax.Array` (or a `jax.ShapeDtypeStruct`) but
  there is a weaker relationship between the indices of the `Fragment`
  instances that it carries than there is between the indices of the shards
  of a `jax.Array` (fragments are not required to have the same shape, or to map
  to a device mesh).
  """
  shape: Shape
  dtype: np.dtype
  fragments: Sequence[Fragment]

  def __post_init__(self):
    for fragment in self.fragments:
      if not isinstance(fragment, Fragment):
        raise TypeError(
            f'Fragments must contain Fragment, not {type(fragment)}.')

  def is_degenerate(self) -> bool:
    """Whether this contains only degenerate fragments."""
    return all(f.is_degenerate() for f in self.fragments)

  @property
  def nbytes(self) -> int:
    """The total number of bytes for the global shape of this object."""
    return np.prod((self.dtype.itemsize, *self.shape))

  @property
  def addressable_nbytes(self) -> int:
    """The total number of bytes for the fragments collected in this object."""
    return sum(f.nbytes_astype(self.dtype) for f in self.fragments)

  def __array__(self) -> np.ndarray:
    for f in self.fragments:
      if f.value is None:
        raise ValueError(
            f"Can\'t convert abstract fragments to array: {self!r}.'"
        )
    non_degenerate_fragments = [
        f for f in self.fragments if not f.is_degenerate()
    ]
    for f in non_degenerate_fragments:
      # Fast path, avoiding creating a copy if we just have a single fragment
      # which has all the data.
      # TODO(b/330745907): we can relax/rewrite this check to search for at
      # least one fragment that covers the target shape fully (omitting the
      # fragments that have step > 1 on any dimension).
      if f.shape == self.shape:
        return f.value  # pytype: disable=bad-return-type
    if not _is_full(self):
      raise ValueError(
          f'Attempt to convert non-full Fragments to array: {self}.')
    result = np.empty(self.shape, dtype=self.dtype)
    for f in non_degenerate_fragments:
      result[f.index] = f.value
    return result


def _is_full(fragments: Fragments) -> bool:
  """True iff every array element is covered by some fragment."""
  present = np.zeros(fragments.shape, dtype=bool)
  for f in fragments.fragments:
    present[f.index] = True
  return np.all(present)


def _normalize(idx: Index, shape: Shape) -> Index:
  return tuple(
      slice(
          s.start if s.start is not None else 0,
          s.stop if s.stop is not None else dim,
          s.step if s.step is not None else 1,
      )
      for s, dim in zip(idx, shape)
  )


def addressable_shards(
    x: Union[jax.Array, jax.ShapeDtypeStruct],
) -> list[Index]:
  sharding = getattr(x, 'sharding', None)
  shape = x.shape
  if not sharding:
    return [tuple(slice(0, dim, 1) for dim in shape)]
  return [
      _normalize(idx, shape) for idx in
      sharding.addressable_devices_indices_map(shape).values()
  ]


def abstract_fragments(
    x: Union[jax.Array, jax.ShapeDtypeStruct, Fragments]
) -> Fragments:
  if isinstance(x, Fragments):
    return x
  return Fragments(
      x.shape,
      x.dtype,
      [Fragment(index=index, value=None) for index in addressable_shards(x)])


def validate_fragments_can_be_stacked(fragments: Fragments) -> None:
  """Validates that the given fragments can be stacked."""
  if not fragments.fragments:
    raise ValueError('No fragments to stack.')
  fragment_arrays = [fragment.value for fragment in fragments.fragments
                     if fragment.value is not None]
  if len(fragment_arrays) != len(fragments.fragments):
    raise ValueError(f'Not all fragments have values: {fragments}')
  fragment_shape = fragment_arrays[0].shape
  if not all(a.shape == fragment_shape for a in fragment_arrays):
    raise ValueError(
        f"Differently-shaped fragments can't be cast to shards: {fragments}"
    )


def stack_fragments(fragments: Optional[Fragments]) -> Optional[np.ndarray]:
  """Stacks the given fragments, which must all have the same shape."""
  if fragments is None:
    return fragments
  validate_fragments_can_be_stacked(fragments)
  fragment_arrays = [fragment.value for fragment in fragments.fragments]
  return (
      np.expand_dims(fragment_arrays[0], axis=0)
      if len(fragment_arrays) == 1
      else np.stack(fragment_arrays)
  )
