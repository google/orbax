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

"""Management of fragments of arrays.

A fragment is a lot like a shard but its shape is not constrained by any
relationship to a mesh of devices, or to other fragments.
"""
# TODO(b/465196209): Remove when support for Python 3.10 is dropped.
from __future__ import annotations

import dataclasses
from typing import Any, ClassVar, Generic, Literal, Sequence, TypeAlias, TypeVar

import jax
import numpy as np
from orbax.checkpoint._src.arrays import numpy_utils as np_utils
from orbax.checkpoint._src.arrays import types

Shape: TypeAlias = types.Shape
Index: TypeAlias = types.Index
NpIndex: TypeAlias = np.ndarray  # shape=[{rank}, 3], dtype=int


Module: TypeAlias = type(dataclasses)
A = TypeVar('A', bound=(np.ndarray | jax.Array | None))
Aconcrete = TypeVar('Aconcrete', bound=(np.ndarray | jax.Array))


def _ndarray_from_index(idx: Index) -> NpIndex:
  if idx:
    return np.stack([np_utils.int_tuple_from_slice(s) for s in idx])
  else:
    return np.empty([0, 3], dtype=int)


def _index_from_ndarray(a: NpIndex) -> Index:
  return tuple(slice(*xs) for xs in a)


def _qualified_name(cls):
  return f'{cls.__module__}.{cls.__name__}'


@dataclasses.dataclass(frozen=True, init=False)
class _GenericFragment(Generic[A]):
  """One of a collection of slices into the same (abstract or concrete) array.

  Fields:
    np_index: A (start, stop, step) triple for each dimension, that locates
      the fragment within the global shape. A sequence of `slice` objects can be
      used to express this as well (see the `index` parameter to `__init__`) but
      this form is more convenient because it can't contain `None`.
    value: The data for this fragment. If this is `None`, the fragment is
      abstract.
  """
  ARRAY_T: ClassVar[type[A]]  # The type of fragment values.
  NP_API: ClassVar[Module | None]  # NumPy-like API, if any, for A instances.

  np_index: NpIndex  # shape=[{rank}, 3], dtype=int
  value: A

  def __init__(
      self,
      *,
      index: Index | None = None,
      np_index: NpIndex | None = None,
      value: A,
  ):
    if not isinstance(value, self.ARRAY_T):
      raise TypeError(
          f'Fragment value must be a {_qualified_name(self.ARRAY_T)},'
          f' not {type(value)}.'
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
    elif not isinstance(np_index, NpIndex):
      raise TypeError(
          f'Fragment np_index must be an np.ndarray, got {type(np_index)}.'
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

  def is_degenerate(self) -> bool:
    """Whether the index has any slices of length zero."""
    return (self.stop == self.start).any()

  def nbytes_astype(self, dtype: np.dtype) -> int:
    return np.prod([dtype.itemsize, *self.shape])

  def offset_by(
      self,
      delta: np.ndarray,  # shape=[{rank}], dtype=int
  ) -> _GenericFragment[A]:  # Use typing.Self once 3.11 is minimum.
    out_idx = self.np_index.copy()
    out_idx[:, :2] += np.expand_dims(delta, axis=1)
    return type(self)(np_index=out_idx, value=self.value)

  def intersect(
      self,
      np_index: NpIndex,  # shape=[{rank}, 3], dtype=int
  ) -> _GenericFragment[A] | None:
    """Intersects this fragment with the given NpIndex.

    The result is in this fragment's coordinate space. For example,
    intersecting a fragment with its own index gives an identical fragment.

    Args:
      np_index: The NpIndex to intersect with.

    Returns:
      A new fragment representing the intersection, or None if there is no
      overlap.
    """
    if (self.step != 1).any() or (np_index[:, 2] != 1).any():
      raise NotImplementedError('index steps other than 1 are not supported.')

    out_np_index = np_index.copy()
    start = out_np_index[:, 0] = np.maximum(out_np_index[:, 0], self.start)
    stop = out_np_index[:, 1] = np.minimum(out_np_index[:, 1], self.stop)
    if not (start < stop).all():
      return None
    return type(self)(
        np_index=out_np_index, value=self.slice_of_value(out_np_index)
    )

  def slice(
      self,
      np_index: NpIndex,  # shape=[{rank}, 3], dtype=int
  ) -> _GenericFragment[A] | None:  # Use typing.Self once 3.11 is minimum.
    """Slices this fragment by the given NpIndex.

    The result is in the slice's coordinate space. For example, slicing a
    fragment by its own index gives a fragment whose start is zero.

    Args:
      np_index: The NpIndex to slice by.

    Returns:
      A new fragment representing the slice, or None if there is no overlap.
    """
    intersection = self.intersect(np_index)
    return intersection.offset_by(-np_index[:, 0]) if intersection else None

  def slice_of_value(self, np_index: NpIndex) -> A:
    """Takes a slice of the value of this fragment.

    It is required that `np_index` has already been clamped to the fragment's
    bounds; otherwise a ValueError will result.

    Args:
      np_index: The NpIndex to slice by.

    Returns:
      A slice of the fragment's value.
    """
    raise NotImplementedError()


@dataclasses.dataclass(frozen=True, init=False, eq=False, repr=False)
class AbstractFragment(_GenericFragment[type(None)]):
  """An abstract fragment."""
  ARRAY_T = type(None)
  NP_API = None

  def __init__(
      self,
      *,
      index: Index | None = None,
      np_index: NpIndex | None = None,
      value: Literal[None] = None,
  ):
    super().__init__(index=index, np_index=np_index, value=value)

  def __eq__(self, other: AbstractFragment):  # Use typing.Self once on 3.11+.
    if not isinstance(other, type(self)):
      return False
    if not np.array_equal(self.np_index, other.np_index):
      return False
    return True

  def __repr__(self):
    return (
        f'{type(self).__name__}(index={np_utils.pretty_nd_slice(self.index)})'
    )

  def offset_by(
      self,
      delta: np.ndarray,  # shape=[{rank}], dtype=int
  ) -> 'AbstractFragment':
    out_idx = self.np_index.copy()
    out_idx[:, :2] += np.expand_dims(delta, axis=1)
    return type(self)(np_index=out_idx)

  def slice_of_value(self, np_index: NpIndex) -> None:
    del np_index
    return None


@dataclasses.dataclass(frozen=True, init=False)
class _ConcreteFragment(_GenericFragment[Aconcrete]):
  """A fragment whose value is an array."""
  ARRAY_T: ClassVar[type[Aconcrete]]  # The type of fragment values.
  NP_API: ClassVar[Module]  # NumPy-like API, for A instances.

  def __eq__(self, other: ConcreteFragment):  # Use typing.Self once on 3.11+.
    if not isinstance(other, type(self)):
      return False
    if not np.array_equal(self.np_index, other.np_index):
      return False
    self_value = self.value
    other_value = other.value
    if self_value is None:
      return other_value is None
    else:
      return (
          other_value is not None
          and self_value.dtype == other_value.dtype
          and self.NP_API.array_equal(self_value, other_value)
      )

  def __repr__(self):
    return (
        f'{type(self).__name__}(index={np_utils.pretty_nd_slice(self.index)},'
        ' value=...)'
    )

  def __array__(
      self,
      dtype: np.dtype | None = None,
      *,
      copy: bool | None = None,
  ) -> np.ndarray:
    return np.asarray(self.value, dtype=dtype, copy=copy)

  @property
  def nbytes(self) -> int:
    return self.value.nbytes

  def slice_of_value(self, np_index: NpIndex) -> Aconcrete:
    # This is just a convenient way to construct the required tuple of slices.
    f = AbstractFragment(np_index=np_index).offset_by(-self.start)
    if (f.start < 0).any() or (f.stop > self.value.shape).any():
      raise ValueError(
          f'Attempt to slice fragment value of shape {self.shape} with'
          f' out-of-bounds index {f}'
      )
    return self.value[f.index or ...]


@dataclasses.dataclass(frozen=True, init=False, eq=False, repr=False)
class NpFragment(_ConcreteFragment[np.ndarray]):
  """One of a collection of slices into the same concrete array."""
  ARRAY_T = np.ndarray
  NP_API = np


@dataclasses.dataclass(frozen=True, init=False, eq=False, repr=False)
class JaxFragment(_ConcreteFragment[jax.Array]):
  """One of a collection of slices into the same JAX array."""
  ARRAY_T = jax.Array
  NP_API = jax.numpy


_F = TypeVar('_F', bound=_GenericFragment)
F = TypeVar('F', bound=(AbstractFragment | NpFragment | JaxFragment))
Fconcrete = TypeVar('Fconcrete', bound=(NpFragment | JaxFragment))


@dataclasses.dataclass(frozen=True, eq=False, repr=False)
class _GenericFragments(Generic[_F]):
  """An abstract or concrete collection of fragments.

  A `Fragments` is a lot like a `jax.Array` (or a `jax.ShapeDtypeStruct`) but
  there is a weaker relationship between the indices of the `Fragment`
  instances that it carries than there is between the indices of the shards
  of a `jax.Array` (fragments are not required to have the same shape, or to map
  to a device mesh).
  """
  FRAGMENT_T: ClassVar[type[_F]]  # The type of Fragment instances.

  shape: Shape
  dtype: np.dtype
  fragments: Sequence[_F]

  def __post_init__(self):
    fragment_t = self.FRAGMENT_T
    for fragment in self.fragments:
      if not isinstance(fragment, fragment_t):
        raise TypeError(
            f'Fragments must contain {_qualified_name(fragment_t)}, not'
            f' {type(fragment)}.'
        )

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

  def __array__(
      self,
      dtype: np.dtype | None = None,
      *,
      copy: bool | None = None,
  ) -> np.ndarray:
    for f in self.fragments:
      if f.value is None:
        raise ValueError(
            f"Can't convert abstract fragments to array: {self!r}.'"
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
        return np.asarray(f.value, dtype=dtype, copy=copy)
    if copy is False:  # pylint: disable=g-bool-id-comparison; None is different
      raise ValueError(
          'Attempt to convert Fragments to array without copying. This is'
          ' only possible if there is a single fragment that spans the entire'
          f' shape, but there are {len(self.fragments)} fragments.'
      )
    if not _is_full(self):
      raise ValueError(
          f'Attempt to convert non-full Fragments to array: {self}.'
      )
    result = np.empty(self.shape, dtype=dtype or self.dtype)
    for f in non_degenerate_fragments:
      result[f.index] = f.value
    return result

  def slice(
      self,
      index: NpIndex | Index,  # shape=[{rank}, 3], dtype=int
  ) -> '_GenericFragments[_GenericFragment[A]]':  # Use typing.Self once >=3.11.
    """Returns a slice of this object."""
    if not isinstance(index, np.ndarray):
      index = np_utils.resolve_slice(index, self.shape)
      index = _ndarray_from_index(index)

    if not (index[:, 2] == 1).all():
      raise NotImplementedError('Coming ... soon?')
    sliced_shape = np.minimum(self.shape, index[:, 1]) - index[:, 0]
    if (sliced_shape < 0).any():
      raise ValueError(
          f'Attempt to slice Fragments of shape {self.shape} '
          f'with out-of-bounds index {_index_from_ndarray(index)}'
      )

    return type(self)(
        tuple(d.item() for d in sliced_shape),
        self.dtype,
        [
            f
            for f in [fragment.slice(index) for fragment in self.fragments]
            if f is not None
        ],
    )


@dataclasses.dataclass(frozen=True, init=False)
class AbstractFragments(_GenericFragments[AbstractFragment]):
  """A collection of abstract fragments."""
  FRAGMENT_T = AbstractFragment


@dataclasses.dataclass(frozen=True, init=False)
class NpFragments(_GenericFragments[NpFragment]):
  """A collection of fragments whose values are of type `np.ndarray`."""
  FRAGMENT_T = NpFragment


@dataclasses.dataclass(frozen=True, init=False)
class JaxFragments(_GenericFragments[JaxFragment]):
  """A collection of fragments whose values are of type `jax.Array`."""
  FRAGMENT_T = JaxFragment


# Extra names for backwards compatibility. Most loading and saving code still
# wants to deal with NumPy arrays so that views and operations on them
# (in particular including assignment to slices) work as expected.
ConcreteFragment = NpFragment
ConcreteFragments = NpFragments


FS: TypeAlias = TypeVar(
    'FS', bound=(AbstractFragments | NpFragments | JaxFragments)
)
FSconcrete: TypeAlias = TypeVar(
    'FSconcrete', bound=(NpFragments | JaxFragments)
)


def _is_full(fragments: _GenericFragments[Any]) -> bool:
  """True iff every array element is covered by some fragment."""
  present = np.zeros(fragments.shape, dtype=bool)
  for f in fragments.fragments:
    present[f.index] = True
  return np.all(present)


def normalize(idx: Index, shape: Shape) -> Index:
  return tuple(
      slice(
          s.start if s.start is not None else 0,
          s.stop if s.stop is not None else dim,
          s.step if s.step is not None else 1,
      )
      for s, dim in zip(idx, shape)
  )


def addressable_shards(x: jax.Array | jax.ShapeDtypeStruct) -> list[Index]:
  """Computes list of fragment indices for addressable shards of a JAX array."""
  sharding = getattr(x, 'sharding', None)
  shape = x.shape
  if not sharding:
    return [tuple(slice(0, dim, 1) for dim in shape)]
  return [
      normalize(idx, shape)
      for idx in sharding.addressable_devices_indices_map(shape).values()
  ]


def abstract_fragments(
    x: jax.Array | jax.ShapeDtypeStruct | FS,
) -> AbstractFragments:
  """Returns abstract fragments matching the given object."""
  if isinstance(x, AbstractFragments):
    return x
  else:
    if isinstance(x, _GenericFragments):
      indices = (fragment.index for fragment in x.fragments)
    else:
      indices = addressable_shards(x)
    return AbstractFragments(x.shape, x.dtype, [
        AbstractFragment(index=index) for index in indices
    ])


def validate_fragments_can_be_stacked(fragments: FSconcrete) -> None:
  """Validates that the given fragments can be stacked."""
  if not fragments.fragments:
    raise ValueError('No fragments to stack.')
  fragment_arrays = [
      fragment.value
      for fragment in fragments.fragments
      if fragment.value is not None
  ]
  if len(fragment_arrays) != len(fragments.fragments):
    raise ValueError(f'Not all fragments have values: {fragments}')
  fragment_shape = fragment_arrays[0].shape
  if not all(a.shape == fragment_shape for a in fragment_arrays):
    raise ValueError(
        f"Differently-shaped fragments can't be cast to shards: {fragments}"
    )


def stack_fragments(fragments: FSconcrete | None) -> Aconcrete | None:
  """Stacks the given fragments, which must all have the same shape."""
  if fragments is None:
    return fragments
  validate_fragments_can_be_stacked(fragments)
  fragment_arrays = [fragment.value for fragment in fragments.fragments]
  np_api = fragments.FRAGMENT_T.NP_API
  return (
      np_api.expand_dims(fragment_arrays[0], axis=0)
      if len(fragment_arrays) == 1
      else np_api.stack(fragment_arrays)
  )
