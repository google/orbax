# Copyright 2026 The Orbax Authors.
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

"""Utilities for working with fragments."""

import dataclasses
import itertools
from typing import Collection, Generic, Iterator, Sequence, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint._src.arrays import fragments as array_fragments

AbstractFragment = array_fragments.AbstractFragment
NpFragment = array_fragments.NpFragment
JaxFragment = array_fragments.JaxFragment
AbstractFragments = array_fragments.AbstractFragments
NpFragments = array_fragments.NpFragments
JaxFragments = array_fragments.JaxFragments

F = TypeVar('F', bound=(AbstractFragment | NpFragment | JaxFragment))
FS = TypeVar('FS', bound=(AbstractFragments | NpFragments | JaxFragments))

Fconcrete = TypeVar('Fconcrete', bound=(NpFragment | JaxFragment))
FSconcrete = TypeVar('FSconcrete', bound=(NpFragments | JaxFragments))

FragmentList = Sequence[F]


class _HashedByIndex(Generic[F]):
  """A box to cache the hash of a fragment's index."""

  __slots__ = ['fragment', 'hash']

  def __init__(self, fragment: F):
    self.fragment = fragment
    self.hash = hash(tuple(fragment.np_index.flat))

  def __hash__(self):
    return self.hash

  def __eq__(self, other):
    return (self.fragment.np_index == other.fragment.np_index).all()


_RANK_0_ABSTRACT = _HashedByIndex(AbstractFragment(index=(), value=None))


def _gen_overlap_fragments(
    fragment_t: type[Fconcrete],
    source_fragments: FragmentList[Fconcrete],
    required_fragment: AbstractFragment,
) -> Iterator[Fconcrete]:
  """Yields slices of fragments that overlap the required fragment.

  The resulting fragments are in the same coordinate space as the required
  fragment. In particular, if one source fragment exactly covers the required
  fragment then it will be returned directly.

  Args:
    fragment_t: The type of the fragments to be returned.
    source_fragments: The fragments to be sliced.
    required_fragment: The fragment to be extracted.
  """
  for source_fragment in source_fragments:
    overlap_start = np.maximum(required_fragment.start, source_fragment.start)
    overlap_stop = np.minimum(required_fragment.stop, source_fragment.stop)

    overlap = AbstractFragment(
        np_index=np.stack(
            [overlap_start, overlap_stop, source_fragment.step], axis=1
        )
    )

    if any(x <= 0 for x in overlap.shape):
      # This source fragment doesn't overlap at all.
      continue

    if overlap.shape == required_fragment.shape:
      # This source fragment supplies everything we need.
      if overlap.shape == source_fragment.shape:
        # The source fragment fits exactly.
        yield source_fragment
        return
      else:
        # The source fragment is too big.
        overlap_value = source_fragment.value[
            overlap.offset_by(-source_fragment.start).index
        ]
        yield fragment_t(
            np_index=required_fragment.np_index, value=overlap_value
        )
        return

    overlap_value = source_fragment.value[
        overlap.offset_by(-source_fragment.start).index
    ]
    yield fragment_t(np_index=overlap.np_index, value=overlap_value)


def extract_fragment(
    source_fragments: FragmentList[Fconcrete],
    required_fragment: AbstractFragment,
    fragment_t: type[Fconcrete] | None = None,
) -> Fconcrete:
  """Given concrete fragments, construct the required fragment."""
  if fragment_t is None:
    assert source_fragments, 'source_fragments cannot be empty'
    fragment_t = type(source_fragments[0])

  assert not required_fragment.is_degenerate()

  overlaps = [
      *_gen_overlap_fragments(fragment_t, source_fragments, required_fragment)
  ]

  if len(overlaps) == 1:
    return overlaps[0]
  else:
    np_api = fragment_t.NP_API
    value = np_api.empty(required_fragment.shape, overlaps[0].value.dtype)
    value = jax.ref.new_ref(value) if np_api is jnp else value
    for overlap in overlaps:
      value[overlap.offset_by(-required_fragment.start).index] = overlap.value
    value = jax.ref.freeze(value) if np_api is jnp else value
    return fragment_t(np_index=required_fragment.np_index, value=value)


def _union_fragments(
    fs: Collection[_HashedByIndex[AbstractFragment]],
    axis: int,
    ndim: int,
) -> Iterator[_HashedByIndex[AbstractFragment]]:
  """Implementation of `union_fragments()`."""
  if axis == ndim:
    if fs:
      yield _RANK_0_ABSTRACT
    return

  if len(fs) == 1:
    f = next(iter(fs)).fragment
    if not f.is_degenerate():
      yield _HashedByIndex(
          AbstractFragment(np_index=f.np_index[axis:])
      )
    return
  get_x = lambda x_and_fragment: x_and_fragment[0]
  edges = sorted(
      itertools.chain.from_iterable(
          ((f.fragment.start[axis], f), (f.fragment.stop[axis], f)) for f in fs
      ),
      key=get_x,
  )

  active_in_fragments = set()
  active_out_fragments = {}

  for x, edge_group in itertools.groupby(edges, key=get_x):
    for _, in_fragment in edge_group:
      if in_fragment.fragment.start[axis] == x:
        active_in_fragments.add(in_fragment)
      if in_fragment.fragment.stop[axis] == x:
        active_in_fragments.discard(in_fragment)

    next_active = {}
    for f in _union_fragments(active_in_fragments, axis + 1, ndim):
      next_active[f] = active_out_fragments.get(f, x)

    for out_fragment, start in active_out_fragments.items():
      if out_fragment not in next_active and start != x:
        yield _HashedByIndex(
            AbstractFragment(
                np_index=np.r_[[[start, x, 1]], out_fragment.fragment.np_index]
            )
        )

    active_out_fragments = next_active


def union_fragment_list(
    fs: FragmentList[F],
) -> Iterator[F]:
  """Calculates fragments that span the same indices as `fs` without overlaps.

  Args:
    fs: Fragments whose union to take.

  Yields:
    Fragments that, taken together, cover the same indices as `fs` but that
    do not overlap each other.
  """
  if not fs:
    return
  one_fragment = next(iter(fs))
  ndim = len(one_fragment.np_index)
  concrete = one_fragment.value is not None
  for f in _union_fragments(
      [_HashedByIndex(AbstractFragment(np_index=frag.np_index)) for frag in fs],
      0,
      ndim,
  ):
    if concrete:
      yield extract_fragment(fs, f.fragment)
    else:
      yield f.fragment


def union_fragments(fs: FS) -> FS:
  return dataclasses.replace(fs, fragments=[*union_fragment_list(fs.fragments)])
