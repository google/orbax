# Copyright 2023 The Orbax Authors.
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

"""Metadata describing PyTree values.""" ''

import dataclasses
from typing import Optional

from jax import numpy as jnp


@dataclasses.dataclass
class Metadata:
  """Metadata describing PyTree values."""

  def __eq__(self, other: 'Metadata') -> bool:
    return isinstance(other, Metadata)


@dataclasses.dataclass
class ArrayMetadata(Metadata):
  """Metadata describing an array.

  shape:
    Tuple of integers describing the array shape.
  shards:
    Tuple of integers indicating how many shards each dimension is divided
    into. May be None if the array is not sharded.
  dtype:
    Dtype of array elements.
  """

  shape: tuple[int, ...]
  shards: Optional[tuple[int, ...]]
  dtype: Optional[jnp.dtype]

  def __eq__(self, other: 'Metadata') -> bool:
    return (
        isinstance(other, ArrayMetadata)
        and self.shape == other.shape
        and self.shards == other.shards
        and self.dtype == other.dtype
    )


@dataclasses.dataclass
class ScalarMetadata(ArrayMetadata):
  """Metadata describing a scalar value.

  dtype:
    Scalar dtype.
  """

  shape: tuple[int, ...] = tuple([])
  shards: Optional[tuple[int, ...]] = None
  dtype: Optional[jnp.dtype] = None

  def __eq__(self, other: 'Metadata') -> bool:
    return isinstance(other, ScalarMetadata) and self.dtype == other.dtype


@dataclasses.dataclass
class StringMetadata(Metadata):
  """Metadata describing a string value."""

  def __eq__(self, other: 'Metadata') -> bool:
    return isinstance(other, StringMetadata)
