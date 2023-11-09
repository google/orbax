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

from etils import epath
import jax
from jax import numpy as jnp


@dataclasses.dataclass
class Metadata:
  """Metadata describing PyTree values.

  name:
    A string representing the original name of the parameter.
  directory:
    The directory where the parameter can be found, after taking `name` into
    account.
  """

  name: str
  directory: Optional[epath.Path]

  def __eq__(self, other: 'Metadata') -> bool:
    return isinstance(other, Metadata)


@dataclasses.dataclass
class ArrayMetadata(Metadata):
  """Metadata describing an array.

  shape:
    Tuple of integers describing the array shape.
  sharding:
    jax.sharding.Sharding to indicate how the array is sharded. In most of the
    cases, it's NamedSharding.
    May be None if the array is not sharded.
  dtype:
    Dtype of array elements.
  """

  shape: tuple[int, ...]
  sharding: Optional[jax.sharding.Sharding]
  dtype: Optional[jnp.dtype]

  def __eq__(self, other: 'Metadata') -> bool:
    return (
        isinstance(other, ArrayMetadata)
        and self.shape == other.shape
        and self.sharding == other.sharding
        and self.dtype == other.dtype
    )

  @classmethod
  def from_shape_dtype_struct(
      cls,
      s: jax.ShapeDtypeStruct,
      name: Optional[str] = None,
      directory: Optional[epath.Path] = None,
  ) -> 'ArrayMetadata':
    return cls(
        name=name,
        directory=directory,
        shape=s.shape,
        sharding=s.sharding,
        dtype=s.dtype,
    )


@dataclasses.dataclass
class ScalarMetadata(ArrayMetadata):
  """Metadata describing a scalar value.

  dtype:
    Scalar dtype.
  """

  shape: tuple[int, ...] = tuple([])
  sharding: Optional[jax.sharding.Sharding] = None
  dtype: Optional[jnp.dtype] = None

  def __eq__(self, other: 'Metadata') -> bool:
    return isinstance(other, ScalarMetadata) and self.dtype == other.dtype


@dataclasses.dataclass
class StringMetadata(Metadata):
  """Metadata describing a string value."""

  def __eq__(self, other: 'Metadata') -> bool:
    return isinstance(other, StringMetadata)
