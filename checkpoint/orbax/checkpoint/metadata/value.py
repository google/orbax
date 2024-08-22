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

"""Metadata describing PyTree values.""" ''

import dataclasses
from typing import Optional

from etils import epath
import jax
from jax import numpy as jnp
from orbax.checkpoint.metadata import sharding as sharding_metadata


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


@dataclasses.dataclass(frozen=True)
class StorageMetadata:
  """Metadata describing how arrays are stored in a checkpoint."""
  chunk_shape: Optional[tuple[int, ...]]


@dataclasses.dataclass
class ArrayMetadata(Metadata):
  """Metadata describing an array.

  shape:
    Tuple of integers describing the array shape.
  sharding:
    ShardingMetadata to indicate how the array is sharded. ShardingMetadata is
    an orbax representation of `jax.sharding.Sharding` which stores the same
    properties but not require accessing real devices.
  dtype:
    Dtype of array elements.
  storage:
    Optional metadata describing how the array is stored in a checkpoint.
  """

  shape: tuple[int, ...]
  sharding: Optional[sharding_metadata.ShardingMetadata]
  dtype: Optional[jnp.dtype]
  storage: Optional[StorageMetadata] = None

  def __eq__(self, other: 'Metadata') -> bool:
    return (
        isinstance(other, ArrayMetadata)
        and self.shape == other.shape
        and self.sharding == other.sharding
        and self.dtype == other.dtype
        and self.storage == other.storage
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
  sharding: Optional[sharding_metadata.ShardingMetadata] = None
  dtype: Optional[jnp.dtype] = None

  def __eq__(self, other: 'Metadata') -> bool:
    return isinstance(other, ScalarMetadata) and self.dtype == other.dtype


@dataclasses.dataclass
class StringMetadata(Metadata):
  """Metadata describing a string value."""

  def __eq__(self, other: 'Metadata') -> bool:
    return isinstance(other, StringMetadata)
