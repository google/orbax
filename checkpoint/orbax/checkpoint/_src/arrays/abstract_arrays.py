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

"""Utilities for dealing with abstract arrays."""

from typing import Protocol, Type
import jax
from jax import numpy as jnp
import numpy as np
from orbax.checkpoint._src.arrays import types
from orbax.checkpoint._src.metadata import sharding as sharding_metadata


ScalarType = Type[float] | Type[int]


class AbstractArrayLike(Protocol):
  """Abstract representation of an array.

  Can include objects like jax.Array, jax.ShapeDtypeStruct,
  ArrayRestoreArgs, and value_metadata.ArrayMetadata.
  """

  shape: types.Shape
  dtype: jnp.dtype | None
  sharding: jax.sharding.Sharding | sharding_metadata.ShardingMetadata | None


def _is_scalar(x):
  return isinstance(x, (int, float, np.number))


def to_shape_dtype_struct(
    x: AbstractArrayLike | np.ndarray,
    dtype: jnp.dtype | None = None,
    scalar_dtype: ScalarType | None = None,
):
  """Get ShapeDtypeStruct from array."""
  if isinstance(x, np.ndarray):
    dtype = dtype or x.dtype
    return jax.ShapeDtypeStruct(x.shape, dtype)
  elif _is_scalar(x):
    if scalar_dtype is not None:
      return scalar_dtype(x)
    return x
  else:
    shape = x.shape
    dtype = dtype or x.dtype
    sharding = x.sharding
    if isinstance(sharding, sharding_metadata.ShardingMetadata):
      sharding = sharding.to_jax_sharding()
    return jax.ShapeDtypeStruct(shape, dtype, sharding=sharding)
