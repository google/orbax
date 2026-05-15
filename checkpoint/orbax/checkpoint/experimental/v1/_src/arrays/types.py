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

"""Array type definitions."""

from typing import Protocol, TypeAlias

import jax
from jax import numpy as jnp
import jax.experimental.layout as jax_layout
import numpy as np

Shape = tuple[int, ...]
DType = jnp.dtype | np.dtype

if jax.__version_info__ >= (0, 6, 2):
  Format = jax_layout.Format
else:
  Format = jax_layout.Layout


Scalar: TypeAlias = int | float | np.number | bytes | bool
AbstractScalar = Scalar


class AbstractArray(Protocol):
  """Abstract representation of an array.

  This is a protocol for an abstract array that can be used to represent
  the metadata belonging to an array.

  shape:
    Tuple of integers describing the array shape.
  dtype:
    Dtype of array elements.
  """

  shape: Shape | None
  dtype: DType | None


class AbstractShardedArray(Protocol):
  """Abstract representation of an array.

  This is a protocol for an abstract array that can be used to represent various
  metadata types such as :py:class:`jax.ShapeDtypeStruct` and
  :py:class:`~orbax.checkpoint.metadata.ArrayMetadata`.

  #TODO(dnlng): All attributes are made optional to support the case where
  # the ArrayMetadata is passed into the metadata() call to pass only the
  # `write_shape`.  Optional attributes are not needed once write_shape is
  # refactored.


  shape:
    Tuple of integers describing the array shape.
  dtype:
    Dtype of array elements.
  Sharding:
    Sharding to indicate how the array is sharded. This can be jax's Sharding or
    Layout or None.
  """

  shape: Shape | None
  dtype: DType | None
  sharding: jax.sharding.Sharding | Format | None = None  # pytype: disable=invalid-annotation


ArrayLike: TypeAlias = AbstractArray | AbstractShardedArray
