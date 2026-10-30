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

#: Tuple of integers describing an array shape.
Shape = tuple[int, ...]

#: Array element data type (``jax.numpy.dtype`` or ``numpy.dtype``).
DType = jnp.dtype | np.dtype

if jax.__version_info__ >= (0, 6, 2):
  Format = jax_layout.Format
else:
  Format = jax_layout.Layout


#: Concrete scalar leaf type supported in checkpoints.
Scalar: TypeAlias = int | float | np.number | bytes | bool

#: Abstract representation of a scalar leaf (alias for :py:data:`~.Scalar`).
AbstractScalar = Scalar


class AbstractArray(Protocol):
  """Abstract representation of an array.

  This is a protocol for an abstract array that can be used to represent
  the metadata belonging to an array.

  Attributes:
    shape: Tuple of integers describing the array shape.
    dtype: Dtype of array elements.
  """

  shape: Shape | None
  dtype: DType | None


# TODO(dnlng): All attributes are made optional to support the case where
# the ArrayMetadata is passed into the metadata() call to pass only the
# `write_shape`. Optional attributes are not needed once write_shape is
# refactored.
class AbstractShardedArray(Protocol):
  """Abstract representation of a sharded array.

  This is a protocol for an abstract array that can be used to represent various
  metadata types such as :py:class:`jax.ShapeDtypeStruct` and
  :py:class:`~orbax.checkpoint.metadata.ArrayMetadata`.

  Attributes:
    shape: Tuple of integers describing the array shape.
    dtype: Dtype of array elements.
    sharding: Sharding to indicate how the array is sharded. This can be JAX's
      ``Sharding`` or ``Format``/``Layout`` or ``None``.
  """

  shape: Shape | None
  dtype: DType | None
  sharding: jax.sharding.Sharding | Format | None = None


#: Union of abstract array types (:py:class:`~.AbstractArray` or
#: :py:class:`~.AbstractShardedArray`).
ArrayLike: TypeAlias = AbstractArray | AbstractShardedArray
