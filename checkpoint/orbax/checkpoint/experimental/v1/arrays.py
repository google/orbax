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

"""Public API for ``arrays`` package."""

# pylint: disable=g-importing-member, g-multiple-import, unused-import, g-bad-import-order

from orbax.checkpoint.experimental.v1._src.arrays import types as _arrays_types
from orbax.checkpoint.experimental.v1._src.arrays.abstract_arrays import (
    to_shape_dtype_struct,
)
from orbax.checkpoint.experimental.v1._src.arrays.types import (
    AbstractArray,
    AbstractShardedArray,
)

#: Tuple of integers describing an array shape.
Shape = _arrays_types.Shape

#: Array element data type (``jax.numpy.dtype`` or ``numpy.dtype``).
DType = _arrays_types.DType

#: Concrete scalar leaf type supported in checkpoints.
Scalar = _arrays_types.Scalar

#: Abstract representation of a scalar leaf (alias for :py:data:`Scalar`).
AbstractScalar = _arrays_types.AbstractScalar

#: Union of abstract array types (:py:class:`AbstractArray` or
#: :py:class:`AbstractShardedArray`).
ArrayLike = _arrays_types.ArrayLike
