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

"""The `Function` base class."""

import dataclasses
import enum
from typing import Any, Optional, Sequence, Tuple, TypeAlias

import numpy as np
from orbax.experimental.model.core.python import tree_util

from orbax.experimental.model.core.protos import xla_data_pb2


Sharding: TypeAlias = xla_data_pb2.OpSharding
Layout: TypeAlias = xla_data_pb2.LayoutProto
ShloDimSize: TypeAlias = Optional[int]
ShloShape: TypeAlias = Optional[Sequence[ShloDimSize]]


# pylint: disable=invalid-name
# Copied from /third_party/py/jax/_src/export/serialization.fbs
class ShloDType(enum.Enum):  # pylint: disable=missing-class-docstring
  bool = 0
  i8 = 1
  i16 = 2
  i32 = 3
  i64 = 4
  ui8 = 5
  ui16 = 6
  ui32 = 7
  ui64 = 8
  f16 = 9
  f32 = 10
  f64 = 11
  c64 = 12
  c128 = 13

  bf16 = 14

  i4 = 15
  ui4 = 16

  f8_e4m3b11fnuz = 17
  f8_e4m3fn = 18
  f8_e4m3fnuz = 19
  f8_e5m2 = 20
  f8_e5m2fnuz = 21

  str = 100


_NP_DTYPE_TO_SHLO_DTYPE: dict[np.dtype[Any], ShloDType] = {
    np.dtype(np.bool): ShloDType.bool,
    np.dtype(np.int8): ShloDType.i8,
    np.dtype(np.int16): ShloDType.i16,
    np.dtype(np.int32): ShloDType.i32,
    np.dtype(np.int64): ShloDType.i64,
    np.dtype(np.uint8): ShloDType.ui8,
    np.dtype(np.uint16): ShloDType.ui16,
    np.dtype(np.uint32): ShloDType.ui32,
    np.dtype(np.uint64): ShloDType.ui64,
    np.dtype(np.float16): ShloDType.f16,
    np.dtype(np.float32): ShloDType.f32,
    np.dtype(np.float64): ShloDType.f64,
    np.dtype(np.complex64): ShloDType.c64,
    np.dtype(np.complex128): ShloDType.c128,
}


_SHLO_DTYPE_TO_NP_DTYPE = {v: k for k, v in _NP_DTYPE_TO_SHLO_DTYPE.items()}


def np_dtype_to_shlo_dtype(dtype: np.dtype[Any]) -> ShloDType:
  return _NP_DTYPE_TO_SHLO_DTYPE[dtype]


def shlo_dtype_to_np_dtype(dtype: ShloDType) -> np.dtype[Any]:
  return _SHLO_DTYPE_TO_NP_DTYPE[dtype]


# TODO(wangpeng): value.py needs this class, so we should move this class out
#   of function.py .
@dataclasses.dataclass
class ShloTensorSpec:

  """A specification for the shape, dtype, sharding, and layout of a StableHLO tensor.

  Attributes:
    shape: The shape of the tensor.
    dtype: The dtype of the tensor.
    sharding: The sharding of the tensor. None means unspecified sharding.
    layout: The layout of the tensor. None means the default layout is used.
    name: The name of the tensor.
  """
  shape: ShloShape
  dtype: ShloDType
  sharding: Sharding | None = None
  layout: Layout | None = None
  name: str| None = None


@dataclasses.dataclass(kw_only=True)
class Function:
  """An abstract base class for functions whose signatures are StableHLO types.

  Attributes:
    input_signature: the input signature of the function.
    output_signature: the output signature of the function.
    data_names: checkpoint data names used by the function.
    signature: the pair `(input_signature, output_signature)`.
  """

  input_signature: tree_util.Tree[ShloTensorSpec]
  output_signature: tree_util.Tree[ShloTensorSpec]
  data_names: Sequence[str] | None = None
  # TODO(b/372084833): Add `vjp_name``.

  @property
  def signature(
      self,
  ) -> Tuple[tree_util.Tree[ShloTensorSpec], tree_util.Tree[ShloTensorSpec]]:
    return self.input_signature, self.output_signature
