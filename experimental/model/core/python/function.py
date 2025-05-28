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

"""The `Function` base class."""

from dataclasses import dataclass  # pylint: disable=g-importing-member
import enum
from typing import Any, Optional, Sequence

import numpy as np
from orbax.experimental.model.core.python.tree_util import Tree

from tensorflow.compiler.xla import xla_data_pb2  # pylint: disable=g-direct-tensorflow-import


ShloDimSize = Optional[int]
ShloShape = Optional[Sequence[ShloDimSize]]


# pylint: disable=invalid-name
# Copied from jax/experimental/export/serialization.fbs
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


Sharding = xla_data_pb2.OpSharding


@dataclass
class ShloTensorSpec:

  shape: ShloShape
  dtype: ShloDType
  # None means unspecified sharding
  sharding: Optional[Sharding] = None


class Absence:
  pass


@dataclass(kw_only=True)
class Function:
  """An abstract base class for functions whose signatures are StableHLO types.

  Attributes:
    input_signature: the input signature of the function.
    output_signature: the output signature of the function.
  """

  # We can't use `None` to indicate absence because `None` is a valid tree.
  input_signature: Tree[ShloTensorSpec] | Absence = Absence()
  output_signature: Tree[ShloTensorSpec] | Absence = Absence()
  # TODO(b/372084833): Add `vjp_name``.
