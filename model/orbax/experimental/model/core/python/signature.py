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

"""Classes needed to describe function signatures."""

# pylint: disable=g-importing-member
from dataclasses import dataclass
from typing import Optional, Tuple

from orbax.experimental.model.core.protos import xla_data_pb2
from orbax.experimental.model.core.protos.saved_model import types_pb2

OpSharding = xla_data_pb2.OpSharding
# TODO(b/329309575): Decide whether to use None or -1 for unknown dim size.
DimSize = Optional[int]
UNKNOWN_DIM_SIZE: DimSize = None


def is_dim_size_unknown(dim_size: DimSize) -> bool:
  return dim_size is None or dim_size < 0


Shape = Optional[Tuple[DimSize, ...]]
DType = types_pb2.DataType


float32: DType = types_pb2.DT_FLOAT
float64: DType = types_pb2.DT_DOUBLE
int32: DType = types_pb2.DT_INT32


Sharding = OpSharding


# TODO(b/329741928): Add sharding spec to `TensorSpec`.
# TODO(b/329894394): Consider renaming `TensorSpec` to `TensorType`.
@dataclass
class TensorSpec:

  shape: Shape
  dtype: DType
  # None means unspecified sharding
  sharding: Optional[Sharding] = None

  def with_sharding(self, sharding: Optional[Sharding]) -> "TensorSpec":
    return TensorSpec(shape=self.shape, dtype=self.dtype, sharding=sharding)

  def with_dtype(self, dtype: DType) -> "TensorSpec":
    return TensorSpec(shape=self.shape, dtype=dtype, sharding=self.sharding)


# TODO(b/329305005): Support nested structures.
TreeOfTensorSpecs = TensorSpec


Signature = Tuple[TreeOfTensorSpecs, ...]


def flatten(*args: TreeOfTensorSpecs) -> Tuple[TensorSpec, ...]:
  return args


def assert_sub_dim_size(sub: DimSize, super_: DimSize) -> None:
  if is_dim_size_unknown(super_):
    return
  assert sub == super_


def assert_sub_shape(sub: Shape, super_: Shape) -> None:  # pylint: disable=g-doc-args
  """Asserts that `sub` is a sub-shape of `super_`.

  `A` is a sub-shape of `B` if `A` is equal to or a refinement of `B`. For
  examples:
  * Any shape is a sub-shape of `None` (the rank-unknown shape).
  * `[2, 3]` is a sub-shape of `[None, 3]`.
  * `[None, 3]` is not a sub-shape of `[2, 3]` .
  * `[2, 3]` is a sub-shape of `[2, 3]` (itself).
  * `[2, 3]` is not a sub-shape of `[1, 2, 3]` because they have different
      ranks.
  """
  if super_ is None:
    return
  assert sub is not None
  assert len(sub) == len(super_)
  for sub_dim, super_dim in zip(sub, super_):
    assert_sub_dim_size(sub_dim, super_dim)


def assert_sub_type(sub: TensorSpec, super_: TensorSpec) -> None:  # pylint: disable=g-doc-args
  """Asserts that `sub` is a sub-type (aka sub-spec) of `super_`.

  `A` is a sub-type of `B` if `A.dtype == B.dtype` and `A.shape` is a sub-shape
  of `B.shape`.
  """
  assert sub.dtype == super_.dtype
  assert_sub_shape(sub.shape, super_.shape)


def assert_sub_specs(sub: TreeOfTensorSpecs, super_: TreeOfTensorSpecs) -> None:
  assert isinstance(sub, TensorSpec)
  assert isinstance(super_, TensorSpec)
  assert_sub_type(sub, super_)


def assert_sub_signature(sub: Signature, super_: Signature) -> None:
  assert len(sub) == len(super_)
  for sub_spec, super_spec in zip(sub, super_):
    assert_sub_specs(sub_spec, super_spec)
