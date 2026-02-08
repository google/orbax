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

"""Some utilities for type.proto ."""

import jax
from orbax.experimental.model.core.protos import type_pb2
from orbax.experimental.model.core.python.function import ShloDimSize
from orbax.experimental.model.core.python.function import ShloDType
from orbax.experimental.model.core.python.function import ShloShape
from orbax.experimental.model.core.python.function import ShloTensorSpec
from orbax.experimental.model.core.python.tree_util import Tree
# TODO(wangpeng): Replace all "manifest" with "type_proto" in this file.


_SHLO_DTYPE_TO_MANIFEST_DTYPE: dict[ShloDType, type_pb2.DType] = {
    ShloDType.bool: type_pb2.DType.i1,
    ShloDType.i4: type_pb2.DType.si4,
    ShloDType.i8: type_pb2.DType.si8,
    ShloDType.i16: type_pb2.DType.si16,
    ShloDType.i32: type_pb2.DType.si32,
    ShloDType.i64: type_pb2.DType.si64,
    ShloDType.ui4: type_pb2.DType.ui4,
    ShloDType.ui8: type_pb2.DType.ui8,
    ShloDType.ui16: type_pb2.DType.ui16,
    ShloDType.ui32: type_pb2.DType.ui32,
    ShloDType.ui64: type_pb2.DType.ui64,
    ShloDType.f16: type_pb2.DType.f16,
    ShloDType.f32: type_pb2.DType.f32,
    ShloDType.f64: type_pb2.DType.f64,
    ShloDType.bf16: type_pb2.DType.bf16,
    ShloDType.str: type_pb2.DType.str,
}


def to_dtype_proto(shlo_dtype: ShloDType) -> type_pb2.DType:
  return _SHLO_DTYPE_TO_MANIFEST_DTYPE[shlo_dtype]


_MANIFEST_DTYPE_TO_SHLO_DTYPE = {
    v: k for k, v in _SHLO_DTYPE_TO_MANIFEST_DTYPE.items()
}


def from_dtype_proto(
    manifest_dtype: type_pb2.DType,
) -> ShloDType:
  """Converts manifest DType to ShloDType."""
  return _MANIFEST_DTYPE_TO_SHLO_DTYPE[manifest_dtype]


def to_shape_proto(
    shape: ShloShape,
) -> type_pb2.Shape:
  """Converts `ShloShape` to manifest `Shape`."""
  shape_proto = type_pb2.Shape()
  if shape is not None:
    shape_proto.shape_with_known_rank.SetInParent()
    for dim in shape:
      dim: ShloDimSize
      ds_proto = type_pb2.DimensionSize()
      if dim is not None:
        ds_proto.size = dim
      shape_proto.shape_with_known_rank.dimension_sizes.append(ds_proto)
  return shape_proto


def to_tensor_type(
    spec: ShloTensorSpec,
) -> type_pb2.TensorType:
  """Converts ShloTensorSpec to manifest TensorType."""
  manifest_spec = type_pb2.TensorType()
  manifest_spec.shape.CopyFrom(to_shape_proto(spec.shape))
  manifest_spec.dtype = to_dtype_proto(spec.dtype)
  if spec.sharding:
    manifest_spec.sharding.CopyFrom(spec.sharding)
  if spec.layout:
    manifest_spec.layout.CopyFrom(spec.layout)
  return manifest_spec


# TODO(b/356174487): Add more support for ordered structures.
def to_type_proto(
    tree: Tree[ShloTensorSpec],
) -> type_pb2.Type:
  """Recursively translates a PyTree of ShloTensorSpecs to manifest Type."""
  result = type_pb2.Type()
  if isinstance(tree, ShloTensorSpec):
    result.leaf.tensor_type.CopyFrom(to_tensor_type(tree))
  elif tree is None:
    result.none.SetInParent()
  # list/tuple/dict can be empty, but we still want to record the type, so we
  # need to call SetInParent() for them.
  elif isinstance(tree, list):
    result.list.SetInParent()
    for x in tree:
      result.list.elements.append(to_type_proto(x))
  elif isinstance(tree, tuple):
    result.tuple.SetInParent()
    for x in tree:
      result.tuple.elements.append(to_type_proto(x))
  elif isinstance(tree, dict):
    tree: dict[str, Tree[ShloTensorSpec]]
    result.dict.SetInParent()
    for key, value in tree.items():
      result.dict.string_to_type[key].CopyFrom(to_type_proto(value))
  else:
    # TODO: b/444497750 - Now we flattern the tree as list, but the custom
    # pytree node sometime can be respresented as other structures, like dict,
    # we should use the PytreeDef to decide how to represent it.
    leaves, tree_def = jax.tree_util.tree_flatten(tree)
    # If tree_def.node_data() is empty, it means `tree` is a leaf node.
    # Since we have handled all supported leaf types (ShloTensorSpec and None)
    # above, this indicates an unsupported leaf type.
    if not tree_def.node_data():
      raise ValueError(f"Unsupported tree type: {type(tree)}")
    result.list.SetInParent()
    for x in leaves:
      result.list.elements.append(to_type_proto(x))
  return result


def to_function_signature_proto(
    input_signature: Tree[ShloTensorSpec],
    output_signature: Tree[ShloTensorSpec],
) -> type_pb2.FunctionSignature:
  signature = type_pb2.FunctionSignature()
  signature.input.CopyFrom(to_type_proto(input_signature))
  signature.output.CopyFrom(to_type_proto(output_signature))
  return signature


def from_shape_proto(
    shape: type_pb2.Shape,
) -> ShloShape:
  """Converts manifest `Shape` to `ShloShape`."""
  if shape.HasField("shape_with_known_rank"):
    return tuple(
        dim.size if dim.HasField("size") else None
        for dim in shape.shape_with_known_rank.dimension_sizes
    )
  return None


def from_tensor_type_proto(
    manifest_tensor_type: type_pb2.TensorType,
) -> ShloTensorSpec:
  """Converts manifest `TensorType` to `ShloTensorSpec`."""
  sharding = (
      manifest_tensor_type.sharding
      if manifest_tensor_type.HasField("sharding")
      else None
  )
  return ShloTensorSpec(
      shape=from_shape_proto(manifest_tensor_type.shape),
      dtype=from_dtype_proto(manifest_tensor_type.dtype),
      sharding=sharding,
  )


def from_type_proto(
    manifest_type: type_pb2.Type,
) -> Tree[ShloTensorSpec]:
  """Recursively translates manifest `Type` to a PyTree of `ShloTensorSpec`s."""
  if manifest_type.HasField("leaf"):
    if manifest_type.leaf.HasField("tensor_type"):
      return from_tensor_type_proto(manifest_type.leaf.tensor_type)
    elif manifest_type.leaf.HasField("token_type"):
      raise NotImplementedError("TokenType is not supported yet")
  elif manifest_type.HasField("none"):
    return None
  elif manifest_type.HasField("list"):
    return [from_type_proto(element) for element in manifest_type.list.elements]
  elif manifest_type.HasField("tuple"):
    return tuple(
        from_type_proto(element) for element in manifest_type.tuple.elements
    )
  elif manifest_type.HasField("dict"):
    return {
        key: from_type_proto(value)
        for key, value in manifest_type.dict.string_to_type.items()
    }
  else:
    raise ValueError(f"Unsupported manifest type: {manifest_type}")
