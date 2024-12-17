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

"""Utilities for building Manifest proto."""
# TODO(b/356174487):  Add unit tests.

from typing import Mapping
from absl import logging
from orbax.experimental.model.core.protos import manifest_pb2
from orbax.experimental.model.core.python import module
from orbax.experimental.model.core.python import unstructured_data
from orbax.experimental.model.core.python.function import Absence
from orbax.experimental.model.core.python.function import Function
from orbax.experimental.model.core.python.function import ShloDimSize
from orbax.experimental.model.core.python.function import ShloDType
from orbax.experimental.model.core.python.function import ShloShape
from orbax.experimental.model.core.python.function import ShloTensorSpec
from orbax.experimental.model.core.python.serializable_function import SerializableFunction
from orbax.experimental.model.core.python.shlo_function import ShloFunction
from orbax.experimental.model.core.python.tree_util import Tree
from orbax.experimental.model.core.python.unstructured_data import UnstructuredData
from orbax.experimental.model.core.python.value import ExternalValue


_SHLO_DTYPE_TO_MANIFEST_DTYPE: dict[ShloDType, manifest_pb2.DType] = {
    ShloDType.i8: manifest_pb2.DType.si8,
    ShloDType.i16: manifest_pb2.DType.si16,
    ShloDType.i32: manifest_pb2.DType.si32,
    ShloDType.i64: manifest_pb2.DType.si64,
    ShloDType.ui8: manifest_pb2.DType.ui8,
    ShloDType.ui16: manifest_pb2.DType.ui16,
    ShloDType.ui32: manifest_pb2.DType.ui32,
    ShloDType.ui64: manifest_pb2.DType.ui64,
    ShloDType.f16: manifest_pb2.DType.f16,
    ShloDType.f32: manifest_pb2.DType.f32,
    ShloDType.f64: manifest_pb2.DType.f64,
    ShloDType.bf16: manifest_pb2.DType.bf16,
}


def shlo_dtype_to_manifest_dtype(shlo_dtype: ShloDType) -> manifest_pb2.DType:
  return _SHLO_DTYPE_TO_MANIFEST_DTYPE[shlo_dtype]


_MANIFEST_DTYPE_TO_SHLO_DTYPE = {
    v: k for k, v in _SHLO_DTYPE_TO_MANIFEST_DTYPE.items()
}


def manifest_dtype_to_shlo_dtype(
    manifest_dtype: manifest_pb2.DType,
) -> ShloDType:
  """Converts manifest DType to ShloDType."""
  return _MANIFEST_DTYPE_TO_SHLO_DTYPE[manifest_dtype]


def shlo_shape_to_manifest_shape(
    shape: ShloShape,
) -> manifest_pb2.Shape:
  """Converts `ShloShape` to manifest `Shape`."""
  shape_proto = manifest_pb2.Shape()
  if shape is not None:
    shape_proto.shape_with_known_rank.SetInParent()
    for dim in shape:
      dim: ShloDimSize
      ds_proto = manifest_pb2.DimensionSize()
      if dim is not None:
        ds_proto.size = dim
      shape_proto.shape_with_known_rank.dimension_sizes.append(ds_proto)
  return shape_proto


def shlo_tensor_spec_to_manifest_tensor_type(
    spec: ShloTensorSpec,
) -> manifest_pb2.TensorType:
  """Converts ShloTensorSpec to manifest TensorType."""
  manifest_spec = manifest_pb2.TensorType()
  manifest_spec.shape.CopyFrom(shlo_shape_to_manifest_shape(spec.shape))
  manifest_spec.dtype = shlo_dtype_to_manifest_dtype(spec.dtype)
  if spec.sharding:
    manifest_spec.sharding.CopyFrom(spec.sharding)
  return manifest_spec


# TODO(b/356174487): Add more support for ordered structures.
def shlo_tensor_spec_pytree_to_manifest_type(
    tree: Tree[ShloTensorSpec],
) -> manifest_pb2.Type:
  """Recursively translates a PyTree of ShloTensorSpecs to manifest Type."""
  result = manifest_pb2.Type()
  if isinstance(tree, ShloTensorSpec):
    result.leaf.tensor_type.CopyFrom(
        shlo_tensor_spec_to_manifest_tensor_type(tree)
    )
  elif tree is None:
    result.none = manifest_pb2.NoneType()
  # list/tuple/dict can be empty, but we still want to record the type, so we
  # need to call SetInParent() for them.
  elif isinstance(tree, list):
    result.list.SetInParent()
    for x in tree:
      result.list.elements.append(shlo_tensor_spec_pytree_to_manifest_type(x))
  elif isinstance(tree, tuple):
    result.tuple.SetInParent()
    for x in tree:
      result.tuple.elements.append(shlo_tensor_spec_pytree_to_manifest_type(x))
  elif isinstance(tree, dict):
    tree: dict[str, Tree[ShloTensorSpec]]
    result.dict.SetInParent()
    for key, value in tree.items():
      result.dict.string_to_type[key].CopyFrom(
          shlo_tensor_spec_pytree_to_manifest_type(value)
      )
  else:
    raise ValueError(f"Unsupported tree type: {type(tree)}")
  return result


def _ignore_absence(fn, a):
  if isinstance(a, Absence):
    return None
  return fn(a)


def _maybe_copy_from(dst, src):
  if src is not None:
    dst.CopyFrom(src)


def build_function(fn: Function, path: str, name: str) -> manifest_pb2.Function:
  """Builds a TopLevelObject proto from a ShloFunction."""
  fn_proto = manifest_pb2.Function()
  # TODO(b/356174487): allow passing in options to control visibility.
  fn_proto.visibility = manifest_pb2.PUBLIC

  # Add input/output signature.
  input_type = _ignore_absence(
      shlo_tensor_spec_pytree_to_manifest_type, fn.input_signature
  )
  output_type = _ignore_absence(
      shlo_tensor_spec_pytree_to_manifest_type, fn.output_signature
  )
  fn_proto.signature.SetInParent()
  _maybe_copy_from(fn_proto.signature.input, input_type)
  _maybe_copy_from(fn_proto.signature.output, output_type)

  if isinstance(fn, ShloFunction):
    stable_hlo_body = fn_proto.body.stable_hlo_body
    # TODO(b/356174487): allow customized `file_system_location`,
    #   `mime_type`, `version`.
    stable_hlo_body.stable_hlo.inlined_bytes = fn.mlir_module_serialized
    stable_hlo_body.stable_hlo.mime_type = "mlir_stablehlo"
    stable_hlo_body.stable_hlo.version = "1.0"

    stable_hlo_body.calling_convention_version = fn.calling_convention_version
    for lowering_platform in fn.lowering_platforms:
      stable_hlo_body.lowering_platforms.append(lowering_platform)
    stable_hlo_body.module_kept_var_idx.extend(fn.module_kept_var_idx)
    if fn.supplemental_info is not None:
      # Sets up `stable_hlo_body.supplemental_info`
      supp = fn.supplemental_info.serializable_to_proto()
      supp_proto = supp.proto
      if supp.ext_name is not None:
        filename = unstructured_data.build_filename_from_extension(
            name + "_supplemental", supp.ext_name
        )
        supp_proto = unstructured_data.write_inlined_data_to_file(
            supp_proto, path, filename
        )
      stable_hlo_body.supplemental_info.CopyFrom(supp_proto)
  elif isinstance(fn, SerializableFunction):
    body_proto = fn.body.proto
    if fn.body.ext_name is not None:
      filename = unstructured_data.build_filename_from_extension(
          name, fn.body.ext_name
      )
      body_proto = unstructured_data.write_inlined_data_to_file(
          body_proto, path, filename
      )
    fn_proto.body.other.CopyFrom(body_proto)
  else:
    raise ValueError(f"Unsupported subclass of `Function`: {type(fn)}")

  return fn_proto


def build_manifest_proto(
    em_module: module.Module,
    path: str,
    supplemental_info: (
        UnstructuredData | Mapping[str, UnstructuredData] | None
    ) = None,
) -> manifest_pb2.Manifest:
  """Builds a Manifest proto from EM functions."""
  manifest_proto = manifest_pb2.Manifest()
  for name, obj in em_module.get_dict().items():
    if isinstance(obj, Function):
      fn = obj
      fn_proto = build_function(fn, path, name)
      manifest_proto.objects[name].function.CopyFrom(fn_proto)
    elif isinstance(obj, ExternalValue):
      value = obj
      if value.type is not None:
        raise NotImplementedError(
            "Serializing `ExternalValue.type` is not supported yet."
        )
      manifest_proto.objects[name].value.external.data.CopyFrom(value.data)
    else:
      raise ValueError(
          f"Unsupported module value type at key {name}: {type(obj)}"
      )

  if supplemental_info is not None:
    if isinstance(supplemental_info, UnstructuredData):
      manifest_proto.supplemental_info.single.CopyFrom(supplemental_info)
    else:
      manifest_proto.supplemental_info.multiple.SetInParent()
      for name, supp in supplemental_info.items():
        manifest_proto.supplemental_info.multiple.map[name].CopyFrom(supp)

  if logging.vlog_is_on(3):
    logging.vlog(3, f"Final manifest proto: {manifest_proto}")

  return manifest_proto


def manifest_shape_to_shlo_shape(
    shape: manifest_pb2.Shape,
) -> ShloShape:
  """Converts manifest `Shape` to `ShloShape`."""
  if shape.HasField("shape_with_known_rank"):
    return tuple(
        dim.size if dim.HasField("size") else None
        for dim in shape.shape_with_known_rank.dimension_sizes
    )
  return None


def manifest_tensor_type_to_shlo_tensor_spec(
    manifest_tensor_type: manifest_pb2.TensorType,
) -> ShloTensorSpec:
  """Converts manifest `TensorType` to `ShloTensorSpec`."""
  sharding = (
      manifest_tensor_type.sharding
      if manifest_tensor_type.HasField("sharding")
      else None
  )
  return ShloTensorSpec(
      shape=manifest_shape_to_shlo_shape(manifest_tensor_type.shape),
      dtype=manifest_dtype_to_shlo_dtype(manifest_tensor_type.dtype),
      sharding=sharding,
  )


def manifest_type_to_shlo_tensor_spec_pytree(
    manifest_type: manifest_pb2.Type,
) -> Tree[ShloTensorSpec]:
  """Recursively translates manifest `Type` to a PyTree of `ShloTensorSpec`s."""
  if manifest_type.HasField("leaf"):
    if manifest_type.leaf.HasField("tensor_type"):
      return manifest_tensor_type_to_shlo_tensor_spec(
          manifest_type.leaf.tensor_type
      )
    elif manifest_type.leaf.HasField("token_type"):
      raise NotImplementedError("TokenType is not supported yet")
  elif manifest_type.HasField("none"):
    return None
  elif manifest_type.HasField("list"):
    return [
        manifest_type_to_shlo_tensor_spec_pytree(element)
        for element in manifest_type.list.elements
    ]
  elif manifest_type.HasField("tuple"):
    return tuple(
        manifest_type_to_shlo_tensor_spec_pytree(element)
        for element in manifest_type.tuple.elements
    )
  elif manifest_type.HasField("dict"):
    return {
        key: manifest_type_to_shlo_tensor_spec_pytree(value)
        for key, value in manifest_type.dict.string_to_type.items()
    }
  else:
    raise ValueError(f"Unsupported manifest type: {manifest_type}")
