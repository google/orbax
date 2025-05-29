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

"""Utilities for building Manifest proto."""
# TODO(b/356174487):  Add unit tests.

from typing import Mapping
from absl import logging
from orbax.experimental.model.core.protos import manifest_pb2
from orbax.experimental.model.core.python import unstructured_data
from orbax.experimental.model.core.python.function import Function
from orbax.experimental.model.core.python.saveable import Saveable
from orbax.experimental.model.core.python.serializable_function import SerializableFunction
from orbax.experimental.model.core.python.shlo_function import ShloFunction
from orbax.experimental.model.core.python.type_proto_util import to_function_signature_proto
from orbax.experimental.model.core.python.unstructured_data import UnstructuredData
from orbax.experimental.model.core.python.value import ExternalValue


def build_function(fn: Function, path: str, name: str) -> manifest_pb2.Function:
  """Builds a TopLevelObject proto from a ShloFunction."""
  fn_proto = manifest_pb2.Function()
  # TODO(b/356174487): allow passing in options to control visibility.
  fn_proto.visibility = manifest_pb2.PUBLIC

  # Add input/output signature.
  fn_proto.signature.CopyFrom(
      to_function_signature_proto(fn.input_signature, fn.output_signature)
  )

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
      for supp_name, supp in fn.supplemental_info.items():
        supp = supp.serializable_to_proto()
        supp_proto = supp.proto
        if supp.ext_name is not None:
          filename = unstructured_data.build_filename_from_extension(
              name + "_supplemental", supp.ext_name
          )
          supp_proto = unstructured_data.write_inlined_data_to_file(
              supp_proto, path, filename
          )
        stable_hlo_body.supplemental_info[supp_name].CopyFrom(supp_proto)
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
    obm_module: dict[str, Saveable],
    path: str,
    supplemental_info: Mapping[str, UnstructuredData] | None = None,
) -> manifest_pb2.Manifest:
  """Builds a Manifest proto from EM functions."""
  manifest_proto = manifest_pb2.Manifest()
  for name, obj in obm_module.items():
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
    for name, supp in supplemental_info.items():
      manifest_proto.supplemental_info[name].CopyFrom(supp)

  if logging.vlog_is_on(3):
    logging.vlog(3, f"Final manifest proto: {manifest_proto}")

  return manifest_proto
