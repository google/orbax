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

from collections.abc import Mapping, Sequence
from absl import logging
from orbax.experimental.model.core.protos import manifest_pb2
from orbax.experimental.model.core.python import unstructured_data
from orbax.experimental.model.core.python.device_assignment import DeviceAssignment
from orbax.experimental.model.core.python.function import Function
from orbax.experimental.model.core.python.saveable import Saveable
from orbax.experimental.model.core.python.serializable_function import SerializableFunction
from orbax.experimental.model.core.python.shlo_function import ShloFunction
from orbax.experimental.model.core.python.type_proto_util import to_function_signature_proto
from orbax.experimental.model.core.python.unstructured_data import UnstructuredData
from orbax.experimental.model.core.python.value import ExternalValue


def _build_function(
    fn: Function,
    path: str,
    name: str,
    visibility: manifest_pb2.Visibility,
) -> manifest_pb2.Function:
  """Builds a `Function` proto from a `Function` object."""
  fn_proto = manifest_pb2.Function()
  fn_proto.visibility = visibility

  # Add input/output signature.
  fn_proto.signature.CopyFrom(
      to_function_signature_proto(fn.input_signature, fn.output_signature)
  )

  if isinstance(fn, ShloFunction):
    stable_hlo_body = fn_proto.body.stable_hlo_body
    # TODO(b/356174487): allow customized `file_system_location`,
    #   `mime_type`, `version`.
    stable_hlo_body.stable_hlo.inlined_bytes = fn.mlir_module_serialized
    stable_hlo_body.stable_hlo.mime_type = "application/x.mlir-stablehlo"
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
              name + "_" + supp_name + "_supplemental", supp.ext_name
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


def build_device_assignment_by_coords_proto(
    device_assignment_by_coords: Sequence[DeviceAssignment],
) -> manifest_pb2.DeviceAssignmentByCoords:
  """Builds a DeviceAssignmentByCoords proto from a sequence of DeviceAssignment.

  Args:
    device_assignment_by_coords: A sequence of DeviceAssignment objects.

  Returns:
    A DeviceAssignmentByCoords proto.
  """
  device_assignment_by_coords_proto = manifest_pb2.DeviceAssignmentByCoords()
  for device_assignment in device_assignment_by_coords:
    device_proto = manifest_pb2.DeviceAssignmentByCoords.Device()
    device_proto.id = device_assignment.id
    if device_assignment.coords is not None:
      device_proto.coords.extend(device_assignment.coords)
    if device_assignment.core_on_chip is not None:
      device_proto.core_on_chip = device_assignment.core_on_chip
    device_assignment_by_coords_proto.devices.append(device_proto)
  return device_assignment_by_coords_proto


def _is_seq_of_functions(obj: Saveable) -> bool:
  """Checks if the object is a sequence of `Function`s."""
  return isinstance(obj, Sequence) and all(
      isinstance(elem, Function) for elem in obj
  )


def build_manifest_proto(
    obm_module: dict[str, Saveable],
    path: str,
    supplemental_info: Mapping[str, UnstructuredData] | None = None,
    names_to_visibilities: Mapping[str, manifest_pb2.Visibility] | None = None,
    device_assignment_by_coords: Sequence[DeviceAssignment] | None = None,
) -> manifest_pb2.Manifest:
  """Builds a Manifest proto from EM functions."""
  if names_to_visibilities is None:
    names_to_visibilities = {}
  manifest_proto = manifest_pb2.Manifest()
  for name, obj in obm_module.items():
    if isinstance(obj, Function):
      fn = obj
      fn_proto = _build_function(
          fn, path, name, names_to_visibilities.get(name, manifest_pb2.PUBLIC)
      )
      manifest_proto.objects[name].function.CopyFrom(fn_proto)
    elif _is_seq_of_functions(obj):
      fn_protos = [
          _build_function(
              fn,
              path,
              f"__{name}_{i}",
              names_to_visibilities.get(name, manifest_pb2.PUBLIC),
          )
          for i, fn in enumerate(obj)
      ]
      manifest_proto.objects[name].poly_fn.concrete_functions.extend(fn_protos)
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

  if device_assignment_by_coords is not None:
    device_assignment_by_coords_proto = build_device_assignment_by_coords_proto(
        device_assignment_by_coords
    )
    manifest_proto.device_assignment_by_coords.CopyFrom(
        device_assignment_by_coords_proto
    )

  if logging.vlog_is_on(3):
    logging.vlog(3, f"Final manifest proto: {manifest_proto}")

  return manifest_proto
