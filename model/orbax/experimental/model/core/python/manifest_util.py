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

"""Utilities for building Manifest proto."""

# TODO(b/356174487):  Add unit tests.

from collections.abc import Mapping, Sequence
from absl import logging
from orbax.experimental.model.core.protos import manifest_pb2
from orbax.experimental.model.core.python import device_assignment
from orbax.experimental.model.core.python import function
from orbax.experimental.model.core.python import saveable
from orbax.experimental.model.core.python import serializable_function
from orbax.experimental.model.core.python import shlo_function
from orbax.experimental.model.core.python import type_proto_util
from orbax.experimental.model.core.python import unstructured_data
from orbax.experimental.model.core.python import value


def _build_function(
    fn: function.Function,
    target_dir: str,
    name: str,
    visibility: manifest_pb2.Visibility,
) -> manifest_pb2.Function:
  """Builds a `manifest_pb2.Function` proto from a `function.Function` object."""

  fn_proto = manifest_pb2.Function()
  fn_proto.visibility = visibility

  if fn.data_names is not None:
    fn_proto.data_names.extend(fn.data_names)

  fn_proto.signature.CopyFrom(
      type_proto_util.to_function_signature_proto(
          fn.input_signature, fn.output_signature
      )
  )

  if isinstance(fn, shlo_function.ShloFunction):
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

    # Serialize and add all supplementals to the function body proto.
    if fn.supplemental_info is not None:
      for supp_name, supp in fn.supplemental_info.items():
        supp = supp.serializable_to_proto()
        supp_proto = supp.proto

        if supp.ext_name is not None:
          filename = unstructured_data.build_relative_filepath_from_extension(
              name + "_" + supp_name + "_supplemental", supp.ext_name
          )
          supp_proto = unstructured_data.write_inlined_data_to_file(
              supp_proto, target_dir, filename
          )
        stable_hlo_body.supplemental_info[supp_name].CopyFrom(supp_proto)

  elif isinstance(fn, serializable_function.SerializableFunction):
    body_proto = fn.body.proto

    if fn.body.ext_name is not None:
      relative_filepath = (
          unstructured_data.build_relative_filepath_from_extension(
              name, fn.body.ext_name, subfolder=fn.body.subfolder
          )
      )
      body_proto = unstructured_data.write_inlined_data_to_file(
          body_proto, target_dir, relative_filepath
      )

    fn_proto.body.other.CopyFrom(body_proto)

  else:
    raise ValueError(f"Unsupported subclass of `Function`: {type(fn)}")

  return fn_proto


def build_device_assignment_by_coords_proto(
    device_assignments: Sequence[device_assignment.DeviceAssignment],
) -> manifest_pb2.DeviceAssignmentByCoords:
  """Builds a DeviceAssignmentByCoords proto from a sequence of DeviceAssignment.

  Args:
    device_assignments: A sequence of DeviceAssignment objects.

  Returns:
    A DeviceAssignmentByCoords proto.
  """
  proto = manifest_pb2.DeviceAssignmentByCoords()

  for assignment in device_assignments:
    device_proto = proto.devices.add(id=assignment.id)
    if assignment.coords is not None:
      device_proto.coords.extend(assignment.coords)
    if assignment.core_on_chip is not None:
      device_proto.core_on_chip = assignment.core_on_chip

  return proto


def _is_seq_of_functions(obj: saveable.Saveable) -> bool:
  """Checks if the object is a sequence of `Function`s."""
  return isinstance(obj, Sequence) and all(
      isinstance(elem, function.Function) for elem in obj
  )


def build_manifest_proto(
    obm_module: dict[str, saveable.Saveable],
    target_dir: str,
    supplementals: (
        Mapping[str, unstructured_data.UnstructuredData] | None
    ) = None,
    visibilities: Mapping[str, manifest_pb2.Visibility] | None = None,
    device_assignments: (
        Sequence[device_assignment.DeviceAssignment] | None
    ) = None,
) -> manifest_pb2.Manifest:
  """Builds a manifest proto from an OBM module."""

  if visibilities is None:
    visibilities = {}
  manifest = manifest_pb2.Manifest()

  for name, obj in obm_module.items():
    visibility = visibilities.get(name, manifest_pb2.PUBLIC)

    if isinstance(obj, function.Function):
      manifest.objects[name].function.CopyFrom(
          _build_function(obj, target_dir, name, visibility)
      )

    elif _is_seq_of_functions(obj):
      fn_protos = [
          _build_function(
              fn,
              target_dir,
              f"__{name}_{i}",
              visibility,
          )
          for i, fn in enumerate(obj)
      ]
      manifest.objects[name].poly_fn.concrete_functions.extend(fn_protos)

    elif isinstance(obj, value.ExternalValue):
      if obj.type is not None:
        raise NotImplementedError(
            "Serializing `ExternalValue.type` is not supported yet."
        )
      manifest.objects[name].value.external.data.CopyFrom(obj.data)

    else:
      raise ValueError(
          f"Unsupported module value type at key {name}: {type(obj)}"
      )

  if supplementals is not None:
    for name, supp in supplementals.items():
      manifest.supplemental_info[name].CopyFrom(supp)

  if device_assignments is not None:
    manifest.device_assignment_by_coords.CopyFrom(
        build_device_assignment_by_coords_proto(device_assignments)
    )

  logging.vlog(3, f"Final manifest proto: {manifest}")

  return manifest
