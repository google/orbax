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

"""Utility functions for XLA compile options."""

from collections.abc import Mapping, Sequence
import logging
from typing import Any

from absl import flags
from google.protobuf import descriptor
from google.protobuf import text_format
import jax
from orbax.experimental.model.core.protos import manifest_pb2

from .google.protobuf import any_pb2
from .platforms.xla.service.jellyfish import tpu_compilation_environment_pb2 as tpu_comp_env_pb2
from .platforms.xla.service.jellyfish.python import tpu_compilation_environment as tpu_comp_env
from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.compiler.xla import xla_pb2
from tensorflow.compiler.xla.pjrt.proto import compile_options_pb2

# A mapping between XLAflag names and protobuf field names.
_XLA_FLAG_TO_FIELD_MAP = {
    field.name: field
    for field in tpu_comp_env_pb2.TpuCompilationEnvironment.DESCRIPTOR.fields
}


def generate_tpu_compilation_env(
    xla_flags: Sequence[str] | None = None,
) -> xla_pb2.CompilationEnvironmentsProto:
  """Generates the TPU compilation environment."""
  # Get default TPU compilation environment.
  tpu_compilation_env_str = tpu_comp_env.create_default_tpu_comp_env()
  env = tpu_comp_env_pb2.TpuCompilationEnvironment.FromString(
      tpu_compilation_env_str
  )
  # Override with supplied XLA flags if any is provided.
  if xla_flags:
    is_proto_formatted = False if xla_flags[0].startswith('--') else True
    if is_proto_formatted:
      merge_proto_formatted_flags_into_compile_options(xla_flags, env)
    else:
      parsed_flags = {}
      for flag in xla_flags:
        if not flag.startswith('--'):
          raise ValueError(
              f"Flag {flag} does not start with '--'. All flags must be in the"
              ' format of --flag_name=flag_value.'
          )
        flag_name, flag_value = flag[2:].split('=', 1)
        parsed_flags[flag_name] = flag_value
      merge_flags_into_compile_options(parsed_flags, env)

  # Pack the TPU compilation environment into a compilation env proto.
  any_proto = any_pb2.Any()
  any_proto.Pack(env)
  compilation_env_proto = xla_pb2.CompilationEnvironmentsProto()
  compilation_env_proto.environments.append(any_proto)
  return compilation_env_proto


def generate_compilation_options(
    compile_environment: xla_pb2.CompilationEnvironmentsProto | None = None,
    jax_mesh: jax.sharding.Mesh | None = None,
) -> compile_options_pb2.CompileOptionsProto:
  """Generates the compilation options for the given compilation environment."""
  compile_options = compile_options_pb2.CompileOptionsProto()
  executable_build_options = compile_options_pb2.ExecutableBuildOptionsProto()
  if compile_environment is not None:
    executable_build_options.comp_envs.CopyFrom(compile_environment)
    executable_build_options.num_replicas = 1
    executable_build_options.num_partitions = 1
    executable_build_options.device_ordinal = -1
    executable_build_options.process_count = 1
    executable_build_options.allow_spmd_sharding_propagation_to_parameters.append(
        False
    )
    executable_build_options.allow_spmd_sharding_propagation_to_output.append(
        False
    )
    # Set the device assignment to a single-replica. If the jax mesh is
    # provided, we will set the device assignment to the jax mesh.
    if jax_mesh is not None:
      partition = 0
      device_assignment = xla_data_pb2.DeviceAssignmentProto()
      device_assignment.replica_count = 1
      for d in jax_mesh.devices.flat:
        computation_device = (
            xla_data_pb2.DeviceAssignmentProto.ComputationDevice()
        )
        computation_device.replica_device_ids.append(d.id)
        device_assignment.computation_devices.append(computation_device)
        partition += 1
      # Reset num_partitions to the number of devices in the mesh.
      executable_build_options.num_partitions = partition
      device_assignment.computation_count = partition
      executable_build_options.device_assignment.CopyFrom(device_assignment)
  executable_build_options.use_shardy_partitioner = (
      jax.config.jax_use_shardy_partitioner
  )
  compile_options.executable_build_options.CopyFrom(executable_build_options)
  return compile_options


def generate_xla_compile_options(
    native_serialization_platforms: Sequence[str] | None,
    xla_flags_per_platform: Mapping[str, Sequence[str] | None],
    jax_mesh: jax.sharding.Mesh | None = None,
) -> manifest_pb2.CompileOptionsProtoMap:
  """Sets the XLA compilation options.

  Args:
    native_serialization_platforms: A sequence of platform names that the
      compile options will be set for. If None, the compile options will be set
      for TPU only.
    xla_flags_per_platform: A mapping from platform name to a list of xla flags
      which will be used to override the default XLA compilation flags.
    jax_mesh: The JAX mesh used for sharding. If None, the compile options will
      be set for a default single-replica.

  Returns:
    A `CompileOptionsProtoMap` containing the XLA compilation options per
    platform.

  Raises:
    ValueError: If the supplied XLA flag overrides cannot be parsed.
  """

  compile_options_map = manifest_pb2.CompileOptionsProtoMap()
  if native_serialization_platforms is None:
    # If no native serialization platforms are specified, we will set the
    # compilation environment for TPU only.
    xla_flags = None
    if xla_flags_per_platform is not None:
      logging.info('Setting XLA flags per platform: %s', xla_flags_per_platform)
      xla_flags = xla_flags_per_platform.get(
          manifest_pb2.Platform.Name(manifest_pb2.Platform.TPU).lower(),
          None,
      )
    tpu_compilation_env = generate_tpu_compilation_env(xla_flags)
    compilation_options = generate_compilation_options(
        tpu_compilation_env, jax_mesh
    )
    compile_options_map.map[
        manifest_pb2.Platform.Name(manifest_pb2.Platform.TPU).lower()
    ].CopyFrom(compilation_options)
  else:
    for platform in native_serialization_platforms:
      compile_environment = None
      tpu_platform = manifest_pb2.Platform.Name(
          manifest_pb2.Platform.TPU
      ).lower()
      if platform == tpu_platform:
        # Adding a TPU compilation environment with all values set from global
        # flag values.
        compile_environment = generate_tpu_compilation_env(
            xla_flags_per_platform.get(platform, None)
            if xla_flags_per_platform is not None
            else None
        )
      compile_options_map.map.get_or_create(platform.lower()).CopyFrom(
          generate_compilation_options(compile_environment, jax_mesh)
      )
  return compile_options_map


def get_field_for_flag(flag_name: str) -> descriptor.FieldDescriptor:
  """Gets the protobuf field descriptor for a given flag name."""
  if flag_name not in _XLA_FLAG_TO_FIELD_MAP:
    raise ValueError(
        f'No TpuCompilationEnvironment field matching flag {flag_name}'
    )
  return _XLA_FLAG_TO_FIELD_MAP[flag_name]


def parse_flag_from_string(flag_name: str, value: str) -> Any:
  """Parses a string value for a given flag and normalizes it for a proto field.

  This is a Python implementation of the C++ function
  TpuCompEnvReflection::ParseFlagFromString.

  Args:
    flag_name: The name of the flag.
    value: The string value of the flag.

  Returns:
    The parsed and normalized value suitable for setting the corresponding field
    in `TpuCompilationEnvironment`. This can be a primitive type (int, bool,
    str), float, an enum's integer value, or a proto message instance.

  Raises:
    ValueError: If the flag is not found, or if a proto message value cannot
      be parsed.
  """
  try:
    flag_holder = flags.FLAGS[flag_name]
  except KeyError:
    raise ValueError(f'Flag not found: {flag_name}')

  parsed_value = flag_holder.parser.parse(value)
  field = get_field_for_flag(flag_name)

  if field.type == descriptor.FieldDescriptor.TYPE_MESSAGE:
    message_instance = field.message_type._concrete_class()
    try:
      text_format.Parse(value, message_instance)
      return message_instance
    except text_format.ParseError as e:
      raise ValueError(
          f'Error parsing proto value for flag {flag_name}: {value}'
      ) from e
  if field.type == descriptor.FieldDescriptor.TYPE_ENUM:
    if isinstance(parsed_value, str):
      return field.enum_type.values_by_name[parsed_value].number
    # If it's already an int, assume it's the correct value.
    return parsed_value
  if field.type in (
      descriptor.FieldDescriptor.TYPE_FLOAT,
      descriptor.FieldDescriptor.TYPE_DOUBLE,
  ):
    return float(parsed_value)
  return parsed_value


def merge_flags_into_compile_options(
    xla_flags: Mapping[str, str],
    env: tpu_comp_env_pb2.TpuCompilationEnvironment,
):
  """Merges flags into a TpuCompilationEnvironment proto.

  Args:
    xla_flags: A mapping of XLA flag names to their string values. These flags
      will be parsed and merged into the `env` proto.
    env: The TpuCompilationEnvironment proto to merge the flags into. This
      proto will be modified in place.
  """
  env_override = tpu_comp_env_pb2.TpuCompilationEnvironment()
  for flag_name, value in xla_flags.items():
    field_descriptor = get_field_for_flag(flag_name)
    parsed_value = parse_flag_from_string(flag_name, value)
    if field_descriptor.type == descriptor.FieldDescriptor.TYPE_MESSAGE:
      # For message types, we need to copy the parsed message.
      getattr(env_override, field_descriptor.name).CopyFrom(parsed_value)
    else:
      # For scalar types, we can set the attribute directly.
      setattr(env_override, field_descriptor.name, parsed_value)
  env.MergeFrom(env_override)


# TODO(b/438187387): remove this path and only allow the "--flag=value" format.
def merge_proto_formatted_flags_into_compile_options(
    xla_flags: Sequence[str],
    env: tpu_comp_env_pb2.TpuCompilationEnvironment,
):
  """Merges flags into a proto."""
  env_override = tpu_comp_env_pb2.TpuCompilationEnvironment()
  xla_flags_str = '\n'.join(xla_flags)
  try:
    text_format.Parse(xla_flags_str, env_override)
  except text_format.ParseError as e:
    raise ValueError(
        f'Error parsing supplied XLA flag overrides {xla_flags_str}.'
    ) from e
  env.MergeFrom(env_override)
