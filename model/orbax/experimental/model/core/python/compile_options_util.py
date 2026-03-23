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

"""Utility functions for XLA compile options."""

from collections.abc import Mapping, Sequence
import logging
import re

from google.protobuf import descriptor
import jax
from orbax.experimental.model.core.protos import manifest_pb2

from .google.protobuf import any_pb2
from .platforms.xla.service.jellyfish import tpu_compilation_environment_pb2 as tpu_comp_env_pb2
from .platforms.xla.service.jellyfish.python import tpu_compilation_environment as tpu_comp_env
from tensorflow.compiler.xla import xla_data_pb2  # pylint: disable=g-direct-tensorflow-import
from tensorflow.compiler.xla import xla_pb2  # pylint: disable=g-direct-tensorflow-import
from tensorflow.compiler.xla.pjrt.proto import compile_options_pb2  # pylint: disable=g-direct-tensorflow-import

# A mapping between XLAflag names and protobuf field names.
_XLA_FLAG_TO_FIELD_MAP = {
    field.name: field
    for field in tpu_comp_env_pb2.TpuCompilationEnvironment.DESCRIPTOR.fields
}


def _generate_tpu_compilation_env(
    xla_flags_overrides: Sequence[str] | None = None,
) -> xla_pb2.CompilationEnvironmentsProto:
  """Generates the TPU compilation environment.

  It starts with default TPU compilation environment settings and overrides them
  with any flags provided in `xla_flags_overrides`.

  Args:
    xla_flags_overrides: A sequence of strings, where each string is an XLA flag
      in the format '--flag_name=flag_value'. These flags override default TPU
      compilation settings.

  Returns:
    An `xla_pb2.CompilationEnvironmentsProto` containing the TPU compilation
    environment.

  Raises:
    ValueError: If a flag in `xla_flags_overrides` is malformed (e.g., does
      not start with '--'), if a flag is deprecated, or a flag cannot be parsed.
  """
  # Get default TPU compilation environment.
  tpu_compilation_env_str = tpu_comp_env.create_default_tpu_comp_env()
  env = tpu_comp_env_pb2.TpuCompilationEnvironment.FromString(
      tpu_compilation_env_str
  )
  # Override with supplied XLA flags if any is provided.
  if xla_flags_overrides:
    parsed_flags = {}
    for flag in xla_flags_overrides:
      if not flag.startswith('--'):
        raise ValueError(
            f"Flag {flag} does not start with '--'. All flags must be in the"
            ' format of --flag_name=flag_value.'
        )
      flag_name, flag_value = flag[2:].split('=', 1)
      field_descriptor = get_field_for_flag(flag_name)
      if field_descriptor.GetOptions().deprecated:
        raise ValueError(
            '[DEPRECATED_XLA_TPU_FLAG_USE] TpuCompilationEnvironment has'
            f' deprecated fields in use: {flag_name}'
        )
      parsed_flags[flag_name] = flag_value
    env_override = parse_tpu_flags(parsed_flags)
    env.MergeFrom(env_override)

  # Pack the TPU compilation environment into a compilation env proto.
  any_proto = any_pb2.Any()
  any_proto.Pack(env)
  compilation_env_proto = xla_pb2.CompilationEnvironmentsProto()
  compilation_env_proto.environments.append(any_proto)
  return compilation_env_proto


def _generate_compilation_options(
    compile_environment: xla_pb2.CompilationEnvironmentsProto | None = None,
    jax_mesh: jax.sharding.Mesh | None = None,
    populate_xla_build_options: bool = False,
) -> compile_options_pb2.CompileOptionsProto:
  """Generates the compilation options for the given compilation environment."""
  compile_options = compile_options_pb2.CompileOptionsProto()
  executable_build_options = compile_options_pb2.ExecutableBuildOptionsProto()
  if compile_environment is not None:
    executable_build_options.comp_envs.CopyFrom(compile_environment)
  if populate_xla_build_options:
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
    xla_flags_per_platform: Mapping[str, Sequence[str]] | None = None,
    jax_mesh: jax.sharding.Mesh | None = None,
    persist_xla_flags: bool = True,
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
    persist_xla_flags: Whether to persist XLA flags in the compile options.

  Returns:
    A `CompileOptionsProtoMap` containing the XLA compilation options per
    platform.

  Raises:
    ValueError: If an unknown platform is provided as a native serialization
      platform.
    ValueError: If an unknown platform is provided as a platform for XLA flags.
    ValueError: If a platform is provided for XLA flags which is not provided
      in the native serialization platforms.
    ValueError: If the supplied XLA flag overrides cannot be parsed.
    ValueError: If `xla_flags` are provided but `persist_xla_flags` is False.
      This ensures that the XLA flags are persisted in the compile options,
      otherwise they would be lost, leading to unexpected behavior.
  """
  tpu_platform_name = manifest_pb2.Platform.Name(
      manifest_pb2.Platform.TPU
  ).lower()
  cuda_platform_name = manifest_pb2.Platform.Name(
      manifest_pb2.Platform.CUDA
  ).lower()

  compile_options_map = manifest_pb2.CompileOptionsProtoMap()
  if native_serialization_platforms is None:
    # If no native serialization platforms are specified, we will set the
    # compilation environment for TPU only.
    platforms = [tpu_platform_name]
  else:
    platforms = native_serialization_platforms

  # Validate the platform provided are valid.
  valid_platforms = {
      p.lower() for p in manifest_pb2.Platform.DESCRIPTOR.values_by_name
  }
  for platform in platforms:
    if platform.lower() not in valid_platforms:
      raise ValueError(
          f'Platform "{platform}" is not a valid platform. Valid platforms are:'
          f' {sorted(list(valid_platforms))}'
      )

  if xla_flags_per_platform:
    logging.info('Setting XLA flags per platform: %s', xla_flags_per_platform)
    # validate XLA flags provided are for a provided platform.
    for xla_platform in xla_flags_per_platform.keys():
      if xla_platform.lower() not in platforms:
        raise ValueError(
            f'Platform "{xla_platform}" used for XLA flags is not provided in'
            ' the native_serialization_platforms. '
        )

  for platform in platforms:
    if xla_flags_per_platform:
      xla_flags_overrides = xla_flags_per_platform.get(platform, None)
      if xla_flags_overrides:
        _validate_xla_flags_setting(xla_flags_overrides, persist_xla_flags)
    else:
      xla_flags_overrides = None

    platform_lower = platform.lower()
    if platform_lower == tpu_platform_name:
      compile_environment = _generate_tpu_compilation_env(xla_flags_overrides)
    else:
      # CPU Path: Leave as None to preserve legacy portable execution behavior.
      # CUDA Path: No specialized compiler environment needed by default.
      compile_environment = None

    compile_options = _generate_compilation_options(
        compile_environment,
        jax_mesh,
        populate_xla_build_options=(
            platform_lower in (tpu_platform_name, cuda_platform_name)
        ),
    )

    # Inject env_option_overrides natively for GPU using a dedicated helper.
    if platform_lower == cuda_platform_name and xla_flags_overrides:
      _apply_gpu_compilation_env_options(compile_options, xla_flags_overrides)

    compile_options_map.map[platform_lower].CopyFrom(compile_options)

  if not persist_xla_flags:
    for compile_options in compile_options_map.map.values():
      compile_options.executable_build_options.comp_envs.Clear()
      compile_options.env_option_overrides.clear()
  return compile_options_map


def _apply_gpu_compilation_env_options(
    compile_options: compile_options_pb2.CompileOptionsProto,
    xla_flags_overrides: Sequence[str],
) -> None:
  """Applies XLA flag overrides generically for GPU platforms.

  Args:
    compile_options: The compilation options proto to be modified.
    xla_flags_overrides: A sequence of XLA flags to apply as option overrides.
  """
  overrides_map = _parse_env_option_overrides_for_gpu(xla_flags_overrides)
  for k, v in overrides_map.items():
    compile_options.env_option_overrides[k].CopyFrom(v)


def _parse_env_option_overrides_for_gpu(
    xla_flags: Sequence[str],
) -> dict[str, compile_options_pb2.OptionOverrideProto]:
  """Parses a list of XLA GPU flags into a dictionary of OptionOverrideProto."""
  overrides = {}
  for flag in xla_flags:
    if not flag.startswith("--"):
      raise ValueError(f"Flag {flag} must start with '--'")

    # Ensure consistent policy enforcement.
    _validate_xla_gpu_flag(flag, strict=True)

    key, value = flag[2:].split("=", 1)
    override_proto = compile_options_pb2.OptionOverrideProto()

    # Infer type (True/False/Int/Float/String)
    if value.lower() == "true":
      override_proto.bool_field = True
    elif value.lower() == "false":
      override_proto.bool_field = False
    elif value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
      override_proto.int_field = int(value)
    else:
      try:
        override_proto.double_field = float(value)
      except ValueError:
        override_proto.string_field = value

    overrides[key] = override_proto
  return overrides


def _validate_xla_flags_setting(
    xla_flags_overrides: Sequence[str] | None, persist_xla_flags: bool
) -> None:
  """Validates the XLA flags setting.

  XLA flag overrides are allowed only when XLA flags are required to be
  persisted.

  Args:
    xla_flags_overrides: A sequence of XLA flags provided for overriding. Can be
      None.
    persist_xla_flags: A boolean indicating whether the XLA flags should be
      persisted in the compile options.

  Raises:
    ValueError: If `xla_flags_overrides` is not None but `persist_xla_flags` is
    False.
  """
  if xla_flags_overrides and not persist_xla_flags:
    raise ValueError(
        'persist_xla_flags must be True if xla_flags_overrides are provided.'
    )


def get_field_for_flag(flag_name: str) -> descriptor.FieldDescriptor:
  """Gets the protobuf field descriptor for a given flag name."""
  if flag_name not in _XLA_FLAG_TO_FIELD_MAP:
    raise ValueError(
        f'No TpuCompilationEnvironment field matching flag {flag_name}'
    )
  return _XLA_FLAG_TO_FIELD_MAP[flag_name]


def parse_tpu_flags(
    flags: Mapping[str, str],
) -> tpu_comp_env_pb2.TpuCompilationEnvironment:
  """Parses a dictionary of flags into a TpuCompilationEnvironment proto.

  Args:
    flags: A mapping of XLA flag names to their string values (as seen on
      the command line).

  Returns:
    A TpuCompilationEnvironment proto with the given flags set.
    All other fields are unset.

  Raises:
    ValueError: If the field is not found or its value cannot be parsed.
  """
  try:
    env_bytes = tpu_comp_env.parse_tpu_flags(flags)
  except RuntimeError as exc:
    raise ValueError(f'Error parsing TPU flags: {exc}') from exc
  return tpu_comp_env_pb2.TpuCompilationEnvironment.FromString(env_bytes)
