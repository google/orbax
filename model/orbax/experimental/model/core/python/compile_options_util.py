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

from google.protobuf import text_format
import jax
from orbax.experimental.model.core.protos import manifest_pb2

from .google.protobuf import any_pb2
from .platforms.xla.service.jellyfish import tpu_compilation_environment_pb2 as tpu_comp_env_pb2
from .platforms.xla.service.jellyfish.python import tpu_compilation_environment as tpu_comp_env
from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.compiler.xla import xla_pb2
from tensorflow.compiler.xla.pjrt.proto import compile_options_pb2


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
  if xla_flags is not None:
    env_override = tpu_comp_env_pb2.TpuCompilationEnvironment()
    xla_flags_str = '\n'.join(xla_flags)
    try:
      text_format.Parse(xla_flags_str, env_override)
    except text_format.ParseError as e:
      raise ValueError(
          f'Error parsing supplied XLA flag overrides {xla_flags_str}.'
      ) from e
    env.MergeFrom(env_override)

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
    xla_flags_per_platform: A mapping from platform name to a list of xla flags
      which will be used to override the default XLA compilation flags.

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
