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

from typing import Any
from absl.testing import absltest
from absl.testing import parameterized
from google.protobuf import text_format
from orbax.experimental.model.core.python import compile_options_util
from .platforms.xla.service.jellyfish import tpu_compilation_environment_pb2 as tpu_comp_env_pb2
from .platforms.xla.service.jellyfish.python import tpu_compilation_environment as tpu_comp_env


def _get_expected_proto_from_tpu_comp_env(field_str: str, proto_str: str):
  field = tpu_comp_env_pb2.TpuCompilationEnvironment.DESCRIPTOR.fields_by_name[
      field_str
  ]
  message_type = field.message_type._concrete_class
  expected_proto = message_type()
  text_format.Parse(proto_str, expected_proto)
  return expected_proto


XLA_FLAGS_DICT = {
    'xla_jf_rematerialization_percent_shared_memory_limit': '99',
    'xla_tpu_allocate_scoped_vmem_at_same_offset': 'false',
    'xla_tpu_alternate_memory_benefit_scaling_factor_for_large_buffers': (
        'NO_SCALE'
    ),
    'xla_tpu_memory_bound_loop_optimizer_options': 'enabled:false',
    'xla_tpu_async_copy_bandwidth_scaling_factor': '0.19125064716453793',
}


EXPECTED_ENV = tpu_comp_env_pb2.TpuCompilationEnvironment(
    xla_jf_rematerialization_percent_shared_memory_limit=99,
    xla_tpu_allocate_scoped_vmem_at_same_offset=False,
    xla_tpu_alternate_memory_benefit_scaling_factor_for_large_buffers=(
        'NO_SCALE'
    ),
    xla_tpu_memory_bound_loop_optimizer_options=_get_expected_proto_from_tpu_comp_env(
        'xla_tpu_memory_bound_loop_optimizer_options', 'enabled:false'
    ),
    xla_tpu_async_copy_bandwidth_scaling_factor=0.19125064716453793,
)


class CompileOptionsUtilTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('bool', 'xla_sc_poison_buffers', 'false', False),
      ('bool_true', 'xla_sc_poison_buffers', 'true', True),
      ('int', 'xla_jf_rematerialization_percent_shared_memory_limit', '99', 99),
      (
          'float',
          'xla_tpu_async_copy_bandwidth_scaling_factor',
          '0.19125064716453793',
          0.19125064716453793,
      ),
      (
          'string',
          'xla_tpu_alternate_memory_benefit_scaling_factor_for_large_buffers',
          'NO_SCALE',
          'NO_SCALE',
      ),
      (
          'string_looks_like_number',
          'xla_tpu_alternate_memory_benefit_scaling_factor_for_large_buffers',
          '123',
          '123',
      ),
      (
          'string_looks_like_bool',
          'xla_tpu_alternate_memory_benefit_scaling_factor_for_large_buffers',
          'true',
          'true',
      ),
      (
          'proto',
          'xla_tpu_memory_bound_loop_optimizer_options',
          'enabled:false',
          _get_expected_proto_from_tpu_comp_env(
              'xla_tpu_memory_bound_loop_optimizer_options', 'enabled:false'
          ),
      ),
      (
          'enum',
          'xla_memory_scheduler',
          'DFS',
          tpu_comp_env_pb2.MemorySchedulerProto.DFS,
      ),
  )
  def test_parse_flag_from_string(
      self, flag_name: str, string_value: str, expected: Any
  ):
    result = compile_options_util.parse_flag_from_string(
        flag_name, string_value
    )
    self.assertEqual(result, expected)

  def test_parse_flag_from_string_nonexistent_flag(self):
    with self.assertRaisesRegex(ValueError, 'Flag not found: nonexistent_flag'):
      compile_options_util.parse_flag_from_string('nonexistent_flag', 'value')

  def test_merge_flags_into_compile_options(self):
    xla_flags = XLA_FLAGS_DICT
    # Initialize the environment with some values.
    env = tpu_comp_env_pb2.TpuCompilationEnvironment()
    # Values that should be overridden.
    env.xla_jf_rematerialization_percent_shared_memory_limit = 10
    env.xla_tpu_memory_bound_loop_optimizer_options.enabled = True
    # Value that should not be overridden.
    env.xla_tpu_wait_n_cycles_before_program_termination = 1234

    # Merge the flags into the environment.
    compile_options_util.merge_flags_into_compile_options(xla_flags, env)
    self.assertEqual(
        env.xla_jf_rematerialization_percent_shared_memory_limit, 99
    )
    self.assertEqual(env.xla_tpu_allocate_scoped_vmem_at_same_offset, False)
    self.assertEqual(
        env.xla_tpu_alternate_memory_benefit_scaling_factor_for_large_buffers,
        'NO_SCALE',
    )
    self.assertEqual(
        env.xla_tpu_memory_bound_loop_optimizer_options.enabled, False
    )
    self.assertAlmostEqual(
        env.xla_tpu_async_copy_bandwidth_scaling_factor,
        0.19125064716453793,
    )

    # Value that should not be overridden.
    self.assertEqual(env.xla_tpu_wait_n_cycles_before_program_termination, 1234)

  @parameterized.named_parameters(
      dict(
          testcase_name='dict_xla_flags',
          xla_flags=[f'--{k}={v}' for k, v in XLA_FLAGS_DICT.items()],
          expected_env=EXPECTED_ENV,
      ),
      dict(
          testcase_name='no_xla_flags',
          xla_flags=None,
          expected_env=None,
      ),
  )
  def test_generate_tpu_compilation_env(self, xla_flags, expected_env):
    env = compile_options_util.generate_tpu_compilation_env(xla_flags=xla_flags)
    self.assertLen(env.environments, 1)
    actual_env_proto = tpu_comp_env_pb2.TpuCompilationEnvironment()
    env.environments[0].Unpack(actual_env_proto)

    expected_env_proto = tpu_comp_env_pb2.TpuCompilationEnvironment()
    expected_env_proto.ParseFromString(
        tpu_comp_env.create_default_tpu_comp_env()
    )
    if expected_env is not None:
      expected_env_proto.MergeFrom(expected_env)

    self.assertEqual(
        text_format.MessageToString(actual_env_proto),
        text_format.MessageToString(expected_env_proto),
    )

  def test_generate_tpu_compilation_env_invalid_flag_format(self):
    with self.assertRaisesRegex(
        ValueError,
        'Flag xla_tpu_allocate_scoped_vmem_at_same_offset: false does not start'
        " with '--'. All flags must be in the format of"
        ' --flag_name=flag_value.',
    ):
      compile_options_util.generate_tpu_compilation_env(
          xla_flags=[
              '--xla_tpu_memory_bound_loop_optimizer_options=enabled:false',
              'xla_tpu_allocate_scoped_vmem_at_same_offset: false',
          ]
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='no_native_serialization_platforms_no_xla_flags',
          native_serialization_platforms=None,
          xla_flags_per_platform=None,
          expected_platforms=['tpu'],
      ),
      dict(
          testcase_name='no_native_serialization_platforms_with_xla_flags',
          native_serialization_platforms=None,
          xla_flags_per_platform={
              'tpu': [f'--{k}={v}' for k, v in XLA_FLAGS_DICT.items()]
          },
          expected_platforms=['tpu'],
      ),
      dict(
          testcase_name='with_native_serialization_platforms_no_xla_flags',
          native_serialization_platforms=['gpu', 'cpu', 'tpu'],
          xla_flags_per_platform=None,
          expected_platforms=['gpu', 'cpu', 'tpu'],
      ),
      dict(
          testcase_name='with_native_serialization_platforms_with_xla_flags',
          native_serialization_platforms=['gpu', 'cpu', 'tpu'],
          xla_flags_per_platform={
              'tpu': [f'--{k}={v}' for k, v in XLA_FLAGS_DICT.items()]
          },
          expected_platforms=['gpu', 'cpu', 'tpu'],
      ),
      dict(
          testcase_name='platforms_without_tpu_with_tpu_xla_flags',
          native_serialization_platforms=['gpu', 'cpu'],
          xla_flags_per_platform={
              'tpu': [f'--{k}={v}' for k, v in XLA_FLAGS_DICT.items()]
          },
          expected_platforms=['gpu', 'cpu'],
      ),
  )
  def test_generate_xla_compile_options_flags_and_platforms(
      self,
      native_serialization_platforms,
      xla_flags_per_platform,
      expected_platforms,
  ):
    compile_options_map = compile_options_util.generate_xla_compile_options(
        native_serialization_platforms=native_serialization_platforms,
        xla_flags_per_platform=xla_flags_per_platform,
    )
    self.assertLen(compile_options_map.map, len(expected_platforms))

    for platform in expected_platforms:
      self.assertIn(platform, compile_options_map.map)
      compile_options = compile_options_map.map[platform]

      if platform != 'tpu':
        self.assertEmpty(
            compile_options.executable_build_options.comp_envs.environments
        )
      else:
        # For TPU platform, verify the compilation environment.
        self.assertLen(
            compile_options.executable_build_options.comp_envs.environments, 1
        )
        actual_env_proto = tpu_comp_env_pb2.TpuCompilationEnvironment()
        compile_options.executable_build_options.comp_envs.environments[
            0
        ].Unpack(actual_env_proto)

        expected_env_overrides = (
            EXPECTED_ENV
            if xla_flags_per_platform and xla_flags_per_platform.get(platform)
            else None
        )
        expected_env_proto = tpu_comp_env_pb2.TpuCompilationEnvironment()
        expected_env_proto.ParseFromString(
            tpu_comp_env.create_default_tpu_comp_env()
        )
        if expected_env_overrides is not None:
          expected_env_proto.MergeFrom(expected_env_overrides)

        self.assertEqual(
            text_format.MessageToString(actual_env_proto),
            text_format.MessageToString(expected_env_proto),
        )


# TODO(b/439870345): add tests with different jax meshes and make sure the
# generated compile options are correct.

if __name__ == '__main__':
  absltest.main()
