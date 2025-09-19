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

from absl.testing import absltest
from absl.testing import parameterized
from orbax.experimental.model.core.python import compile_options_util
from .platforms.xla.service.jellyfish import tpu_compilation_environment_pb2 as tpu_comp_env_pb2


class CompileOptionsUtilTest(parameterized.TestCase):

  def test_parse_flag_from_string_bool(self):
    result = compile_options_util.parse_flag_from_string(
        'xla_sc_poison_buffers', 'false'
    )
    self.assertEqual(result, False)

  def test_parse_flag_from_string_int(self):
    result = compile_options_util.parse_flag_from_string(
        'xla_jf_rematerialization_percent_shared_memory_limit', '99'
    )
    self.assertEqual(result, 99)

  def test_parse_flag_from_string_float(self):
    result = compile_options_util.parse_flag_from_string(
        'xla_tpu_async_copy_bandwidth_scaling_factor', '0.19125064716453793'
    )
    self.assertEqual(result, 0.19125064716453793)

  def test_parse_flag_from_string_string(self):
    result = compile_options_util.parse_flag_from_string(
        'xla_tpu_alternate_memory_benefit_scaling_factor_for_large_buffers',
        'NO_SCALE',
    )
    self.assertEqual(result, 'NO_SCALE')

  def test_parse_flag_from_string_proto(self):
    compile_options_util.parse_flag_from_string(
        'xla_tpu_memory_bound_loop_optimizer_options', 'enabled:false'
    )

  def test_parse_flag_from_string_enum(self):
    result = compile_options_util.parse_flag_from_string(
        'xla_memory_scheduler', 'DFS'
    )
    expected = tpu_comp_env_pb2.MemorySchedulerProto.DFS
    self.assertEqual(result, expected)

  def test_parse_flag_from_string_nonexistent_flag(self):
    with self.assertRaisesRegex(ValueError, 'Flag not found: nonexistent_flag'):
      compile_options_util.parse_flag_from_string('nonexistent_flag', 'value')

  @parameterized.named_parameters(
      (
          'dict_xla_flags',
          {
              'xla_jf_rematerialization_percent_shared_memory_limit': '99',
              'xla_tpu_allocate_scoped_vmem_at_same_offset': 'false',
              'xla_tpu_alternate_memory_benefit_scaling_factor_for_large_buffers': (
                  'NO_SCALE'
              ),
              'xla_tpu_memory_bound_loop_optimizer_options': 'enabled:false',
              'xla_tpu_async_copy_bandwidth_scaling_factor': (
                  '0.19125064716453793'
              ),
          },
          compile_options_util.merge_flags_into_compile_options,
      ),
      (
          'proto_formatted_xla_flags',
          [
              'xla_jf_rematerialization_percent_shared_memory_limit: 99',
              'xla_tpu_allocate_scoped_vmem_at_same_offset: false',
              (
                  'xla_tpu_alternate_memory_benefit_scaling_factor_for_large_buffers:'
                  " 'NO_SCALE'"
              ),
              'xla_tpu_memory_bound_loop_optimizer_options: {enabled:false}',
              (
                  'xla_tpu_async_copy_bandwidth_scaling_factor:'
                  ' 0.19125064716453793'
              ),
          ],
          compile_options_util.merge_proto_formatted_flags_compile_option,
      ),
  )
  def test_merge_flags_into_compile_options(self, xla_flags, merge_fn):
    # Initialize the environment with some values.
    env = tpu_comp_env_pb2.TpuCompilationEnvironment()
    # Values that should be overridden.
    env.xla_jf_rematerialization_percent_shared_memory_limit = 10
    env.xla_tpu_memory_bound_loop_optimizer_options.enabled = True
    # Value that should not be overridden.
    env.xla_tpu_wait_n_cycles_before_program_termination = 1234

    # Merge the flags into the environment.
    merge_fn(xla_flags, env)
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


if __name__ == '__main__':
  absltest.main()
