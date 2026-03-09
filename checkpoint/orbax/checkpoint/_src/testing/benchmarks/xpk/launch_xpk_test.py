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

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from orbax.checkpoint._src.testing.benchmarks.xpk import launch_xpk

HardwareType = launch_xpk.HardwareType

# satisfy required flags and validators
flags.FLAGS.set_default('cluster_name', 'dummy-cluster')
flags.FLAGS.set_default('config_file', __file__)
flags.FLAGS.set_default('output_directory', 'gs://dummy-bucket')


class LaunchXpkTest(parameterized.TestCase):

  @parameterized.named_parameters(
      # TPU Cases
      dict(
          testcase_name='tpu_v2',
          tpu='v2',
          device=None,
          expected=HardwareType.TPU,
      ),
      dict(
          testcase_name='tpu_v3', tpu='v3', device='', expected=HardwareType.TPU
      ),
      dict(
          testcase_name='tpu_v4',
          tpu='v4-8',
          device=None,
          expected=HardwareType.TPU,
      ),
      dict(
          testcase_name='tpu_v5e',
          tpu='v5e-4',
          device=None,
          expected=HardwareType.TPU,
      ),
      dict(
          testcase_name='tpu_v5p',
          tpu='v5p-8',
          device=None,
          expected=HardwareType.TPU,
      ),
      dict(
          testcase_name='tpu_explicit',
          tpu=None,
          device='tpu-v5-litepod-8',
          expected=HardwareType.TPU,
      ),
      dict(
          testcase_name='tpu_mixed',
          tpu='v5p',
          device='something-else',
          expected=HardwareType.TPU,
      ),
      # GPU Cases
      dict(
          testcase_name='gpu_h100',
          tpu=None,
          device='h100',
          expected=HardwareType.GPU,
      ),
      dict(
          testcase_name='gpu_a100',
          tpu='',
          device='a100',
          expected=HardwareType.GPU,
      ),
      dict(
          testcase_name='gpu_v100',
          tpu=None,
          device='v100',
          expected=HardwareType.GPU,
      ),
      dict(
          testcase_name='gpu_l4',
          tpu=None,
          device='l4',
          expected=HardwareType.GPU,
      ),
      dict(
          testcase_name='gpu_a2_instance',
          tpu=None,
          device='a2-highgpu-1g',
          expected=HardwareType.GPU,
      ),
      dict(
          testcase_name='gpu_g2_instance',
          tpu=None,
          device='g2-standard-4',
          expected=HardwareType.GPU,
      ),
      dict(
          testcase_name='gpu_p4_instance',
          tpu=None,
          device='p4.2xlarge',
          expected=HardwareType.GPU,
      ),
      dict(
          testcase_name='gpu_explicit',
          tpu=None,
          device='nvidia-gpu',
          expected=HardwareType.GPU,
      ),
      # CPU Cases
      dict(
          testcase_name='cpu_n1',
          tpu=None,
          device='n1-standard-4',
          expected=HardwareType.CPU,
      ),
      dict(
          testcase_name='cpu_n2',
          tpu=None,
          device='n2-standard-32',
          expected=HardwareType.CPU,
      ),
      dict(
          testcase_name='cpu_c3',
          tpu=None,
          device='c3-standard-4',
          expected=HardwareType.CPU,
      ),
      dict(
          testcase_name='cpu_m1',
          tpu=None,
          device='m1-ultramem-40',
          expected=HardwareType.CPU,
      ),
      dict(
          testcase_name='cpu_t2_aws',
          tpu=None,
          device='t2.micro',
          expected=HardwareType.CPU,
      ),
      dict(
          testcase_name='cpu_explicit',
          tpu=None,
          device='google-cpu',
          expected=HardwareType.CPU,
      ),
      # Unknown Cases
      dict(
          testcase_name='unknown_junk',
          tpu=None,
          device='junk-string',
          expected=HardwareType.UNKNOWN,
      ),
      dict(
          testcase_name='unknown_empty',
          tpu='',
          device='',
          expected=HardwareType.UNKNOWN,
      ),
      dict(
          testcase_name='unknown_none',
          tpu=None,
          device=None,
          expected=HardwareType.UNKNOWN,
      ),
  )
  def test_get_hardware_type(self, tpu, device, expected):
    self.assertEqual(launch_xpk.get_hardware_type(tpu, device), expected)


if __name__ == '__main__':
  absltest.main()
