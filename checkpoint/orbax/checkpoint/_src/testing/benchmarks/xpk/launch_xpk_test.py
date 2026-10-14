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
from absl.testing import flagsaver
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


class ValidateSimDeviceTest(parameterized.TestCase):

  def _flags(self, **overrides):
    return dict(
        dict(
            enable_simdevice=True,
            tpu_type=None,
            device_type='n2-standard-32-2',
            enable_pathways=False,
        ),
        **overrides,
    )

  def test_cpu_nodes(self):
    self.assertTrue(launch_xpk._validate_simdevice(self._flags()))

  @parameterized.named_parameters(
      dict(testcase_name='tpu_type', overrides=dict(tpu_type='v6e-8')),
      dict(testcase_name='tpu_nodes', overrides=dict(device_type='v6e-8')),
      dict(testcase_name='no_device_type', overrides=dict(device_type=None)),
      dict(testcase_name='pathways', overrides=dict(enable_pathways=True)),
  )
  def test_invalid(self, overrides):
    self.assertFalse(launch_xpk._validate_simdevice(self._flags(**overrides)))

  def test_disabled(self):
    self.assertTrue(
        launch_xpk._validate_simdevice(
            self._flags(enable_simdevice=False, tpu_type='v6e-8')
        )
    )


class ConstructWorkloadCommandTest(absltest.TestCase):

  def _command(self, **kwargs):
    return launch_xpk.construct_workload_command(
        workload_name='bench',
        config_file='gs://bucket/config.yaml',
        output_directory='gs://bucket/out',
        run_id='run',
        enable_pathways=False,
        benchmark_binary_path='run_benchmarks.py',
        hardware_type=HardwareType.CPU,
        v_level=None,
        **kwargs,
    )

  def test_cpu(self):
    command = self._command()

    self.assertIn('export JOBSET_NAME=bench', command)
    self.assertIn('export JAX_PLATFORMS=cpu', command)
    self.assertIn('--jax_cpu_collectives_implementation=gloo', command)
    # run_benchmarks works out the process id and coordinator itself.
    self.assertNotIn('JAX_PROCESS_ID', command)
    self.assertNotIn('JAX_COORDINATOR_ADDRESS', command)

  def test_simdevice(self):
    command = self._command(enable_simdevice=True)

    self.assertIn('export JOBSET_NAME=bench', command)
    self.assertNotIn('JAX_PLATFORMS', command)
    self.assertNotIn('gloo', command)
    self.assertNotIn('SIMDEVICE_', command)


class ConstructXpkCommandTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    flags.FLAGS.set_default('xpk_path', 'xpk')
    flags.FLAGS.set_default('cluster_name', 'test-cluster')
    flags.FLAGS.set_default('project', 'test-project')
    flags.FLAGS.set_default('zone', 'test-zone')
    flags.FLAGS.set_default('num_slices', 1)
    flags.FLAGS.set_default('priority', 'medium')
    flags.FLAGS.set_default('storage', 'local')

  def test_simdevice_uses_device_type(self):
    with flagsaver.flagsaver(
        enable_simdevice=True, tpu_type=None, device_type='n2-standard-4'
    ):
      xpk_cmd = launch_xpk.construct_xpk_command(
          workload_name='test-workload',
          workload_command='python3 benchmark.py',
      )

    self.assertIn('--device-type=n2-standard-4', xpk_cmd)
    self.assertNotIn('--tpu-type', ' '.join(xpk_cmd))

  def test_normal_tpu_uses_tpu_type_in_xpk_flags(self):
    flags.FLAGS.set_default('enable_simdevice', False)
    flags.FLAGS.set_default('tpu_type', 'v6e-8')

    xpk_cmd = launch_xpk.construct_xpk_command(
        workload_name='test-workload',
        workload_command='python3 benchmark.py',
    )

    # We should see tpu-type in the XPK command flags
    self.assertIn('--tpu-type=v6e-8', xpk_cmd)


if __name__ == '__main__':
  absltest.main()
