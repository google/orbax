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

"""Tests for simdevice_backend, with the SimDevice plugin mocked out."""

import atexit
import dataclasses
import os
import sys
import types
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from etils import epath
import jax
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.testing.benchmarks.core import configs
from orbax.checkpoint._src.testing.benchmarks.core import simdevice_backend
import yaml

_CONFIG = configs.SimDeviceConfig(device_type='v6e', topology=[2, 2, 1])

# What the real plugin leaves in JAX_PLATFORMS once it is imported.
_SIMDEVICE_PLATFORMS = 'simdevice,cpu'

# Exits with the simulated host id, so host 0 succeeds and host 1 fails.
_EXIT_WITH_HOST_ID = ('/bin/sh', '-c', 'exit "$SIMDEVICE_PROCESS_ID"')

# Writes a simulated host's environment to a file named after its host id, in
# the directory passed as $0.
_WRITE_HOST_ENV = ('/bin/sh', '-c', 'env > "$0/$SIMDEVICE_PROCESS_ID"')


#  JAX defines this flag; OSS JAX does not.
if 'jax_controller_address' not in flags.FLAGS:
  flags.DEFINE_string(
      'jax_controller_address', '', 'Stand-in for the  JAX flag.'
  )


# Environment variables a launcher sets; cleared before each cluster test.
_LAUNCH_ENVS = (
    'SIMDEVICE_PROCESS_ID',
    'SIMDEVICE_TASK_ID',
    'SIMDEVICE_NUM_TASKS',
    'SIMDEVICE_BARRIER_ADDRESS',
    'SIMDEVICE_BARRIER_PORT',
    'JAX_PROCESS_ID',
    'JAX_COORDINATOR_ADDRESS',
    'JOBSET_NAME',
)

# What XPK sets on the second pod of a two-pod CPU JobSet.
_XPK_POD_1 = {
    'JOBSET_NAME': 'bench',
    'REPLICATED_JOB_NAME': 'slice-job',
    'JOB_INDEX': '0',
    'JOB_COMPLETION_INDEX': '1',
    'PROCESSES_IN_JOB': '2',
    'JAX_PROCESS_COUNT': '2',
    'JAX_COORDINATOR_ADDRESS': 'bench-slice-job-0-0.bench',
}


def _read_env_file(path: epath.Path) -> dict[str, str]:
  lines = path.read_text().splitlines()
  return dict(line.split('=', 1) for line in lines if '=' in line)


class RenderTopologyYamlTest(absltest.TestCase):

  def test_render(self):
    rendered = simdevice_backend.render_topology_yaml(
        configs.SimDeviceConfig(
            device_type='v7x', topology=[4, 4, 1], num_processes=2
        )
    )

    self.assertEqual(
        yaml.safe_load(rendered),
        {
            'device': {'type': 'v7x'},
            'topology': {'mesh_dimensions': [4, 4, 1]},
            'orchestration': {'num_processes': 2},
        },
    )

  def test_render_slices(self):
    rendered = simdevice_backend.render_topology_yaml(
        configs.SimDeviceConfig(
            device_type='v6e', topology=[2, 2, 1], num_processes=4, num_slices=2
        )
    )

    self.assertEqual(
        yaml.safe_load(rendered)['topology'],
        {'mesh_dimensions': [2, 2, 1], 'num_slices': 2},
    )


class ActivateTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(
        mock.patch.object(simdevice_backend, '_active_config', None)
    )
    self.enter_context(
        mock.patch.dict(os.environ, {'JAX_PLATFORMS': _SIMDEVICE_PLATFORMS})
    )
    os.environ.pop('SIMDEVICE_FABRICATED_FILL_SEED', None)
    # Stands in for the plugin, so `import simdevice` does not load it.
    self.enter_context(
        mock.patch.dict(sys.modules, {'simdevice': types.ModuleType('fake')})
    )
    self.mock_config_update = self.enter_context(
        mock.patch.object(jax.config, 'update', autospec=True)
    )
    self.mock_atexit_register = self.enter_context(
        mock.patch.object(atexit, 'register', autospec=True)
    )
    self.enter_context(flagsaver.flagsaver())

  def _activate(self, config: configs.SimDeviceConfig) -> str:
    """Activates SimDevice and returns the topology file path."""
    try:
      simdevice_backend.activate(config)
    finally:
      if 'SIMDEVICE_CONFIG_PATH' in os.environ:
        self.addCleanup(os.remove, os.environ['SIMDEVICE_CONFIG_PATH'])
    return os.environ['SIMDEVICE_CONFIG_PATH']

  def test_writes_topology_file(self):
    topology_path = self._activate(_CONFIG)

    with open(topology_path) as f:
      self.assertEqual(
          f.read(), simdevice_backend.render_topology_yaml(_CONFIG)
      )
    self.mock_atexit_register.assert_called_once_with(os.remove, topology_path)

  def test_selects_simdevice_platform(self):
    self._activate(_CONFIG)

    self.mock_config_update.assert_called_once_with(
        'jax_platforms', _SIMDEVICE_PLATFORMS
    )

  def test_sets_fill_seed(self):
    self._activate(dataclasses.replace(_CONFIG, fabricated_fill_seed=7))

    self.assertEqual(os.environ['SIMDEVICE_FABRICATED_FILL_SEED'], '7')

  def test_no_fill_seed(self):
    self._activate(_CONFIG)

    self.assertNotIn('SIMDEVICE_FABRICATED_FILL_SEED', os.environ)

  def test_activate_twice_raises(self):
    self._activate(_CONFIG)

    with self.assertRaisesRegex(RuntimeError, 'already active'):
      simdevice_backend.activate(_CONFIG)

  def test_failed_import_raises(self):
    # A None entry makes `import simdevice` raise ImportError.
    with mock.patch.dict(sys.modules, {'simdevice': None}):
      with self.assertRaises(ImportError):
        self._activate(_CONFIG)

    self.mock_config_update.assert_not_called()
    # A second attempt is not blocked by the failed one.
    self._activate(_CONFIG)

  def test_single_process_keeps_default_barrier(self):
    self._activate(_CONFIG)

    self.assertFalse(multihost.EXPERIMENTAL_ORBAX_USE_DISTRIBUTED_BARRIER.value)

  def test_multi_process_uses_distributed_barrier(self):
    self._activate(dataclasses.replace(_CONFIG, num_processes=2))

    self.assertTrue(multihost.EXPERIMENTAL_ORBAX_USE_DISTRIBUTED_BARRIER.value)


class CheckProcessCountTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_process_count = self.enter_context(
        mock.patch.object(jax, 'process_count', autospec=True, return_value=1)
    )

  def test_inactive_passes(self):
    self.mock_process_count.return_value = 2
    with mock.patch.object(simdevice_backend, '_active_config', None):
      simdevice_backend.check_process_count()

  def test_matching_count_passes(self):
    with mock.patch.object(simdevice_backend, '_active_config', _CONFIG):
      simdevice_backend.check_process_count()

  def test_mismatch_raises(self):
    config = dataclasses.replace(_CONFIG, num_processes=2)
    with mock.patch.object(simdevice_backend, '_active_config', config):
      with self.assertRaisesRegex(RuntimeError, 'expects 2 processes'):
        simdevice_backend.check_process_count()


class NeedsLocalClusterTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(mock.patch.dict(os.environ))
    for name in _LAUNCH_ENVS:
      os.environ.pop(name, None)
    self.enter_context(flagsaver.flagsaver(jax_controller_address=''))
    self.config = dataclasses.replace(_CONFIG, num_processes=2)

  def test_single_process(self):
    self.assertFalse(simdevice_backend.needs_local_cluster(_CONFIG))

  def test_multi_process_without_host_id(self):
    self.assertTrue(simdevice_backend.needs_local_cluster(self.config))

  def test_multi_process_with_host_id(self):
    os.environ['SIMDEVICE_PROCESS_ID'] = '1'

    self.assertFalse(simdevice_backend.needs_local_cluster(self.config))

  def test_jax_launcher_without_host_id_raises(self):
    os.environ['JAX_PROCESS_ID'] = '1'

    with self.assertRaisesRegex(RuntimeError, 'SIMDEVICE_PROCESS_ID'):
      simdevice_backend.needs_local_cluster(self.config)

  @flagsaver.flagsaver(jax_controller_address='host:1234')
  def test_borg_launcher_without_host_id_raises(self):
    with self.assertRaisesRegex(RuntimeError, 'SIMDEVICE_PROCESS_ID'):
      simdevice_backend.needs_local_cluster(self.config)

  def test_launcher_task_without_host_id(self):
    os.environ.update(
        SIMDEVICE_TASK_ID='1',
        SIMDEVICE_NUM_TASKS='2',
        SIMDEVICE_BARRIER_ADDRESS='task0:barrier',
        JAX_COORDINATOR_ADDRESS='task0:jax',
    )

    self.assertTrue(simdevice_backend.needs_local_cluster(self.config))

  def test_incomplete_launcher_task_raises(self):
    os.environ['SIMDEVICE_TASK_ID'] = '1'

    with self.assertRaisesRegex(ValueError, 'SIMDEVICE_NUM_TASKS'):
      simdevice_backend.needs_local_cluster(self.config)

  def test_xpk_pod_without_host_id(self):
    os.environ.update(_XPK_POD_1)

    self.assertTrue(simdevice_backend.needs_local_cluster(self.config))

  def test_jax_launcher_with_host_id(self):
    os.environ['JAX_PROCESS_ID'] = '1'
    os.environ['SIMDEVICE_PROCESS_ID'] = '1'

    self.assertFalse(simdevice_backend.needs_local_cluster(self.config))


class RunLocalClusterTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(mock.patch.dict(os.environ))
    for name in _LAUNCH_ENVS:
      os.environ.pop(name, None)

  def test_hosts_get_their_identity(self):
    output_dir = epath.Path(self.create_tempdir().full_path)
    config = dataclasses.replace(_CONFIG, num_processes=2)

    simdevice_backend.run_local_cluster(
        config, command=[*_WRITE_HOST_ENV, str(output_dir)]
    )

    host_envs = [_read_env_file(output_dir / str(i)) for i in range(2)]
    for host_id, host_env in enumerate(host_envs):
      with self.subTest(host_id=host_id):
        self.assertEqual(host_env['SIMDEVICE_PROCESS_ID'], str(host_id))
        self.assertEqual(host_env['JAX_PROCESS_ID'], str(host_id))
        self.assertEqual(host_env['JAX_NUM_PROCESSES'], '2')
    self.assertEqual(
        host_envs[0]['JAX_COORDINATOR_ADDRESS'],
        host_envs[1]['JAX_COORDINATOR_ADDRESS'],
    )
    self.assertEqual(
        host_envs[0]['SIMDEVICE_BARRIER_ADDRESS'],
        host_envs[1]['SIMDEVICE_BARRIER_ADDRESS'],
    )

  def test_host_that_never_exits_times_out(self):
    config = dataclasses.replace(_CONFIG, num_processes=2, timeout_seconds=1)
    # Host 0 exits at once; host 1 hangs.
    command = ('/bin/sh', '-c', '[ "$SIMDEVICE_PROCESS_ID" = 0 ] || sleep 600')

    with self.assertRaisesRegex(RuntimeError, r'\[1\] did not finish'):
      simdevice_backend.run_local_cluster(config, command=command)

  def test_failed_host_raises(self):
    config = dataclasses.replace(_CONFIG, num_processes=2)

    with self.assertRaisesRegex(RuntimeError, r'\{1: 1\}'):
      simdevice_backend.run_local_cluster(config, command=_EXIT_WITH_HOST_ID)

  def test_launcher_task_runs_its_share(self):
    os.environ.update(
        SIMDEVICE_TASK_ID='1',
        SIMDEVICE_NUM_TASKS='2',
        SIMDEVICE_BARRIER_ADDRESS='task0:barrier',
        JAX_COORDINATOR_ADDRESS='task0:jax',
    )
    output_dir = epath.Path(self.create_tempdir().full_path)
    config = dataclasses.replace(_CONFIG, num_processes=4)

    simdevice_backend.run_local_cluster(
        config, command=[*_WRITE_HOST_ENV, str(output_dir)]
    )

    self.assertCountEqual(
        [path.name for path in output_dir.iterdir()], ['2', '3']
    )
    for host_id in (2, 3):
      host_env = _read_env_file(output_dir / str(host_id))
      with self.subTest(host_id=host_id):
        self.assertEqual(host_env['JAX_PROCESS_ID'], str(host_id))
        self.assertEqual(host_env['JAX_NUM_PROCESSES'], '4')
        self.assertEqual(host_env['JAX_COORDINATOR_ADDRESS'], 'task0:jax')
        self.assertEqual(host_env['SIMDEVICE_BARRIER_ADDRESS'], 'task0:barrier')

  def test_xpk_pod_runs_its_share(self):
    os.environ.update(_XPK_POD_1)
    output_dir = epath.Path(self.create_tempdir().full_path)
    config = dataclasses.replace(_CONFIG, num_processes=4)

    simdevice_backend.run_local_cluster(
        config, command=[*_WRITE_HOST_ENV, str(output_dir)]
    )

    self.assertCountEqual(
        [path.name for path in output_dir.iterdir()], ['2', '3']
    )
    host_env = _read_env_file(output_dir / '2')
    self.assertEqual(
        host_env['JAX_COORDINATOR_ADDRESS'], 'bench-slice-job-0-0.bench:1234'
    )
    self.assertEqual(
        host_env['SIMDEVICE_BARRIER_ADDRESS'], 'bench-slice-job-0-0.bench:50051'
    )
    self.assertEqual(host_env['SIMDEVICE_BARRIER_PORT'], '50051')

  def test_uneven_split_raises(self):
    os.environ.update(
        SIMDEVICE_TASK_ID='0',
        SIMDEVICE_NUM_TASKS='2',
        SIMDEVICE_BARRIER_ADDRESS='task0:barrier',
        JAX_COORDINATOR_ADDRESS='task0:jax',
    )
    config = dataclasses.replace(_CONFIG, num_processes=3)

    with self.assertRaisesRegex(ValueError, 'multiple'):
      simdevice_backend.run_local_cluster(config, command=_EXIT_WITH_HOST_ID)

  def test_launcher_task_without_addresses_raises(self):
    os.environ.update(SIMDEVICE_TASK_ID='0', SIMDEVICE_NUM_TASKS='2')
    config = dataclasses.replace(_CONFIG, num_processes=4)

    with self.assertRaisesRegex(ValueError, 'JAX_COORDINATOR_ADDRESS'):
      simdevice_backend.run_local_cluster(config, command=_EXIT_WITH_HOST_ID)


if __name__ == '__main__':
  absltest.main()
