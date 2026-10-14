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

from absl.testing import absltest
from orbax.checkpoint._src.testing.benchmarks.core import launch_env

_EXPLICIT = {
    'SIMDEVICE_TASK_ID': '1',
    'SIMDEVICE_NUM_TASKS': '2',
    'JAX_COORDINATOR_ADDRESS': '10.0.0.2:1234',
    'SIMDEVICE_BARRIER_ADDRESS': '10.0.0.2:50051',
}

_JOBSET = {
    'JOBSET_NAME': 'bench',
    'REPLICATED_JOB_NAME': 'slice-job',
    'JOB_INDEX': '1',
    'JOB_COMPLETION_INDEX': '1',
    'PROCESSES_IN_JOB': '2',
    'JAX_PROCESS_COUNT': '4',
    # XPK sets this without a port; the JobSet source ignores it.
    'JAX_COORDINATOR_ADDRESS': 'bench-slice-job-0-0.bench',
}


def _resolve(environ: dict[str, str]) -> launch_env.LaunchEnv:
  env = launch_env.resolve(environ)
  assert env is not None
  return env


class ResolveTest(absltest.TestCase):

  def test_single_task(self):
    self.assertIsNone(launch_env.resolve({}))

  def test_explicit(self):
    self.assertEqual(
        launch_env.resolve(dict(_EXPLICIT, SIMDEVICE_BARRIER_PORT='50051')),
        launch_env.LaunchEnv(
            source='explicit',
            task_id=1,
            num_tasks=2,
            coordinator_address='10.0.0.2:1234',
            barrier_address='10.0.0.2:50051',
            barrier_port=50051,
        ),
    )

  def test_explicit_without_barrier_port(self):
    self.assertIsNone(_resolve(_EXPLICIT).barrier_port)

  def test_explicit_wins_over_jobset(self):
    self.assertEqual(
        _resolve(dict(_JOBSET, **_EXPLICIT)).source, 'explicit'
    )

  def test_explicit_incomplete(self):
    environ = dict(_EXPLICIT)
    del environ['SIMDEVICE_BARRIER_ADDRESS']
    with self.assertRaisesRegex(ValueError, 'SIMDEVICE_BARRIER_ADDRESS'):
      launch_env.resolve(environ)

  def test_explicit_task_out_of_range(self):
    with self.assertRaisesRegex(ValueError, 'Invalid task 2 of 2'):
      launch_env.resolve(dict(_EXPLICIT, SIMDEVICE_TASK_ID='2'))

  def test_jobset(self):
    self.assertEqual(
        launch_env.resolve(_JOBSET),
        launch_env.LaunchEnv(
            source='JobSet',
            task_id=3,
            num_tasks=4,
            coordinator_address='bench-slice-job-0-0.bench:1234',
            barrier_address='bench-slice-job-0-0.bench:50051',
            barrier_port=50051,
        ),
    )

  def test_jobset_task0_uses_localhost(self):
    env = _resolve(dict(_JOBSET, JOB_INDEX='0', JOB_COMPLETION_INDEX='0'))
    self.assertEqual(env.task_id, 0)
    self.assertEqual(env.coordinator_address, 'localhost:1234')
    self.assertEqual(env.barrier_address, 'localhost:50051')

  def test_jobset_coordinator_port(self):
    env = _resolve(dict(_JOBSET, JAX_COORDINATOR_PORT='8476'))
    self.assertEqual(env.coordinator_address, 'bench-slice-job-0-0.bench:8476')

  def test_partial_jobset_is_single_task(self):
    # JobSet sets some variables on every pod, e.g. TPU pods.
    environ = dict(_JOBSET)
    del environ['PROCESSES_IN_JOB']
    self.assertIsNone(launch_env.resolve(environ))


if __name__ == '__main__':
  absltest.main()
