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

"""Tests for deletion's use of shared async execution and process barriers."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.deletion import execution
from orbax.checkpoint.experimental.v1._src.synchronization import multihost
from orbax.checkpoint.experimental.v1._src.synchronization import synchronization
from orbax.checkpoint.experimental.v1._src.synchronization import thread_utils


class ExecutionTest(parameterized.TestCase):

  @parameterized.product(primary=(True, False), absent=(True, False))
  def test_uses_shared_runner_and_barriers(self, primary, absent):
    context = context_lib.Context()
    context.multiprocessing_options.barrier_sync_key_prefix = 'run'
    plan = None if absent else execution._Plan(epath.Path('/checkpoint'), None)
    events = []

    async def sync(key, **_):
      events.append(key)

    with (
        mock.patch.object(
            synchronization,
            'synchronize_next_operation_id',
            new_callable=mock.AsyncMock,
        ),
        mock.patch.object(multihost, 'is_primary_host', return_value=primary),
        mock.patch.object(multihost, 'sync_global_processes', side_effect=sync),
        mock.patch.object(execution, '_prepare', return_value=plan),
        mock.patch.object(
            execution, '_begin', side_effect=lambda *_: events.append('begin')
        ),
        mock.patch.object(
            execution,
            '_execute',
            side_effect=lambda *_: events.append('execute'),
        ),
    ):
      with context:
        response = execution.start(
            '/checkpoint', checkpointable_name=None, missing_ok=False
        )
      self.assertIsInstance(response, thread_utils.BackgroundThreadRunner)
      self.assertIs(response.result(), not absent)
    expected = ['run_delete:prepare']
    if primary:
      expected.append('begin')
    expected.append('run_delete:begin')
    if primary:
      expected.append('execute')
    expected.append('run_delete:finalize')
    self.assertEqual(events, expected)

  def test_prepare_barrier_failure_does_not_publish_or_delete(self):
    with (
        mock.patch.object(
            synchronization,
            'synchronize_next_operation_id',
            new_callable=mock.AsyncMock,
        ),
        mock.patch.object(
            execution,
            '_prepare',
            return_value=execution._Plan(epath.Path('/checkpoint'), None),
        ),
        mock.patch.object(
            multihost,
            'sync_global_processes',
            side_effect=TimeoutError('peer did not prepare'),
        ),
        mock.patch.object(execution, '_begin') as begin,
        mock.patch.object(execution, '_execute') as execute,
    ):
      with self.assertRaises(TimeoutError):
        execution.start(
            '/checkpoint', checkpointable_name=None, missing_ok=False
        )
    begin.assert_not_called()
    execute.assert_not_called()


if __name__ == '__main__':
  absltest.main()
