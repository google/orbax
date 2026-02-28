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

import time
from unittest import mock

from absl.testing import flagsaver
from absl.testing import parameterized
from etils import epath
import jax
import numpy as np
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.testing import multiprocess_test


class MultihostUtilsTestBase:

  class Test(parameterized.TestCase):

    def setUp(self):
      super().setUp()
      self.assertEqual(jax.device_count(), 8)
      self.assertEqual(jax.process_count(), 4)
      self.assertEqual(jax.local_device_count(), 2)

      if not multihost.is_runtime_to_distributed_ids_initialized():
        multihost.initialize_runtime_to_distributed_ids()

      self.tmpdir = epath.Path(
          self.create_tempdir(name='multihost_test').full_path
      )
      test_utils.sync_global_processes('setUp')

    def tearDown(self):
      test_utils.sync_global_processes('tearDown')
      super().tearDown()

    def test_process_errors(self):
      if multihost.process_index() == 1:
        with self.assertRaises(ValueError):
          multihost.sync_global_processes(
              'test_process_errors_1', processes={0}
          )

    def test_sync_global_processes(self):
      if multihost.process_index() == 0:
        time.sleep(2)
        (self.tmpdir / 'dummy').mkdir(parents=False, exist_ok=False)
      multihost.sync_global_processes('test_sync_global_processes')
      self.assertTrue((self.tmpdir / 'dummy').exists())

    def test_sync_global_processes_partial(self):
      participating_processes = {0, 2}
      primary_process = 0
      non_primary_process = 1

      directory = self.tmpdir / 'testdir'
      if multihost.process_index() == primary_process:
        directory.mkdir(parents=False, exist_ok=False)
      test_utils.sync_global_processes(
          'test_sync_global_processes_partial_setup'
      )

      if multihost.process_index() == primary_process:
        time.sleep(2)
        (directory / 'dummy').mkdir(parents=False, exist_ok=False)
      if multihost.process_index() in participating_processes:
        multihost.sync_global_processes(
            'test_sync_global_processes_partial',
            processes=participating_processes,
        )
      if multihost.process_index() in participating_processes:
        self.assertTrue((directory / 'dummy').exists())
      else:
        self.assertFalse((directory / 'dummy').exists())

      if multihost.process_index() == primary_process:
        time.sleep(2)
        (directory / 'foo').mkdir(parents=False, exist_ok=False)
      if multihost.process_index() in participating_processes:
        multihost.sync_global_processes(
            'test_sync_global_processes_partial_second',
            processes=participating_processes,
        )
      if multihost.process_index() in participating_processes:
        self.assertTrue((directory / 'foo').exists())
      else:
        self.assertFalse((directory / 'foo').exists())

      multihost.sync_global_processes('test_sync_global_processes_partial_all')
      # If non-primary processes get past the above barrier without waiting for
      # all, then an error would happen for the primary process when trying to
      # create subdirectories.
      if multihost.process_index() == non_primary_process:
        directory.rmtree()

    def test_different_barriers(self):
      slice1 = {0, 2}
      slice2 = {1, 3}
      primary_processes = [0, 1]

      if multihost.process_index() in primary_processes:
        # Don't sleep for slice1, but do sleep for slice2, so that when slice1
        # finishes waiting at the barrier, one file exists but the other does
        # not.
        time.sleep(3 * multihost.process_index())
        (self.tmpdir / f'dummy_{multihost.process_index()}').mkdir(
            parents=False, exist_ok=False
        )

      if multihost.process_index() in slice1:
        multihost.sync_global_processes(
            'test_different_barriers_slice1',
            processes=slice1,
        )
      else:
        multihost.sync_global_processes(
            'test_different_barriers_slice2',
            processes=slice2,
        )
      if multihost.process_index() in slice1:
        self.assertTrue((self.tmpdir / 'dummy_0').exists())
        self.assertFalse((self.tmpdir / 'dummy_1').exists())
      else:
        self.assertTrue((self.tmpdir / 'dummy_0').exists())
        self.assertTrue((self.tmpdir / 'dummy_1').exists())

    def test_broadcast_one_to_all(self):
      if multihost.process_index() == 0:
        tree = {'bar': [5, 12]}
      else:
        tree = {'bar': [0, 0]}
      result = multihost.broadcast_one_to_all(tree)

      expected = {
          'bar': [np.asarray(5, dtype=np.int32), np.asarray(12, dtype=np.int32)]
      }
      test_utils.assert_tree_equal(self, expected, result)


    def test_sync_global_processes_with_distributed_barrier(self):
      with flagsaver.flagsaver(
          experimental_orbax_use_distributed_barrier=True
      ), mock.patch.object(
          multihost.multihost_utils, 'sync_global_devices', autospec=True
      ) as mock_sync_global_devices, mock.patch.object(
          multihost, 'get_barrier_sync_fn', autospec=True
      ) as mock_get_barrier_sync_fn, mock.patch.object(
          multihost, 'should_skip_process_sync', return_value=False
      ):
        multihost.sync_global_processes('test_barrier')

        mock_sync_global_devices.assert_not_called()
        mock_get_barrier_sync_fn.assert_called_once_with(processes=None)
        mock_get_barrier_sync_fn.return_value.assert_called_once_with(
            key='test_barrier', timeout_ms=300000
        )

    def test_sync_global_processes_without_distributed_barrier(self):
      with flagsaver.flagsaver(
          experimental_orbax_use_distributed_barrier=False
      ), mock.patch.object(
          multihost.multihost_utils, 'sync_global_devices', autospec=True
      ) as mock_sync_global_devices, mock.patch.object(
          multihost, 'get_barrier_sync_fn', autospec=True
      ) as mock_get_barrier_sync_fn, mock.patch.object(
          multihost, 'should_skip_process_sync', return_value=False
      ):
        multihost.sync_global_processes('test_barrier')

        mock_sync_global_devices.assert_called_once()
        mock_get_barrier_sync_fn.assert_not_called()


class MultihostUtilsTestStandard(MultihostUtilsTestBase.Test):

  def setUp(self):
    self.enter_context(
        flagsaver.flagsaver(experimental_orbax_use_distributed_process_id=False)
    )
    super().setUp()

  def test_sync_global_processes_partial(self):
    self.skipTest('Fix this scenario.')

  def test_different_barriers(self):
    self.skipTest('Fix this scenario.')


class MultihostUtilsTestDistributedId(MultihostUtilsTestBase.Test):

  def setUp(self):
    self.enter_context(
        flagsaver.flagsaver(experimental_orbax_use_distributed_process_id=True)
    )
    super().setUp()


if __name__ == '__main__':
  multiprocess_test.main()
