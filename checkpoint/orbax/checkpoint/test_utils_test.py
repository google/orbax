# Copyright 2024 The Orbax Authors.
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

"""To test Orbax in single-host setup."""

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from orbax.checkpoint import multihost
from orbax.checkpoint import test_utils


@test_utils.barrier_compatible_test
class BarrierCompatibleTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.ckpt_dir = epath.Path(self.create_tempdir('ckpt').full_path)

  def test_unique_barrier_key(self):
    expected_key = 'BarrierCompatibleTest.test_unique_barrier_key'
    self.assertIn(
        expected_key,
        multihost.utils._unique_barrier_key('foo'),
    )
    self.assertIn(
        'foo',
        multihost.utils._unique_barrier_key('footest_path_permission_mode'),
    )

  def test_unique_counter_primary(self):
    expected_key = 'BarrierCompatibleTest.test_unique_counter_primary'
    self.assertIn(f'{expected_key}_0', multihost.counters.async_save_counter())
    self.assertIn(f'{expected_key}_1', multihost.counters.async_save_counter())

  def test_unique_counter_secondary(self):
    expected_key = 'BarrierCompatibleTest.test_unique_counter_secondary'
    self.assertIn(f'{expected_key}_0', multihost.counters.async_save_counter())
    self.assertIn(f'{expected_key}_1', multihost.counters.async_save_counter())


if __name__ == '__main__':
  absltest.main()
