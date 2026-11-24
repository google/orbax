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

"""Tests for the benchmark multihost helpers."""

from unittest import mock

from absl.testing import absltest
from orbax.checkpoint._src.multihost import multihost as orbax_multihost
from orbax.checkpoint._src.testing.benchmarks.core import multihost


class GatherOnPrimaryTest(absltest.TestCase):

  def test_primary_gets_every_value_in_process_order(self):
    client = mock.Mock()
    client.key_value_dir_get.return_value = [
        ('benchmark_gather/x/2', '"c"'),
        ('benchmark_gather/x/0', '"a"'),
        ('benchmark_gather/x/1', '"b"'),
    ]
    self.enter_context(
        mock.patch.object(orbax_multihost, 'process_count', return_value=3)
    )
    self.enter_context(
        mock.patch.object(
            orbax_multihost, 'get_jax_distributed_client', return_value=client
        )
    )
    self.enter_context(
        mock.patch.object(multihost, 'get_process_index', return_value=0)
    )
    self.enter_context(mock.patch.object(multihost, 'sync_global_processes'))

    self.assertEqual(multihost.gather_on_primary('x', 'a'), ['a', 'b', 'c'])

  def test_single_process_needs_no_store(self):
    self.enter_context(
        mock.patch.object(orbax_multihost, 'process_count', return_value=1)
    )
    client = self.enter_context(
        mock.patch.object(orbax_multihost, 'get_jax_distributed_client')
    )

    self.assertEqual(multihost.gather_on_primary('x', 3), [3])
    client.assert_not_called()


if __name__ == '__main__':
  absltest.main()
