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

from unittest import mock
from absl import flags
from orbax.checkpoint._src.checkpointers import checkpointer as checkpointer_lib
from orbax.checkpoint._src.checkpointers import checkpointer_test_utils
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.testing import multiprocess_test


FLAGS = flags.FLAGS


class CheckpointerTest(
    checkpointer_test_utils.CheckpointerTestBase.Test,
    multiprocess_test.MultiProcessTest,
):

  def checkpointer(self, handler, **kwargs):
    return checkpointer_lib.Checkpointer(handler, **kwargs)

  def test_save_metrics(self):
    handler = checkpointer_test_utils.PyTreeCheckpointHandler()
    checkpointer = self.checkpointer(handler)
    with mock.patch(
        'jax.monitoring.record_event_duration_secs'
    ) as mock_record_duration:
      checkpointer.save(
          self.directory,
          args=checkpointer_test_utils.args.PyTreeSave(self.pytree),
      )
      recorded_metrics = [
          call[0][0] for call in mock_record_duration.call_args_list
      ]
      if multihost.is_primary_host(checkpointer._primary_host):
        self.assertIn(
            '/jax/orbax/write/blocking_duration_secs', recorded_metrics
        )
      self.assertIn(
          '/jax/orbax/write/blocking_tree_map_duration_secs', recorded_metrics
      )
      self.assertIn(
          '/jax/orbax/write/blocking_d2h_duration_secs', recorded_metrics
      )
    checkpointer.close()


if __name__ == '__main__':
  multiprocess_test.main()
