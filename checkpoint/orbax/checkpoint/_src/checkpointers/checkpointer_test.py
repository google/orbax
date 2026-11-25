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

from absl import flags
import jax
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
    orig_sync_op_id = (
        checkpointer.synchronize_next_awaitable_signal_operation_id
    )

    def _delayed_sync_op_id():
      time.sleep(0.05)
      orig_sync_op_id()

    with (
        mock.patch.object(
            jax.monitoring,
            'record_event_duration_secs',
            autospec=True,
        ) as mock_record_duration,
        mock.patch.object(
            checkpointer,
            'synchronize_next_awaitable_signal_operation_id',
            autospec=True,
            side_effect=_delayed_sync_op_id,
        ),
    ):
      checkpointer.save(
          self.directory,
          args=checkpointer_test_utils.args.PyTreeSave(self.pytree),
      )
      recorded_durations = {
          call.args[0]: call.args[1]
          for call in mock_record_duration.call_args_list
      }
      if multihost.is_primary_host(checkpointer._primary_host):
        self.assertIn(
            '/jax/orbax/write/blocking_duration_secs', recorded_durations
        )
      self.assertIn(
          '/jax/orbax/write/checkpoint_start_sync_duration_secs',
          recorded_durations,
      )
      self.assertGreaterEqual(
          recorded_durations[
              '/jax/orbax/write/checkpoint_start_sync_duration_secs'
          ],
          0.05,
      )
      self.assertIn(
          '/jax/orbax/write/blocking_tree_map_duration_secs', recorded_durations
      )
      self.assertIn(
          '/jax/orbax/write/blocking_d2h_duration_secs', recorded_durations
      )
    checkpointer.close()


if __name__ == '__main__':
  multiprocess_test.main()
