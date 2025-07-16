# Copyright 2025 The Orbax Authors.
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

import queue
import threading
import time
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from orbax.checkpoint._src.path import deleter as deleter_lib
from orbax.checkpoint._src.path import step as step_lib


class CheckpointDeleterTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.ckpt_dir = epath.Path(self.create_tempdir('ckpt').full_path)

  def _get_save_diretory(self, step: int, directory: epath.Path) -> epath.Path:
    return directory / step_lib.standard_name_format().build_name(step)

  @parameterized.product(
      threaded=(False, True),
      todelete_subdir=(None, 'some_delete_dir'),
  )
  def test_checkpoint_deleter_delete(
      self, threaded, todelete_subdir, enable_hns_rmtree: bool = False
  ):
    """Test regular CheckpointDeleter."""
    deleter = deleter_lib.create_checkpoint_deleter(
        primary_host=None,
        directory=self.ckpt_dir,
        todelete_subdir=todelete_subdir,
        name_format=step_lib.standard_name_format(),
        enable_hns_rmtree=enable_hns_rmtree,
        enable_background_delete=threaded,
    )

    step = 1
    step_dir = self._get_save_diretory(step, self.ckpt_dir)
    step_dir.mkdir()
    self.assertTrue(step_dir.exists())
    deleter.delete(step)
    deleter.close()

    # assert the step_dir is deleted
    self.assertFalse(step_dir.exists())

    # In case of rename, check if the new folder exists
    if todelete_subdir is not None:
      self.assertTrue((self.ckpt_dir / todelete_subdir / str(step)).exists())

    deleter.close()

  @parameterized.parameters((1,), (3,), (5,))
  def test_checkpoint_deleter_thread_count(self, thread_count):
    """Test CheckpointDeleter with different thread counts."""

    thread_ids = queue.Queue()  # thread safe
    original_delete = deleter_lib.StandardCheckpointDeleter.delete

    def mock_delete(self, step):
      thread_ids.put(threading.get_ident())
      time.sleep(0.01)  # yield control to other threads.
      original_delete(self, step)

    with mock.patch.object(
        deleter_lib.StandardCheckpointDeleter, 'delete', new=mock_delete
    ):
      deleter = deleter_lib.create_checkpoint_deleter(
          primary_host=None,
          directory=self.ckpt_dir,
          todelete_subdir=None,
          name_format=step_lib.standard_name_format(),
          enable_hns_rmtree=False,
          enable_background_delete=True,
          background_thread_count=thread_count,
      )

      steps = list(range(thread_count * 5))
      step_dirs = []
      for step in steps:
        step_dir = self._get_save_diretory(step, self.ckpt_dir)
        step_dir.mkdir()
        self.assertTrue(step_dir.exists())
        step_dirs.append(step_dir)

      deleter.delete_steps(steps)
      deleter.close()

      # assert the step_dirs are deleted
      for step_dir in step_dirs:
        self.assertFalse(step_dir.exists())

      deleter.close()

      # make sure different threads are involved deletion.
      self.assertLen(set(thread_ids.queue), thread_count)


if __name__ == '__main__':
  absltest.main()
