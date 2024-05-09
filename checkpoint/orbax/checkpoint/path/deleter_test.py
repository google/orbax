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
from orbax.checkpoint.path import deleter as deleter_lib
from orbax.checkpoint.path import step as step_lib


class CheckpointDeleterTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.ckpt_dir = epath.Path(self.create_tempdir('ckpt').full_path)

  def _get_save_diretory(self, step: int, directory: epath.Path) -> epath.Path:
    return directory / str(step)

  @parameterized.product(
      threaded=(False, True),
      todelete_subdir=(None, 'some_delete_dir'),
  )
  def test_checkpoint_deleter_delete(self, threaded, todelete_subdir):
    """Test regular CheckpointDeleter."""
    deleter = deleter_lib.create_checkpoint_deleter(
        primary_host=None,
        directory=self.ckpt_dir,
        todelete_subdir=todelete_subdir,
        name_format=step_lib.standard_name_format(),
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


if __name__ == '__main__':
  absltest.main()
