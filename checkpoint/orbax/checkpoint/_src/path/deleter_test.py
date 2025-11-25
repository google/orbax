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

import unittest
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
    return directory / str(step)

  @parameterized.product(
      threaded=(False, True),
      todelete_subdir=(None, 'some_delete_dir'),
  )
  def test_checkpoint_deleter_delete(
      self, threaded, todelete_subdir, enable_hns: bool = False
  ):
    """Test regular CheckpointDeleter."""
    deleter = deleter_lib.create_checkpoint_deleter(
        self.ckpt_dir,
        name_format=step_lib.standard_name_format(),
        primary_host=None,
        todelete_subdir=todelete_subdir,
        todelete_full_path=None,
        enable_hns=enable_hns,
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


class GcsRenameTest(unittest.TestCase):

  @mock.patch('orbax.checkpoint._src.path.deleter.epath.Path')
  def test_gcs_rename_logic_directly(self, mock_epath_constructor):
    """Tests path construction and rename call logic."""
    standard_checkpoint_deleter = deleter_lib.StandardCheckpointDeleter

    deleter = standard_checkpoint_deleter(
        directory=mock.MagicMock(),
        name_format=step_lib.standard_name_format(),
        primary_host=None,
        todelete_subdir=None,
        todelete_full_path='trash_bin',
        enable_hns=False,
    )
    # When epath.Path() is called inside the code, it returns this mock parent
    mock_dest_parent = mock.MagicMock()
    mock_epath_constructor.return_value = mock_dest_parent

    # When the code does (parent / child), return a specific final mock
    mock_final_dest = mock.MagicMock()
    mock_final_dest.__str__.return_value = 'gs://mocked/final/destination'
    mock_dest_parent.__truediv__.return_value = mock_final_dest

    # Setup the "Source" Mock (The step being deleted)
    mock_step_path = mock.MagicMock()
    mock_step_path.__str__.return_value = 'gs://my-bucket/checkpoints/step_10'
    mock_step_path.name = 'step_10'

    deleter._gcs_rename_step(step=10, delete_target=mock_step_path)

    # Verify mkdir was called on the destination parent.
    mock_dest_parent.mkdir.assert_called_with(parents=True, exist_ok=True)

    # Verify the Parent Path string was constructed correctly
    # The code does: epath.Path(f'gs://{bucket}/{todelete_full_path}')
    (parent_path_arg,), _ = mock_epath_constructor.call_args
    self.assertEqual(parent_path_arg, 'gs://my-bucket/trash_bin')

    # Verify the Child Filename was constructed correctly
    (child_name_arg,), _ = mock_dest_parent.__truediv__.call_args
    self.assertIn('step_10-', child_name_arg)

    # Verify the Rename was actually called
    mock_step_path.rename.assert_called_with(mock_final_dest)

if __name__ == '__main__':
  absltest.main()
