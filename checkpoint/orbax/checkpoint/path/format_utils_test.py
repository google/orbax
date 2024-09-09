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

"""Tests for atomicity.py."""

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import numpy as np
from orbax.checkpoint import args as args_lib
from orbax.checkpoint import checkpoint_manager
from orbax.checkpoint import checkpointer
from orbax.checkpoint._src.handlers import pytree_checkpoint_handler
from orbax.checkpoint._src.handlers import standard_checkpoint_handler
from orbax.checkpoint.metadata import checkpoint as checkpoint_metadata
from orbax.checkpoint.path import format_utils



Checkpointer = checkpointer.Checkpointer
CheckpointManager = checkpoint_manager.CheckpointManager
PyTreeCheckpointHandler = pytree_checkpoint_handler.PyTreeCheckpointHandler
StandardCheckpointHandler = (
    standard_checkpoint_handler.StandardCheckpointHandler
)
CheckpointArgs = args_lib.CheckpointArgs
PyTreeSave = args_lib.PyTreeSave
StandardSave = args_lib.StandardSave
Composite = args_lib.Composite


class FormatUtilsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(
        self.create_tempdir('format_utils_test').full_path
    )
    (self.directory / 'dummy_file.txt').write_text('foo')
    self.pytree = {'a': 1, 'b': np.arange(8)}

  @parameterized.parameters(
      (PyTreeCheckpointHandler,), (StandardCheckpointHandler,)
  )
  def test_standard_format(self, handler_type):
    ckpt = self.directory / 'ckpt'
    if handler_type == PyTreeCheckpointHandler:
      args_type = PyTreeSave
    else:
      args_type = StandardSave
    with Checkpointer(handler_type()) as ckptr:
      ckptr.save(ckpt, args=args_type(self.pytree))

    self.assertTrue(format_utils.is_orbax_checkpoint(ckpt))
    self.assertTrue(format_utils.is_orbax_checkpoint(ckpt.parent))
    self.assertFalse(format_utils.is_orbax_checkpoint(ckpt.parent.parent))

  @parameterized.parameters((True,), (False,))
  def test_checkpoint_manager_single_item(self, with_checkpoint_metadata):
    with CheckpointManager(self.directory) as mngr:
      self.assertTrue(mngr.save(0, args=StandardSave(self.pytree)))
    if not with_checkpoint_metadata:
      checkpoint_metadata.metadata_file_path(self.directory / '0').unlink(
          missing_ok=False
      )

    self.assertFalse(format_utils.is_orbax_checkpoint(self.directory))
    self.assertTrue(format_utils.is_orbax_checkpoint(self.directory / '0'))
    self.assertTrue(
        format_utils.is_orbax_checkpoint(self.directory / '0' / 'default')
    )

  @parameterized.parameters((True,), (False,))
  def test_checkpoint_manager_multiple_items(self, with_checkpoint_metadata):
    item_names = ('params', 'state')
    with CheckpointManager(self.directory, item_names=item_names) as mngr:
      self.assertTrue(
          mngr.save(
              0,
              args=Composite(
                  params=StandardSave(self.pytree),
                  state=StandardSave(self.pytree),
              ),
          )
      )
    if not with_checkpoint_metadata:
      checkpoint_metadata.metadata_file_path(self.directory / '0').unlink(
          missing_ok=False
      )

    self.assertFalse(format_utils.is_orbax_checkpoint(self.directory))
    self.assertTrue(format_utils.is_orbax_checkpoint(self.directory / '0'))
    for item_name in item_names:
      self.assertTrue(
          format_utils.is_orbax_checkpoint(self.directory / '0' / item_name)
      )

  def test_path_does_not_exist(self):
    with self.assertRaises(FileNotFoundError):
      format_utils.is_orbax_checkpoint(self.directory / '/foo/bar')

  def test_path_is_not_a_directory(self):
    with self.assertRaises(NotADirectoryError):
      format_utils.is_orbax_checkpoint(self.directory / 'dummy_file.txt')



if __name__ == '__main__':
  absltest.main()
