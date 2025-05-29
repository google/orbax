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

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from orbax.checkpoint._src.checkpointers import standard_checkpointer
from orbax.checkpoint.experimental.v1._src.path import format_utils
from orbax.checkpoint.experimental.v1._src.saving import saving
from orbax.checkpoint.experimental.v1._src.testing import array_utils as array_test_utils


class FormatUtilsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(self.create_tempdir().full_path) / 'ckpt'
    self.pytree, _ = array_test_utils.create_numpy_pytree()
    saving.save_pytree(self.directory, self.pytree)
    # Save a checkpoint with a checkpointable name, `state`.
    with standard_checkpointer.StandardCheckpointer() as checkpointer:
      checkpointer.save(self.directory / 'state', self.pytree)
    # Save a checkpoint with no checkpointable subdir.
    self.ckpt_dir = epath.Path(self.create_tempdir().full_path) / 'direct'
    with standard_checkpointer.StandardCheckpointer() as checkpointer:
      checkpointer.save(self.ckpt_dir, self.pytree)

  @parameterized.product(checkpointable_name=['pytree', 'state', None])
  def test_nonexistent_path(self, checkpointable_name: str | None):
    directory = self.ckpt_dir if checkpointable_name is None else self.directory

    with self.assertRaises(FileNotFoundError):
      format_utils.validate_checkpoint_directory(
          directory / 'foo'
      )

  @parameterized.product(checkpointable_name=['pytree', 'state', None])
  def test_not_a_directory(self, checkpointable_name: str | None):
    directory = self.ckpt_dir if checkpointable_name is None else self.directory
    (directory / 'foo').write_text('foo')

    with self.assertRaises(NotADirectoryError):
      format_utils.validate_checkpoint_directory(
          directory / 'foo'
      )

  @parameterized.product(checkpointable_name=['pytree', 'state', None])
  def test_invalid_metadata(self, checkpointable_name: str | None):
    directory = self.ckpt_dir if checkpointable_name is None else self.directory
    (directory / '_CHECKPOINT_METADATA').unlink()

    format_utils.validate_pytree_checkpoint(
        directory, checkpointable_name=checkpointable_name
    )

    with self.assertRaises(FileNotFoundError):
      format_utils.validate_checkpoint_metadata(directory)

  @parameterized.product(checkpointable_name=['pytree', 'state', None])
  def test_invalid_pytree(self, checkpointable_name: str | None):
    directory = self.ckpt_dir if checkpointable_name is None else self.directory
    if checkpointable_name is None:
      directory.rmtree()
    else:
      (directory / checkpointable_name).rmtree()

    with self.assertRaises(FileNotFoundError):
      format_utils.validate_pytree_checkpoint(
          directory, checkpointable_name=checkpointable_name
      )

  @parameterized.product(checkpointable_name=['pytree', 'state', None])
  def test_no_pytree_metadata(self, checkpointable_name: str | None):
    directory = self.ckpt_dir if checkpointable_name is None else self.directory
    if checkpointable_name is None:
      (directory / '_METADATA').unlink()
    else:
      (directory / checkpointable_name / '_METADATA').unlink()

    with self.assertRaises(FileNotFoundError):
      format_utils.validate_pytree_checkpoint(
          directory, checkpointable_name=checkpointable_name
      )

  @parameterized.product(checkpointable_name=['pytree', 'state', None])
  def test_valid_pytree(self, checkpointable_name: str | None):
    directory = self.ckpt_dir if checkpointable_name is None else self.directory

    format_utils.validate_pytree_checkpoint(
        directory, checkpointable_name=checkpointable_name
    )


if __name__ == '__main__':
  absltest.main()
