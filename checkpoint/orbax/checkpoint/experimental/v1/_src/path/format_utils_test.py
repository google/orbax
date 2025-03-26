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

from absl.testing import absltest
from etils import epath
from orbax.checkpoint.experimental.v1._src.path import format_utils
from orbax.checkpoint.experimental.v1._src.saving import saving
from orbax.checkpoint.experimental.v1._src.testing import array_utils as array_test_utils


class FormatUtilsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(self.create_tempdir().full_path) / 'ckpt'
    self.pytree, _ = array_test_utils.create_numpy_pytree()
    saving.save_pytree(self.directory, self.pytree)

  def test_nonexistent_path(self):
    with self.assertRaises(FileNotFoundError):
      format_utils.validate_pytree_checkpoint(self.directory / 'foo')

  def test_not_a_directory(self):
    (self.directory / 'foo').write_text('foo')
    with self.assertRaises(NotADirectoryError):
      format_utils.validate_pytree_checkpoint(self.directory / 'foo')

  def test_invalid_metadata(self):
    (self.directory / '_CHECKPOINT_METADATA').unlink()
    with self.assertRaises(FileNotFoundError):
      format_utils.validate_pytree_checkpoint(self.directory)

  def test_invalid_pytree(self):
    (self.directory / format_utils.PYTREE_CHECKPOINTABLE_KEY).rmtree()
    with self.assertRaises(FileNotFoundError):
      format_utils.validate_pytree_checkpoint(self.directory)

  def test_no_pytree_metadata(self):
    (self.directory / 'pytree' / '_METADATA').unlink()
    with self.assertRaises(FileNotFoundError):
      format_utils.validate_pytree_checkpoint(self.directory)

  def test_valid_pytree(self):
    format_utils.validate_pytree_checkpoint(self.directory)


if __name__ == '__main__':
  absltest.main()
