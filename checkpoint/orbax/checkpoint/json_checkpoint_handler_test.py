# Copyright 2023 The Orbax Authors.
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

"""Tests for JsonCheckpointHandler."""
from absl import flags
from absl.testing import absltest
from etils import epath
import jax
from orbax.checkpoint.json_checkpoint_handler import JsonCheckpointHandler

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()

FLAGS = flags.FLAGS


class JsonCheckpointHandlerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(
        self.create_tempdir(name='checkpointing_test').full_path)

  def test_save_restore(self):
    item = {'a': 1, 'b': {'c': 'test1', 'b': 'test2'}, 'd': 5.5}
    checkpointer = JsonCheckpointHandler()
    checkpointer.save(self.directory, item)
    restored = checkpointer.restore(self.directory)
    self.assertEqual(item, restored)

  def test_save_restore_filename(self):
    item = {'a': 1, 'b': {'c': 'test1', 'b': 'test2'}, 'd': 5.5}
    checkpointer = JsonCheckpointHandler(filename='file')
    checkpointer.save(self.directory, item)
    restored = checkpointer.restore(self.directory)
    self.assertEqual(item, restored)
    self.assertTrue((self.directory / 'file').exists())
    self.assertFalse((self.directory / 'metadata').exists())


if __name__ == '__main__':
  absltest.main()
