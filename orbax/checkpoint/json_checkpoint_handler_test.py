# Copyright 2022 The Orbax Authors.
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
import jax
from orbax.checkpoint.json_checkpoint_handler import JsonCheckpointHandler
import tensorflow as tf

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()

FLAGS = flags.FLAGS


class JsonCheckpointHandlerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.directory = self.create_tempdir(name='checkpointing_test').full_path

  def test_save_restore(self):
    item = {'a': 1, 'b': {'c': 'test1', 'b': 'test2'}, 'd': 5.5}
    ckptr = JsonCheckpointHandler()
    ckptr.save(self.directory, item)
    restored = ckptr.restore(self.directory)
    self.assertEqual(item, restored)

  def test_save_restore_filename(self):
    item = {'a': 1, 'b': {'c': 'test1', 'b': 'test2'}, 'd': 5.5}
    ckptr = JsonCheckpointHandler(filename='file')
    ckptr.save(self.directory, item)
    restored = ckptr.restore(self.directory)
    self.assertEqual(item, restored)
    self.assertTrue(
        tf.io.gfile.exists(tf.io.gfile.join(self.directory, 'file')))
    self.assertFalse(
        tf.io.gfile.exists(tf.io.gfile.join(self.directory, 'metadata')))


if __name__ == '__main__':
  absltest.main()
