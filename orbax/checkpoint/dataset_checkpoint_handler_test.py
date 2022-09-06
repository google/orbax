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

"""Tests for DatasetCheckpointHandler."""

from unittest import mock

from absl import flags
from absl.testing import absltest
from etils import epath
import jax
from orbax.checkpoint import dataset_checkpoint_handler
import tensorflow as tf

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()

FLAGS = flags.FLAGS


class DatasetCheckpointHandlerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    self.directory = epath.Path(
        self.create_tempdir(name='checkpointing_test').full_path)
    self.dataset = tf.data.Dataset.range(64)

  def test_save_restore(self):
    checkpointer = dataset_checkpoint_handler.DatasetCheckpointHandler()
    iterator = iter(self.dataset)
    # Change iterator state to check restoration of original state.
    for _ in range(10):
      next(iterator)
    checkpointer.save(self.directory, iterator)
    restored = checkpointer.restore(self.directory, iter(self.dataset))
    self.assertEqual(10, next(restored).numpy())

  # We stub out jax.process_index() and jax.process_count() with the
  # dataset_checkpoint_handler module. This will not affect other modules
  # (which would break JAX multihost utils).
  @mock.patch.object(dataset_checkpoint_handler, 'jax')
  def test_save_restore_multihost(self, jax_mock):
    jax_mock.process_count.return_value = 2
    handler = dataset_checkpoint_handler.DatasetCheckpointHandler()

    # Process 0 - save().
    jax_mock.process_index.return_value = 0
    iterator = iter(self.dataset)
    # Change iterator state to check restoration of original state.
    for _ in range(10):
      next(iterator)
    handler.save(self.directory, iterator)
    # Sub-directory with checkpoint for this host was created.
    self.assertIn('process_0-of-2', [p.name for p in self.directory.iterdir()])

    # Process 1 - save().
    jax_mock.process_index.return_value = 1
    iterator = iter(self.dataset)
    # Change iterator state to check restoration of original state.
    for _ in range(5):
      next(iterator)
    handler.save(self.directory, iterator)
    # Sub-directory with checkpoint for this host was created.
    self.assertIn('process_1-of-2', [p.name for p in self.directory.iterdir()])

    # Process 0 - restore()
    jax_mock.process_index.return_value = 0
    restored = handler.restore(self.directory, iter(self.dataset))
    self.assertEqual(10, next(restored).numpy())

    # Process 1 - restore()
    jax_mock.process_index.return_value = 1
    restored = handler.restore(self.directory, iter(self.dataset))
    self.assertEqual(5, next(restored).numpy())


if __name__ == '__main__':
  absltest.main()
