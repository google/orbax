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

"""Tests for DatasetCheckpointer."""

from absl import flags
from absl.testing import absltest
import jax
from orbax.checkpoint.dataset_checkpointer import DatasetCheckpointer
import tensorflow as tf

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()

FLAGS = flags.FLAGS


class DatasetCheckpointerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    self.directory = self.create_tempdir(name='checkpointing_test').full_path
    self.dataset = tf.data.Dataset.range(64)

  def test_save_restore(self):
    ckptr = DatasetCheckpointer()
    iterator = iter(self.dataset)
    # change iterator state
    for _ in range(10):
      next(iterator)
    ckptr.save(self.directory, iterator)
    restored = ckptr.restore(self.directory, iter(self.dataset))
    self.assertEqual(10, next(restored).numpy())


if __name__ == '__main__':
  absltest.main()
