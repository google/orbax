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

import os

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
from orbax.export.examples.flax_mnist_main import export_mnist

import tensorflow as tf


_FLAGS = flags.FLAGS


class SavedModelMainTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._export_dir = self.create_tempdir().full_path
    self.enter_context(flagsaver.flagsaver(output_dir=self._export_dir))

  @parameterized.named_parameters(
      dict(testcase_name="fixed batch size", batch_size=8),
      dict(testcase_name="polymorphic batch size", batch_size=None))
  def test_export_mnist(self, batch_size):
    with flagsaver.flagsaver(batch_size=batch_size):
      export_mnist()
      self.assertTrue(
          tf.io.gfile.exists(os.path.join(self._export_dir, "saved_model.pb"))
      )


if __name__ == "__main__":
  _FLAGS.set_default("output_dir", "default-in-test")
  absltest.main()
