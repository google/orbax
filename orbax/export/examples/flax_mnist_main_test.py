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

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
from orbax.export.examples.flax_mnist_main import export_mnist


_FLAGS = flags.FLAGS


class SavedModelMainTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(
        flagsaver.flagsaver(output_dir=self.create_tempdir().full_path))

  @parameterized.named_parameters(
      dict(testcase_name="fixed batch size", batch_size=8),
      dict(testcase_name="polymorphic batch size", batch_size=None))
  def test_export_mnist(self, batch_size):
    with flagsaver.flagsaver(batch_size=batch_size):
      export_mnist()


if __name__ == "__main__":
  _FLAGS.set_default("output_dir", "default-in-test")
  absltest.main()
