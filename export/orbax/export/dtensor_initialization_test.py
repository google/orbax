# Copyright 2026 The Orbax Authors.
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
from orbax.export import dtensor_utils
import tensorflow as tf


class DtensorInitializationTest(absltest.TestCase):

  def test_initialize_dtensor(self):
    self.assertFalse(dtensor_utils.dtensor_initialized())
    dtensor_utils.initialize_dtensor()
    self.assertTrue(dtensor_utils.dtensor_initialized())
    dtensor_utils.shutdown_dtensor()
    self.assertFalse(dtensor_utils.dtensor_initialized())

    _ = tf.constant(1)  # Trigger an implicit TF context init.
    with self.assertRaisesRegex(
        ValueError, "TensorFlow has already been initialized"
    ):
      dtensor_utils.initialize_dtensor()
    dtensor_utils.initialize_dtensor(reset_context=True)
    self.assertTrue(dtensor_utils.dtensor_initialized())


if __name__ == "__main__":
  absltest.main()
