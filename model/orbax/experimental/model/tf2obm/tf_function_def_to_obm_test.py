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

"""tf2obm test."""

from orbax.experimental.model.tf2obm import tf_function_def_to_obm
import tensorflow as tf
from absl.testing import absltest


class TfFunctionDefToObmTest(absltest.TestCase):

  def test_tf_concrete_function_to_unstructured_data_success(self):

    @tf.function
    def tf_pow(a):
      degree = 2 if a.dtype == tf.float32 else 3
      return a**degree

    tf_input_tensor_spec = tf.TensorSpec((4, 5), tf.float32)
    tf_square = tf_pow.get_concrete_function(
        a=tf_input_tensor_spec,
    )

    fn = tf_function_def_to_obm.tf_concrete_function_to_obm_function(tf_square)

    file_extension = fn.body.ext_name
    unstructured_data = fn.body.proto
    self.assertEqual(
        file_extension,
        "pb",
    )
    self.assertEqual(
        unstructured_data.mime_type,
        "tf_function_def",
    )
    self.assertEqual(
        unstructured_data.version,
        "0.0.1",
    )
    self.assertEqual(
        unstructured_data.inlined_bytes,
        tf_square.function_def.SerializeToString(),
    )


if __name__ == "__main__":
  absltest.main()
