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
from orbax.experimental.model import core as obm
from orbax.experimental.model.tf2obm._src import utils
import tensorflow as tf


class UtilsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="fully_ranked",
          spec=tf.TensorSpec(shape=(2, 3), dtype=tf.float32, name="t1"),
          expected=obm.ShloTensorSpec(
              shape=(2, 3), dtype=obm.ShloDType.f32, name="t1"
          ),
      ),
      dict(
          testcase_name="unknown_dim",
          spec=tf.TensorSpec(shape=(None,), dtype=tf.int32, name="t1"),
          expected=obm.ShloTensorSpec(
              shape=(None,), dtype=obm.ShloDType.i32, name="t1"
          )
      ),
      dict(
          testcase_name="partially_unknown_dim",
          spec=tf.TensorSpec(shape=(None, 3, 5), dtype=tf.int32, name="t1"),
          expected=obm.ShloTensorSpec(
              shape=(None, 3, 5), dtype=obm.ShloDType.i32, name="t1"
          ),
      ),
      dict(
          testcase_name="scalar",
          spec=tf.TensorSpec(shape=(), dtype=tf.string, name="t1"),
          expected=obm.ShloTensorSpec(
              shape=(), dtype=obm.ShloDType.str, name="t1"
          ),
      ),
      dict(
          testcase_name="unranked",
          spec=tf.TensorSpec(shape=None, dtype=tf.string, name="t1"),
          expected=obm.ShloTensorSpec(
              shape=None, dtype=obm.ShloDType.str, name="t1"
          ),
      ),
  )
  def test_tf_tensor_spec_to_obm(self, spec, expected):
    self.assertEqual(utils.tf_tensor_spec_to_obm(spec), expected)


if __name__ == "__main__":
  absltest.main()
