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

import numpy as np
from orbax.export import utils
import tensorflow as tf


TensorSpecWithDefault = utils.TensorSpecWithDefault


class TensorSpecWithDefaultTest(tf.test.TestCase):

  def test_with_default_args(self):
    input_signature = [
        TensorSpecWithDefault(
            tf.TensorSpec([None], tf.int32),
            np.asarray([1, 2]),
        )
    ]
    tf_f = utils.with_default_args(
        tf.reduce_sum,
        input_signature,
    )
    self.assertEqual(tf_f(), 3)

  def test_bad_order(self):
    input_signature = [
        TensorSpecWithDefault(
            tf.TensorSpec([None], tf.int32),
            np.asarray([1, 2]),
        ),
        tf.TensorSpec([None], tf.int32),
    ]
    with self.assertRaisesRegex(
        ValueError,
        'non-default argument follows default argument',
    ):
      utils.with_default_args(lambda x, y: x + y, input_signature)

  def test_missing_default(self):
    input_signature = [[
        TensorSpecWithDefault(
            tf.TensorSpec([None], tf.int32),
            np.asarray([1, 2]),
        ),
        tf.TensorSpec([None], tf.int32),
    ]]
    with self.assertRaisesRegex(
        ValueError,
        'TensorSpecWithDefault must be defined for each tensor in the structure'
        ' for the Python arg',
    ):
      utils.with_default_args(lambda x: x[0] + x[1], input_signature)

  def test_with_default_args_nested(self):
    def f(required_arg, optional_args):
      return (
          required_arg
          + optional_args['foo']
          + optional_args['bar'][0]
          + optional_args['bar'][1]
      )

    input_signature = (
        tf.TensorSpec([2], tf.int32),
        dict(
            foo=TensorSpecWithDefault(
                tf.TensorSpec([2], tf.int32),
                np.asarray([0, 1]),
            ),
            bar=[
                TensorSpecWithDefault(
                    tf.TensorSpec([2], tf.int32),
                    np.asarray([2, 3]),
                ),
                TensorSpecWithDefault(
                    tf.TensorSpec([2], tf.int32),
                    np.asarray([4, 5]),
                ),
            ],
        ),
    )

    tf_f = utils.with_default_args(f, input_signature)
    self.assertAllEqual(tf_f(np.asarray([6, 7])), np.asarray([12, 16]))


if __name__ == '__main__':
  tf.test.main()
