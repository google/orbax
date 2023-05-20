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

from orbax.export.serving_config import ServingConfig
from orbax.export.serving_config import TensorSpecWithDefault
from orbax.export.serving_config import with_default_args

import tensorflow as tf


class ServingConfigTest(tf.test.TestCase):

  def test_with_default_args(self):
    sc = ServingConfig(
        signature_key='f',
        input_signature=[
            TensorSpecWithDefault(
                tf.TensorSpec([None], tf.int32),
                np.asarray([1, 2]),
            )
        ],
    )
    tf_f = with_default_args(
        sc.bind(
            {'f': tf.reduce_sum},
            require_numpy=False,
        )['f'],
        sc.get_input_signature(),
    )
    self.assertEqual(tf_f(required=[]), 3)

  def test_with_default_args_nested(self):
    def preprocess(required_arg, optional_args):
      return (
          required_arg
          + optional_args['foo']
          + optional_args['bar'][0]
          + optional_args['bar'][1]
      )

    sc = ServingConfig(
        signature_key='f',
        input_signature=(
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
        ),
        tf_preprocessor=preprocess,
    )

    tf_f = with_default_args(
        sc.bind({'f': lambda x: x}, require_numpy=False)['f'],
        sc.get_input_signature(),
    )
    self.assertAllEqual(
        tf_f(required=[np.asarray([6, 7])]), np.asarray([12, 16])
    )


if __name__ == '__main__':
  tf.test.main()
