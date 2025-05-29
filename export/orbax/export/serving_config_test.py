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

import jax.numpy as jnp
import numpy as np
from orbax.export import serving_config
import tensorflow as tf


ServingConfig = serving_config.ServingConfig


class ServingConfigTest(tf.test.TestCase):

  def test_bind_tf(self):
    sc = ServingConfig(
        signature_key='f',
        input_signature=[
            tf.TensorSpec([None], tf.int32),
        ],
    )
    tf_f = sc.bind(
        {'f': tf.reduce_sum},
        require_numpy=False,
    )['f']
    self.assertEqual(tf_f(np.asarray([1, 2])), 3)

  def test_bind_jax(self):
    sc = ServingConfig(
        signature_key='f',
        input_signature=[
            tf.TensorSpec([None], tf.int32),
        ],
    )
    tf_f = sc.bind(
        {'f': jnp.sum},
        require_numpy=True,
    )['f']
    self.assertEqual(tf_f(np.asarray([1, 2])), 3)


if __name__ == '__main__':
  tf.test.main()
