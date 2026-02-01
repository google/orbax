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

"""Tests for validate module."""
import json

from absl.testing import absltest
import jax  # pylint: disable=unused-import
import jax.numpy as jnp
import numpy as np
from orbax.export.validate import validation_utils
import tensorflow as tf


class ValidationUtilsTest(tf.test.TestCase):

  def test_enhanced_json_encoder(self):
    """Make sure json.dumps support tf tensor, np array, bytes format."""
    input_dict = {
        "tf": tf.convert_to_tensor(np.arange(8)),
        "np": np.arange(8),
        "bytes": bytes("test message", "utf-8")
    }
    json.dumps(input_dict, cls=validation_utils.EnhancedJSONEncoder)

  def test_get_latency_stat(self):
    latencies = [88.0, 88.0, 90.0, 91.0, 88.5, 88.7, 87.0]
    num_batches, avg_in_ms, p90_in_ms, p99_in_ms = (
        validation_utils.get_latency_stat(latencies=latencies)
    )
    self.assertEqual(num_batches, 7)
    self.assertAlmostEqual(avg_in_ms, 88742.85714285714)
    self.assertAlmostEqual(p90_in_ms, 90400.0)
    self.assertAlmostEqual(p99_in_ms, 90940.0)

  def test_split_tf_floating_and_discrete_groups(self):
    """Test split_tf_floating_and_discrete_groups accept np,tf and jax array."""
    arrays = [np.array([0.1, 0.2, 0.3], dtype=np.float64), np.array([1, 2, 3])]
    tf_arrays = jax.tree_util.tree_map(tf.convert_to_tensor, arrays)
    jax_array = jax.tree_util.tree_map(jnp.array, arrays)
    expected_float_vals = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    for arr in (arrays, tf_arrays, jax_array):
      float_vals, _ = validation_utils.split_tf_floating_and_discrete_groups(
          arr
      )
      self.assertAllClose(float_vals, expected_float_vals)

  def test_split_tf_floating_and_discrete_groups_2(self):
    """Test split_tf_floating_and_discrete_groups accept non-inhomogeneous array."""
    arrays = [np.zeros((1,)), np.zeros((2048,)), np.zeros((1, 1, 1))]
    float_vals, _ = validation_utils.split_tf_floating_and_discrete_groups(
        arrays
    )
    self.assertLen(float_vals, 2050)


if __name__ == "__main__":
  absltest.main()
