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

"""Tests for model surgery repeating transformations."""

from absl.testing import absltest
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint.experimental.model_surgery.transformations import repeating


class RepeatingTest(absltest.TestCase):

  def test_repeat_by_pattern(self):
    params = {
        "layers.0.weight": jnp.array([[1, 2], [3, 4]]),
        "other.param": jnp.array([5]),
    }
    transform = repeating.repeat_by_pattern(
        pattern=r"^(layers\.\d+)\.weight$",
        dimension=1,
        repeat_count=2,
    )
    result = transform(params)

    self.assertIn("layers.0.weight", result)
    self.assertIn("other.param", result)
    np.testing.assert_array_equal(
        result["layers.0.weight"], jnp.array([[1, 1, 2, 2], [3, 3, 4, 4]])
    )
    np.testing.assert_array_equal(result["other.param"], jnp.array([5]))

  def test_repeat_by_keys(self):
    params = {
        "a": jnp.array([[1, 2], [3, 4]]),
        "b": jnp.array([5]),
    }
    transform = repeating.repeat_by_keys(
        target_keys=["a"], dimension=1, repeat_count=2
    )
    result = transform(params)

    self.assertIn("a", result)
    self.assertIn("b", result)
    np.testing.assert_array_equal(
        result["a"], jnp.array([[1, 1, 2, 2], [3, 3, 4, 4]])
    )
    np.testing.assert_array_equal(result["b"], jnp.array([5]))

  def test_repeat_by_keys_missing(self):
    params = {
        "a": jnp.array([[1, 2], [3, 4]]),
    }
    transform = repeating.repeat_by_keys(
        target_keys=["missing_key"], dimension=1, repeat_count=2
    )
    with self.assertLogs(level="WARNING"):
      result = transform(params)

    self.assertIn("a", result)
    self.assertNotIn("missing_key", result)

  def test_repeat_multiple_dimensions(self):
    params = {
        "block_param": jnp.array([[1]]),
    }
    transform_dim1 = repeating.repeat_by_pattern(
        pattern=r"^block_param$",
        dimension=1,
        repeat_count=2,
    )
    transform_dim0 = repeating.repeat_by_pattern(
        pattern=r"^block_param$",
        dimension=0,
        repeat_count=2,
    )

    result = transform_dim0(transform_dim1(params))

    self.assertIn("block_param", result)
    np.testing.assert_array_equal(
        result["block_param"], jnp.array([[1, 1], [1, 1]])
    )


if __name__ == "__main__":
  absltest.main()
