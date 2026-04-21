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

"""Tests for model surgery fusing transformations."""

from absl.testing import absltest
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint.experimental.model_surgery.transformations import fusing


class FusingTest(absltest.TestCase):

  def test_fuse_by_pattern_not_enough_keys(self):
    params = {
        "layers.0.gate_proj.weight": jnp.array([[1, 2]]),
        "other.param": jnp.array([5]),
    }
    transform = fusing.fuse_by_pattern(
        pattern=r"^(layers\.\d+)\.(gate_proj|up_proj)\.weight$",
        unique_parts=["gate_proj", "up_proj"],
        fused_unique_part="gate_up_proj",
    )
    result = transform(params)

    self.assertNotIn("layers.0.gate_up_proj.weight", result)
    self.assertIn("layers.0.gate_proj.weight", result)
    self.assertIn("other.param", result)
    np.testing.assert_array_equal(
        result["layers.0.gate_proj.weight"], jnp.array([[1, 2]])
    )

  def test_fuse_by_pattern_suffix(self):
    params = {
        "layers.0.gate_proj.weight": jnp.array([[1, 2]]),
        "layers.0.up_proj.weight": jnp.array([[3, 4]]),
        "other.param": jnp.array([5]),
    }
    transform = fusing.fuse_by_pattern(
        pattern=r"^(layers\.\d+)\.(gate_proj|up_proj)\.weight$",
        unique_parts=["gate_proj", "up_proj"],
        fused_unique_part="gate_up_proj",
    )
    result = transform(params)

    self.assertIn("layers.0.gate_up_proj.weight", result)
    self.assertIn("other.param", result)
    np.testing.assert_array_equal(
        result["layers.0.gate_up_proj.weight"], jnp.array([[1, 2], [3, 4]])
    )

  def test_fuse_by_pattern_prefix(self):
    params = {
        "part1.layer0.weight": jnp.array([[1, 2]]),
        "part2.layer0.weight": jnp.array([[3, 4]]),
        "other.param": jnp.array([5]),
    }
    transform = fusing.fuse_by_pattern(
        pattern=r"^(part1|part2)\.(layer0\.weight)$",
        unique_parts=["part1", "part2"],
        fused_unique_part="fused",
    )
    result = transform(params)

    self.assertIn("fused.layer0.weight", result)
    self.assertIn("other.param", result)
    np.testing.assert_array_equal(
        result["fused.layer0.weight"], jnp.array([[1, 2], [3, 4]])
    )

  def test_fuse_by_keys_missing_keys(self):
    params = {
        "a": jnp.array([1]),
        "c": jnp.array([3]),
    }
    transform = fusing.fuse_by_keys(source_keys=["a", "b"], target_key="ab")

    with self.assertLogs(level="WARNING"):
      result = transform(params)

    self.assertNotIn("ab", result)
    self.assertIn("a", result)
    self.assertIn("c", result)
    np.testing.assert_array_equal(result["a"], jnp.array([1]))

  def test_fuse_by_keys(self):
    params = {
        "a": jnp.array([1]),
        "b": jnp.array([2]),
        "c": jnp.array([3]),
    }
    transform = fusing.fuse_by_keys(source_keys=["a", "b"], target_key="ab")
    result = transform(params)

    self.assertIn("ab", result)
    self.assertIn("c", result)
    self.assertNotIn("a", result)
    self.assertNotIn("b", result)
    np.testing.assert_array_equal(result["ab"], jnp.array([1, 2]))


if __name__ == "__main__":
  absltest.main()
