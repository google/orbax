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

"""Tests for model surgery stacking transformations."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint.experimental.model_surgery.transformations import stacking


Mesh = jax.sharding.Mesh
NamedSharding = jax.sharding.NamedSharding
PartitionSpec = jax.sharding.PartitionSpec


class StackingTest(absltest.TestCase):

  def test_stack(self):
    params = {
        "layers.0.mlp.experts.0.weight": jnp.array([1, 2]),
        "layers.0.mlp.experts.1.weight": jnp.array([3, 4]),
        "layers.1.mlp.experts.0.weight": jnp.array([5, 6]),
        "layers.1.mlp.experts.1.weight": jnp.array([7, 8]),
        "other.param": jnp.array([9]),
    }
    transform = stacking.stack(r"experts\.(\d+\.)")
    result = transform(params)

    self.assertIn("layers.0.mlp.experts.weight", result)
    self.assertIn("layers.1.mlp.experts.weight", result)
    self.assertIn("other.param", result)

    np.testing.assert_array_equal(
        result["layers.0.mlp.experts.weight"], jnp.array([[1, 2], [3, 4]])
    )
    np.testing.assert_array_equal(
        result["layers.1.mlp.experts.weight"], jnp.array([[5, 6], [7, 8]])
    )

  def test_stack_numpy(self):
    params = {
        "layers.0.q_proj": np.array([1]),
        "layers.1.q_proj": np.array([2]),
        "other": np.array([3]),
    }
    transform = stacking.stack(r"layers\.(\d+\.)")
    result = transform(params)

    self.assertIn("layers.q_proj", result)
    self.assertIn("other", result)
    self.assertIsInstance(result["layers.q_proj"], np.ndarray)
    np.testing.assert_array_equal(result["layers.q_proj"], np.array([[1], [2]]))

  def test_stack_jax(self):
    params = {
        "layers.0.q_proj": jnp.array([1]),
        "layers.1.q_proj": jnp.array([2]),
    }
    transform = stacking.stack(pattern=r"layers\.(\d+\.)")
    result = transform(params)

    self.assertIn("layers.q_proj", result)
    self.assertNotIsInstance(result["layers.q_proj"], np.ndarray)
    np.testing.assert_array_equal(
        result["layers.q_proj"], jnp.array([[1], [2]])
    )

  def test_stack_padding(self):
    params = {
        "layers.0.q_proj": np.array([1]),
        "layers.1.q_proj": np.array([2]),
        "layers.0.weight_scale": np.array([10]),
    }
    transform_1 = stacking.stack(
        pattern=r"layers\.(\d+\.)q_proj",
        expected_count=3,
        default_filler=0.0,
    )
    transform_2 = stacking.stack(
        pattern=r"layers\.(\d+\.)weight_scale",
        expected_count=3,
        filler_mapping={"weight_scale": 1.0},
        default_filler=0.0,
    )
    with self.assertLogs(level="WARNING") as cm:
      result = transform_2(transform_1(params))

    self.assertIn("layers.q_proj", result)
    self.assertIn("layers.weight_scale", result)

    np.testing.assert_array_equal(
        result["layers.q_proj"], np.array([[1], [2], [0]])
    )
    np.testing.assert_array_equal(
        result["layers.weight_scale"], np.array([[10], [1], [1]])
    )

    warnings = [r.message for r in cm.records]
    self.assertLen(warnings, 2)
    self.assertIn(
        "Stacking layers.q_proj: Found 2 items, expected 3. Padded with 0.0.",
        warnings,
    )
    self.assertIn(
        "Stacking layers.weight_scale: Found 1 items, expected 3. Padded with"
        " 1.0.",
        warnings,
    )

  def test_stack_expected_count_none_with_padding(self):
    params = {
        "layers.0.q_proj": np.array([1]),
        "layers.2.q_proj": np.array([3]),
        "layers.0.weight_scale": np.array([10]),
    }
    transform = stacking.stack(
        pattern=r"layers\.(\d+\.)",
        expected_count=None,
        filler_mapping={"weight_scale": 1.0},
        default_filler=0.0,
    )
    with self.assertLogs(level="WARNING") as cm:
      result = transform(params)

    self.assertIn("layers.q_proj", result)
    self.assertIn("layers.weight_scale", result)

    np.testing.assert_array_equal(
        result["layers.q_proj"], np.array([[1], [0], [3]])
    )
    np.testing.assert_array_equal(
        result["layers.weight_scale"], np.array([[10], [1], [1]])
    )

    warnings = [r.message for r in cm.records]
    self.assertLen(warnings, 2)
    self.assertIn(
        "Stacking layers.q_proj: Found 2 items, expected 3. Padded with 0.0.",
        warnings,
    )
    self.assertIn(
        "Stacking layers.weight_scale: Found 1 items, expected 3. Padded with"
        " 1.0.",
        warnings,
    )

  def test_stack_missing_filler_fails(self):
    params = {
        "layers.0.q_proj": np.array([1]),
        "layers.2.q_proj": np.array([3]),
    }
    transform = stacking.stack(
        pattern=r"layers\.(\d+\.)",
        expected_count=3,
        default_filler=None,
        filler_mapping=None,
    )
    with self.assertRaisesRegex(
        ValueError,
        r'Stacking "layers\.q_proj": Found keys \[0, 2\], but expected indices'
        r" 0\.\.2 when padding is disabled",
    ):
      transform(params)

  def test_stack_inplace(self):
    params = {
        "layers.0.q_proj": np.array([1]),
        "layers.1.q_proj": np.array([2]),
    }
    transform = stacking.stack(
        pattern=r"layers\.(\d+\.)",
        expected_count=2,
        inplace=True,
    )
    params_dict = dict(params)
    result = transform(params_dict)

    self.assertNotIn("layers.0.q_proj", params_dict)
    self.assertNotIn("layers.1.q_proj", params_dict)
    self.assertIn("layers.q_proj", result)


if __name__ == "__main__":
  absltest.main()
