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

"""Tests for model surgery renaming transformations."""

from absl.testing import absltest
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint.experimental.model_surgery.transformations import renaming


class RenamingTest(absltest.TestCase):

  def test_rename_by_regex(self):
    params = {
        "params.weight": jnp.array([1]),
        "other.weight_scale_inv": jnp.array([2]),
        "keep.this": jnp.array([3]),
    }
    rules = [
        (r"^params\.", "carried_state."),
        (r"\.weight_scale_inv$", ".weight_scale"),
    ]
    transform = renaming.rename_by_regex(rules)
    result = transform(params)

    self.assertIn("carried_state.weight", result)
    self.assertIn("other.weight_scale", result)
    self.assertIn("keep.this", result)

    np.testing.assert_array_equal(
        result["carried_state.weight"], jnp.array([1])
    )
    np.testing.assert_array_equal(result["other.weight_scale"], jnp.array([2]))
    np.testing.assert_array_equal(result["keep.this"], jnp.array([3]))


if __name__ == "__main__":
  absltest.main()
