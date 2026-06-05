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

"""Tests for tree types validation."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint.experimental.v1._src.tree import types_validation


class TypesValidationTest(parameterized.TestCase):

  @parameterized.parameters(
      (np.ones((2, 2)), True),
      (1, True),
      (1.0, True),
      ('test', True),
      (b'test', True),
      (True, True),
      (None, False),
      (object(), False),
  )
  def test_is_supported_leaf(self, x, expected):
    self.assertEqual(types_validation.is_supported_leaf(x), expected)

  def test_is_supported_leaf_jax(self):
    self.assertTrue(types_validation.is_supported_leaf(jnp.ones((2, 2))))

  @parameterized.parameters(
      (int, True),
      (1, True),
      (str, True),
      ('test', True),
      (..., True),
      (object, False),
      (object(), False),
  )
  def test_is_supported_abstract_leaf(self, x, expected):
    self.assertEqual(types_validation.is_supported_abstract_leaf(x), expected)

  def test_is_supported_abstract_leaf_jax(self):
    self.assertTrue(
        types_validation.is_supported_abstract_leaf(
            jax.ShapeDtypeStruct((2, 2), np.float32)
        )
    )
    self.assertTrue(
        types_validation.is_supported_abstract_leaf(jax.ShapeDtypeStruct)
    )


if __name__ == '__main__':
  absltest.main()
