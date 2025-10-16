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
import jax
import numpy as np
from orbax.checkpoint._src.testing.benchmarks.core import pytree_utils


class PytreeUtilsTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          v_expected=1,
          v_actual='1',
          expected_regex='Type mismatch',
      ),
      dict(
          v_expected=np.array([1], dtype=np.int32),
          v_actual=np.array([1], dtype=np.int64),
          expected_regex='Dtype mismatch',
      ),
      dict(
          v_expected=np.arange(16),
          v_actual=np.arange(16) + 1,
          expected_regex='Error at',
      ),
      dict(
          v_expected=1,
          v_actual=2,
          expected_regex='Value mismatch',
      ),
  )
  def test_assert_pytree_equal_raises_error_on_mismatch(
      self, v_expected, v_actual, expected_regex
  ):
    with self.assertRaisesRegex(AssertionError, expected_regex):
      pytree_utils.assert_pytree_equal(v_expected, v_actual)

  @parameterized.parameters(
      dict(
          sharding1=jax.sharding.PartitionSpec('data'),
          sharding2=jax.sharding.PartitionSpec('data'),
          arr1=np.arange(16),
          arr2=np.arange(16) + 1,
          expected_regex='Error at',
      ),
  )
  def test_assert_pytree_equal_raises_error_on_jax_array_mismatch(
      self, sharding1, sharding2, arr1, arr2, expected_regex
  ):
    mesh = jax.sharding.Mesh(np.array(jax.devices()), ('data',))
    s1 = jax.sharding.NamedSharding(mesh, sharding1)
    s2 = jax.sharding.NamedSharding(mesh, sharding2)
    a1 = jax.device_put(arr1, s1)
    a2 = jax.device_put(arr2, s2)

    with self.assertRaisesRegex(AssertionError, expected_regex):
      pytree_utils.assert_pytree_equal(a1, a2)

  def test_assert_pytree_equal_passes_on_equal_jax_array(self):
    mesh = jax.sharding.Mesh(np.array(jax.devices()), ('data',))
    sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec('data')
    )
    a1 = jax.device_put(np.arange(16), sharding)
    a2 = jax.device_put(np.arange(16), sharding)

    pytree_utils.assert_pytree_equal(a1, a2)

  @parameterized.parameters(
      dict(v1=np.array([1]), v2=np.array([1])),
      dict(v1=1, v2=1),
  )
  def test_assert_pytree_equal_passes_on_equal_values(self, v1, v2):
    pytree_utils.assert_pytree_equal(v1, v2)


if __name__ == '__main__':
  absltest.main()
