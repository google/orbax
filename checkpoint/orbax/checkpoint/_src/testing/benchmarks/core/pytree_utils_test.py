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


class DigestPytreeTest(absltest.TestCase):

  def test_matching_digests_pass(self):
    tree = {'w': np.arange(16, dtype=np.float32)}
    pytree_utils.assert_digests_match(pytree_utils.digest_pytree(tree), tree)

  def test_digest_mismatch_raises(self):
    ref = {'w': np.arange(16, dtype=np.float32)}
    other = {'w': np.arange(16, dtype=np.float32)[::-1].copy()}
    self.assertRaisesRegex(
        AssertionError,
        'Digest mismatch',
        pytree_utils.assert_digests_match,
        pytree_utils.digest_pytree(ref),
        other,
    )

  def test_extra_leaf_raises(self):
    digests = pytree_utils.digest_pytree({'a': np.zeros(4)})
    self.assertRaisesRegex(
        AssertionError,
        'Digest key mismatch',
        pytree_utils.assert_digests_match,
        digests,
        {'a': np.zeros(4), 'b': np.zeros(4)},
    )

  def test_digest_is_stable_across_calls(self):
    tree = {'w': np.arange(16, dtype=np.int32)}
    self.assertEqual(
        pytree_utils.digest_pytree(tree), pytree_utils.digest_pytree(tree)
    )

  def test_digest_differs_for_dtype_change_only(self):
    a = {'w': np.zeros(4, dtype=np.float32)}
    b = {'w': np.zeros(4, dtype=np.float64)}
    self.assertNotEqual(
        pytree_utils.digest_pytree(a), pytree_utils.digest_pytree(b)
    )


def _linear_apply(params, x):
  return params['W'] @ x


class FunctionalEquivalenceTest(absltest.TestCase):

  def test_identical_params_pass(self):
    w = {'W': np.eye(4, dtype=np.float32)}
    pytree_utils.assert_functional_equivalence(
        _linear_apply, [np.arange(4, dtype=np.float32)], w, w
    )

  def test_divergence_raises(self):
    ref = {'W': np.eye(4, dtype=np.float32)}
    actual = {'W': np.eye(4, dtype=np.float32) * 2}
    self.assertRaisesRegex(
        AssertionError,
        'Functional divergence',
        pytree_utils.assert_functional_equivalence,
        _linear_apply,
        [np.arange(4, dtype=np.float32)],
        ref,
        actual,
    )

  def test_within_tolerance_passes(self):
    ref = {'W': np.eye(4, dtype=np.float32)}
    actual = {'W': np.eye(4, dtype=np.float32) + 1e-6}
    pytree_utils.assert_functional_equivalence(
        _linear_apply,
        [np.arange(4, dtype=np.float32)],
        ref,
        actual,
        tolerance=1e-3,
    )

  def test_shape_change_raises(self):
    ref = {'W': np.eye(4, dtype=np.float32)}
    actual = {'W': np.eye(5, 4, dtype=np.float32)}
    self.assertRaisesRegex(
        AssertionError,
        'shape mismatch',
        pytree_utils.assert_functional_equivalence,
        _linear_apply,
        [np.zeros(4, dtype=np.float32)],
        ref,
        actual,
    )

  def test_empty_inputs_raises(self):
    self.assertRaisesRegex(
        ValueError,
        'non-empty',
        pytree_utils.assert_functional_equivalence,
        _linear_apply,
        [],
        {'W': np.eye(4)},
        {'W': np.eye(4)},
    )


if __name__ == '__main__':
  absltest.main()
