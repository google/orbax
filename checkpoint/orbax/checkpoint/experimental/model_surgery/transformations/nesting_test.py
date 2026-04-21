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

"""Tests for model surgery nesting transformations."""

from absl.testing import absltest
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint.experimental.model_surgery.transformations import nesting


class NestingTest(absltest.TestCase):

  def test_unflatten_jnp(self):
    params = {
        'linear1.kernel.qvalue': jnp.array([1]),
        'linear1.kernel.scale': jnp.array([2]),
        'linear2.kernel.qvalue': jnp.array([3]),
    }
    transform = nesting.unflatten()
    result = transform(params)

    expected = {
        'linear1': {
            'kernel': {
                'qvalue': jnp.array([1]),
                'scale': jnp.array([2]),
            }
        },
        'linear2': {
            'kernel': {
                'qvalue': jnp.array([3]),
            }
        },
    }
    self.assertEqual(result.keys(), expected.keys())
    self.assertEqual(result['linear1'].keys(), expected['linear1'].keys())
    self.assertEqual(
        result['linear1']['kernel'].keys(), expected['linear1']['kernel'].keys()
    )
    np.testing.assert_array_equal(
        result['linear1']['kernel']['qvalue'], jnp.array([1])
    )

  def test_unflatten_np(self):
    params = {
        'a.b': np.array([1]),
        'c': np.array([2]),
    }
    transform = nesting.unflatten()
    result = transform(params)

    self.assertEqual(result.keys(), {'a', 'c'})
    np.testing.assert_array_equal(result['a']['b'], np.array([1]))
    np.testing.assert_array_equal(result['c'], np.array([2]))

  def test_unflatten_inplace(self):
    params = {
        'a.b': jnp.array([1]),
        'c': jnp.array([2]),
    }
    transform = nesting.unflatten(inplace=True)
    result = transform(params)

    self.assertEqual(result.keys(), {'a', 'c'})
    np.testing.assert_array_equal(result['a']['b'], jnp.array([1]))
    self.assertEqual(params, {})  # Cleared

  def test_unflatten_conflict_subtree_leaf(self):
    params = {
        'a.b': jnp.array([1]),
        'a.b.c': jnp.array([2]),
    }
    transform = nesting.unflatten()
    with self.assertRaises((TypeError, AssertionError)):
      transform(params)

  def test_unflatten_conflict_leaf_subtree(self):
    params = {
        'a.b.c': jnp.array([1]),
        'a.b': jnp.array([2]),
    }
    transform = nesting.unflatten()
    with self.assertRaises((TypeError, AssertionError)):
      transform(params)


if __name__ == '__main__':
  absltest.main()
