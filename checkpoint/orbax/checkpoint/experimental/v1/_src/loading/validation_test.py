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

import itertools
import typing
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
import optax
from orbax.checkpoint._src.testing import test_tree_utils
from orbax.checkpoint.experimental.v1._src.loading import validation


class LocalNamedTuple(typing.NamedTuple):
  x: int
  y: float


_VALID_ABSTRACT_LEAVES = (
    1,
    int,
    np.ndarray,
    'string',
    str,
    None,
    optax.EmptyState(),
    ...,
)

_BASIC_CONTAINERS = (dict, list, tuple)


def _generate_basic_combinations(leaves):
  states = []
  for c_type, val in itertools.product(_BASIC_CONTAINERS, leaves):
    if c_type is dict:
      states.append({'a': val, 'b': val})
    else:
      states.append(c_type([val, val]))
  return states


_VALID_ABSTRACT_PYTREES = (
    [1, 2.0, True],
    (b'bytes', 'string', None),
)

_INVALID_ABSTRACT_PYTREES = (
    test_tree_utils.MyClass(a=None, b=None),
    test_tree_utils.MyDataClass(my_jax_array=None, my_np_array=None),
    {'nested_invalid': test_tree_utils.MyClass()},
)


class ValidationTest(parameterized.TestCase):

  def test_validate_state_checkpointable_name(self):
    validation.validate_state_checkpointable_name(None)
    validation.validate_state_checkpointable_name('pytree')
    validation.validate_state_checkpointable_name('a')

    with self.assertRaisesRegex(ValueError, 'Empty string is not supported'):
      validation.validate_state_checkpointable_name('')

    with mock.patch.object(
        validation, 'RESERVED_CHECKPOINTABLE_KEYS', {'reserved'}
    ):
      with self.assertRaisesRegex(ValueError, 'reserved'):
        validation.validate_state_checkpointable_name('reserved')

  def test_validate_abstract_checkpointables_valid(self):
    validation.validate_abstract_checkpointables(None)
    validation.validate_abstract_checkpointables({})
    validation.validate_abstract_checkpointables({'a': {'b': 1}})

  def test_validate_abstract_checkpointables_invalid_structure(self):
    with self.assertRaises(ValueError):
      validation.validate_abstract_checkpointables([{'a': 1}])  # pytype: disable=wrong-arg-types

    with self.assertRaisesRegex(ValueError, 'Empty string is not supported'):
      validation.validate_abstract_checkpointables({'': {'a': 1}})

    with mock.patch.object(
        validation, 'RESERVED_CHECKPOINTABLE_KEYS', {'reserved'}
    ):
      with self.assertRaisesRegex(ValueError, 'reserved'):
        validation.validate_abstract_checkpointables({'reserved': {'a': 1}})

  @parameterized.parameters(
      list(zip(_generate_basic_combinations(_VALID_ABSTRACT_LEAVES)))
  )
  def test_validate_abstract_state_basic_combinations(self, state):
    validation.validate_abstract_state(state)

  def test_validate_abstract_state_empty_nodes_and_none(self):
    validation.validate_abstract_state(optax.EmptyState())  # pyrefly: ignore[bad-argument-type]
    validation.validate_abstract_state(None)
    validation.validate_abstract_state({})  # pyrefly: ignore[bad-argument-type]
    validation.validate_abstract_state([])  # pyrefly: ignore[bad-argument-type]
    validation.validate_abstract_state(())  # pyrefly: ignore[bad-argument-type]

  @parameterized.parameters(list(zip(_VALID_ABSTRACT_PYTREES)))
  def test_validate_abstract_state_valid_specific_containers(self, state):
    validation.validate_abstract_state(state)

  @parameterized.parameters(list(zip(_INVALID_ABSTRACT_PYTREES)))
  def test_validate_abstract_state_invalid_specific_containers(self, state):
    with self.assertRaises(TypeError):
      validation.validate_abstract_state(state)

  def test_validate_abstract_state_invalid_leaves(self):
    with self.assertRaises(TypeError):
      validation.validate_abstract_state(object())  # pyrefly: ignore[bad-argument-type]

    with self.assertRaises(TypeError):
      validation.validate_abstract_state({'a': object()})  # pyrefly: ignore[bad-argument-type]

    with self.assertRaises(TypeError):
      validation.validate_abstract_state({'a': {'b': [object()]}})  # pyrefly: ignore[bad-argument-type]

  def test_validate_abstract_state_shape_dtype_struct(self):
    validation.validate_abstract_state({  # pyrefly: ignore[bad-argument-type]
        'a': jax.ShapeDtypeStruct((2, 2), np.float32),
        'b': jax.ShapeDtypeStruct((2, 2), np.float32),
    })

  def test_validate_abstract_state_jax_array(self):
    validation.validate_abstract_state({'a': jax.Array, 'b': jax.Array})  # pyrefly: ignore[bad-argument-type]


if __name__ == '__main__':
  absltest.main()
