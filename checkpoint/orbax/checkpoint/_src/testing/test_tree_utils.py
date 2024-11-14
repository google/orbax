# Copyright 2024 The Orbax Authors.
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

"""Testing utilities for pytrees."""

import dataclasses
from typing import Any, Callable, Dict, List, NamedTuple, Tuple
import chex
import jax
import jax.numpy as jnp
import numpy as np

PyTree = Any


class MuNu(NamedTuple):
  mu: jax.Array | None
  nu: np.ndarray | None


class EmptyNamedTuple(NamedTuple):
  pass


class NamedTupleWithNestedAttributes(NamedTuple):
  nested_mu_nu: MuNu | None = None
  nested_dict: Dict[str, jax.Array] | None = None
  nested_tuple: Tuple[jax.Array, np.ndarray] | None = None
  nested_empty_named_tuple: EmptyNamedTuple | None = None


@chex.dataclass
class MyEmptyChex:
  pass


@chex.dataclass
class MyChex:
  my_jax_array: jax.Array | None = None
  my_np_array: np.ndarray | None = None


@chex.dataclass
class MyChexWithNestedAttributes:
  my_chex: MyChex | None = None
  my_dict: Dict[str, jax.Array] | None = None
  my_list: List[np.ndarray] | None = None
  my_empty_chex: MyEmptyChex | None = None


class MyEmptyClass:
  pass


class MyClass:
  """Test class.

  Attributes:
    a: optional jax.Array. default=None.
    b: optional np.ndarray. default=None.
  """

  def __init__(
      self,
      a: jax.Array | None = None,
      b: np.ndarray | None = None,
  ):
    self._a = a
    self._b = b

  @property
  def a(self) -> jax.Array | None:
    return self._a

  @property
  def b(self) -> np.ndarray | None:
    return self._b


class MyClassWithNestedAttributes:
  """Test class.

  Attributes:
    my_empty_class: optional `MyEmptyClass`. default=None.
    my_class: optional `MyClass`. default=None.
    my_chex: optional `MyChex`. default=None.
    mu_nu: optional `MuNu`. default=None.
  """

  def __init__(
      self,
      my_empty_class: MyEmptyClass | None = None,
      my_class: MyClass | None = None,
      my_chex: MyChex | None = None,
      mu_nu: MuNu | None = None,
  ):
    self._my_empty_class = my_empty_class
    self._my_class = my_class
    self._my_chex = my_chex
    self._mu_nu = mu_nu


@dataclasses.dataclass
class TestPyTree:
  """Test data class for pytree.

  Attributes:
    unique_name: unique name for the test.
    provide_tree: function to provide the pytree.
    expected_tree_metadata_key_json: expected tree metadata key json dict.
    expected_save_response: expected save response. Can be a BaseException or
      None. None implies that save should succeed.
    expected_restore_response: expected restore response. Can be a
      BaseException, a function to provide the pytree or None. None implies that
      expected restored tree should be the same as the tree returned by
      `provide_tree` function.
  """

  unique_name: str
  # Provide tree lazily via `tree_provider` to avoid error:
  # RuntimeError: Attempted call to JAX before absl.app.run() is called.
  provide_tree: Callable[[], PyTree]
  expected_tree_metadata_key_json: Dict[str, Any]
  expected_save_response: BaseException | None = None
  expected_restore_response: BaseException | Callable[[], PyTree] | None = None

  def __post_init__(self):
    self.expected_restore_response = (
        self.expected_restore_response or self.provide_tree
    )

  def __str__(self):
    return self.unique_name

  def __repr__(self):
    return self.unique_name


TEST_PYTREES = [
    TestPyTree(
        unique_name='empty_pytree',
        provide_tree=lambda: {},
        expected_tree_metadata_key_json={
            '()': {
                'key_metadata': (),
                'value_metadata': {
                    'value_type': 'Dict',
                    'skip_deserialize': True,
                },
            }
        },
        expected_save_response=ValueError(),
        expected_restore_response=ValueError(),
    ),
    TestPyTree(
        unique_name='empty_dict',
        provide_tree=lambda: {'empty_dict': {}},
        expected_tree_metadata_key_json={
            "('empty_dict',)": {
                'key_metadata': ({'key': 'empty_dict', 'key_type': 2},),
                'value_metadata': {
                    'value_type': 'Dict',
                    'skip_deserialize': True,
                },
            },
        },
    ),
    TestPyTree(
        unique_name='empty_list',
        provide_tree=lambda: {'empty_list': []},
        expected_tree_metadata_key_json={
            "('empty_list',)": {
                'key_metadata': ({'key': 'empty_list', 'key_type': 2},),
                'value_metadata': {
                    'value_type': 'List',
                    'skip_deserialize': True,
                },
            },
        },
    ),
    TestPyTree(
        unique_name='empty_tuple',
        provide_tree=lambda: {'empty_tuple': tuple()},
        expected_tree_metadata_key_json={
            "('empty_tuple',)": {
                'key_metadata': ({'key': 'empty_tuple', 'key_type': 2},),
                'value_metadata': {
                    'value_type': 'Tuple',
                    'skip_deserialize': True,
                },
            },
        },
    ),
    TestPyTree(
        unique_name='empty_named_tuple',
        provide_tree=lambda: {'empty_named_tuple': EmptyNamedTuple()},
        expected_tree_metadata_key_json={
            "('empty_named_tuple',)": {
                'key_metadata': ({'key': 'empty_named_tuple', 'key_type': 2},),
                'value_metadata': {
                    # TODO: b/365169723 - Handle empty NamedTuple.
                    'value_type': 'None',
                    'skip_deserialize': True,
                },
            },
        },
    ),
    TestPyTree(
        unique_name='tuple_of_empty_list',
        provide_tree=lambda: {'tuple_of_empty_list': ([],)},
        expected_tree_metadata_key_json={
            "('tuple_of_empty_list', '0')": {
                'key_metadata': (
                    {'key': 'tuple_of_empty_list', 'key_type': 2},
                    {'key': '0', 'key_type': 1},
                ),
                'value_metadata': {
                    'value_type': 'List',
                    'skip_deserialize': True,
                },
            },
        },
    ),
    TestPyTree(
        unique_name='list_of_empty_tuple',
        provide_tree=lambda: {'list_of_empty_tuple': [tuple()]},
        expected_tree_metadata_key_json={
            "('list_of_empty_tuple', '0')": {
                'key_metadata': (
                    {'key': 'list_of_empty_tuple', 'key_type': 2},
                    {'key': '0', 'key_type': 1},
                ),
                'value_metadata': {
                    'value_type': 'Tuple',
                    'skip_deserialize': True,
                },
            },
        },
    ),
    TestPyTree(
        unique_name='none_param',
        provide_tree=lambda: {'none_param': None},
        expected_tree_metadata_key_json={
            "('none_param',)": {
                'key_metadata': ({'key': 'none_param', 'key_type': 2},),
                'value_metadata': {
                    'value_type': 'None',
                    'skip_deserialize': True,
                },
            },
        },
    ),
    TestPyTree(
        unique_name='scalar_param',
        provide_tree=lambda: {'scalar_param': 1},
        expected_tree_metadata_key_json={
            "('scalar_param',)": {
                'key_metadata': ({'key': 'scalar_param', 'key_type': 2},),
                'value_metadata': {
                    'value_type': 'scalar',
                    'skip_deserialize': False,
                },
            },
        },
    ),
    TestPyTree(
        unique_name='nested_scalars',
        provide_tree=lambda: {
            'b': {'scalar_param': 1, 'nested_scalar_param': 3.4}
        },
        expected_tree_metadata_key_json={
            "('b', 'scalar_param')": {
                'key_metadata': (
                    {'key': 'b', 'key_type': 2},
                    {'key': 'scalar_param', 'key_type': 2},
                ),
                'value_metadata': {
                    'value_type': 'scalar',
                    'skip_deserialize': False,
                },
            },
            "('b', 'nested_scalar_param')": {
                'key_metadata': (
                    {'key': 'b', 'key_type': 2},
                    {'key': 'nested_scalar_param', 'key_type': 2},
                ),
                'value_metadata': {
                    'value_type': 'scalar',
                    'skip_deserialize': False,
                },
            },
        },
    ),
    TestPyTree(
        unique_name='list_of_arrays',
        provide_tree=lambda: {'list_of_arrays': [np.arange(8), jnp.arange(8)]},
        expected_tree_metadata_key_json={
            "('list_of_arrays', '0')": {
                'key_metadata': (
                    {'key': 'list_of_arrays', 'key_type': 2},
                    {'key': '0', 'key_type': 1},
                ),
                'value_metadata': {
                    'value_type': 'np.ndarray',
                    'skip_deserialize': False,
                },
            },
            "('list_of_arrays', '1')": {
                'key_metadata': (
                    {'key': 'list_of_arrays', 'key_type': 2},
                    {'key': '1', 'key_type': 1},
                ),
                'value_metadata': {
                    'value_type': 'jax.Array',
                    'skip_deserialize': False,
                },
            },
        },
    ),
    TestPyTree(
        unique_name='tuple_of_arrays',
        provide_tree=lambda: {'tuple_of_arrays': (np.arange(8), jnp.arange(8))},
        expected_tree_metadata_key_json={
            "('tuple_of_arrays', '0')": {
                'key_metadata': (
                    {'key': 'tuple_of_arrays', 'key_type': 2},
                    {'key': '0', 'key_type': 1},
                ),
                'value_metadata': {
                    'value_type': 'np.ndarray',
                    'skip_deserialize': False,
                },
            },
            "('tuple_of_arrays', '1')": {
                'key_metadata': (
                    {'key': 'tuple_of_arrays', 'key_type': 2},
                    {'key': '1', 'key_type': 1},
                ),
                'value_metadata': {
                    'value_type': 'jax.Array',
                    'skip_deserialize': False,
                },
            },
        },
    ),
    TestPyTree(
        unique_name='mu_nu',
        provide_tree=lambda: {'mu_nu': MuNu(mu=jnp.arange(8), nu=np.arange(8))},
        expected_tree_metadata_key_json={
            "('mu_nu', 'mu')": {
                'key_metadata': (
                    {'key': 'mu_nu', 'key_type': 2},
                    {'key': 'mu', 'key_type': 2},
                ),
                'value_metadata': {
                    'value_type': 'jax.Array',
                    'skip_deserialize': False,
                },
            },
            "('mu_nu', 'nu')": {
                'key_metadata': (
                    {'key': 'mu_nu', 'key_type': 2},
                    {'key': 'nu', 'key_type': 2},
                ),
                'value_metadata': {
                    'value_type': 'np.ndarray',
                    'skip_deserialize': False,
                },
            },
        },
    ),
    TestPyTree(
        unique_name='my_empty_chex',
        provide_tree=lambda: {'my_empty_chex': MyEmptyChex()},
        expected_tree_metadata_key_json={
            "('my_empty_chex',)": {
                'key_metadata': ({'key': 'my_empty_chex', 'key_type': 2},),
                'value_metadata': {
                    'value_type': 'Dict',
                    'skip_deserialize': True,
                },
            },
        },
    ),
    TestPyTree(
        unique_name='default_my_chex',
        provide_tree=lambda: {'default_my_chex': MyChex()},
        expected_tree_metadata_key_json={
            "('default_my_chex', 'my_jax_array')": {
                'key_metadata': (
                    {'key': 'default_my_chex', 'key_type': 2},
                    {'key': 'my_jax_array', 'key_type': 2},
                ),
                'value_metadata': {
                    'value_type': 'None',  # value set is None.
                    'skip_deserialize': True,
                },
            },
            "('default_my_chex', 'my_np_array')": {
                'key_metadata': (
                    {'key': 'default_my_chex', 'key_type': 2},
                    {'key': 'my_np_array', 'key_type': 2},
                ),
                'value_metadata': {
                    'value_type': 'None',  # value set is None.
                    'skip_deserialize': True,
                },
            },
        },
    ),
    TestPyTree(
        unique_name='my_chex',
        provide_tree=lambda: {
            'my_chex': MyChex(
                my_jax_array=jnp.arange(8),
                my_np_array=np.arange(8),
            )
        },
        expected_tree_metadata_key_json={
            "('my_chex', 'my_jax_array')": {
                'key_metadata': (
                    {'key': 'my_chex', 'key_type': 2},
                    {'key': 'my_jax_array', 'key_type': 2},
                ),
                'value_metadata': {
                    'value_type': 'jax.Array',
                    'skip_deserialize': False,
                },
            },
            "('my_chex', 'my_np_array')": {
                'key_metadata': (
                    {'key': 'my_chex', 'key_type': 2},
                    {'key': 'my_np_array', 'key_type': 2},
                ),
                'value_metadata': {
                    'value_type': 'np.ndarray',
                    'skip_deserialize': False,
                },
            },
        },
    ),
    TestPyTree(
        unique_name='default_my_chex_with_nested_attrs',
        provide_tree=lambda: {
            'my_chex_with_nested_attrs': MyChexWithNestedAttributes()
        },
        expected_tree_metadata_key_json={
            "('my_chex_with_nested_attrs', 'my_chex')": {
                'key_metadata': (
                    {'key': 'my_chex_with_nested_attrs', 'key_type': 2},
                    {'key': 'my_chex', 'key_type': 2},
                ),
                'value_metadata': {
                    'value_type': 'None',
                    'skip_deserialize': True,
                },
            },
            "('my_chex_with_nested_attrs', 'my_dict')": {
                'key_metadata': (
                    {'key': 'my_chex_with_nested_attrs', 'key_type': 2},
                    {'key': 'my_dict', 'key_type': 2},
                ),
                'value_metadata': {
                    'value_type': 'None',
                    'skip_deserialize': True,
                },
            },
            "('my_chex_with_nested_attrs', 'my_empty_chex')": {
                'key_metadata': (
                    {'key': 'my_chex_with_nested_attrs', 'key_type': 2},
                    {'key': 'my_empty_chex', 'key_type': 2},
                ),
                'value_metadata': {
                    'value_type': 'None',
                    'skip_deserialize': True,
                },
            },
            "('my_chex_with_nested_attrs', 'my_list')": {
                'key_metadata': (
                    {'key': 'my_chex_with_nested_attrs', 'key_type': 2},
                    {'key': 'my_list', 'key_type': 2},
                ),
                'value_metadata': {
                    'value_type': 'None',
                    'skip_deserialize': True,
                },
            },
        },
    ),
    TestPyTree(
        unique_name='my_chex_with_nested_attrs',
        provide_tree=lambda: {
            'my_chex_with_nested_attrs': MyChexWithNestedAttributes(
                my_chex=MyChex(
                    my_jax_array=jnp.arange(8), my_np_array=np.arange(8)
                ),
                my_dict={'a': jnp.arange(8), 'b': np.arange(8)},
                my_list=[jnp.arange(8), np.arange(8)],
                my_empty_chex=MyEmptyChex(),
            )
        },
        expected_tree_metadata_key_json={
            "('my_chex_with_nested_attrs', 'my_chex', 'my_jax_array')": {
                'key_metadata': (
                    {'key': 'my_chex_with_nested_attrs', 'key_type': 2},
                    {'key': 'my_chex', 'key_type': 2},
                    {'key': 'my_jax_array', 'key_type': 2},
                ),
                'value_metadata': {
                    'value_type': 'jax.Array',
                    'skip_deserialize': False,
                },
            },
            "('my_chex_with_nested_attrs', 'my_chex', 'my_np_array')": {
                'key_metadata': (
                    {'key': 'my_chex_with_nested_attrs', 'key_type': 2},
                    {'key': 'my_chex', 'key_type': 2},
                    {'key': 'my_np_array', 'key_type': 2},
                ),
                'value_metadata': {
                    'value_type': 'np.ndarray',
                    'skip_deserialize': False,
                },
            },
            "('my_chex_with_nested_attrs', 'my_dict', 'a')": {
                'key_metadata': (
                    {'key': 'my_chex_with_nested_attrs', 'key_type': 2},
                    {'key': 'my_dict', 'key_type': 2},
                    {'key': 'a', 'key_type': 2},
                ),
                'value_metadata': {
                    'value_type': 'jax.Array',
                    'skip_deserialize': False,
                },
            },
            "('my_chex_with_nested_attrs', 'my_dict', 'b')": {
                'key_metadata': (
                    {'key': 'my_chex_with_nested_attrs', 'key_type': 2},
                    {'key': 'my_dict', 'key_type': 2},
                    {'key': 'b', 'key_type': 2},
                ),
                'value_metadata': {
                    'value_type': 'np.ndarray',
                    'skip_deserialize': False,
                },
            },
            "('my_chex_with_nested_attrs', 'my_empty_chex')": {
                'key_metadata': (
                    {'key': 'my_chex_with_nested_attrs', 'key_type': 2},
                    {'key': 'my_empty_chex', 'key_type': 2},
                ),
                'value_metadata': {
                    'value_type': 'Dict',
                    'skip_deserialize': True,
                },
            },
            "('my_chex_with_nested_attrs', 'my_list', '0')": {
                'key_metadata': (
                    {'key': 'my_chex_with_nested_attrs', 'key_type': 2},
                    {'key': 'my_list', 'key_type': 2},
                    {'key': '0', 'key_type': 1},
                ),
                'value_metadata': {
                    'value_type': 'jax.Array',
                    'skip_deserialize': False,
                },
            },
            "('my_chex_with_nested_attrs', 'my_list', '1')": {
                'key_metadata': (
                    {'key': 'my_chex_with_nested_attrs', 'key_type': 2},
                    {'key': 'my_list', 'key_type': 2},
                    {'key': '1', 'key_type': 1},
                ),
                'value_metadata': {
                    'value_type': 'np.ndarray',
                    'skip_deserialize': False,
                },
            },
        },
    ),
    TestPyTree(
        unique_name='my_empty_class',
        provide_tree=lambda: {'my_empty_class': MyEmptyClass()},
        expected_tree_metadata_key_json={
            "('my_empty_class',)": {
                'key_metadata': ({'key': 'my_empty_class', 'key_type': 2},),
                'value_metadata': {
                    'value_type': 'None',
                    'skip_deserialize': True,
                },
            },
        },
        expected_save_response=ValueError(),
        expected_restore_response=ValueError(),
    ),
    TestPyTree(
        unique_name='default_my_class',
        provide_tree=lambda: {'my_class': MyClass()},
        expected_tree_metadata_key_json={
            "('my_class',)": {
                'key_metadata': ({'key': 'my_class', 'key_type': 2},),
                'value_metadata': {
                    'value_type': 'None',
                    'skip_deserialize': True,
                },
            },
        },
        expected_save_response=ValueError(),
        expected_restore_response=ValueError(),
    ),
    TestPyTree(
        unique_name='my_class',
        provide_tree=lambda: {
            'my_class': MyClass(a=jnp.arange(8), b=np.arange(8))
        },
        expected_tree_metadata_key_json={
            "('my_class',)": {
                'key_metadata': ({'key': 'my_class', 'key_type': 2},),
                'value_metadata': {
                    'value_type': 'None',
                    'skip_deserialize': True,
                },
            },
        },
        expected_save_response=ValueError(),
        expected_restore_response=ValueError(),
    ),
    TestPyTree(
        unique_name='default_my_class_with_nested_attrs',
        provide_tree=lambda: {
            'my_class_with_nested_attrs': MyClassWithNestedAttributes()
        },
        expected_tree_metadata_key_json={
            "('my_class_with_nested_attrs',)": {
                'key_metadata': (
                    {'key': 'my_class_with_nested_attrs', 'key_type': 2},
                ),
                'value_metadata': {
                    'value_type': 'None',
                    'skip_deserialize': True,
                },
            }
        },
        expected_save_response=ValueError(),
        expected_restore_response=ValueError(),
    ),
    TestPyTree(
        unique_name='my_class_with_nested_attrs',
        provide_tree=lambda: {
            'my_class_with_nested_attrs': MyClassWithNestedAttributes(
                my_empty_class=MyEmptyClass(),
                my_class=MyClass(a=jnp.arange(8), b=np.arange(8)),
                my_chex=MyChex(
                    my_jax_array=jnp.arange(8), my_np_array=np.arange(8)
                ),
                mu_nu=MuNu(mu=jnp.arange(8), nu=np.arange(8)),
            )
        },
        expected_tree_metadata_key_json={
            "('my_class_with_nested_attrs',)": {
                'key_metadata': (
                    {'key': 'my_class_with_nested_attrs', 'key_type': 2},
                ),
                'value_metadata': {
                    'value_type': 'None',
                    'skip_deserialize': True,
                },
            }
        },
        expected_save_response=ValueError(),
        expected_restore_response=ValueError(),
    ),
    TestPyTree(
        unique_name='default_named_tuple_with_nested_attrs',
        provide_tree=lambda: {
            'named_tuple_with_nested_attrs': NamedTupleWithNestedAttributes()
        },
        expected_tree_metadata_key_json={
            "('named_tuple_with_nested_attrs', 'nested_mu_nu')": {
                'key_metadata': (
                    {'key': 'named_tuple_with_nested_attrs', 'key_type': 2},
                    {'key': 'nested_mu_nu', 'key_type': 2},
                ),
                'value_metadata': {
                    'value_type': 'None',
                    'skip_deserialize': True,
                },
            },
            "('named_tuple_with_nested_attrs', 'nested_dict')": {
                'key_metadata': (
                    {'key': 'named_tuple_with_nested_attrs', 'key_type': 2},
                    {'key': 'nested_dict', 'key_type': 2},
                ),
                'value_metadata': {
                    'value_type': 'None',
                    'skip_deserialize': True,
                },
            },
            "('named_tuple_with_nested_attrs', 'nested_tuple')": {
                'key_metadata': (
                    {'key': 'named_tuple_with_nested_attrs', 'key_type': 2},
                    {'key': 'nested_tuple', 'key_type': 2},
                ),
                'value_metadata': {
                    'value_type': 'None',
                    'skip_deserialize': True,
                },
            },
            "('named_tuple_with_nested_attrs', 'nested_empty_named_tuple')": {
                'key_metadata': (
                    {'key': 'named_tuple_with_nested_attrs', 'key_type': 2},
                    {'key': 'nested_empty_named_tuple', 'key_type': 2},
                ),
                'value_metadata': {
                    'value_type': 'None',
                    'skip_deserialize': True,
                },
            },
        },
    ),
    TestPyTree(
        unique_name='named_tuple_with_nested_attrs',
        provide_tree=lambda: {
            'named_tuple_with_nested_attrs': NamedTupleWithNestedAttributes(
                nested_mu_nu=MuNu(mu=jnp.arange(8), nu=np.arange(8)),
                nested_dict={'a': jnp.arange(8), 'b': np.arange(8)},
                nested_tuple=(jnp.arange(8), np.arange(8)),
                nested_empty_named_tuple=EmptyNamedTuple(),
            )
        },
        expected_tree_metadata_key_json={
            "('named_tuple_with_nested_attrs', 'nested_mu_nu', 'mu')": {
                'key_metadata': (
                    {'key': 'named_tuple_with_nested_attrs', 'key_type': 2},
                    {'key': 'nested_mu_nu', 'key_type': 2},
                    {'key': 'mu', 'key_type': 2},
                ),
                'value_metadata': {
                    'value_type': 'jax.Array',
                    'skip_deserialize': False,
                },
            },
            "('named_tuple_with_nested_attrs', 'nested_mu_nu', 'nu')": {
                'key_metadata': (
                    {'key': 'named_tuple_with_nested_attrs', 'key_type': 2},
                    {'key': 'nested_mu_nu', 'key_type': 2},
                    {'key': 'nu', 'key_type': 2},
                ),
                'value_metadata': {
                    'value_type': 'np.ndarray',
                    'skip_deserialize': False,
                },
            },
            "('named_tuple_with_nested_attrs', 'nested_dict', 'a')": {
                'key_metadata': (
                    {'key': 'named_tuple_with_nested_attrs', 'key_type': 2},
                    {'key': 'nested_dict', 'key_type': 2},
                    {'key': 'a', 'key_type': 2},
                ),
                'value_metadata': {
                    'value_type': 'jax.Array',
                    'skip_deserialize': False,
                },
            },
            "('named_tuple_with_nested_attrs', 'nested_dict', 'b')": {
                'key_metadata': (
                    {'key': 'named_tuple_with_nested_attrs', 'key_type': 2},
                    {'key': 'nested_dict', 'key_type': 2},
                    {'key': 'b', 'key_type': 2},
                ),
                'value_metadata': {
                    'value_type': 'np.ndarray',
                    'skip_deserialize': False,
                },
            },
            "('named_tuple_with_nested_attrs', 'nested_tuple', '0')": {
                'key_metadata': (
                    {'key': 'named_tuple_with_nested_attrs', 'key_type': 2},
                    {'key': 'nested_tuple', 'key_type': 2},
                    {'key': '0', 'key_type': 1},
                ),
                'value_metadata': {
                    'value_type': 'jax.Array',
                    'skip_deserialize': False,
                },
            },
            "('named_tuple_with_nested_attrs', 'nested_tuple', '1')": {
                'key_metadata': (
                    {'key': 'named_tuple_with_nested_attrs', 'key_type': 2},
                    {'key': 'nested_tuple', 'key_type': 2},
                    {'key': '1', 'key_type': 1},
                ),
                'value_metadata': {
                    'value_type': 'np.ndarray',
                    'skip_deserialize': False,
                },
            },
            "('named_tuple_with_nested_attrs', 'nested_empty_named_tuple')": {
                'key_metadata': (
                    {'key': 'named_tuple_with_nested_attrs', 'key_type': 2},
                    {'key': 'nested_empty_named_tuple', 'key_type': 2},
                ),
                'value_metadata': {
                    # TODO: b/365169723 - Handle empty NamedTuple.
                    'value_type': 'None',
                    'skip_deserialize': True,
                },
            },
        },
    ),
]

# Suitable for parameterized.named_parameters.
TEST_PYTREES_FOR_NAMED_PARAMETERS = [
    (test_pytree.unique_name, test_pytree) for test_pytree in TEST_PYTREES
]
