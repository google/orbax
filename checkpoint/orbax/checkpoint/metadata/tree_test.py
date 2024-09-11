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

from collections.abc import Callable
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
from jax import numpy as jnp
import numpy as np
from orbax.checkpoint import tree as tree_utils
from orbax.checkpoint import type_handlers
from orbax.checkpoint.metadata import tree as tree_metadata


@chex.dataclass
class TransformByLrAndWdScheduleState:
  pass


@chex.dataclass
class CustomDataClass:
  jax_array: Optional[jax.Array] = None
  np_array: Optional[np.ndarray] = None


@chex.dataclass
class CustomDataClassWithNestedAttributes:
  nested_data_class: Optional[CustomDataClass] = None
  nested_dict: Optional[Dict[str, jax.Array]] = None
  nested_list: Optional[List[np.ndarray]] = None
  nested_empty_data_class: Optional[TransformByLrAndWdScheduleState] = None


class MuNu(NamedTuple):
  mu: Optional[jax.Array]
  nu: Optional[np.ndarray]


class EmptyNamedTuple(NamedTuple):
  pass


class NamedTupleWithNestedAttributes(NamedTuple):
  nested_mu_nu: Optional[MuNu] = None
  nested_dict: Optional[Dict[str, jax.Array]] = None
  nested_tuple: Optional[Tuple[str, np.ndarray]] = None
  nested_empty_named_tuple: Optional[EmptyNamedTuple] = None


class NoAttributeCustom:
  pass


class Custom:

  def __init__(
      self,
      a: Optional[jax.Array] = None,
      b: Optional[np.ndarray] = None,
  ):
    self._a = a
    self._b = b

  @property
  def a(self) -> Optional[jax.Array]:
    return self._a

  @property
  def b(self) -> Optional[np.ndarray]:
    return self._b


class CustomWithNestedAttributes:

  def __init__(
      self,
      no_attribute_custom: Optional[NoAttributeCustom] = None,
      custom: Optional[Custom] = None,
      custom_data_class: Optional[CustomDataClass] = None,
      mu_nu: Optional[MuNu] = None,
  ):
    self._no_attribute_custom = no_attribute_custom
    self._custom = custom
    self._custom_data_class = custom_data_class
    self._mu_nu = mu_nu


class TreeMetadataTest(parameterized.TestCase):

  def _to_param_infos(self, tree: Any):
    return jax.tree.map(
        # Other properties are not relevant.
        lambda x: type_handlers.ParamInfo(
            value_typestr=type_handlers.get_param_typestr(
                x, type_handlers.GLOBAL_TYPE_HANDLER_REGISTRY
            )
        ),
        tree,
        is_leaf=tree_utils.is_empty_or_leaf,
    )

  # Pass tree lazily via `tree_provider` to avoid error:
  # RuntimeError: Attempted call to JAX before absl.app.run() is called.
  @parameterized.parameters(
      {
          'tree_provider': lambda: {'scalar_param': 1},
          'expected_tree_json': {
              "('scalar_param',)": {
                  'key_metadata': ({'key': 'scalar_param', 'key_type': 2},),
                  'value_metadata': {
                      'value_type': 'scalar',
                      'skip_deserialize': False,
                  },
              },
          },
      },
      {
          'tree_provider': lambda: {
              'b': {'string_param': 'hi', 'nested_scalar_param': 3.4}
          },
          'expected_tree_json': {
              "('b', 'string_param')": {
                  'key_metadata': (
                      {'key': 'b', 'key_type': 2},
                      {'key': 'string_param', 'key_type': 2},
                  ),
                  'value_metadata': {
                      'value_type': 'string',
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
      },
      {
          'tree_provider': lambda: {
              'list_of_arrays': [np.arange(8), jnp.arange(8)]
          },
          'expected_tree_json': {
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
      },
      {
          'tree_provider': lambda: {'none_param': None},
          'expected_tree_json': {
              "('none_param',)": {
                  'key_metadata': ({'key': 'none_param', 'key_type': 2},),
                  'value_metadata': {
                      'value_type': 'None',
                      'skip_deserialize': True,
                  },
              },
          },
      },
      {
          'tree_provider': lambda: {'empty_dict': {}},
          'expected_tree_json': {
              "('empty_dict',)": {
                  'key_metadata': ({'key': 'empty_dict', 'key_type': 2},),
                  'value_metadata': {
                      'value_type': 'Dict',
                      'skip_deserialize': True,
                  },
              },
          },
      },
      {
          'tree_provider': lambda: {'empty_list': []},
          'expected_tree_json': {
              "('empty_list',)": {
                  'key_metadata': ({'key': 'empty_list', 'key_type': 2},),
                  'value_metadata': {
                      'value_type': 'List',
                      'skip_deserialize': True,
                  },
              },
          },
      },
      {
          'tree_provider': lambda: {'tuple_of_empty_list': tuple([])},
          'expected_tree_json': {
              "('tuple_of_empty_list',)": {
                  'key_metadata': (
                      {'key': 'tuple_of_empty_list', 'key_type': 2},
                  ),
                  'value_metadata': {
                      'value_type': 'None',  # TODO: b/365169723 - Fix this.
                      'skip_deserialize': True,
                  },
              },
          },
      },
      {
          'tree_provider': lambda: {
              'tuple_of_arrays': (np.arange(8), jnp.arange(8))
          },
          'expected_tree_json': {
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
      },
      {
          'tree_provider': lambda: {
              'named_tuple_param': MuNu(nu=np.arange(8), mu=jnp.arange(8))
          },
          'expected_tree_json': {
              "('named_tuple_param', 'mu')": {
                  'key_metadata': (
                      {'key': 'named_tuple_param', 'key_type': 2},
                      {'key': 'mu', 'key_type': 2},
                  ),
                  'value_metadata': {
                      'value_type': 'jax.Array',
                      'skip_deserialize': False,
                  },
              },
              "('named_tuple_param', 'nu')": {
                  'key_metadata': (
                      {'key': 'named_tuple_param', 'key_type': 2},
                      {'key': 'nu', 'key_type': 2},
                  ),
                  'value_metadata': {
                      'value_type': 'np.ndarray',
                      'skip_deserialize': False,
                  },
              },
          },
      },
      {
          'tree_provider': lambda: {'empty_tuple': tuple()},
          'expected_tree_json': {
              "('empty_tuple',)": {
                  'key_metadata': ({'key': 'empty_tuple', 'key_type': 2},),
                  'value_metadata': {
                      'value_type': 'None',  # TODO: b/365169723 - Fix this.
                      'skip_deserialize': True,
                  },
              },
          },
      },
      {
          'tree_provider': lambda: {'list_of_empty_tuple': [tuple()]},
          'expected_tree_json': {
              "('list_of_empty_tuple', '0')": {
                  'key_metadata': (
                      {'key': 'list_of_empty_tuple', 'key_type': 2},
                      {'key': '0', 'key_type': 1},
                  ),
                  'value_metadata': {
                      'value_type': 'None',  # TODO: b/365169723 - Fix this.
                      'skip_deserialize': True,
                  },
              },
          },
      },
      {
          'tree_provider': lambda: {'empty_named_tuple': EmptyNamedTuple()},
          'expected_tree_json': {
              "('empty_named_tuple',)": {
                  'key_metadata': (
                      {'key': 'empty_named_tuple', 'key_type': 2},
                  ),
                  'value_metadata': {
                      'value_type': 'None',  # TODO: b/365169723 - Fix this.
                      'skip_deserialize': True,
                  },
              },
          },
      },
      {
          'tree_provider': lambda: {
              'no_attribute_chex_dataclass': TransformByLrAndWdScheduleState()
          },
          'expected_tree_json': {
              "('no_attribute_chex_dataclass',)": {
                  'key_metadata': (
                      {'key': 'no_attribute_chex_dataclass', 'key_type': 2},
                  ),
                  'value_metadata': {
                      'value_type': 'Dict',
                      'skip_deserialize': True,
                  },
              },
          },
      },
      {
          'tree_provider': lambda: {
              'default_chex_dataclass': CustomDataClass()
          },
          'expected_tree_json': {
              "('default_chex_dataclass', 'jax_array')": {
                  'key_metadata': (
                      {'key': 'default_chex_dataclass', 'key_type': 2},
                      {'key': 'jax_array', 'key_type': 2},
                  ),
                  'value_metadata': {
                      'value_type': 'None',  # value set is None.
                      'skip_deserialize': True,
                  },
              },
              "('default_chex_dataclass', 'np_array')": {
                  'key_metadata': (
                      {'key': 'default_chex_dataclass', 'key_type': 2},
                      {'key': 'np_array', 'key_type': 2},
                  ),
                  'value_metadata': {
                      'value_type': 'None',  # value set is None.
                      'skip_deserialize': True,
                  },
              },
          },
      },
      {
          'tree_provider': lambda: {
              'chex_dataclass': CustomDataClass(
                  jax_array=jnp.arange(8), np_array=np.arange(8)
              )
          },
          'expected_tree_json': {
              "('chex_dataclass', 'jax_array')": {
                  'key_metadata': (
                      {'key': 'chex_dataclass', 'key_type': 2},
                      {'key': 'jax_array', 'key_type': 2},
                  ),
                  'value_metadata': {
                      'value_type': 'jax.Array',
                      'skip_deserialize': False,
                  },
              },
              "('chex_dataclass', 'np_array')": {
                  'key_metadata': (
                      {'key': 'chex_dataclass', 'key_type': 2},
                      {'key': 'np_array', 'key_type': 2},
                  ),
                  'value_metadata': {
                      'value_type': 'np.ndarray',
                      'skip_deserialize': False,
                  },
              },
          },
      },
      {
          'tree_provider': lambda: {
              'no_attribute_custom_object': NoAttributeCustom()
          },
          'expected_tree_json': {
              "('no_attribute_custom_object',)": {
                  'key_metadata': (
                      {'key': 'no_attribute_custom_object', 'key_type': 2},
                  ),
                  'value_metadata': {
                      'value_type': 'None',
                      'skip_deserialize': True,
                  },
              },
          },
      },
      {
          'tree_provider': lambda: {'default_custom_object': Custom()},
          'expected_tree_json': {
              "('default_custom_object',)": {
                  'key_metadata': (
                      {'key': 'default_custom_object', 'key_type': 2},
                  ),
                  'value_metadata': {
                      'value_type': 'None',
                      'skip_deserialize': True,
                  },
              },
          },
      },
      {
          'tree_provider': lambda: {
              'custom_object': Custom(a=jnp.arange(8), b=np.arange(8))
          },
          'expected_tree_json': {
              "('custom_object',)": {
                  'key_metadata': ({'key': 'custom_object', 'key_type': 2},),
                  'value_metadata': {
                      'value_type': 'None',
                      'skip_deserialize': True,
                  },
              },
          },
      },
      {
          'tree_provider': lambda: {
              'custom_data_class': CustomDataClassWithNestedAttributes()
          },
          'expected_tree_json': {
              "('custom_data_class', 'nested_data_class')": {
                  'key_metadata': (
                      {'key': 'custom_data_class', 'key_type': 2},
                      {'key': 'nested_data_class', 'key_type': 2},
                  ),
                  'value_metadata': {
                      'value_type': 'None',
                      'skip_deserialize': True,
                  },
              },
              "('custom_data_class', 'nested_dict')": {
                  'key_metadata': (
                      {'key': 'custom_data_class', 'key_type': 2},
                      {'key': 'nested_dict', 'key_type': 2},
                  ),
                  'value_metadata': {
                      'value_type': 'None',
                      'skip_deserialize': True,
                  },
              },
              "('custom_data_class', 'nested_empty_data_class')": {
                  'key_metadata': (
                      {'key': 'custom_data_class', 'key_type': 2},
                      {'key': 'nested_empty_data_class', 'key_type': 2},
                  ),
                  'value_metadata': {
                      'value_type': 'None',
                      'skip_deserialize': True,
                  },
              },
              "('custom_data_class', 'nested_list')": {
                  'key_metadata': (
                      {'key': 'custom_data_class', 'key_type': 2},
                      {'key': 'nested_list', 'key_type': 2},
                  ),
                  'value_metadata': {
                      'value_type': 'None',
                      'skip_deserialize': True,
                  },
              },
          },
      },
      {
          'tree_provider': lambda: {
              'custom_data_class': CustomDataClassWithNestedAttributes(
                  nested_data_class=CustomDataClass(
                      jax_array=jnp.arange(8), np_array=np.arange(8)
                  ),
                  nested_dict={'a': jnp.arange(8), 'b': np.arange(8)},
                  nested_list=[jnp.arange(8), np.arange(8)],
                  nested_empty_data_class=TransformByLrAndWdScheduleState(),
              )
          },
          'expected_tree_json': {
              "('custom_data_class', 'nested_data_class', 'jax_array')": {
                  'key_metadata': (
                      {'key': 'custom_data_class', 'key_type': 2},
                      {'key': 'nested_data_class', 'key_type': 2},
                      {'key': 'jax_array', 'key_type': 2},
                  ),
                  'value_metadata': {
                      'value_type': 'jax.Array',
                      'skip_deserialize': False,
                  },
              },
              "('custom_data_class', 'nested_data_class', 'np_array')": {
                  'key_metadata': (
                      {'key': 'custom_data_class', 'key_type': 2},
                      {'key': 'nested_data_class', 'key_type': 2},
                      {'key': 'np_array', 'key_type': 2},
                  ),
                  'value_metadata': {
                      'value_type': 'np.ndarray',
                      'skip_deserialize': False,
                  },
              },
              "('custom_data_class', 'nested_dict', 'a')": {
                  'key_metadata': (
                      {'key': 'custom_data_class', 'key_type': 2},
                      {'key': 'nested_dict', 'key_type': 2},
                      {'key': 'a', 'key_type': 2},
                  ),
                  'value_metadata': {
                      'value_type': 'jax.Array',
                      'skip_deserialize': False,
                  },
              },
              "('custom_data_class', 'nested_dict', 'b')": {
                  'key_metadata': (
                      {'key': 'custom_data_class', 'key_type': 2},
                      {'key': 'nested_dict', 'key_type': 2},
                      {'key': 'b', 'key_type': 2},
                  ),
                  'value_metadata': {
                      'value_type': 'np.ndarray',
                      'skip_deserialize': False,
                  },
              },
              "('custom_data_class', 'nested_empty_data_class')": {
                  'key_metadata': (
                      {'key': 'custom_data_class', 'key_type': 2},
                      {'key': 'nested_empty_data_class', 'key_type': 2},
                  ),
                  'value_metadata': {
                      'value_type': 'Dict',
                      'skip_deserialize': True,
                  },
              },
              "('custom_data_class', 'nested_list', '0')": {
                  'key_metadata': (
                      {'key': 'custom_data_class', 'key_type': 2},
                      {'key': 'nested_list', 'key_type': 2},
                      {'key': '0', 'key_type': 1},
                  ),
                  'value_metadata': {
                      'value_type': 'jax.Array',
                      'skip_deserialize': False,
                  },
              },
              "('custom_data_class', 'nested_list', '1')": {
                  'key_metadata': (
                      {'key': 'custom_data_class', 'key_type': 2},
                      {'key': 'nested_list', 'key_type': 2},
                      {'key': '1', 'key_type': 1},
                  ),
                  'value_metadata': {
                      'value_type': 'np.ndarray',
                      'skip_deserialize': False,
                  },
              },
          },
      },
      {
          'tree_provider': lambda: {
              'custom_named_tuple': NamedTupleWithNestedAttributes()
          },
          'expected_tree_json': {
              "('custom_named_tuple', 'nested_mu_nu')": {
                  'key_metadata': (
                      {'key': 'custom_named_tuple', 'key_type': 2},
                      {'key': 'nested_mu_nu', 'key_type': 2},
                  ),
                  'value_metadata': {
                      'value_type': 'None',
                      'skip_deserialize': True,
                  },
              },
              "('custom_named_tuple', 'nested_dict')": {
                  'key_metadata': (
                      {'key': 'custom_named_tuple', 'key_type': 2},
                      {'key': 'nested_dict', 'key_type': 2},
                  ),
                  'value_metadata': {
                      'value_type': 'None',
                      'skip_deserialize': True,
                  },
              },
              "('custom_named_tuple', 'nested_tuple')": {
                  'key_metadata': (
                      {'key': 'custom_named_tuple', 'key_type': 2},
                      {'key': 'nested_tuple', 'key_type': 2},
                  ),
                  'value_metadata': {
                      'value_type': 'None',
                      'skip_deserialize': True,
                  },
              },
              "('custom_named_tuple', 'nested_empty_named_tuple')": {
                  'key_metadata': (
                      {'key': 'custom_named_tuple', 'key_type': 2},
                      {'key': 'nested_empty_named_tuple', 'key_type': 2},
                  ),
                  'value_metadata': {
                      'value_type': 'None',
                      'skip_deserialize': True,
                  },
              },
          },
      },
      {
          'tree_provider': lambda: {
              'custom_named_tuple': NamedTupleWithNestedAttributes(
                  nested_mu_nu=MuNu(mu=jnp.arange(8), nu=np.arange(8)),
                  nested_dict={'a': jnp.arange(8), 'b': np.arange(8)},
                  nested_tuple=('np_array', np.arange(8)),
                  nested_empty_named_tuple=EmptyNamedTuple(),
              )
          },
          'expected_tree_json': {
              "('custom_named_tuple', 'nested_mu_nu', 'mu')": {
                  'key_metadata': (
                      {'key': 'custom_named_tuple', 'key_type': 2},
                      {'key': 'nested_mu_nu', 'key_type': 2},
                      {'key': 'mu', 'key_type': 2},
                  ),
                  'value_metadata': {
                      'value_type': 'jax.Array',
                      'skip_deserialize': False,
                  },
              },
              "('custom_named_tuple', 'nested_mu_nu', 'nu')": {
                  'key_metadata': (
                      {'key': 'custom_named_tuple', 'key_type': 2},
                      {'key': 'nested_mu_nu', 'key_type': 2},
                      {'key': 'nu', 'key_type': 2},
                  ),
                  'value_metadata': {
                      'value_type': 'np.ndarray',
                      'skip_deserialize': False,
                  },
              },
              "('custom_named_tuple', 'nested_dict', 'a')": {
                  'key_metadata': (
                      {'key': 'custom_named_tuple', 'key_type': 2},
                      {'key': 'nested_dict', 'key_type': 2},
                      {'key': 'a', 'key_type': 2},
                  ),
                  'value_metadata': {
                      'value_type': 'jax.Array',
                      'skip_deserialize': False,
                  },
              },
              "('custom_named_tuple', 'nested_dict', 'b')": {
                  'key_metadata': (
                      {'key': 'custom_named_tuple', 'key_type': 2},
                      {'key': 'nested_dict', 'key_type': 2},
                      {'key': 'b', 'key_type': 2},
                  ),
                  'value_metadata': {
                      'value_type': 'np.ndarray',
                      'skip_deserialize': False,
                  },
              },
              "('custom_named_tuple', 'nested_tuple', '0')": {
                  'key_metadata': (
                      {'key': 'custom_named_tuple', 'key_type': 2},
                      {'key': 'nested_tuple', 'key_type': 2},
                      {'key': '0', 'key_type': 1},
                  ),
                  'value_metadata': {
                      'value_type': 'string',
                      'skip_deserialize': False,
                  },
              },
              "('custom_named_tuple', 'nested_tuple', '1')": {
                  'key_metadata': (
                      {'key': 'custom_named_tuple', 'key_type': 2},
                      {'key': 'nested_tuple', 'key_type': 2},
                      {'key': '1', 'key_type': 1},
                  ),
                  'value_metadata': {
                      'value_type': 'np.ndarray',
                      'skip_deserialize': False,
                  },
              },
              "('custom_named_tuple', 'nested_empty_named_tuple')": {
                  'key_metadata': (
                      {'key': 'custom_named_tuple', 'key_type': 2},
                      {'key': 'nested_empty_named_tuple', 'key_type': 2},
                  ),
                  'value_metadata': {
                      'value_type': 'None',
                      'skip_deserialize': True,
                  },
              },
          },
      },
      {
          'tree_provider': lambda: {
              'custom_object': CustomWithNestedAttributes()
          },
          'expected_tree_json': {
              "('custom_object',)": {
                  'key_metadata': ({'key': 'custom_object', 'key_type': 2},),
                  'value_metadata': {
                      'value_type': 'None',
                      'skip_deserialize': True,
                  },
              }
          },
      },
      {
          'tree_provider': lambda: {
              'custom_object': CustomWithNestedAttributes(
                  no_attribute_custom=NoAttributeCustom(),
                  custom=Custom(a=jnp.arange(8), b=np.arange(8)),
                  custom_data_class=CustomDataClass(
                      jax_array=jnp.arange(8), np_array=np.arange(8)
                  ),
                  mu_nu=MuNu(mu=jnp.arange(8), nu=np.arange(8)),
              )
          },
          'expected_tree_json': {
              "('custom_object',)": {
                  'key_metadata': ({'key': 'custom_object', 'key_type': 2},),
                  'value_metadata': {
                      'value_type': 'None',
                      'skip_deserialize': True,
                  },
              }
          },
      },
  )
  def test_json_conversion(
      self,
      # Pass tree lazily via `tree_provider` to avoid error:
      # RuntimeError: Attempted call to JAX before absl.app.run() is called.
      tree_provider: Callable[[], Any],
      expected_tree_json: Dict[str, Any],
  ):
    tree = tree_provider()
    expected_tree_json = {
        'tree_metadata': expected_tree_json,
        'use_zarr3': True,
    }
    metadata = tree_metadata.TreeMetadata.build(
        self._to_param_infos(tree),
        use_zarr3=True,
    )

    self.assertDictEqual(expected_tree_json, metadata.to_json())
    self.assertCountEqual(
        tree_metadata.TreeMetadata.from_json(
            expected_tree_json
        ).tree_metadata_entries,
        metadata.tree_metadata_entries,
    )


if __name__ == '__main__':
  absltest.main()
