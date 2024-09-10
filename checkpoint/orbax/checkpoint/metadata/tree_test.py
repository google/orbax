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

from typing import NamedTuple, Optional
from absl.testing import absltest
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


class MuNu(NamedTuple):
  mu: Optional[jax.Array]
  nu: Optional[np.ndarray]


class EmptyNamedTuple(NamedTuple):
  pass


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


class TreeMetadataTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    arr = jnp.arange(8)
    assert isinstance(arr, jax.Array)
    self.tree = {
        'scalar_param': 1,
        'b': {'string_param': 'hi', 'nested_scalar_param': 3.4},
        'list_of_arrays': [np.arange(8), arr],
        'none_param': None,
        'empty_dict': {},
        'empty_list': [],
        'tuple_of_empty_list': tuple([]),
        'no_attribute_chex_dataclass': TransformByLrAndWdScheduleState(),
        'default_chex_dataclass': CustomDataClass(),
        'chex_dataclass': CustomDataClass(jax_array=arr, np_array=np.arange(8)),
        'tuple_of_arrays': (np.arange(8), arr),
        'named_tuple_param': MuNu(nu=np.arange(8), mu=arr),
        'empty_tuple': tuple(),
        'list_of_empty_tuple': [tuple()],
        'empty_named_tuple': EmptyNamedTuple(),
        'no_attribute_custom_object': NoAttributeCustom(),
        'default_custom_object': Custom(),
        'custom_object': Custom(a=arr, b=np.arange(8)),
    }
    self.tree_json = {
        'tree_metadata': {
            "('scalar_param',)": {
                'key_metadata': ({'key': 'scalar_param', 'key_type': 2},),
                'value_metadata': {
                    'value_type': 'scalar',
                    'skip_deserialize': False,
                },
            },
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
            "('none_param',)": {
                'key_metadata': ({'key': 'none_param', 'key_type': 2},),
                'value_metadata': {
                    'value_type': 'None',
                    'skip_deserialize': True,
                },
            },
            "('empty_dict',)": {
                'key_metadata': ({'key': 'empty_dict', 'key_type': 2},),
                'value_metadata': {
                    'value_type': 'Dict',
                    'skip_deserialize': True,
                },
            },
            "('empty_list',)": {
                'key_metadata': ({'key': 'empty_list', 'key_type': 2},),
                'value_metadata': {
                    'value_type': 'List',
                    'skip_deserialize': True,
                },
            },
            "('tuple_of_empty_list',)": {
                'key_metadata': (
                    {'key': 'tuple_of_empty_list', 'key_type': 2},
                ),
                'value_metadata': {
                    'value_type': 'None',  # TODO: b/365169723 - Fix this.
                    'skip_deserialize': True,
                },
            },
            "('no_attribute_chex_dataclass',)": {
                'key_metadata': (
                    {'key': 'no_attribute_chex_dataclass', 'key_type': 2},
                ),
                'value_metadata': {
                    'value_type': 'None',
                    'skip_deserialize': True,
                },
            },
            "('default_chex_dataclass', 'jax_array')": {
                'key_metadata': (
                    {'key': 'default_chex_dataclass', 'key_type': 2},
                    {'key': 'jax_array', 'key_type': 2},
                ),
                'value_metadata': {
                    'value_type': 'None',
                    'skip_deserialize': True,
                },
            },
            "('default_chex_dataclass', 'np_array')": {
                'key_metadata': (
                    {'key': 'default_chex_dataclass', 'key_type': 2},
                    {'key': 'np_array', 'key_type': 2},
                ),
                'value_metadata': {
                    'value_type': 'None',
                    'skip_deserialize': True,
                },
            },
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
            "('empty_tuple',)": {
                'key_metadata': ({'key': 'empty_tuple', 'key_type': 2},),
                'value_metadata': {
                    'value_type': 'None',  # TODO: b/365169723 - Fix this.
                    'skip_deserialize': True,
                },
            },
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
            "('empty_named_tuple',)": {
                'key_metadata': ({'key': 'empty_named_tuple', 'key_type': 2},),
                'value_metadata': {
                    'value_type': 'None',  # TODO: b/365169723 - Fix this.
                    'skip_deserialize': True,
                },
            },
            "('no_attribute_custom_object',)": {
                'key_metadata': (
                    {'key': 'no_attribute_custom_object', 'key_type': 2},
                ),
                'value_metadata': {
                    'value_type': 'None',
                    'skip_deserialize': True,
                },
            },
            "('default_custom_object',)": {
                'key_metadata': (
                    {'key': 'default_custom_object', 'key_type': 2},
                ),
                'value_metadata': {
                    'value_type': 'None',
                    'skip_deserialize': True,
                },
            },
            "('custom_object',)": {
                'key_metadata': ({'key': 'custom_object', 'key_type': 2},),
                'value_metadata': {
                    'value_type': 'None',
                    'skip_deserialize': True,
                },
            },
        },
        'use_zarr3': True,
    }
    self.param_infos = jax.tree.map(
        # Other properties are not relevant.
        lambda x: type_handlers.ParamInfo(
            value_typestr=type_handlers.get_param_typestr(
                x, type_handlers.GLOBAL_TYPE_HANDLER_REGISTRY
            )
        ),
        self.tree,
        is_leaf=tree_utils.is_empty_or_leaf,
    )

  def test_json_conversion(self):
    metadata = tree_metadata.TreeMetadata.build(
        self.param_infos,
        use_zarr3=True,
    )
    self.assertDictEqual(self.tree_json, metadata.to_json())
    self.assertCountEqual(
        metadata.tree_metadata_entries,
        tree_metadata.TreeMetadata.from_json(
            self.tree_json
        ).tree_metadata_entries,
    )


if __name__ == '__main__':
  absltest.main()
