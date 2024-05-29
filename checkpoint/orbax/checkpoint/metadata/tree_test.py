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

from absl.testing import absltest
import jax
from jax import numpy as jnp
import numpy as np
from orbax.checkpoint import type_handlers
from orbax.checkpoint.metadata import tree as tree_metadata


class TreeMetadataTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    arr = jnp.arange(8)
    assert isinstance(arr, jax.Array)
    self.tree = {'a': 1, 'b': {'c': 'hi', 'd': 3.4}, 'e': [np.arange(8), arr]}
    self.tree_json = {
        'tree_metadata': {
            "('a',)": {
                'key_metadata': ({'key': 'a', 'key_type': 2},),
                'value_metadata': {
                    'value_type': 'scalar',
                    'skip_deserialize': False,
                },
            },
            "('b', 'c')": {
                'key_metadata': (
                    {'key': 'b', 'key_type': 2},
                    {'key': 'c', 'key_type': 2},
                ),
                'value_metadata': {
                    'value_type': 'string',
                    'skip_deserialize': False,
                },
            },
            "('b', 'd')": {
                'key_metadata': (
                    {'key': 'b', 'key_type': 2},
                    {'key': 'd', 'key_type': 2},
                ),
                'value_metadata': {
                    'value_type': 'scalar',
                    'skip_deserialize': False,
                },
            },
            "('e', '0')": {
                'key_metadata': (
                    {'key': 'e', 'key_type': 2},
                    {'key': '0', 'key_type': 1},
                ),
                'value_metadata': {
                    'value_type': 'np.ndarray',
                    'skip_deserialize': False,
                },
            },
            "('e', '1')": {
                'key_metadata': (
                    {'key': 'e', 'key_type': 2},
                    {'key': '1', 'key_type': 1},
                ),
                'value_metadata': {
                    'value_type': 'jax.Array',
                    'skip_deserialize': False,
                },
            },
        },
        'use_zarr3': True,
    }

  def test_json_conversion(self):
    metadata = tree_metadata.TreeMetadata.build(
        self.tree,
        type_handler_registry=type_handlers.GLOBAL_TYPE_HANDLER_REGISTRY,
        use_zarr3=True,
    )
    self.assertDictEqual(self.tree_json, metadata.to_json())
    self.assertEqual(
        metadata, tree_metadata.TreeMetadata.from_json(self.tree_json)
    )

  def test_aggregate_option(self):
    save_args = jax.tree.map(
        lambda x: type_handlers.SaveArgs(), self.tree
    )
    save_args['a'] = type_handlers.SaveArgs(aggregate=True)
    metadata = tree_metadata.TreeMetadata.build(
        self.tree,
        type_handler_registry=type_handlers.GLOBAL_TYPE_HANDLER_REGISTRY,
        save_args=save_args,
        use_zarr3=True,
    )
    tree_json = dict(**self.tree_json)
    tree_json['tree_metadata']["('a',)"]['value_metadata'][
        'skip_deserialize'
    ] = True
    self.assertDictEqual(self.tree_json, metadata.to_json())
    self.assertEqual(
        metadata, tree_metadata.TreeMetadata.from_json(self.tree_json)
    )


if __name__ == '__main__':
  absltest.main()
