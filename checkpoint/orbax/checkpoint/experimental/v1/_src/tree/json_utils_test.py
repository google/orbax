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

import json
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint.experimental.v1._src.tree import json_utils


class _SelfRendering:
  """A leaf that renders itself, like `ArrayMetadata`."""

  def __init__(self, value: int):
    self.value = value

  def to_json_dict(self) -> dict[str, json_utils.JsonValue]:
    return {'value': self.value}


class LeafToJsonTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('none', None, None),
      ('int', 0, {'value_type': 'scalar', 'python_type': 'int'}),
      ('float', 0.0, {'value_type': 'scalar', 'python_type': 'float'}),
      ('bool', False, {'value_type': 'scalar', 'python_type': 'bool'}),
      (
          'np_number',
          np.float32(0),
          {'value_type': 'scalar', 'python_type': 'float32'},
      ),
      ('string', 'string', {'value_type': 'string'}),
      ('self_rendering', _SelfRendering(3), {'value': 3}),
  )
  def test_leaf_to_json(self, leaf: Any, expected: json_utils.JsonValue):
    self.assertEqual(json_utils.leaf_to_json(leaf), expected)

  def test_classes_and_unknown_leaves_fall_back_to_repr(self):
    for leaf in (_SelfRendering, object()):
      with self.subTest(repr(leaf)):
        self.assertEqual(
            json_utils.leaf_to_json(leaf),
            {'value_type': 'unknown', 'repr': repr(leaf)},
        )

  @parameterized.named_parameters(
      ('none', None, None),
      ('np_dtype', np.dtype('float32'), 'float32'),
      ('scalar_type', np.int32, 'int32'),
      ('bfloat16_type', jnp.bfloat16, 'bfloat16'),
      ('bfloat16_dtype', jnp.dtype(jnp.bfloat16), 'bfloat16'),
      ('unparsable', 'not-a-dtype', 'not-a-dtype'),
  )
  def test_dtype_to_json(self, dtype: Any, expected: str | None):
    self.assertEqual(json_utils.dtype_to_json(dtype), expected)

  def test_shape_to_json(self):
    self.assertIsNone(json_utils.shape_to_json(None))
    self.assertEqual(json_utils.shape_to_json(()), [])
    self.assertEqual(json_utils.shape_to_json((2, 3)), [2, 3])


class TreeToJsonTest(absltest.TestCase):

  def test_renders_nested_containers(self):
    tree = {'a': [0, ('string', None)], 1: {}, 'r': _SelfRendering(1)}

    rendered = json_utils.tree_to_json(tree)

    self.assertEqual(
        rendered,
        {
            'a': [
                {'value_type': 'scalar', 'python_type': 'int'},
                [{'value_type': 'string'}, None],
            ],
            '1': {},
            'r': {'value': 1},
        },
    )
    self.assertEqual(json.loads(json.dumps(rendered)), rendered)

  def test_leaf_fn_overrides_leaf_rendering(self):
    rendered = json_utils.tree_to_json({'a': [1, (2,)]}, leaf_fn=str)
    self.assertEqual(rendered, {'a': ['1', ['2']]})


if __name__ == '__main__':
  absltest.main()
