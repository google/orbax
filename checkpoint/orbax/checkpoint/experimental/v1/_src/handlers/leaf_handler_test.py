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

from typing import Any, Awaitable

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import jax
from jax import numpy as jnp
import numpy as np
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.arrays import abstract_arrays
from orbax.checkpoint.experimental.v1._src.handlers import leaf_handler
from orbax.checkpoint.experimental.v1._src.path import types as path_types
from orbax.checkpoint.experimental.v1._src.serialization import registry
from orbax.checkpoint.experimental.v1._src.testing import handler_utils as handler_test_utils
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types


PathAwaitingCreation = path_types.PathAwaitingCreation
PathLike = path_types.PathLike
Path = path_types.Path
Json = tree_types.JsonType
create_test_handler = handler_test_utils.create_test_handler

Leaf = leaf_handler.Leaf
AbstractLeaf = leaf_handler.AbstractLeaf


async def _run_awaitable(awaitable: Awaitable[Any]) -> Any:
  return await awaitable


class FooHandler(
    leaf_handler._LeafHandler[
        handler_test_utils.Foo, handler_test_utils.AbstractFoo
    ]
):

  def is_handleable(self, checkpointable: Any) -> bool:
    return isinstance(checkpointable, handler_test_utils.Foo)

  def is_abstract_handleable(self, abstract_checkpointable: Any) -> bool | None:
    return isinstance(abstract_checkpointable, handler_test_utils.AbstractFoo)


class LeafHandlerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(
        self.create_tempdir(name='checkpointing_test').full_path
    )
    self.values_and_abstract_values = {
        leaf_handler.ShardedArrayHandler: [
            (
                jnp.arange(8),
                abstract_arrays.to_shape_dtype_struct(jnp.arange(8)),
            ),
            (jnp.arange(8), jax.ShapeDtypeStruct),
        ],
        leaf_handler.ArrayHandler: [
            (np.arange(8), np.empty_like(np.arange(8)))
        ],
        leaf_handler.ScalarHandler: [
            (123, int),
            (123, 0),
            (123.456, float),
            (123.456, 0.0),
        ],
        leaf_handler.StringHandler: [('test', str), ('test', '_')],
    }

  def validate_load(
      self,
      handler: handler_test_utils.TestHandler[Leaf, AbstractLeaf],
      value: Leaf,
      abstract_value: AbstractLeaf,
      directory: Path | None = None,
  ):
    directory = directory or self.directory
    with self.subTest('load_with_abstract'):
      restored = handler.load(directory, abstract_value)
      test_utils.assert_array_equal(self, value, restored)
    with self.subTest('load_without_abstract'):
      restored = handler.load(directory)
      test_utils.assert_array_equal(self, value, restored)

  @parameterized.parameters(
      leaf_handler.ShardedArrayHandler,
      leaf_handler.ArrayHandler,
      leaf_handler.ScalarHandler,
      leaf_handler.StringHandler,
  )
  def test_save_load(self, handler_cls):
    handler = create_test_handler(handler_cls)
    test_cases = self.values_and_abstract_values[handler_cls]

    self.assertFalse(handler.is_handleable(handler_test_utils.Foo(1, 'hi')))
    self.assertFalse(
        handler.is_abstract_handleable(handler_test_utils.AbstractFoo())
    )

    for i, (value, abstract_value) in enumerate(test_cases):
      name = str(i)
      with self.subTest(f'value={value}, abstract_value={abstract_value}'):
        logging.info(
            'Subtest: value=%s, abstract_value=%s', value, abstract_value
        )
        self.assertTrue(handler.is_handleable(value))
        self.assertTrue(handler.is_abstract_handleable(abstract_value))
        handler.save(self.directory / name, value)
        self.validate_load(
            handler, value, abstract_value, directory=self.directory / name
        )

  def test_unregistered_type(self):
    handler = create_test_handler(FooHandler)
    with self.assertRaises(registry.UnregisteredTypeError):
      handler.save(self.directory, handler_test_utils.Foo(1, 'hi'))

    with self.assertRaises(registry.UnregisteredTypeError):
      handler.load(self.directory, handler_test_utils.AbstractFoo())

if __name__ == '__main__':
  absltest.main()
