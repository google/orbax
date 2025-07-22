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

"""Unit tests for the LeafHandlerRegistry."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.serialization import array_leaf_handler
from orbax.checkpoint.experimental.v1._src.serialization import registry
from orbax.checkpoint.experimental.v1._src.serialization import types


class DummyIntHandlerInt(types.LeafHandler[int, int]):

  def __init__(self, context: context_lib.Context | None = None):
    del context


class DummyJaxHandler(
    types.LeafHandler[jax.Array, array_leaf_handler.AbstractArray]
):

  def __init__(self, context: context_lib.Context | None = None):
    del context


class RegistryTest(absltest.TestCase):

  def test_simple_add_and_get(self):
    reg = registry.BaseLeafHandlerRegistry()
    reg.add(int, int, DummyIntHandlerInt)
    reg.add(jax.Array, array_leaf_handler.AbstractArray, DummyJaxHandler)

    self.assertEqual(DummyIntHandlerInt, reg.get(int))
    self.assertEqual(DummyIntHandlerInt, reg.get(type(0)))
    self.assertEqual(DummyIntHandlerInt, reg.get_abstract(int))

    self.assertEqual(DummyJaxHandler, reg.get(jax.Array))
    self.assertEqual(reg.get(type(jnp.asarray([1, 2, 3]))), DummyJaxHandler)
    self.assertEqual(reg.get(type(jax.random.key(0))), DummyJaxHandler)
    self.assertEqual(
        reg.get_abstract(type(jax.ShapeDtypeStruct((), jnp.float32))),
        DummyJaxHandler,
    )

    with self.assertRaisesRegex(ValueError, "Unknown Leaf type"):
      reg.get(float)

    with self.assertRaisesRegex(ValueError, "Unknown AbstractLeaf type"):
      reg.get_abstract(float)

    with self.assertRaisesRegex(ValueError, "Unknown AbstractLeaf type"):
      reg.get_abstract(type(0.0))

  def test_replace(self):
    reg = registry.StandardLeafHandlerRegistry()

    orig_array_handler_type = reg.get(jax.Array)
    orig_int_handler_type = reg.get(int)

    self.assertIsNotNone(reg.get(jax.Array))
    self.assertIsNotNone(reg.get_abstract(jax.ShapeDtypeStruct))
    self.assertIsNotNone(reg.get(int))
    self.assertIsNotNone(reg.get_abstract(int))

    reg.add(int, int, DummyIntHandlerInt, override=True)
    reg.add(
        jax.Array,
        array_leaf_handler.AbstractArray,
        DummyJaxHandler,
        override=True,
    )

    self.assertEqual(DummyIntHandlerInt, reg.get(int))
    self.assertEqual(DummyJaxHandler, reg.get(jax.Array))

    self.assertEqual(DummyIntHandlerInt, reg.get_abstract(int))
    self.assertEqual(DummyJaxHandler, reg.get_abstract(jax.ShapeDtypeStruct))

    self.assertNotEqual(reg.get(jax.Array), orig_array_handler_type)
    self.assertNotEqual(
        reg.get_abstract(type(jax.ShapeDtypeStruct((), jnp.float32))),
        orig_array_handler_type,
    )
    self.assertNotEqual(reg.get(int), orig_int_handler_type)
    self.assertNotEqual(reg.get_abstract(int), orig_int_handler_type)

  def test_replace_with_different_abstract_type(self):

    class DummyIntHandlerInt2(types.LeafHandler[int, type(None)]):

      def __init__(self, context: context_lib.Context | None = None):
        del context

    reg = registry.BaseLeafHandlerRegistry()
    reg.add(int, int, DummyIntHandlerInt)
    self.assertEqual(reg.get(int), DummyIntHandlerInt)
    self.assertEqual(reg.get_abstract(int), DummyIntHandlerInt)

    # replace AbstractInt
    reg.add(int, type(None), DummyIntHandlerInt2, override=True)
    self.assertEqual(DummyIntHandlerInt2, reg.get(int))
    self.assertEqual(DummyIntHandlerInt2, reg.get_abstract(type(None)))

    with self.assertRaisesRegex(ValueError, "Unknown AbstractLeaf type"):
      print(f"{reg.get_abstract(int)=}")

  def test_get_all(self):
    reg = registry.BaseLeafHandlerRegistry()
    reg.add(int, int, DummyIntHandlerInt)
    reg.add(jax.Array, array_leaf_handler.AbstractArray, DummyJaxHandler)

    all_items = reg.get_all()
    self.assertLen(all_items, 2)

    for item in all_items:
      if item[0] == int:
        self.assertEqual(item, (int, int, DummyIntHandlerInt))
      elif item[0] == jax.Array:
        self.assertEqual(
            item,
            (jax.Array, array_leaf_handler.AbstractArray, DummyJaxHandler),
        )
      else:
        self.fail(f"Unexpected item: {item}")


if __name__ == "__main__":
  absltest.main()
