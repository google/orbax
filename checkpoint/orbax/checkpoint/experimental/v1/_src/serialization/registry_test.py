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

"""Unit tests for the LeafHandlerRegistry."""

from typing import Any
from absl.testing import absltest
import jax
import jax.numpy as jnp
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.serialization import registry
from orbax.checkpoint.experimental.v1._src.serialization import types


class DummyIntHandlerInt(types.LeafHandler[int, int]):

  def __init__(self, context: context_lib.Context | None = None):
    del context


class DummyJaxHandler(types.LeafHandler[jax.Array, types.AbstractShardedArray]):

  def __init__(self, context: context_lib.Context | None = None):
    del context


class DummyGenericAbstractType:
  pass


class DummyGenericHandler(types.LeafHandler[Any, DummyGenericAbstractType]):
  def __init__(self, context: context_lib.Context | None = None):
    del context


class RegistryTest(absltest.TestCase):

  def test_simple_add_and_get(self):
    reg = registry.BaseLeafHandlerRegistry()
    reg.add(int, int, DummyIntHandlerInt)
    reg.add(jax.Array, types.AbstractShardedArray, DummyJaxHandler)

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
        types.AbstractShardedArray,
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
    reg.add(jax.Array, types.AbstractShardedArray, DummyJaxHandler)

    all_items = reg.get_all()
    self.assertLen(all_items, 2)

    for item in all_items:
      if item[0] == int:
        self.assertEqual(item, (int, int, DummyIntHandlerInt))
      elif item[0] == jax.Array:
        self.assertEqual(
            item,
            (jax.Array, types.AbstractShardedArray, DummyJaxHandler),
        )
      else:
        self.fail(f"Unexpected item: {item}")

  def test_secondary_typestrs(self):
    reg = registry.BaseLeafHandlerRegistry()
    reg.add(
        int,
        int,
        DummyIntHandlerInt,
        secondary_typestrs=["alias1", "alias2"],
    )
    self.assertEqual(
        reg.get_secondary_typestrs(DummyIntHandlerInt), ["alias1", "alias2"]
    )

    reg.add(jax.Array, types.AbstractShardedArray, DummyJaxHandler)
    self.assertEqual(reg.get_secondary_typestrs(DummyJaxHandler), [])

  def test_multiple_concrete_to_abstract(self):
    reg = registry.BaseLeafHandlerRegistry()
    reg.add(int, DummyGenericAbstractType, DummyGenericHandler)
    reg.add(float, DummyGenericAbstractType, DummyGenericHandler)

    # Both concrete types should reliably use DummyGenericHandler
    self.assertEqual(reg.get(int), DummyGenericHandler)
    self.assertEqual(reg.get(float), DummyGenericHandler)
    self.assertEqual(
        reg.get_abstract(DummyGenericAbstractType), DummyGenericHandler
    )

  def test_override_abstract_with_different_handler(self):
    class DummyGenericHandler2(
        types.LeafHandler[Any, DummyGenericAbstractType]
    ):
      def __init__(self, context: context_lib.Context | None = None):
        del context

    reg = registry.BaseLeafHandlerRegistry()
    reg.add(str, DummyGenericAbstractType, DummyGenericHandler)
    reg.add(float, DummyGenericAbstractType, DummyGenericHandler)

    # Try registering with an abstract type that is already registered.
    with self.assertRaisesRegex(
        ValueError, r"abstract_type\[.*\] is already handled by.*"
    ):
      reg.add(int, DummyGenericAbstractType, DummyGenericHandler2)

    reg.add(int, int, DummyIntHandlerInt)
    self.assertEqual(reg.get(int), DummyIntHandlerInt)

    # Override both the concrete and the abstract type with a different handler.
    reg.add(int, DummyGenericAbstractType, DummyGenericHandler2, override=True)
    self.assertEqual(reg.get(int), DummyGenericHandler2)
    self.assertEqual(
        reg.get_abstract(DummyGenericAbstractType), DummyGenericHandler2
    )

    with self.assertRaisesRegex(ValueError, "Unknown Leaf type"):
      reg.get(str)
    with self.assertRaisesRegex(ValueError, "Unknown Leaf type"):
      reg.get(float)

  def test_abstract_registration_order(self):
    class DummyArrayType:
      array: jax.Array

    class DummySpecificArrayHandler(
        types.LeafHandler[Any, types.AbstractShardedArray]
    ):
      def __init__(self, context: context_lib.Context | None = None):
        del context

    class DummyGenericArrayHandler(types.LeafHandler[Any, types.AbstractArray]):
      def __init__(self, context: context_lib.Context | None = None):
        del context

    # Test registration order of leaf and abstract types.
    reg = registry.BaseLeafHandlerRegistry()
    reg.add(jax.Array, types.AbstractShardedArray, DummySpecificArrayHandler)
    # Register the more generic abstract type last.
    reg.add(DummyArrayType, types.AbstractArray, DummyGenericArrayHandler)

    # An abstract object that satisfies BOTH protocols. The specific protocol
    # must be matched first because of sorting.
    self.assertEqual(
        reg.get_abstract(jax.ShapeDtypeStruct), DummySpecificArrayHandler
    )

    # An abstract object that ONLY satisfies the generic AbstractArray protocol
    class PureArrayStruct:
      shape = (1,)
      dtype = jnp.float32
    self.assertEqual(
        reg.get_abstract(PureArrayStruct), DummyGenericArrayHandler
    )
    reg2 = registry.BaseLeafHandlerRegistry()
    reg2.add(DummyArrayType, types.AbstractArray, DummyGenericArrayHandler)
    reg2.add(jax.Array, types.AbstractShardedArray, DummySpecificArrayHandler)
    self.assertEqual(
        reg2.get_abstract(jax.ShapeDtypeStruct), DummySpecificArrayHandler
    )
    self.assertEqual(
        reg2.get_abstract(PureArrayStruct), DummyGenericArrayHandler
    )


if __name__ == "__main__":
  absltest.main()
