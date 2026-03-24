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
from absl.testing import parameterized
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


class RegistryTest(parameterized.TestCase):

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

  def test_multiple_leaf_to_abstract_mappings(self):
    reg = registry.BaseLeafHandlerRegistry()
    reg.add(int, DummyGenericAbstractType, DummyGenericHandler)
    reg.add(float, DummyGenericAbstractType, DummyGenericHandler)

    # Both leaf types should reliably use DummyGenericHandler
    self.assertEqual(reg.get(int), DummyGenericHandler)
    self.assertEqual(reg.get(float), DummyGenericHandler)
    self.assertEqual(
        reg.get_abstract(DummyGenericAbstractType), DummyGenericHandler
    )

  def test_add_abstract_type_conflict(self):
    class DummyGenericHandler2(
        types.LeafHandler[Any, DummyGenericAbstractType]
    ):
      def __init__(self, context: context_lib.Context | None = None):
        del context

    reg = registry.BaseLeafHandlerRegistry()
    reg.add(str, DummyGenericAbstractType, DummyGenericHandler)

    # Try registering with an abstract type that is already registered without
    # override=True.
    with self.assertRaisesRegex(
        ValueError, r"abstract_type\[.*\] is already handled by.*"
    ):
      reg.add(int, DummyGenericAbstractType, DummyGenericHandler2)

  def test_override_leaf_and_abstract_type(self):
    class DummyGenericHandler2(
        types.LeafHandler[Any, DummyGenericAbstractType]
    ):
      def __init__(self, context: context_lib.Context | None = None):
        del context

    reg = registry.BaseLeafHandlerRegistry()
    reg.add(int, int, DummyIntHandlerInt)
    self.assertEqual(reg.get(int), DummyIntHandlerInt)
    self.assertEqual(reg.get_abstract(int), DummyIntHandlerInt)

    reg.add(str, DummyGenericAbstractType, DummyGenericHandler)
    self.assertEqual(
        reg.get_abstract(DummyGenericAbstractType), DummyGenericHandler
    )

    # Override both the leaf and the abstract type with a different handler.
    reg.add(int, DummyGenericAbstractType, DummyGenericHandler2, override=True)
    self.assertEqual(reg.get(int), DummyGenericHandler2)
    self.assertEqual(
        reg.get_abstract(DummyGenericAbstractType), DummyGenericHandler2
    )

  def test_override_abstract_removes_old_leaf_mappings(self):
    class DummyGenericHandler2(
        types.LeafHandler[Any, DummyGenericAbstractType]
    ):
      def __init__(self, context: context_lib.Context | None = None):
        del context

    reg = registry.BaseLeafHandlerRegistry()
    reg.add(float, DummyGenericAbstractType, DummyGenericHandler)
    reg.add(str, DummyGenericAbstractType, DummyGenericHandler)
    self.assertEqual(reg.get(float), DummyGenericHandler)
    self.assertEqual(reg.get(str), DummyGenericHandler)

    # Override the handler for DummyGenericAbstractType.
    reg.add(int, DummyGenericAbstractType, DummyGenericHandler2, override=True)

    # The previous leaf types (str, float) associated with
    # DummyGenericAbstractType via DummyGenericHandler are no longer valid.
    with self.assertRaisesRegex(ValueError, "Unknown Leaf type"):
      reg.get(str)
    with self.assertRaisesRegex(ValueError, "Unknown Leaf type"):
      reg.get(float)

  @parameterized.named_parameters(
      ("specific_then_generic", False),
      ("generic_then_specific", True),
  )
  def test_abstract_registration_order(self, reverse_registration):
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

    class PureArrayStruct:
      shape = (1,)
      dtype = jnp.float32

    # Define the entries in the default order
    entries_to_register = [
        (jax.Array, types.AbstractShardedArray, DummySpecificArrayHandler),
        (DummyArrayType, types.AbstractArray, DummyGenericArrayHandler),
    ]

    # Apply the parameterized ordering
    if reverse_registration:
      entries_to_register.reverse()

    # Register the handlers
    reg = registry.BaseLeafHandlerRegistry()
    for leaf, abstract, handler in entries_to_register:
      reg.add(leaf, abstract, handler)

    # Run the assertions (which should pass regardless of registration order)
    # An abstract object that satisfies BOTH protocols. The specific protocol
    # must be matched first because of sorting.
    self.assertEqual(
        reg.get_abstract(jax.ShapeDtypeStruct), DummySpecificArrayHandler
    )

    # An abstract object that ONLY satisfies the generic AbstractArray protocol
    self.assertEqual(
        reg.get_abstract(PureArrayStruct), DummyGenericArrayHandler
    )

  def test_recalculation_correctness(self):
    class A:
      pass

    class B(A):
      pass

    class C(B):
      pass

    class AbstractA:
      pass

    class AbstractB:
      pass

    class AbstractC:
      pass

    class DummyAHandler(types.LeafHandler[A, Any]):
      def __init__(self, context: context_lib.Context | None = None):
        del context

    class DummyBHandler(types.LeafHandler[B, Any]):
      def __init__(self, context: context_lib.Context | None = None):
        del context

    class DummyCHandler(types.LeafHandler[C, Any]):
      def __init__(self, context: context_lib.Context | None = None):
        del context

    reg = registry.BaseLeafHandlerRegistry()

    # Add A
    reg.add(A, AbstractA, DummyAHandler)
    # A should have score 0
    self.assertEqual(reg._entries[0].leaf_specificity_score, 0)

    # Add C
    reg.add(C, AbstractC, DummyCHandler)
    # After adding C, A:0, C:1
    # Sorted Generic -> Specific (ascending)
    self.assertEqual(reg._entries[1].leaf_type, C)
    self.assertEqual(reg._entries[1].leaf_specificity_score, 1)
    self.assertEqual(reg._entries[0].leaf_type, A)
    self.assertEqual(reg._entries[0].leaf_specificity_score, 0)

    # Add B
    reg.add(B, AbstractB, DummyBHandler)
    self.assertEqual(reg._entries[0].leaf_type, A)
    self.assertEqual(reg._entries[0].leaf_specificity_score, 0)
    self.assertEqual(reg._entries[1].leaf_type, B)
    self.assertEqual(reg._entries[1].leaf_specificity_score, 1)
    self.assertEqual(reg._entries[2].leaf_type, C)
    self.assertEqual(reg._entries[2].leaf_specificity_score, 2)

    # Verify resolution
    self.assertEqual(reg.get(C), DummyCHandler)
    self.assertEqual(reg.get(B), DummyBHandler)
    self.assertEqual(reg.get(A), DummyAHandler)

  def test_recalculate_with_non_class_types(self):
    reg = registry.BaseLeafHandlerRegistry()
    # Adding Any as a leaf_type should trigger TypeError in issubclass
    reg.add(int, int, DummyIntHandlerInt)
    # This shouldn't crash
    reg.add(Any, Any, DummyGenericHandler)
    # Verify scores are still sensible for the valid types
    for entry in reg._entries:
      if entry.leaf_type == int:
        self.assertEqual(entry.leaf_specificity_score, 0)

  def test_get_with_non_class_type(self):
    reg = registry.BaseLeafHandlerRegistry()
    reg.add(int, int, DummyIntHandlerInt)
    # issubclass(Any, int) or similar might be called in _try_get
    # which should be caught by the try-except block
    with self.assertRaisesRegex(ValueError, "Unknown Leaf type"):
      reg.get(Any)

  def test_get_abstract_with_non_class_type(self):
    reg = registry.BaseLeafHandlerRegistry()
    reg.add(int, int, DummyIntHandlerInt)
    with self.assertRaisesRegex(ValueError, "Unknown AbstractLeaf type"):
      reg.get_abstract(Any)

  def test_add_duplicate_registration(self):
    reg = registry.BaseLeafHandlerRegistry()
    reg.add(int, int, DummyIntHandlerInt)
    reg.add(int, int, DummyIntHandlerInt)
    self.assertLen(reg._entries, 1)

  def test_secondary_typestr_override(self):
    reg = registry.BaseLeafHandlerRegistry()
    reg.add(
        int,
        int,
        DummyIntHandlerInt,
        secondary_typestrs=["original"],
    )
    self.assertEqual(
        reg.get_secondary_typestrs(DummyIntHandlerInt), ["original"]
    )
    # This should not trigger an update because override=False.
    reg.add(
        int,
        int,
        DummyIntHandlerInt,
        secondary_typestrs=["updated"],
    )
    self.assertEqual(
        reg.get_secondary_typestrs(DummyIntHandlerInt), ["original"]
    )
    reg.add(
        int,
        int,
        DummyIntHandlerInt,
        secondary_typestrs=["updated"],
        override=True,
    )
    self.assertEqual(
        reg.get_secondary_typestrs(DummyIntHandlerInt), ["updated"]
    )

  def test_standard_registrations(self):
    reg = registry.StandardLeafHandlerRegistry()
    self.assertEqual(
        reg.get_secondary_typestrs(
            registry.array_leaf_handler.ArrayLeafHandler
        ),
        ["jax.Array"],
    )
    self.assertEqual(
        reg.get_secondary_typestrs(
            registry.numpy_leaf_handler.NumpyLeafHandler
        ),
        ["np.ndarray"],
    )
    self.assertEqual(
        reg.get_secondary_typestrs(
            registry.scalar_leaf_handler.ScalarLeafHandler
        ),
        ["scalar"],
    )
    self.assertEqual(
        reg.get_secondary_typestrs(
            registry.string_leaf_handler.StringLeafHandler
        ),
        ["string"],
    )


if __name__ == "__main__":
  absltest.main()
