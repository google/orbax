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
from orbax.checkpoint.experimental.v1._src.serialization import array_leaf_handler
from orbax.checkpoint.experimental.v1._src.serialization import registry
from orbax.checkpoint.experimental.v1._src.serialization import types


class DummyIntHandlerInt(types.LeafHandler[int, int]):
  pass


class DummyJaxHandler(
    types.LeafHandler[jax.Array, array_leaf_handler.AbstractArray]
):
  pass


class RegistryTest(absltest.TestCase):

  def test_simple_add_and_get(self):
    reg = registry.BaseLeafHandlerRegistry()
    int_handler = DummyIntHandlerInt()
    array_handler = DummyJaxHandler()
    reg.add(int, int, int_handler)
    reg.add(jax.Array, array_leaf_handler.AbstractArray, array_handler)

    self.assertEqual(reg.get(type(0)), int_handler)
    self.assertEqual(reg.get_abstract(int), int_handler)

    self.assertEqual(reg.get(jax.Array), array_handler)
    self.assertEqual(reg.get(type(jnp.asarray([1, 2, 3]))), array_handler)
    self.assertEqual(reg.get(type(jax.random.key(0))), array_handler)
    self.assertEqual(
        reg.get_abstract(type(jax.ShapeDtypeStruct((), jnp.float32))),
        array_handler,
    )

    with self.assertRaisesRegex(ValueError, "Unknown Leaf type"):
      reg.get(float)

    with self.assertRaisesRegex(ValueError, "Unknown AbstractLeaf type"):
      reg.get_abstract(float)

    with self.assertRaisesRegex(ValueError, "Unknown AbstractLeaf type"):
      reg.get_abstract(type(0.0))

  def test_replace(self):
    reg = registry.StandardLeafHandlerRegistry()

    orig_array_handler = reg.get(jax.Array)
    orig_int_handler = reg.get(int)

    self.assertIsNotNone(reg.get(jax.Array))
    self.assertIsNotNone(reg.get_abstract(jax.ShapeDtypeStruct))
    self.assertIsNotNone(reg.get(int))
    self.assertIsNotNone(reg.get_abstract(int))

    int_handler = DummyIntHandlerInt()
    array_handler = DummyJaxHandler()

    reg.add(int, int, int_handler, override=True)
    reg.add(
        jax.Array,
        array_leaf_handler.AbstractArray,
        array_handler,
        override=True,
    )

    self.assertEqual(reg.get(int), int_handler)
    self.assertEqual(reg.get_abstract(int), int_handler)
    self.assertEqual(reg.get(jax.Array), array_handler)
    self.assertEqual(reg.get_abstract(jax.ShapeDtypeStruct), array_handler)

    self.assertNotEqual(reg.get(jax.Array), orig_array_handler)
    self.assertNotEqual(
        reg.get_abstract(type(jax.ShapeDtypeStruct((), jnp.float32))),
        orig_array_handler,
    )
    self.assertNotEqual(reg.get(int), orig_int_handler)
    self.assertNotEqual(reg.get_abstract(int), orig_int_handler)

  def test_replace_with_different_abstract_type(self):
    reg = registry.BaseLeafHandlerRegistry()
    int_handler = DummyIntHandlerInt()
    array_handler = DummyJaxHandler()
    reg.add(int, int, int_handler, override=True)
    reg.add(
        jax.Array,
        array_leaf_handler.AbstractArray,
        array_handler,
        override=True,
    )

    self.assertEqual(reg.get(int), int_handler)
    self.assertEqual(reg.get_abstract(int), int_handler)

    int_handler2 = DummyIntHandlerInt()
    reg.add(int, jnp.int64, int_handler2, override=True)

    with self.assertRaisesRegex(ValueError, "Unknown AbstractLeaf type"):
      reg.get_abstract(int)
    self.assertEqual(reg.get(int), int_handler2)
    self.assertEqual(reg.get_abstract(jnp.int64), int_handler2)

  def test_get_all(self):
    reg = registry.BaseLeafHandlerRegistry()
    int_handler = DummyIntHandlerInt()
    array_handler = DummyJaxHandler()
    reg.add(int, int, int_handler)
    reg.add(jax.Array, array_leaf_handler.AbstractArray, array_handler)

    all_items = reg.get_all()
    self.assertLen(all_items, 2)

    for item in all_items:
      if item[0] == int:
        self.assertEqual(item, (int, int, int_handler))
      elif item[0] == jax.Array:
        self.assertEqual(
            item,
            (jax.Array, array_leaf_handler.AbstractArray, array_handler),
        )
      else:
        self.fail(f"Unexpected item: {item}")


if __name__ == "__main__":
  absltest.main()
