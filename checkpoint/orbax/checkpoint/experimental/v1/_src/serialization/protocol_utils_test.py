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

"""Unittests for protocol_utils."""

import dataclasses
from typing import Protocol, Sequence
from absl.testing import absltest
import jax
import numpy as np
from orbax.checkpoint.experimental.v1._src.serialization import numpy_leaf_handler
from orbax.checkpoint.experimental.v1._src.serialization import protocol_utils
from orbax.checkpoint.experimental.v1._src.serialization import types as serialization_types


class ProtocolUtilsTest(absltest.TestCase):

  def test_is_define_protocol_attributes_simple(self):
    class AbstractProtocolAttr(Protocol):
      attr1: int
      attr2: float

    @dataclasses.dataclass
    class Child1:
      attr1: int
      attr2: float

    c = Child1(1, 11)

    self.assertTrue(
        protocol_utils.is_subclass_protocol(type(c), AbstractProtocolAttr)
    )

  def test_is_define_protocol_attributes_with_extra_attributes(self):
    class AbstractProtocolAttr(Protocol):
      attr1: int
      attr2: float

    class Child1:
      attr1: int
      attr2: float
      attr3: str
      attr4: Sequence[int]

    self.assertTrue(
        protocol_utils.is_subclass_protocol(Child1, AbstractProtocolAttr)
    )

  def test_is_define_protocol_attributes_not_match(self):
    class AbstractProtocolAttr(Protocol):
      attr1: int
      attr2: float

    class Child1:
      attr1: int
      attr3: float

    self.assertFalse(
        protocol_utils.is_subclass_protocol(Child1, AbstractProtocolAttr)
    )

    self.assertFalse(
        protocol_utils.is_subclass_protocol(int, AbstractProtocolAttr)
    )

    self.assertFalse(
        protocol_utils.is_subclass_protocol(np.float32, AbstractProtocolAttr)
    )

  def test_with_methods(self):
    class AbstractProtocolAttr(Protocol):
      attr1: int
      attr2: float

      def foo(self):
        ...

    class ChildWithFoo:
      attr1: int
      attr2: float

      def foo(self):
        pass

    class ChildWithOutFoo:
      attr1: int
      attr2: float

      def other_foo(self):
        pass

    self.assertTrue(
        protocol_utils.is_subclass_protocol(ChildWithFoo, AbstractProtocolAttr)
    )
    self.assertFalse(
        protocol_utils.is_subclass_protocol(
            ChildWithOutFoo, AbstractProtocolAttr
        )
    )

  def test_exception_for_non_protocol(self):
    with self.assertRaisesRegex(ValueError, "Protocol .* is not a Protocol"):
      protocol_utils.is_subclass_protocol(int, int)

  def test_complex_type(self):
    class AbstractProtocol(Protocol):
      attr1: jax.Array
      attr2: np.ndarray

    class Child(AbstractProtocol):
      attr1: jax.Array
      attr2: np.ndarray

    self.assertTrue(
        protocol_utils.is_subclass_protocol(Child, AbstractProtocol)
    )

  def test_with_property(self):
    class AbstractProtocol(Protocol):
      attr1: jax.Array

      @property
      def foo(self) -> int:
        ...

    @dataclasses.dataclass
    class ChildWithFoo:
      attr1: jax.Array
      foo: int

    self.assertTrue(
        protocol_utils.is_subclass_protocol(ChildWithFoo, AbstractProtocol)
    )

    @dataclasses.dataclass
    class ChildWithoutFoo:
      attr1: jax.Array
      not_foo: int

    self.assertFalse(
        protocol_utils.is_subclass_protocol(ChildWithoutFoo, AbstractProtocol)
    )

  def test_same_protocol(self):
    class AbstractProtocol(Protocol):
      attr1: jax.Array
      attr2: int

    self.assertTrue(
        protocol_utils.is_subclass_protocol(AbstractProtocol, AbstractProtocol)
    )

  def test_abstract_protocol(self):
    npint = np.int64(1)

    self.assertTrue(
        protocol_utils.is_subclass_protocol(
            type(npint), serialization_types.AbstractArray
        )
    )

    self.assertTrue(
        protocol_utils.is_subclass_protocol(
            type(
                numpy_leaf_handler.NumpyShapeDtype(
                    shape=(), dtype=np.dtype("int32")
                )
            ),
            serialization_types.AbstractArray,
        )
    )


if __name__ == "__main__":
  absltest.main()
