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

from typing import Awaitable

from absl.testing import absltest
from orbax.checkpoint.experimental.v1._src.handlers import types as handler_types
from orbax.checkpoint.experimental.v1._src.path import types as path_types


class Foo:
  pass


class FooHandler(handler_types.CheckpointableHandler[Foo, None]):

  async def save(
      self,
      directory: path_types.PathAwaitingCreation,
      checkpointable: Foo,
  ) -> Awaitable[None]:
    raise NotImplementedError()

  async def load(
      self,
      directory: path_types.Path,
      abstract_checkpointable: None = None,
  ) -> Awaitable[Foo]:
    raise NotImplementedError()

  async def metadata(self, directory: path_types.Path) -> None:
    return

  def is_handleable(self, checkpointable: Foo) -> bool:
    return isinstance(checkpointable, Foo)

  def is_abstract_handleable(self, abstract_checkpointable: None) -> bool:
    return abstract_checkpointable is None


class Bar:
  pass


class AbstractBar:
  pass


class BarMetadata(AbstractBar):
  pass


class BarHandler(handler_types.CheckpointableHandler[Bar, AbstractBar]):

  async def save(
      self,
      directory: path_types.PathAwaitingCreation,
      checkpointable: Bar,
  ) -> Awaitable[None]:
    raise NotImplementedError()

  async def load(
      self,
      directory: path_types.Path,
      abstract_checkpointable: AbstractBar | None = None,
  ) -> Awaitable[Bar]:
    raise NotImplementedError()

  async def metadata(self, directory: path_types.Path) -> BarMetadata:
    return BarMetadata()

  def is_handleable(self, checkpointable: Bar) -> bool:
    return isinstance(checkpointable, Bar)

  def is_abstract_handleable(
      self, abstract_checkpointable: AbstractBar
  ) -> bool:
    return isinstance(abstract_checkpointable, AbstractBar)


class TypesTest(absltest.TestCase):

  def test_checkpointable_handler_typing_foo(self):
    handler = FooHandler()
    self.assertTrue(handler.is_handleable(Foo()))
    self.assertFalse(handler.is_handleable(None))  # pytype: disable=wrong-arg-types
    self.assertTrue(handler.is_abstract_handleable(None))
    self.assertFalse(handler.is_abstract_handleable(Foo()))  # pytype: disable=wrong-arg-types

  def test_checkpointable_handler_typing_bar(self):
    handler = BarHandler()
    self.assertTrue(handler.is_handleable(Bar()))
    self.assertFalse(handler.is_handleable(AbstractBar()))  # pytype: disable=wrong-arg-types
    self.assertFalse(handler.is_abstract_handleable(Bar()))  # pytype: disable=wrong-arg-types
    self.assertTrue(handler.is_abstract_handleable(AbstractBar()))


if __name__ == '__main__':
  absltest.main()
