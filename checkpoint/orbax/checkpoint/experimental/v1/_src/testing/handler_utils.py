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

"""Toy CheckpointableHandlers implementations for testing."""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Awaitable, Type

import aiofiles
from etils import epath
from orbax.checkpoint.experimental.v1._src.handlers import types as handler_types
from orbax.checkpoint.experimental.v1._src.path import types as path_types


# pylint: disable=missing-class-docstring


class DataclassHandler:

  async def background_save(
      self,
      directory: path_types.PathLike,
      checkpointable: Any,
  ):
    directory = epath.Path(directory)
    async with aiofiles.open(directory / 'foo.txt', 'w') as f:
      contents = json.dumps(dataclasses.asdict(checkpointable))
      await f.write(contents)

  async def background_load(
      self, directory: path_types.PathLike, checkpointable_type: Type[Any]
  ) -> Any:
    directory = epath.Path(directory)
    async with aiofiles.open(directory / 'foo.txt', 'r') as f:
      contents = json.loads(await f.read())
      return checkpointable_type(*contents.values())


@dataclasses.dataclass
class Foo:
  x: int
  y: str

  def __eq__(self, other: Foo) -> bool:
    return isinstance(other, Foo) and self.x == other.x and self.y == other.y


class AbstractFoo:
  pass


class FooHandler(handler_types.CheckpointableHandler[Foo, AbstractFoo]):

  async def save(
      self,
      directory: path_types.PathLike,
      checkpointable: Foo,
  ) -> Awaitable[None]:
    return DataclassHandler().background_save(
        directory, Foo(**dataclasses.asdict(checkpointable))
    )

  async def load(
      self,
      directory: path_types.PathLike,
      abstract_checkpointable: AbstractFoo | None = None,
  ) -> Awaitable[Foo]:
    return DataclassHandler().background_load(directory, Foo)

  async def metadata(self, directory: path_types.PathLike) -> AbstractFoo:
    return AbstractFoo()

  def is_handleable(self, checkpointable: Foo) -> bool:
    return isinstance(checkpointable, Foo)

  def is_abstract_handleable(
      self, abstract_checkpointable: AbstractFoo
  ) -> bool:
    return isinstance(abstract_checkpointable, AbstractFoo)


@dataclasses.dataclass
class Bar:
  a: int
  b: str

  def __eq__(self, other: Bar) -> bool:
    return isinstance(other, Bar) and self.a == other.a and self.b == other.b


class AbstractBar:
  pass


class BarHandler(handler_types.CheckpointableHandler[Bar, AbstractBar]):

  async def save(
      self,
      directory: path_types.PathLike,
      checkpointable: Bar,
  ) -> Awaitable[None]:
    return DataclassHandler().background_save(
        directory, Bar(**dataclasses.asdict(checkpointable))
    )

  async def load(
      self,
      directory: path_types.PathLike,
      abstract_checkpointable: AbstractBar | None = None,
  ) -> Awaitable[Bar]:
    return DataclassHandler().background_load(directory, Bar)

  async def metadata(self, directory: path_types.PathLike) -> AbstractBar:
    return AbstractBar()

  def is_handleable(self, checkpointable: Bar) -> bool:
    return isinstance(checkpointable, Bar)

  def is_abstract_handleable(
      self, abstract_checkpointable: AbstractBar
  ) -> bool:
    return isinstance(abstract_checkpointable, AbstractBar)


@dataclasses.dataclass
class Baz:
  int_val: int
  str_val: str

  def __eq__(self, other: Baz) -> bool:
    return (
        isinstance(other, Baz)
        and self.int_val == other.int_val
        and self.str_val == other.str_val
    )


class AbstractBaz:
  pass


class BazHandler(handler_types.CheckpointableHandler[Baz, AbstractBaz]):

  async def save(
      self,
      directory: path_types.PathLike,
      checkpointable: Baz,
  ) -> Awaitable[None]:
    return DataclassHandler().background_save(
        directory, Baz(**dataclasses.asdict(checkpointable))
    )

  async def load(
      self,
      directory: path_types.PathLike,
      abstract_checkpointable: AbstractBaz | None = None,
  ) -> Awaitable[Baz]:
    return DataclassHandler().background_load(directory, Baz)

  async def metadata(self, directory: path_types.PathLike) -> AbstractBaz:
    return AbstractBaz()

  def is_handleable(self, checkpointable: Baz) -> bool:
    return isinstance(checkpointable, Baz)

  def is_abstract_handleable(
      self, abstract_checkpointable: AbstractBaz
  ) -> bool:
    return isinstance(abstract_checkpointable, AbstractBaz)


BasicDict = dict[int | str | float, int | str | float]


class DictHandler(handler_types.CheckpointableHandler[BasicDict, None]):

  async def _background_save(self):
    pass

  async def _background_load(self, result: BasicDict) -> BasicDict:
    return result

  async def save(
      self,
      directory: path_types.PathLike,
      checkpointable: BasicDict,
  ) -> Awaitable[None]:
    directory = epath.Path(directory)
    async with aiofiles.open(directory / 'data.txt', 'w') as f:
      await f.write(str(dict(checkpointable)))
    return self._background_save()

  async def load(
      self,
      directory: path_types.PathLike,
      abstract_checkpointable: None = None,
  ) -> Awaitable[BasicDict]:
    directory = epath.Path(directory)
    async with aiofiles.open(directory / 'data.txt', 'r') as f:
      r = await f.read()
      result = dict(**r)
    return self._background_load(result)

  async def metadata(self, directory: path_types.PathLike) -> None:
    return None

  def is_handleable(self, checkpointable: BasicDict) -> bool:
    return isinstance(checkpointable, dict)

  def is_abstract_handleable(self, abstract_checkpointable: None) -> bool:
    return abstract_checkpointable is None
