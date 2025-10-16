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

"""Implementation of `CheckpointableHandler` for PyTrees."""

from __future__ import annotations

import json
from typing import Any, Awaitable

from orbax.checkpoint._src.path import async_path
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.handlers import types as handler_types
from orbax.checkpoint.experimental.v1._src.path import types as path_types
from orbax.checkpoint.experimental.v1._src.synchronization import multihost
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types


CheckpointableHandler = handler_types.CheckpointableHandler
JsonType = tree_types.JsonType


_DATA_FILENAME = 'data.json'


def _get_supported_filenames(filename: str | None = None) -> list[str]:
  filename = filename or _DATA_FILENAME
  return [filename, _DATA_FILENAME, 'metadata']


class JsonHandler(CheckpointableHandler[JsonType, None]):
  """An implementation of `CheckpointableHandler` for Json."""

  def __init__(self, filename: str | None = None):
    self._supported_filenames = _get_supported_filenames(filename)
    self._filename = self._supported_filenames[0]

  async def _background_save(
      self,
      directory: path_types.PathAwaitingCreation,
      checkpointable: JsonType,
      *,
      primary_host: int | None = None,
  ):
    directory = await directory.await_creation()
    if multihost.is_primary_host(primary_host):
      path = directory / self._filename
      await async_path.write_text(path, json.dumps(checkpointable))

  async def save(
      self, directory: path_types.PathAwaitingCreation, checkpointable: JsonType
  ) -> Awaitable[None]:
    context = context_lib.get_context()
    return self._background_save(
        directory=directory,
        checkpointable=checkpointable,
        primary_host=context.multiprocessing_options.primary_host,
    )

  async def _background_load(
      self,
      directory: path_types.Path,
  ):
    for filename in self._supported_filenames:
      path = directory / filename
      if await async_path.exists(path):
        return json.loads(await async_path.read_text(path))
    raise FileNotFoundError(
        f'Unable to parse JSON file in {directory}. Recognized filenames are:'
        f' {self._supported_filenames}'
    )

  async def load(
      self,
      directory: path_types.Path,
      abstract_checkpointable: None = None,
  ) -> Awaitable[JsonType]:
    return self._background_load(directory)

  async def metadata(self, directory: path_types.Path) -> None:
    return None

  def is_handleable(self, checkpointable: Any) -> bool:
    try:
      json.loads(json.dumps(checkpointable))
      return True
    except Exception:  # pylint: disable=broad-exception-caught
      return False

  def is_abstract_handleable(self, abstract_checkpointable: Any) -> bool | None:
    return None


class MetricsHandler(JsonHandler):

  def __init__(self):
    super().__init__(filename='metrics')
