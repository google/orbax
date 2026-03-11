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

"""Implementation of :py:class:`.CheckpointableHandler` for PyTrees."""

from __future__ import annotations

import json
import typing
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


@typing.final
class JsonHandler(CheckpointableHandler[JsonType, None]):
  """An implementation of :py:class:`.CheckpointableHandler` for Json.

  JsonHandler enables the persistence of standard Python structures (dicts,
  lists, and primitives) that are JSON-serializable. It utilizes an asynchronous
  two-tier execution model to offload I/O operations, ensuring background
  writing does not block the main process. It also provides multihost
  coordination to ensure that only the primary host performs the write
  operation.

  **Note: Users are encouraged NEVER to instantiate or use this handler
  directly.** Always use the top-level APIs like `ocp.save_checkpointables` and
  `ocp.load_checkpointables`. Orbax uses this handler by default for standard
  JSON-serializable objects.

  To save a custom JSON-serializable object (like a specific dictionary
  containing metadata) and aggressively force Orbax to use the JsonHandler,
  the recommended approach is to use `ocp.Context` with
  `CheckpointablesOptions`, which only applies to save/load operations
  strictly within the Context scope.

  See :py:class:`~orbax.checkpoint.options.CheckpointablesOptions` for more
  details on handler registration.

  Example Usage:
    Save a dictionary configuration::

      import orbax.checkpoint as ocp

      config = {'learning_rate': 0.01, 'batch_size': 32}

      checkpointables_options = (
          ocp.options.CheckpointablesOptions.create_with_handlers(
              experiment_config=ocp.handlers.JsonHandler(
                  filename='experiment_config.json'
              )
          )
      )
      with ocp.Context(checkpointables_options=checkpointables_options):
          ocp.save_checkpointables(path, dict(experiment_config=config))

  Attributes:
    filename: An optional specific filename to use for saving and loading the
      JSON data. If not provided, the handler will fall back to a default set
      of supported JSON filenames.
  """

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


@typing.final
class MetricsHandler(CheckpointableHandler[JsonType, None]):
  """An implementation of :py:class:`.CheckpointableHandler` for JSON metrics."""

  def __init__(self):
    self._handler = JsonHandler(filename='metrics')

  async def save(
      self, directory: path_types.PathAwaitingCreation, checkpointable: JsonType
  ) -> Awaitable[None]:
    return await self._handler.save(directory, checkpointable)

  async def load(
      self,
      directory: path_types.Path,
      abstract_checkpointable: None = None,
  ) -> Awaitable[JsonType]:
    return await self._handler.load(directory)

  async def metadata(self, directory: path_types.Path) -> None:
    return await self._handler.metadata(directory)

  def is_handleable(self, checkpointable: Any) -> bool:
    return self._handler.is_handleable(checkpointable)

  def is_abstract_handleable(self, abstract_checkpointable: Any) -> bool | None:
    return self._handler.is_abstract_handleable(abstract_checkpointable)
