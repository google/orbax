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

"""V0 / V1 handler compatibility constructs."""

from __future__ import annotations

import abc
import dataclasses
from typing import Any

from etils import epath
from orbax.checkpoint import checkpoint_args
from orbax.checkpoint._src import asyncio_utils
from orbax.checkpoint._src.futures import future
from orbax.checkpoint._src.handlers import async_checkpoint_handler
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.handlers import types as handler_types
from orbax.checkpoint.experimental.v1._src.path import types as path_types
from orbax.checkpoint.experimental.v1._src.synchronization import synchronization


class _PathAwaitingCreation(path_types.PathAwaitingCreation):
  """Implementation of `PathAwaitingCreation` that awaits contracted signals."""

  def __init__(self, path: path_types.Path, operation_id: str):
    self._path = path
    self._operation_id = operation_id

  def __truediv__(
      self, other: path_types.PathAwaitingCreation | path_types.PathLike
  ) -> path_types.PathAwaitingCreation:
    if isinstance(other, path_types.PathAwaitingCreation):
      other = other.path
    return _PathAwaitingCreation(self._path / other, self._operation_id)

  async def await_creation(self) -> path_types.Path:
    await synchronization.await_contracted_signals(self._operation_id)
    return self._path

  @property
  def path(self) -> path_types.Path:
    return self._path


class CompatibilityCheckpointHandler(
    async_checkpoint_handler.AsyncCheckpointHandler
):
  """A V0 CheckpointHandler that wraps a V1 CheckpointableHandler."""

  def __init__(self, handler: handler_types.CheckpointableHandler):
    self._handler = handler

  async def async_save(
      self,
      directory: epath.Path,
      args: Args,
  ) -> list[future.Future] | None:
    async_path = _PathAwaitingCreation(
        directory, context_lib.get_context().operation_id()
    )
    save_awaitable = await self._handler.save(async_path, args.checkpointable)

    async def _background_save():
      await save_awaitable

    return [future.CommitFuture(_background_save())]

  def save(self, directory: epath.Path, *args, **kwargs):
    async def async_save(*args, **kwargs):
      commit_futures = await self.async_save(*args, **kwargs)  # pytype: disable=bad-return-type
      # Futures are already running, so sequential waiting is equivalent to
      # concurrent waiting.
      if commit_futures:  # May be None.
        for f in commit_futures:
          f.result()  # Block on result.

    asyncio_utils.run_sync(async_save(directory, *args, **kwargs))

  def restore(self, directory: epath.Path, args: Args | None = None) -> Any:
    abstract_checkpointable = args.checkpointable if args else None

    async def _synchronous_load():
      load_awaitable = await self._handler.load(
          directory, abstract_checkpointable
      )
      return await load_awaitable

    return asyncio_utils.run_sync(_synchronous_load())

  def metadata(self, directory: epath.Path) -> Any | None:
    return asyncio_utils.run_sync(self._handler.metadata(directory))

  def finalize(self, directory: epath.Path) -> None:
    pass

  def close(self):
    pass

  @classmethod
  @abc.abstractmethod
  def typestr(cls) -> str:
    """A unique identifier for the CheckpointHandler type."""
    ...

  def __repr__(self):
    return f'CompatibilityCheckpointHandler({handler_types.typestr(type(self._handler))})'


@dataclasses.dataclass
class Args(checkpoint_args.CheckpointArgs):
  checkpointable: Any

  def __repr__(self):
    return f'CompatibilityArgs({type(self.checkpointable)})'


def get_compatibility_handler(
    handler: handler_types.CheckpointableHandler,
) -> CompatibilityCheckpointHandler:

  class _CompatibilityHandler(CompatibilityCheckpointHandler):

    @classmethod
    def typestr(cls) -> str:
      return handler_types.typestr(type(handler))

  return _CompatibilityHandler(handler)
