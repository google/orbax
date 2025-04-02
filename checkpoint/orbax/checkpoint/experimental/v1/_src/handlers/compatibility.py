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
from orbax.checkpoint.experimental.v1._src.handlers import types as handler_types


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
    save_awaitable = await self._handler.save(directory, args.checkpointable)

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


@dataclasses.dataclass
class Args(checkpoint_args.CheckpointArgs):
  checkpointable: Any


def get_compatibility_handler(
    handler: handler_types.CheckpointableHandler,
) -> CompatibilityCheckpointHandler:

  class _CompatibilityHandler(CompatibilityCheckpointHandler):

    @classmethod
    def typestr(cls) -> str:
      return handler_types.typestr(type(handler))

  return _CompatibilityHandler(handler)
