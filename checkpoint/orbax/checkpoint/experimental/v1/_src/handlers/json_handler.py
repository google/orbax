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

"""Implementation of `CheckpointableHandler` for PyTrees."""

from __future__ import annotations

import contextlib
from typing import Any, Awaitable, Mapping, Optional, Sequence

from etils import epath
from orbax.checkpoint import options as v0_options_lib
from orbax.checkpoint._src.futures import future
from orbax.checkpoint._src.futures import synchronization
from orbax.checkpoint._src.handlers import json_checkpoint_handler
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.context import options as options_lib
from orbax.checkpoint.experimental.v1._src.handlers import registration
from orbax.checkpoint.experimental.v1._src.handlers import types as handler_types
from orbax.checkpoint.experimental.v1._src.path import types as path_types

PathLike = path_types.PathLike
CheckpointableHandler = handler_types.CheckpointableHandler
Json = Mapping[str, Any]

JSON_CHECKPOINTABLE_KEY = 'json'


def create_v0_handler(
    context: context_lib.Context,
    filename: Optional[str] = None,
) -> json_checkpoint_handler.JsonCheckpointHandler:
  """Creates a V0 handler from a V1 context."""
  return json_checkpoint_handler.JsonCheckpointHandler(
      filename=filename,
      multiprocessing_options=v0_options_lib.MultiprocessingOptions(
          primary_host=context.multiprocessing_options.primary_host,
          active_processes=context.multiprocessing_options.active_processes,
          barrier_sync_key_prefix=context.multiprocessing_options.barrier_sync_key_prefix,
      ),
  )


def create_v0_save_args(
    checkpointable: Json,
) -> json_checkpoint_handler.JsonSaveArgs:
  """Creates v0 CheckpointArgs for saving."""
  return json_checkpoint_handler.JsonSaveArgs(item=checkpointable)


def create_v0_restore_args(
    abstract_checkpointable: Json | None,
) -> json_checkpoint_handler.JsonRestoreArgs:
  """Creates v0 CheckpointArgs for restoration."""
  return json_checkpoint_handler.JsonRestoreArgs(item=abstract_checkpointable)


class JsonHandler(CheckpointableHandler[Json, Json]):
  """An implementation of `CheckpointableHandler` for Json."""

  def __init__(
      self,
      *,
      context: context_lib.Context | None = None,
      filename: Optional[str] = None,
  ):
    context = context_lib.get_context(context)
    self._context = context
    self._filename = filename or 'metadata'
    self._multiprocessing_options = context.multiprocessing_options
    self._handler_impl = create_v0_handler(
        context,
        filename,
    )

  async def _background_save(
      self,
      directory: path_types.Path,
      commit_futures: Sequence[future.Future],
      operation_id: str,
  ):
    active_processes = self._multiprocessing_options.active_processes or set(
        range(multihost.process_count())
    )
    for f in commit_futures:
      f.result()
    # Global sync to ensure all participating processes have completed their
    # save operations before proceeding to finalize.
    barrier_name = f'save_and_finalize_{operation_id}_commit_complete'
    multihost.sync_global_processes(barrier_name, processes=active_processes)
    # Finalize.
    # Global sync to ensure all hosts are aware that the finalize operation
    # has completed before returning to the user.
    barrier_name = f'save_and_finalize_{operation_id}_finalize_complete'
    multihost.sync_global_processes(barrier_name, processes=active_processes)

  async def save(
      self, directory: path_types.PathLike, checkpointable: Json
  ) -> Awaitable[None]:
    directory = epath.Path(directory)
    commit_futures = await self._handler_impl.async_save(
        directory,
        args=create_v0_save_args(checkpointable),
    )
    assert commit_futures

    # TODO(b/398310070): Move operation ID generation to `Context`.
    operation_id = (
        synchronization.HandlerAwaitableSignalOperationIdGenerator.get_current_operation_id()
    )
    # Needed to differentiate between different handlers when we have multiple
    # PyTreeHandlers performing a save.
    operation_id = f'{operation_id}.{directory.name}'
    return self._background_save(
        directory=directory,
        commit_futures=commit_futures,
        operation_id=operation_id,
    )

  async def _background_load(
      self,
      directory: path_types.Path,
      abstract_checkpointable: Json | None = None,
  ) -> Json:
    return self._handler_impl.restore(
        directory,
        args=create_v0_restore_args(abstract_checkpointable),
    )

  async def load(
      self,
      directory: path_types.PathLike,
      abstract_checkpointable: Json | None = None,
  ) -> Awaitable[Json]:
    directory = epath.Path(directory)
    # TODO(b/406252214): Add validation for Json.
    return self._background_load(directory, abstract_checkpointable)


@contextlib.contextmanager
def json_handler_context():
  """Creates a local context where only `JsonHandler` is registered."""
  # TODO(b/398310070): Verify behavior with nested Contexts.
  checkpointables_options = options_lib.CheckpointablesOptions(
      registry=registration.local_registry(include_global_registry=True).add(
          JsonHandler,
          JSON_CHECKPOINTABLE_KEY,
      )
  )
  with context_lib.Context(
      context_lib.get_context(), checkpointables_options=checkpointables_options
  ) as new_context:
    yield new_context
