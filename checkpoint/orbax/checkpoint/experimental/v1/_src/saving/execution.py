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

"""Internal utilities for saving whole and partial checkpoints."""

import asyncio
import time
from typing import Any, Awaitable, Iterable
import uuid

from absl import logging
from etils import epath
import jax
import nest_asyncio
from orbax.checkpoint._src.futures import future
from orbax.checkpoint._src.logging import event_tracking
from orbax.checkpoint._src.metadata import step_metadata_serialization
from orbax.checkpoint._src.path import atomicity
from orbax.checkpoint._src.path import atomicity_types
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.handlers import composite_handler
from orbax.checkpoint.experimental.v1._src.handlers import types as handler_types
from orbax.checkpoint.experimental.v1._src.metadata import serialization as metadata_serialization
from orbax.checkpoint.experimental.v1._src.path import async_utils as path_async_utils
from orbax.checkpoint.experimental.v1._src.path import format_utils
from orbax.checkpoint.experimental.v1._src.path import types as path_types
from orbax.checkpoint.experimental.v1._src.saving import path_utils as saving_path_utils
from orbax.checkpoint.experimental.v1._src.synchronization import multihost
from orbax.checkpoint.experimental.v1._src.synchronization import thread_utils
from orbax.checkpoint.experimental.v1._src.synchronization import types as async_types
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types


InternalCheckpointMetadata = (
    step_metadata_serialization.InternalCheckpointMetadata
)


def add_internal_checkpointables(
    checkpointables: dict[str, Any],
    *,
    context: context_lib.Context,
    metrics: tree_types.JsonType | None = None,
) -> dict[str, Any]:
  """Adds descriptor to checkpointables if enabled."""
  # Global registration ties metrics key to JsonHandler.
  if metrics:
    checkpointables[format_utils.METRICS_CHECKPOINTABLE_KEY] = metrics
  return checkpointables


class SaveResponse(async_types.AsyncResponse[None]):
  """An `AsyncResponse` representing the result of `save_pytree_async`."""

  def __init__(
      self,
      operation_id: str,
      tmp_path: atomicity_types.TemporaryPath,
      handler_typestrs: dict[str, str],
      background_awaitable: Awaitable[None],
      *,
      start_time: float,
      custom_metadata: tree_types.JsonType | None,
      context: context_lib.Context,
      async_origin: bool,
  ):
    self._operation_id = operation_id
    self._tmp_path = tmp_path
    self._handler_typestrs = handler_typestrs
    self._background_awaitable = background_awaitable
    self._start_time = start_time
    self._custom_metadata = custom_metadata
    self._context = context
    self._async_origin = async_origin
    self._thread_runner = thread_utils.BackgroundThreadRunner[None](
        self._finalize_save()
    )

  async def _finalize_save(self):
    logging.info(
        '[process=%s] Finalizing checkpoint on %s',
        multihost.process_index(),
        self._tmp_path.get(),
    )
    await self._background_awaitable
    logging.vlog(
        1,
        '[process=%s] Finished waiting for background save operations.',
        multihost.process_index(),
    )

    if multihost.is_primary_host(
        self._context.multiprocessing_options.primary_host
    ):
      logging.vlog(
          1,
          '[process=%s] Writing checkpoint metadata.',
          multihost.process_index(),
      )
      internal_metadata = InternalCheckpointMetadata.create(
          handler_typestrs=self._handler_typestrs,
          init_timestamp_nsecs=int(self._start_time * 1e9),
          commit_timestamp_nsecs=time.time_ns(),
          custom_metadata=self._custom_metadata,
      )
      await metadata_serialization.write(
          metadata_serialization.checkpoint_metadata_file_path(
              self._tmp_path.get()
          ),
          internal_metadata.serialize(),
      )
      logging.vlog(
          1,
          '[process=%s] Finished writing checkpoint metadata.',
          multihost.process_index(),
      )
      await atomicity.on_commit_callback(
          self._tmp_path,
          checkpoint_start_time=self._start_time,
      )

    # Clean up all awaitable signals for the current operation id as they are
    # no longer needed.
    if self._context.async_options.create_directories_asynchronously:
      future.AwaitableSignalsContract.remove_all_awaitable_signals(
          self._operation_id
      )

    await multihost.sync_global_processes(
        multihost.unique_barrier_key(
            'save_checkpointables_async:finalize',
            prefix=self._context.multiprocessing_options.barrier_sync_key_prefix,
        ),
        processes=self._context.multiprocessing_options.active_processes,
    )
    total_duration_secs = time.time() - self._start_time
    event_tracking.record_save_completion(
        self._tmp_path.get_final(),
        total_duration_secs=total_duration_secs,
        async_origin=self._async_origin,
    )

  def result(self, timeout: float | None = None) -> None:
    return self._thread_runner.result(timeout=timeout)


async def run_blocking_save(
    tmp_path: atomicity_types.TemporaryPath,
    checkpointables: dict[str, Any],
    subdirectories: Iterable[str],
    *,
    overwrite: bool,
    context: context_lib.Context,
    partial_save: bool = False,
) -> Awaitable[None]:
  """Runs the synchronous portion of the save operation.

  This includes directory creation and calling the handler's save method.

  Args:
    tmp_path: The temporary path to save the checkpointables to.
    checkpointables: A mapping from checkpointable name to checkpointable.
    subdirectories: A list of subdirectories to create under `tmp_path`.
    overwrite: Whether to overwrite an existing checkpoint in `tmp_path`.
    context: The context to use for the save operation.
    partial_save: Whether to save the checkpoint in partial mode.

  Returns:
    An awaitable that will be completed when the synchronous portion of the
    save operation is complete.
  """
  await context_lib.synchronize_next_operation_id()

  if not partial_save:
    await saving_path_utils.maybe_overwrite_existing(
        tmp_path.get_final(), overwrite=overwrite, context=context
    )

  handler = composite_handler.CompositeHandler(
      context.checkpointables_options.registry
  )

  # Directory creation is handled here.
  tmp_path_awaiting_creation = path_async_utils.start_async_mkdir(
      tmp_path, subdirectories
  )
  if (
      partial_save
      or not context.async_options.create_directories_asynchronously
  ):
    await tmp_path_awaiting_creation.await_creation()

  # Delegate to the handler to get the background awaitable.
  background_awaitable = await handler.save(
      tmp_path_awaiting_creation, checkpointables
  )
  # Log write event for the final path.
  event_tracking.record_write_event(tmp_path.get_final())

  return background_awaitable


def create_save_response(
    background_awaitable: Awaitable[None],
    checkpointables: dict[str, Any],
    tmp_path: atomicity_types.TemporaryPath,
    start_time: float,
    *,
    context: context_lib.Context,
    custom_metadata: tree_types.JsonType | None,
    async_origin: bool,
) -> async_types.AsyncResponse[None]:
  """Creates and returns the final AsyncResponse for a save operation."""
  blocking_duration_secs = time.time() - start_time
  jax.monitoring.record_event_duration_secs(
      '/jax/checkpoint/write/async/blocking_duration_secs',
      blocking_duration_secs,
  )
  logging.info(
      'Finished blocking save in %.2f seconds. Continuing to write to %s.',
      blocking_duration_secs,
      tmp_path.get_final(),
  )

  handler = composite_handler.CompositeHandler(
      context.checkpointables_options.registry
  )
  handler_typestrs = {
      name: handler_types.typestr(type(handler))
      for name, handler in handler.get_handlers_for_save(
          checkpointables
      ).items()
  }

  return SaveResponse(
      context.operation_id(),
      tmp_path,
      handler_typestrs,
      background_awaitable,
      start_time=start_time,
      custom_metadata=custom_metadata,
      context=context,
      async_origin=async_origin,
  )


def save_checkpointables_impl(
    path: path_types.PathLike,
    checkpointables: dict[str, Any],
    *,
    async_origin: bool,
    overwrite: bool,
    custom_metadata: tree_types.JsonType | None,
    partial_save: bool = False,
) -> async_types.AsyncResponse[None]:
  """See caller docstrings."""
  nest_asyncio.apply()
  context = context_lib.get_context()
  path = epath.Path(path)
  path_exists = path.exists() if partial_save else False
  # Prevent internal mutation from affecting the caller.
  checkpointables = dict(checkpointables)

  start_time = time.time()
  event_tracking.record_save_start(path, async_origin=async_origin)

  tmp_path = saving_path_utils.get_temporary_path(
      path, context=context, use_snapshot=path_exists
  )

  checkpointables = add_internal_checkpointables(
      checkpointables, context=context
  )
  subdirectories = [] if path_exists else checkpointables.keys()

  background_awaitable = asyncio.run(
      run_blocking_save(
          tmp_path,
          checkpointables,
          subdirectories,
          overwrite=overwrite,
          context=context,
          partial_save=partial_save,
      )
  )

  return create_save_response(
      background_awaitable,
      checkpointables,
      tmp_path,
      start_time,
      context=context,
      custom_metadata=custom_metadata,
      async_origin=async_origin,
  )
