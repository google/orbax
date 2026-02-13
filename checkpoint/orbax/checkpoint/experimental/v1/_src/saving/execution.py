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

"""Internal utilities for saving whole and partial checkpoints."""

from __future__ import annotations

import asyncio
import hashlib
import time
from typing import Any, Awaitable, Iterable
import uuid

from absl import logging
import jax
import numpy as np
from orbax.checkpoint._src.futures import future
from orbax.checkpoint._src.logging import event_tracking
from orbax.checkpoint._src.metadata import step_metadata_serialization
from orbax.checkpoint._src.path import atomicity
from orbax.checkpoint._src.path import atomicity_types
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.handlers import composite_handler
from orbax.checkpoint.experimental.v1._src.handlers import types as handler_types
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout
from orbax.checkpoint.experimental.v1._src.layout import registry
from orbax.checkpoint.experimental.v1._src.metadata import serialization as metadata_serialization
from orbax.checkpoint.experimental.v1._src.path import async_utils as path_async_utils
from orbax.checkpoint.experimental.v1._src.path import types as path_types
from orbax.checkpoint.experimental.v1._src.saving import path_utils as saving_path_utils
from orbax.checkpoint.experimental.v1._src.synchronization import multihost
from orbax.checkpoint.experimental.v1._src.synchronization import thread_utils
from orbax.checkpoint.experimental.v1._src.synchronization import types as async_types
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types


InternalCheckpointMetadata = (
    step_metadata_serialization.InternalCheckpointMetadata
)
AsyncResponse = async_types.AsyncResponse


def _should_create_directories_synchronously(
    context: context_lib.Context, partial_save: bool
):
  return (
      partial_save
      or not context.async_options.create_directories_asynchronously
  )


def add_internal_checkpointables(
    checkpointables: dict[str, Any],
    *,
    context: context_lib.Context,
    metrics: tree_types.JsonType | None = None,
) -> dict[str, Any]:
  """Adds a descriptor to checkpointables if enabled.

  Args:
    checkpointables: A dictionary of checkpointables.
    context: The Orbax context.
    metrics: Optional metrics to add to the checkpointables.

  Returns:
    The updated dictionary of checkpointables.
  """
  # Global registration ties metrics key to JsonHandler.
  if metrics:
    checkpointables[checkpoint_layout.METRICS_CHECKPOINTABLE_KEY] = metrics
  return checkpointables


class _SaveResponse(AsyncResponse[None]):
  """An :py:class:`.AsyncResponse` representing the result of:py:func:`.save_pytree_async`."""

  def __init__(
      self,
      operation_id: str,
      temporary_path: _TemporaryPathAwaitingCreation,
      handler_typestrs: dict[str, str],
      background_awaitable: Awaitable[None],
      *,
      start_time: float,
      custom_metadata: tree_types.JsonType | None,
      context: context_lib.Context,
      async_origin: bool,
  ):
    self._operation_id = operation_id
    self._temporary_path = temporary_path
    self._handler_typestrs = handler_typestrs
    self._background_awaitable = background_awaitable
    self._start_time = start_time
    self._custom_metadata = custom_metadata
    self._context = context
    self._async_origin = async_origin
    self._thread_runner = thread_utils.BackgroundThreadRunner[None](
        self._finalize_save()
    )

  @classmethod
  def create(
      cls,
      background_awaitable: Awaitable[None],
      checkpointables: dict[str, Any],
      temporary_path: _TemporaryPathAwaitingCreation,
      start_time: float,
      *,
      context: context_lib.Context,
      custom_metadata: tree_types.JsonType | None,
      async_origin: bool,
  ) -> _SaveResponse:
    """Creates and returns the final AsyncResponse for a save operation."""
    blocking_duration_secs = time.time() - start_time
    jax.monitoring.record_event_duration_secs(
        '/jax/checkpoint/write/async/blocking_duration_secs',
        blocking_duration_secs,
    )
    logging.info(
        'Finished blocking save in %.2f seconds. Continuing to write to %s.',
        blocking_duration_secs,
        temporary_path.temporary_path.get_final(),
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

    return cls(
        context.operation_id(),
        temporary_path,
        handler_typestrs,
        background_awaitable,
        start_time=start_time,
        custom_metadata=custom_metadata,
        context=context,
        async_origin=async_origin,
    )

  async def _finalize_save(self):
    logging.info(
        '[process=%s] Creating directories on %s',
        multihost.process_index(),
        self._temporary_path.temporary_path.get(),
    )
    await self._temporary_path.path_awaiting_creation.create()
    logging.info(
        '[process=%s] Waiting for background save operations',
        multihost.process_index(),
    )
    await self._background_awaitable
    logging.vlog(
        1,
        '[process=%s] Finished waiting for background save operations.',
        multihost.process_index(),
    )
    logging.info(
        '[process=%s] Finalizing checkpoint on %s',
        multihost.process_index(),
        self._temporary_path.temporary_path.get(),
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
              self._temporary_path.temporary_path.get()
          ),
          internal_metadata.serialize(),
      )
      logging.vlog(
          1,
          '[process=%s] Finished writing checkpoint metadata.',
          multihost.process_index(),
      )
      await atomicity.on_commit_callback(
          self._temporary_path.temporary_path,
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
        operation_id=self._operation_id,
        processes=self._context.multiprocessing_options.active_processes,
    )
    total_duration_secs = time.time() - self._start_time
    event_tracking.record_save_completion(
        self._temporary_path.temporary_path.get_final(),
        total_duration_secs=total_duration_secs,
        async_origin=self._async_origin,
    )

  def result(self, timeout: float | None = None) -> None:
    return self._thread_runner.result(timeout=timeout)


async def _run_blocking_save(
    temporary_path: _TemporaryPathAwaitingCreation,
    checkpointables: dict[str, Any],
    *,
    overwrite: bool,
    context: context_lib.Context,
    partial_save: bool = False,
) -> Awaitable[None]:
  """Runs the synchronous portion of the save operation.

  This includes directory creation and calling the handler's save method.

  Args:
    temporary_path: The temporary path to save the checkpointables to.
    checkpointables: A mapping from checkpointable name to checkpointable.
    overwrite: Whether to overwrite an existing checkpoint in `tmp_path`.
    context: The context to use for the save operation.
    partial_save: Whether to save the checkpoint in partial mode.

  Returns:
    An awaitable that will be completed when the synchronous portion of the save
    operation is complete.
  """
  if not partial_save:
    await saving_path_utils.maybe_overwrite_existing(
        temporary_path.temporary_path.get_final(),
        overwrite=overwrite,
        context=context,
    )

  layout_enum = context.checkpoint_layout
  layout_class = await registry.get_layout_class(layout_enum)
  layout = layout_class()
  if (
      partial_save
      or not context.async_options.create_directories_asynchronously
  ):
    await temporary_path.path_awaiting_creation.create()

  if partial_save:
    await multihost.sync_global_processes(
        multihost.unique_barrier_key(
            'save_checkpointables_async:run_blocking_save:partial_save',
            prefix=context.multiprocessing_options.barrier_sync_key_prefix,
        ),
        operation_id=context.operation_id(),
        processes=context.multiprocessing_options.active_processes,
    )

  # Delegate to the handler to get the background awaitable.
  background_awaitable = await layout.save(
      path=temporary_path.path_awaiting_creation,
      checkpointables=checkpointables,
  )
  # Log write event for the final path.
  event_tracking.record_write_event(temporary_path.temporary_path.get_final())

  return background_awaitable


def _check_directory_consistency(directory: path_types.PathLike):
  """Raises error if directory paths are not consistent across processes.

  Args:
    directory: The directory path to check.

  Raises:
    ValueError: If the directory paths are not consistent across processes.
  """
  if multihost.process_count() <= 1:
    return

  path_str = str(directory)
  path_hash = hashlib.sha256(path_str.encode('utf-8')).digest()
  path_hash_arr = np.frombuffer(path_hash, dtype=np.uint8)

  # Broadcast the path hash from process 0 to all other processes.
  broadcasted_hash_arr = multihost.broadcast_one_to_all(path_hash_arr)

  # Gather mismatch status from all processes.
  mismatch_detected = np.array(
      0 if np.array_equal(path_hash_arr, broadcasted_hash_arr) else 1,
      dtype=np.int32,
  )
  all_mismatches = multihost.process_allgather(mismatch_detected)
  total_mismatches = np.sum(np.array(all_mismatches))

  if total_mismatches > 0:
    raise ValueError(
        'Directory path mismatch in multi-process save. '
        f"Process {jax.process_index()} has path '{path_str}'. (See logs from "
        'other processes for their paths.) Ensure all JAX processes are saving '
        'to the exact same directory path. If using create_tempdir in tests, '
        "provide the 'name' argument to ensure all processes generate the same "
        'path.'
    )


class _TemporaryPathAwaitingCreation:
  """A simple container for `PathAwaitingCreation` and `TemporaryPath`."""

  def __init__(
      self,
      path: path_types.Path,
      subdirectories: Iterable[str],
      *,
      use_snapshot: bool,
  ):
    self._temporary_path = saving_path_utils.get_temporary_path(
        path, context=context_lib.get_context(), use_snapshot=use_snapshot
    )
    self._temporary_path_awaiting_creation = (
        path_async_utils.PathAwaitingCreation.build(
            self._temporary_path,
            subdirectories,
        )
    )
    assert (
        self._temporary_path_awaiting_creation.path
        == self._temporary_path.get()
    )

  @property
  def path_awaiting_creation(self) -> path_async_utils.PathAwaitingCreation:
    return self._temporary_path_awaiting_creation

  @property
  def temporary_path(self) -> atomicity_types.TemporaryPath:
    return self._temporary_path


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
  start_time = time.time()
  event_tracking.record_save_start(path, async_origin=async_origin)
  # Ensure the operation ID is incremented as soon as possible. This must be
  # done uniquely for each save operation.
  asyncio.run(context_lib.synchronize_next_operation_id())
  context = context_lib.get_context()

  path = context.file_options.path_class(path)
  _check_directory_consistency(path)
  path_exists = path.exists() if partial_save else False
  # Prevent internal mutation from affecting the caller.
  checkpointables = dict(checkpointables)
  checkpointables = add_internal_checkpointables(
      checkpointables, context=context
  )
  subdirectories = [] if path_exists else checkpointables.keys()
  temporary_path = _TemporaryPathAwaitingCreation(
      path,
      subdirectories=subdirectories,
      use_snapshot=path_exists,
  )
  background_awaitable = asyncio.run(
      _run_blocking_save(
          temporary_path,
          checkpointables,
          overwrite=overwrite,
          context=context,
          partial_save=partial_save,
      )
  )
  return _SaveResponse.create(
      background_awaitable,
      checkpointables,
      temporary_path,
      start_time,
      context=context,
      custom_metadata=custom_metadata,
      async_origin=async_origin,
  )
