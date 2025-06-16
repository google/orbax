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

"""Defines free-function interface for saving."""

import asyncio
import time
from typing import Any, Awaitable
import uuid

from absl import logging
from etils import epath
import jax
from orbax.checkpoint._src.checkpointers import async_checkpointer
from orbax.checkpoint._src.futures import future
from orbax.checkpoint._src.handlers import composite_checkpoint_handler
from orbax.checkpoint._src.handlers import handler_registration as legacy_handler_registration
from orbax.checkpoint._src.metadata import step_metadata_serialization
from orbax.checkpoint._src.path import atomicity
from orbax.checkpoint._src.path import atomicity_defaults
from orbax.checkpoint._src.path import atomicity_types
from orbax.checkpoint._src.path import utils as path_utils
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.handlers import compatibility as handler_compatibility
from orbax.checkpoint.experimental.v1._src.handlers import composite_handler
from orbax.checkpoint.experimental.v1._src.handlers import registration as handler_registration
from orbax.checkpoint.experimental.v1._src.handlers import types as handler_types
import orbax.checkpoint.experimental.v1._src.handlers.global_registration  # pylint: disable=unused-import
from orbax.checkpoint.experimental.v1._src.metadata import serialization as metadata_serialization
from orbax.checkpoint.experimental.v1._src.path import async_utils as path_async_utils
from orbax.checkpoint.experimental.v1._src.path import format_utils
from orbax.checkpoint.experimental.v1._src.path import types as path_types
from orbax.checkpoint.experimental.v1._src.serialization import registration as serialization_registration
from orbax.checkpoint.experimental.v1._src.synchronization import multihost
from orbax.checkpoint.experimental.v1._src.synchronization import thread_utils
from orbax.checkpoint.experimental.v1._src.synchronization import types as async_types
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types


PYTREE_CHECKPOINTABLE_KEY = format_utils.PYTREE_CHECKPOINTABLE_KEY
InternalCheckpointMetadata = (
    step_metadata_serialization.InternalCheckpointMetadata
)


async def _exists(path: path_types.Path) -> bool:
  return await asyncio.to_thread(path.exists)


async def _rmtree(path: path_types.Path) -> None:
  return await asyncio.to_thread(path.rmtree)


def save_pytree(
    path: path_types.PathLike,
    pytree: tree_types.PyTreeOf[tree_types.LeafType],
    *,
    overwrite: bool = False,
    custom_metadata: tree_types.JsonType | None = None,
):
  """Saves a PyTree.

  The operation blocks until complete. For improved performance, consider using
  `save_async` instead.

  Args:
    path: The path to save the checkpoint to.
    pytree: The PyTree to save. This may be any JAX PyTree (including custom
      objects registered as PyTrees) consisting of supported leaf types. Default
      supported leaf types include `jax.Array`, `np.ndarray`, simple types like
      `int`, `float`, `str`, and empty nodes. Support for custom leaves is also
      possible by implementing a `LeafTypeHandler`.
    overwrite: If True, fully overwrites an existing checkpoint in `path`.
      Otherwise, raises an error if the checkpoint already exists.
    custom_metadata: User-provided custom metadata. An arbitrary
      JSON-serializable dictionary the user can use to store additional
      information. The field is treated as opaque by Orbax.
  """
  save_checkpointables(
      path,
      {PYTREE_CHECKPOINTABLE_KEY: pytree},
      overwrite=overwrite,
      custom_metadata=custom_metadata,
  )


def save_checkpointables(
    path: path_types.PathLike,
    checkpointables: dict[str, Any],
    *,
    overwrite: bool = False,
    custom_metadata: tree_types.JsonType | None = None,
) -> None:
  """Saves a dictionary of checkpointables.

  A “checkpointable” refers to a logical piece of the checkpoint that is
  distinct in some way from other pieces. Checkpointables are separable;
  they may or may not be loaded concurrently and some may be omitted from the
  checkpoint entirely. Checkpointables are often represented by different types,
  and have different representations on disk. The quintessential example is
  model params vs. dataset.

  For example, one might do::

    ocp.save_checkpointables(
        path,
        {
            'params': pytree_of_arrays,
            'dataset': pygrain.DatasetIterator(...),
        }
    )

  It is also possible to do::

    train_state = {
        'params': params_pytree_of_arrays,
        'opt_state': opt_state_pytree_of_arrays,
        'step': step,
        ...
    }
    ocp.save_checkpointables(path, train_state)

  This is not the ideal way of doing things because it is then difficult to run
  transformations that involve the entire train state (see the
  `load_and_transform` API).

  Args:
    path: The path to save the checkpoint to.
    checkpointables: A dictionary of checkpointables. Dictionary keys represent
      the names of the checkpointables, while the values are the checkpointable
      objects themselves.
    overwrite: If True, fully overwrites an existing checkpoint in `path`.
      Otherwise, raises an error if the checkpoint already exists.
    custom_metadata: User-provided custom metadata. An arbitrary
      JSON-serializable dictionary the user can use to store additional
      information. The field is treated as opaque by Orbax.
  """
  _save_checkpointables_impl(
      path,
      checkpointables,
      overwrite=overwrite,
      custom_metadata=custom_metadata,
      async_origin=False,
  ).result()


# TODO(b/396190818): Test modification of the context by the user after the
# save operation is scheduled.
def save_pytree_async(
    path: path_types.PathLike,
    pytree: tree_types.PyTreeOf[tree_types.LeafType],
    *,
    overwrite: bool = False,
    custom_metadata: tree_types.JsonType | None = None,
) -> async_types.AsyncResponse[None]:
  """Saves a PyTree asynchronously.

  Unlike `save_pytree`, this function returns immediately after the save
  operation is scheduled
  (except for certain operations, like device-to-host copying of
  on-device arrays, which must happen on the main thread). Further writing
  operations continue in a background thread. An `AsyncResponse` is returned
  that can be used to block until the save is complete (using
  `response.result()`). Make sure to wait for completion before attempting to
  load the checkpoint or exiting the program.

  Args:
    path: The path to save the checkpoint to.
    pytree: The PyTree to save. This may be any JAX PyTree (including custom
      objects registered as PyTrees) consisting of supported leaf types. Default
      supported leaf types include `jax.Array`, `np.ndarray`, simple types like
      `int`, `float`, `str`, and empty nodes. Support for custom leaves is also
      possible by implementing a `LeafTypeHandler`.
    overwrite: If True, fully overwrites an existing checkpoint in `path`.
      Otherwise, raises an error if the checkpoint already exists.
    custom_metadata: User-provided custom metadata. An arbitrary
      JSON-serializable dictionary the user can use to store additional
      information. The field is treated as opaque by Orbax.

  Returns:
    An `AsyncResponse` that can be used to block until the save is complete.
    Blocking can be done using `response.result()`, which returns `None`.
  """
  return save_checkpointables_async(
      path,
      {PYTREE_CHECKPOINTABLE_KEY: pytree},
      overwrite=overwrite,
      custom_metadata=custom_metadata,
  )


def save_checkpointables_async(
    path: path_types.PathLike,
    checkpointables: dict[str, Any],
    *,
    overwrite: bool = False,
    custom_metadata: tree_types.JsonType | None = None,
) -> async_types.AsyncResponse[None]:
  """Saves a dictionary of checkpointables asynchronously.

  See `save_checkpointables` documentation.

  Unlike `save_checkpointables`, this function returns immediately after the
  save operation is scheduled
  (except for certain operations, like device-to-host copying of
  on-device arrays, which must happen on the main thread). Further writing
  operations continue in a background thread. An `AsyncResponse` is returned
  that can be used to block until the save is complete (using
  `response.result()`). Make sure to wait for completion before attempting to
  load the checkpoint or exiting the program.

  Args:
    path: The path to save the checkpoint to.
    checkpointables: A dictionary of checkpointables. Dictionary keys represent
      the names of the checkpointables, while the values are the checkpointable
      objects themselves.
    overwrite: If True, fully overwrites an existing checkpoint in `path`.
      Otherwise, raises an error if the checkpoint already exists.
    custom_metadata: User-provided custom metadata. An arbitrary
      JSON-serializable dictionary the user can use to store additional
      information. The field is treated as opaque by Orbax.

  Returns:
    An `AsyncResponse` that can be used to block until the save is complete.
    Blocking can be done using `response.result()`, which returns `None`.
  """
  return _save_checkpointables_impl(
      path,
      checkpointables,
      overwrite=overwrite,
      custom_metadata=custom_metadata,
      async_origin=True,
  )


def _get_temporary_path(
    path: path_types.Path, *, context: context_lib.Context
) -> atomicity_types.TemporaryPath:
  """Gets a TemporaryPath for the given path."""
  temporary_path_class = (
      context.file_options.temporary_path_class
      or atomicity_defaults.get_default_temporary_path_class(path)
  )
  tmpdir = temporary_path_class.from_final(
      path,
      # Ensure metadata store is NOT passed, to prevent separate metadata
      # writing.
      checkpoint_metadata_store=None,
      multiprocessing_options=context.multiprocessing_options.v0(),
      file_options=context.file_options.v0(),
  )
  return tmpdir


async def _remove_existing_path(
    path: path_types.Path,
    *,
    context: context_lib.Context,
):
  if multihost.is_primary_host(context.multiprocessing_options.primary_host):
    logging.info(
        '[process=%s] Specified `overwrite`: removing existing path.',
        multihost.process_index(),
    )
    await _rmtree(path)
  await multihost.sync_global_processes(
      multihost.unique_barrier_key(
          'save_checkpointables_async:rmtree',
          prefix=context.multiprocessing_options.barrier_sync_key_prefix,
      ),
      processes=context.multiprocessing_options.active_processes,
  )


class _SaveResponse(async_types.AsyncResponse[None]):
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
      # Properly, this should be an async function. For now, it's not a big
      # problem if it isn't though, since we have earlier async executions that
      # will yield control.
      atomicity.on_commit_callback(
          self._tmp_path,
          checkpoint_start_time=self._start_time,
      )

    # Clean up all awaitable signals for the current operation id as they are
    # no longer needed.
    if self._context.async_options.create_directories_asynchronously:
      future.remove_all_awaitable_signals(self._operation_id)

    await multihost.sync_global_processes(
        multihost.unique_barrier_key(
            'save_checkpointables_async:finalize',
            prefix=self._context.multiprocessing_options.barrier_sync_key_prefix,
        ),
        processes=self._context.multiprocessing_options.active_processes,
    )
    total_duration_secs = time.time() - self._start_time
    _record_save_completion(
        self._tmp_path.get_final(),
        total_duration_secs=total_duration_secs,
        async_origin=self._async_origin,
    )

  def result(self, timeout: float | None = None) -> None:
    return self._thread_runner.result(timeout=timeout)


def _record_save_start(path: path_types.Path, *, async_origin: bool):
  """Records the start of a save operation."""
  logging.info(
      '[process=%s] Started %s checkpoint to %s.',
      multihost.process_index(),
      'async saving' if async_origin else 'saving',
      path,
  )
  if async_origin:
    event_name = '/jax/orbax/write/async/start'
  else:
    event_name = '/jax/orbax/write/start'
  jax.monitoring.record_event(event_name)
  jax.monitoring.record_event(
      '/jax/orbax/write/storage_type',
      storage_type=path_utils.get_storage_type(path),
  )


def _record_save_completion(
    path: path_types.Path,
    *,
    total_duration_secs: float,
    async_origin: bool,
):
  """Records the completion of a save operation."""
  logging.info(
      'Finished asynchronous save (blocking + background) in %.2f seconds'
      ' to %s',
      total_duration_secs,
      path,
  )
  # TODO(cpgaffney): No event is currently being recorded for synchronous saves.
  # Consider collecting this information
  if async_origin:
    jax.monitoring.record_event_duration_secs(
        '/jax/checkpoint/write/async/total_duration_secs',
        total_duration_secs,
    )


def _save_checkpointables_impl(
    path: path_types.PathLike,
    checkpointables: dict[str, Any],
    *,
    async_origin: bool,
    overwrite: bool,
    custom_metadata: tree_types.JsonType | None,
) -> async_types.AsyncResponse[None]:
  """See caller docstrings."""
  path = epath.Path(path)
  # Prevent internal mutation from affecting the caller.
  checkpointables = dict(checkpointables)

  start_time = time.time()
  _record_save_start(path, async_origin=async_origin)
  context = context_lib.get_context()
  checkpointables_handler = composite_handler.CompositeHandler(
      context.checkpointables_options.registry
  )
  checkpointables = _add_internal_checkpointables(
      checkpointables, context=context
  )
  tmp_path = _get_temporary_path(path, context=context)

  async def _blocking_save() -> Awaitable[None]:
    await context_lib.synchronize_next_operation_id()
    if await _exists(path):
      if overwrite:
        await _remove_existing_path(path, context=context)
      else:
        raise ValueError(f'Destination {path} already exists.')

    tmp_path_awaiting_creation = path_async_utils.start_async_mkdir(
        tmp_path, checkpointables.keys()
    )
    if not context.async_options.create_directories_asynchronously:
      await tmp_path_awaiting_creation.await_creation()

    # Synchronous portion of the save.
    background_awaitable = await checkpointables_handler.save(
        tmp_path_awaiting_creation, checkpointables
    )
    return background_awaitable

  background_awaitable = asyncio.run(_blocking_save())
  blocking_duration_secs = time.time() - start_time
  jax.monitoring.record_event_duration_secs(
      '/jax/checkpoint/write/async/blocking_duration_secs',
      blocking_duration_secs,
  )
  logging.info(
      'Finished blocking save in %.2f seconds. Continuing to write to %s.',
      blocking_duration_secs,
      path,
  )

  handler_typestrs = {
      name: handler_types.typestr(type(handler))
      for name, handler in checkpointables_handler.get_handlers_for_save(
          checkpointables
      ).items()
  }
  return _SaveResponse(
      context.operation_id(),
      tmp_path,
      handler_typestrs,
      background_awaitable,
      start_time=start_time,
      custom_metadata=custom_metadata,
      context=context,
      async_origin=async_origin,
  )


def _add_internal_checkpointables(
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


def get_v0_checkpointer_and_args(
    checkpointables: dict[str, Any],
    *,
    metrics: tree_types.JsonType | None = None,
    context: context_lib.Context,
) -> tuple[
    async_checkpointer.AsyncCheckpointer,
    composite_checkpoint_handler.CompositeArgs,
]:
  """Construct V0 Checkpointer and Args for saving."""
  if (
      provided_reserved_keys := checkpointables.keys()
      & format_utils.RESERVED_CHECKPOINTABLE_KEYS
  ):
    raise ValueError(
        f'Provided reserved checkpointable keys: {provided_reserved_keys}.'
    )
  checkpointables = _add_internal_checkpointables(
      checkpointables, context=context, metrics=metrics
  )

  handlers = {
      name: handler_registration.resolve_handler_for_save(
          context.checkpointables_options.registry, checkpointable, name=name
      )
      for name, checkpointable in checkpointables.items()
  }
  compatibility_handlers = {
      name: handler_compatibility.get_compatibility_handler(handler)
      for name, handler in handlers.items()
  }
  handler_registry = (
      legacy_handler_registration.DefaultCheckpointHandlerRegistry()
  )
  for name, handler in compatibility_handlers.items():
    handler_registry.add(name, handler_compatibility.Args, handler)
  composite_options = composite_checkpoint_handler.CompositeOptions(
      async_options=context.async_options.v0(),
      file_options=context.file_options.v0(),
      multiprocessing_options=context.multiprocessing_options.v0(),
      temporary_path_class=context.file_options.temporary_path_class,
  )
  ckptr = async_checkpointer.AsyncCheckpointer(
      composite_checkpoint_handler.CompositeCheckpointHandler(
          handler_registry=handler_registry,
          composite_options=composite_options,
      ),
      async_options=context.async_options.v0(),
      multiprocessing_options=context.multiprocessing_options.v0(),
      file_options=context.file_options.v0(),
      temporary_path_class=context.file_options.temporary_path_class,
  )
  args = composite_checkpoint_handler.CompositeArgs(**{
      name: handler_compatibility.Args(checkpointable)
      for name, checkpointable in checkpointables.items()
  })
  return ckptr, args
