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

"""Orbax context for customized checkpointing."""

from __future__ import annotations

from collections.abc import Iterable
import contextvars
import typing

from absl import logging
from etils import epy
from orbax.checkpoint.experimental.v1._src.context import options as options_lib
from orbax.checkpoint.experimental.v1._src.synchronization import multihost
from orbax.checkpoint.experimental.v1._src.synchronization import synchronization


# Each Thread will have its own copy of `Context` object.
# Task and groups will have their own copy of `Context` object.
_CONTEXT: contextvars.ContextVar[Context] = contextvars.ContextVar(
    'orbax_context', default=None
)


def get_context(default: Context | None = None) -> Context:
  """Returns the current `Context` or `default` or `Context()` if not set."""
  default = default or Context()
  return _CONTEXT.get(default)


@typing.final
class Context(epy.ContextManager):
  """Context for customized checkpointing.

  This class manages the configuration options (e.g., async, multiprocessing,
  array handling) used during Orbax checkpoint operations.

  Creating a new :py:class:`.Context` within an existing :py:class:`.Context`
  sets all parameters from scratch by default. To inherit properties from a
  parent :py:class:`.Context`, you must explicitly pass the parent context as
  the first argument. The new context will inherit the parent's properties,
  except for any options explicitly provided as keyword arguments to the child
  context.

  WARNING: The context is thread-local and is not shared across threads. The
  entire context block must be executed within the same thread. If you dispatch
  a checkpointing operation to a worker thread (e.g., via `ThreadPoolExecutor`),
  that thread will not inherit the context and will fall back to default
  settings.

  Example:
    Basic usage and explicit inheritance::

      import orbax.checkpoint as ocp

      # Basic usage
      with ocp.Context(pytree_options=ocp.options.PyTreeOptions()):
        ocp.save_pytree(directory, tree)

      # Inheriting properties from an existing context
      with ocp.Context(pytree_options=ocp.options.PyTreeOptions()) as outer_ctx:
        # inner_ctx inherits pytree_options, but overrides/adds array_options
        with ocp.Context(outer_ctx,
            array_options=ocp.options.ArrayOptions()
            ) as inner_ctx:
          ocp.save_pytree(directory, tree)

    Context is not shared across threads::

      from concurrent.futures import ThreadPoolExecutor
      import orbax.checkpoint as ocp

      executor = ThreadPoolExecutor(max_workers=1)
      with ocp.Context(
          pytree_options=ocp.options.PyTreeOptions()
      ):  # Thread #1 creates Context.
        # The following save_pytree call is executed in Thread #2, which sees
        # a "default" Context, NOT the one created above.
        executor.submit(ocp.save_pytree, directory, tree)


  Attributes:
    pytree_options: Options for PyTree checkpointing.
    array_options: Options for saving and loading array (and array-like
      objects).
    async_options: Options for controlling asynchronous behavior.
    multiprocessing_options: Options for multiprocessing behavior.
    file_options: Options for working with the file system.
    checkpointables_options: Options for controlling checkpointables behavior.
    pathways_options: Options for Pathways checkpointing.
    checkpoint_layout: The layout of the checkpoint. Defaults to ORBAX.
  """

  def __init__(
      self,
      context: Context | None = None,
      *,
      pytree_options: options_lib.PyTreeOptions | None = None,
      array_options: options_lib.ArrayOptions | None = None,
      async_options: options_lib.AsyncOptions | None = None,
      multiprocessing_options: options_lib.MultiprocessingOptions | None = None,
      file_options: options_lib.FileOptions | None = None,
      checkpointables_options: options_lib.CheckpointablesOptions | None = None,
      pathways_options: options_lib.PathwaysOptions | None = None,
      checkpoint_layout: options_lib.CheckpointLayout | None = None,
  ):
    self._pytree_options = pytree_options or (
        context.pytree_options if context else options_lib.PyTreeOptions()
    )
    self._array_options = array_options or (
        context.array_options if context else options_lib.ArrayOptions()
    )
    self._async_options = async_options or (
        context.async_options if context else options_lib.AsyncOptions()
    )
    self._multiprocessing_options = multiprocessing_options or (
        context.multiprocessing_options
        if context
        else options_lib.MultiprocessingOptions()
    )
    self._file_options = file_options or (
        context.file_options if context else options_lib.FileOptions()
    )
    self._checkpointables_options = checkpointables_options or (
        context.checkpointables_options
        if context
        else options_lib.CheckpointablesOptions()
    )
    self._pathways_options = pathways_options or (
        context.pathways_options if context else options_lib.PathwaysOptions()
    )
    self._checkpoint_layout = checkpoint_layout or (
        context.checkpoint_layout
        if context
        else options_lib.CheckpointLayout.ORBAX
    )

  @property
  def pytree_options(self) -> options_lib.PyTreeOptions:
    return self._pytree_options

  @property
  def array_options(self) -> options_lib.ArrayOptions:
    return self._array_options

  @property
  def async_options(self) -> options_lib.AsyncOptions:
    return self._async_options

  @property
  def multiprocessing_options(self) -> options_lib.MultiprocessingOptions:
    return self._multiprocessing_options

  @property
  def file_options(self) -> options_lib.FileOptions:
    return self._file_options

  @property
  def checkpointables_options(self) -> options_lib.CheckpointablesOptions:
    return self._checkpointables_options

  @property
  def pathways_options(self) -> options_lib.PathwaysOptions:
    return self._pathways_options

  @property
  def checkpoint_layout(self) -> options_lib.CheckpointLayout:
    return self._checkpoint_layout

  def operation_id(self) -> str:
    return synchronization.OperationIdGenerator.get_current_operation_id()

  def __contextmanager__(self) -> Iterable[Context]:
    token = _CONTEXT.set(self)
    try:
      yield self
    finally:
      _CONTEXT.reset(token)


async def synchronize_next_operation_id():
  """Obtains the next operation id and synchronizes processes."""
  context = get_context()
  operation_id = synchronization.OperationIdGenerator.next_operation_id()
  await multihost.sync_global_processes(
      multihost.unique_barrier_key(
          'next_awaitable_signal_operation_id:sync',
          prefix=context.multiprocessing_options.barrier_sync_key_prefix,
      ),
      operation_id=context.operation_id(),
      processes=context.multiprocessing_options.active_processes,
  )
  logging.vlog(
      1,
      '[process=%s] Synchronized next awaitable signal operation id to %s',
      multihost.process_index(),
      operation_id,
  )
