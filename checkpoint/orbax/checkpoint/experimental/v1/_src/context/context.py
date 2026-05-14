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
import copy
import dataclasses
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


def _get_option(
    opt_value: typing.Any,
    parent_opt: typing.Any,
    default_factory: typing.Callable[[], typing.Any],
) -> typing.Any:
  if opt_value is not None:
    return copy.deepcopy(opt_value)
  if parent_opt is not None:
    return copy.deepcopy(parent_opt)
  return default_factory()


@typing.final
class Context(epy.ContextManager):
  """Context for customized checkpointing.

  This class manages the configuration options (e.g., async, multiprocessing,
  array handling) used during Orbax checkpoint operations using a mutable
  namespace pattern.

  Creating a new :py:class:`.Context` within an existing :py:class:`.Context`
  sets all parameters from scratch by default. To inherit properties from a
  parent :py:class:`.Context`, pass the parent context as the first positional
  or explicit `context` keyword argument. The new context will inherit
  the parent's properties but can be mutated independently.

  WARNING: The context is thread-local and is not shared across threads. The
  entire context block must be executed within the same thread. If you dispatch
  a checkpointing operation to a worker thread (e.g., via `ThreadPoolExecutor`),
  that thread will not inherit the context and will fall back to default
  settings.

  Note: When testing or mixing checkpointer instances and free functions,
  explicitly wrap free functions inside their own `with ocp.Context(...)` block,
  or pass explicit contexts to Checkpointer constructors, to ensure each actor
  receives its correct active configuration independent of the surrounding
  context.

  Example:
    Basic usage and explicit inheritance::

      from orbax.checkpoint import v1 as ocp

      # Basic usage
      ctx = ocp.Context()
      ctx.pytree.loading.partial_load = True
      with ctx:
        ocp.load_pytree(directory, tree)

      # Inheriting properties from an existing context
      ctx1 = ocp.Context()
      ctx1.pytree.loading.partial_load = True
      with ctx1 as outer_ctx:
        # inner_ctx inherits partial_load, but mutates array saving
        ctx2 = ocp.Context(outer_ctx)
        ctx2.array.saving.use_zarr3 = False
        with ctx2 as inner_ctx:
          ocp.load_pytree(directory, tree)

    Context is not shared across threads::

      from concurrent.futures import ThreadPoolExecutor
      from orbax.checkpoint import v1 as ocp

      executor = ThreadPoolExecutor(max_workers=1)
      ctx = ocp.Context()
      ctx.pytree.loading.partial_load = True
      with ctx:  # Thread #1 creates Context.
        # The following load_pytree call is executed in Thread #2, which sees
        # a "default" Context, NOT the one created above.
        executor.submit(ocp.load_pytree, directory, tree)

  Attributes:
    pytree: Options for PyTree checkpointing. See
      :class:`~orbax.checkpoint.experimental.v1.options.PyTreeOptions`.
    array: Options for saving and loading array (and array-like objects). See
      :class:`~orbax.checkpoint.experimental.v1.options.ArrayOptions`.
    asynchronous: Options for controlling asynchronous behavior. See
      :class:`~orbax.checkpoint.experimental.v1.options.AsyncOptions`.
    multiprocessing: Options for multiprocessing behavior. See
      :class:`~orbax.checkpoint.experimental.v1.options.MultiprocessingOptions`.
    file: Options for working with the file system. See
      :class:`~orbax.checkpoint.experimental.v1.options.FileOptions`.
    checkpointables: Options for controlling checkpointables behavior. See
      :class:`~orbax.checkpoint.experimental.v1.options.CheckpointablesOptions`.
    pathways: Options for Pathways checkpointing. See
      :class:`~orbax.checkpoint.experimental.v1.options.PathwaysOptions`.
    checkpoint_layout: The layout of the checkpoint. Defaults to ORBAX. See
      :class:`~orbax.checkpoint.experimental.v1.options.CheckpointLayout`.
    deletion: Options for controlling deletion behavior. See
      :class:`~orbax.checkpoint.experimental.v1.options.DeletionOptions`.
    memory: Options for controlling memory limits during save / load. See
      :class:`~orbax.checkpoint.experimental.v1.options.MemoryOptions`.
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
      deletion_options: options_lib.DeletionOptions | None = None,
      memory_options: options_lib.MemoryOptions | None = None,
      safetensors_options: options_lib.SafetensorsOptions | None = None,
  ):
    if any(
        opt is not None for opt in (
            pytree_options, array_options, async_options,
            multiprocessing_options, file_options, checkpointables_options,
            pathways_options, checkpoint_layout, deletion_options,
            memory_options, safetensors_options
        )
    ):
      # TODO: b/513156122 - Passing option objects directly to Context.__init__
      # is deprecated in favor of mutable dot-notation configuration (e.g.
      # ctx.array.saving...). Remove these keyword parameters.
      logging.warning(
          'Passing direct option objects to Context.__init__ is deprecated'
          ' in favor of mutable dot-notation configuration (e.g.'
          ' ctx.array.saving.use_ocdbt = ...). These keyword arguments will be'
          ' removed in a future release.'
      )

    self._pytree_options = _get_option(
        pytree_options,
        context.pytree_options if context is not None else None,
        options_lib.PyTreeOptions,
    )
    self._array_options = _get_option(
        array_options,
        context.array_options if context is not None else None,
        options_lib.ArrayOptions,
    )
    self._async_options = _get_option(
        async_options,
        context.async_options if context is not None else None,
        options_lib.AsyncOptions,
    )
    self._multiprocessing_options = _get_option(
        multiprocessing_options,
        context.multiprocessing_options if context is not None else None,
        options_lib.MultiprocessingOptions,
    )
    self._file_options = _get_option(
        file_options,
        context.file_options if context is not None else None,
        options_lib.FileOptions,
    )
    self._checkpointables_options = _get_option(
        checkpointables_options,
        context.checkpointables_options if context is not None else None,
        options_lib.CheckpointablesOptions,
    )
    self._pathways_options = _get_option(
        pathways_options,
        context.pathways_options if context is not None else None,
        options_lib.PathwaysOptions,
    )
    self._checkpoint_layout = _get_option(
        checkpoint_layout,
        context.checkpoint_layout if context is not None else None,
        lambda: options_lib.CheckpointLayout.ORBAX,
    )
    self._deletion_options = _get_option(
        deletion_options,
        context.deletion_options if context is not None else None,
        options_lib.DeletionOptions,
    )
    self._memory_options = _get_option(
        memory_options,
        context.memory_options if context is not None else None,
        options_lib.MemoryOptions,
    )
    self._safetensors_options = _get_option(
        safetensors_options,
        context.safetensors_options if context is not None else None,
        options_lib.SafetensorsOptions,
    )

  def _check_not_frozen(self) -> None:
    if id(self) in options_lib._FROZEN_IDS.get():  # pylint: disable=protected-access
      raise RuntimeError(
          'Cannot mutate an active Context. Ensure all configuration options'
          ' are set before entering the `with` context block.'
      )

  @property
  def array(self) -> options_lib.ArrayOptions:
    return self._array_options

  @property
  def asynchronous(self) -> options_lib.AsyncOptions:
    return self._async_options

  @property
  def pytree(self) -> options_lib.PyTreeOptions:
    return self._pytree_options

  @property
  def file(self) -> options_lib.FileOptions:
    return self._file_options

  @property
  def multiprocessing(self) -> options_lib.MultiprocessingOptions:
    return self._multiprocessing_options

  @property
  def checkpointables(self) -> options_lib.CheckpointablesOptions:
    return self._checkpointables_options

  @property
  def pathways(self) -> options_lib.PathwaysOptions:
    return self._pathways_options

  @property
  def deletion(self) -> options_lib.DeletionOptions:
    return self._deletion_options

  @property
  def memory(self) -> options_lib.MemoryOptions:
    return self._memory_options

  @property
  def safetensors(self) -> options_lib.SafetensorsOptions:
    return self._safetensors_options

  @property
  def checkpoint_layout(self) -> options_lib.CheckpointLayout:
    return self._checkpoint_layout

  @checkpoint_layout.setter
  def checkpoint_layout(self, value: options_lib.CheckpointLayout) -> None:
    self._check_not_frozen()
    self._checkpoint_layout = value

  # TODO: b/513156122 - Migrate internal read sites to short-hand properties and
  # remove legacy aliases in the next refactor.
  # --- Legacy aliases for internal read access compatibility ---

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
  def deletion_options(self) -> options_lib.DeletionOptions:
    return self._deletion_options

  @property
  def memory_options(self) -> options_lib.MemoryOptions:
    return self._memory_options

  @property
  def safetensors_options(self) -> options_lib.SafetensorsOptions:
    return self._safetensors_options

  def operation_id(self) -> str:
    return synchronization.OperationIdGenerator.get_current_operation_id()

  def __contextmanager__(self) -> Iterable[Context]:
    option_ids = _collect_ids(self)
    prev_frozen = options_lib._FROZEN_IDS.get()  # pylint: disable=protected-access
    guard_token = options_lib._FROZEN_IDS.set(prev_frozen | option_ids)  # pylint: disable=protected-access

    token = _CONTEXT.set(self)
    try:
      yield self
    finally:
      _CONTEXT.reset(token)
      options_lib._FROZEN_IDS.reset(guard_token)  # pylint: disable=protected-access


def _collect_ids(ctx: Context) -> frozenset[int]:
  """Collects all object ids from the context and its options.

  This function traverses the context object and all its attributes. If an
  attribute is a dataclass, it recursively traverses the dataclass and collects
  the ids of all its fields. This is used to freeze the context and all its
  options, so that they cannot be modified after the context is entered.

  Args:
    ctx: The context object to collect ids from.

  Returns:
    A frozenset of all object ids from the context and its options.
  """
  ids = {id(ctx)}

  def _traverse(obj: typing.Any) -> None:
    if id(obj) in ids:
      return
    if dataclasses.is_dataclass(obj):
      ids.add(id(obj))
      for field in dataclasses.fields(obj):
        _traverse(getattr(obj, field.name))

  for obj in vars(ctx).values():
    _traverse(obj)
  return frozenset(ids)


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
