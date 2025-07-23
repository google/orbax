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
from typing import Any

from etils import epath
import nest_asyncio
from orbax.checkpoint._src.checkpointers import async_checkpointer
from orbax.checkpoint._src.handlers import composite_checkpoint_handler
from orbax.checkpoint._src.handlers import handler_registration as legacy_handler_registration
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.handlers import compatibility as handler_compatibility
from orbax.checkpoint.experimental.v1._src.handlers import registration as handler_registration
import orbax.checkpoint.experimental.v1._src.handlers.global_registration  # pylint: disable=unused-import
from orbax.checkpoint.experimental.v1._src.path import format_utils
from orbax.checkpoint.experimental.v1._src.path import types as path_types
from orbax.checkpoint.experimental.v1._src.saving import saving_utils
from orbax.checkpoint.experimental.v1._src.synchronization import types as async_types
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types

PYTREE_CHECKPOINTABLE_KEY = format_utils.PYTREE_CHECKPOINTABLE_KEY


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


def _save_checkpointables_impl(
    path: path_types.PathLike,
    checkpointables: dict[str, Any],
    *,
    async_origin: bool,
    overwrite: bool,
    custom_metadata: tree_types.JsonType | None,
) -> async_types.AsyncResponse[None]:
  """See caller docstrings."""
  nest_asyncio.apply()
  context = context_lib.get_context()
  path = epath.Path(path)
  # Prevent internal mutation from affecting the caller.
  checkpointables = dict(checkpointables)

  start_time = time.time()
  saving_utils.record_save_start(path, async_origin=async_origin)

  tmp_path = saving_utils.get_temporary_path(path, context=context)

  checkpointables = saving_utils.add_internal_checkpointables(
      checkpointables, context=context
  )

  background_awaitable = asyncio.run(
      saving_utils.run_blocking_save(
          tmp_path,
          checkpointables,
          overwrite=overwrite,
          context=context,
      )
  )

  return saving_utils.create_save_response(
      background_awaitable,
      checkpointables,
      tmp_path,
      start_time,
      context=context,
      custom_metadata=custom_metadata,
      async_origin=async_origin,
  )


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
  checkpointables = saving_utils.add_internal_checkpointables(
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
