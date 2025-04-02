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

"""Defines free-function interface for saving."""

import threading
from typing import Any

from etils import epath
import orbax.checkpoint as ocp
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.handlers import compatibility as handler_compatibility
from orbax.checkpoint.experimental.v1._src.handlers import pytree_handler
from orbax.checkpoint.experimental.v1._src.handlers import registration
from orbax.checkpoint.experimental.v1._src.path import format_utils
from orbax.checkpoint.experimental.v1._src.path import types as path_types
from orbax.checkpoint.experimental.v1._src.synchronization import types as async_types
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types

PYTREE_CHECKPOINTABLE_KEY = format_utils.PYTREE_CHECKPOINTABLE_KEY


def save_pytree(
    directory: path_types.PathLike,
    pytree: tree_types.PyTreeOf[tree_types.LeafType],
    *,
    force: bool = False,
    custom_metadata: tree_types.JsonType | None = None,
):
  """Saves a PyTree.

  The operation blocks until complete. For improved performance, consider using
  `save_async` instead.

  Args:
    directory: The directory to save the checkpoint to.
    pytree: The PyTree to save. This may be any JAX PyTree (including custom
      objects registered as PyTrees) consisting of supported leaf types. Default
      supported leaf types include `jax.Array`, `np.ndarray`, simple types like
      `int`, `float`, `str`, and empty nodes. Support for custom leaves is also
      possible by implementing a `LeafTypeHandler`.
    force: Whether to allow the save to proceed even if it would fully overwrite
      an existing checkpoint.
    custom_metadata: User-provided custom metadata. An arbitrary
      JSON-serializable dictionary the user can use to store additional
      information. The field is treated as opaque by Orbax.
  """
  with pytree_handler.pytree_handler_context():
    save_checkpointables(
        directory,
        {PYTREE_CHECKPOINTABLE_KEY: pytree},
        force=force,
        custom_metadata=custom_metadata,
    )


def save_checkpointables(
    directory: path_types.PathLike,
    checkpointables: dict[str, Any],
    *,
    force: bool = False,
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
        directory,
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
    ocp.save_checkpointables(directory, train_state)

  This is not the ideal way of doing things because it is then difficult to run
  transformations that involve the entire train state (see the
  `load_and_transform` API).

  Args:
    directory: The directory to save the checkpoint to.
    checkpointables: A dictionary of checkpointables. Dictionary keys represent
      the names of the checkpointables, while the values are the checkpointable
      objects themselves.
    force: Whether to allow the save to proceed even if it would fully overwrite
      an existing checkpoint.
    custom_metadata: User-provided custom metadata. An arbitrary
      JSON-serializable dictionary the user can use to store additional
      information. The field is treated as opaque by Orbax.
  """
  save_checkpointables_async(
      directory,
      checkpointables,
      force=force,
      custom_metadata=custom_metadata,
  ).result()


class _SaveResponse(async_types.AsyncResponse[None]):
  """An `AsyncResponse` representing the result of `save_pytree_async`.

  TODO(cpgaffney): Note that a memory leak is possible if the user does not
  call `result`.
  """

  def __init__(self, checkpointer: ocp.AsyncCheckpointer):
    self._checkpointer = checkpointer
    self._thread = threading.Thread(target=self._wait_for_save)
    self._thread.start()

  def _wait_for_save(self):
    self._checkpointer.wait_until_finished()

  def result(self, timeout: float | None = None) -> None:
    self._thread.join()
    self._checkpointer.close()


def save_pytree_async(
    directory: path_types.PathLike,
    pytree: tree_types.PyTreeOf[tree_types.LeafType],
    *,
    force: bool = False,
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
    directory: The directory to save the checkpoint to.
    pytree: The PyTree to save. This may be any JAX PyTree (including custom
      objects registered as PyTrees) consisting of supported leaf types. Default
      supported leaf types include `jax.Array`, `np.ndarray`, simple types like
      `int`, `float`, `str`, and empty nodes. Support for custom leaves is also
      possible by implementing a `LeafTypeHandler`.
    force: Whether to allow the save to proceed even if it would fully overwrite
      an existing checkpoint.
    custom_metadata: User-provided custom metadata. An arbitrary
      JSON-serializable dictionary the user can use to store additional
      information. The field is treated as opaque by Orbax.

  Returns:
    An `AsyncResponse` that can be used to block until the save is complete.
    Blocking can be done using `response.result()`, which returns `None`.
  """
  with pytree_handler.pytree_handler_context():
    return save_checkpointables_async(
        directory,
        {PYTREE_CHECKPOINTABLE_KEY: pytree},
        force=force,
        custom_metadata=custom_metadata,
    )


def save_checkpointables_async(
    directory: path_types.PathLike,
    checkpointables: dict[str, Any],
    *,
    force: bool = False,
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
    directory: The directory to save the checkpoint to.
    checkpointables: A dictionary of checkpointables. Dictionary keys represent
      the names of the checkpointables, while the values are the checkpointable
      objects themselves.
    force: Whether to allow the save to proceed even if it would fully overwrite
      an existing checkpoint.
    custom_metadata: User-provided custom metadata. An arbitrary
      JSON-serializable dictionary the user can use to store additional
      information. The field is treated as opaque by Orbax.

  Returns:
    An `AsyncResponse` that can be used to block until the save is complete.
    Blocking can be done using `response.result()`, which returns `None`.
  """
  directory = epath.Path(directory)

  directory = epath.Path(directory)
  ckptr, args = get_v0_checkpointer_and_args(checkpointables)
  ckptr.save(directory, args=args, force=force, custom_metadata=custom_metadata)
  return _SaveResponse(ckptr)


def get_v0_checkpointer_and_args(
    checkpointables: dict[str, Any],
) -> tuple[ocp.AsyncCheckpointer, ocp.args.Composite]:
  """Construct V0 Checkpointer and Args for saving."""
  context = context_lib.get_context()
  handlers = {
      name: registration.resolve_handler_for_save(
          context.checkpointables_options.registry, checkpointable, name=name
      )
      for name, checkpointable in checkpointables.items()
  }
  compatibility_handlers = {
      name: handler_compatibility.get_compatibility_handler(handler)
      for name, handler in handlers.items()
  }
  handler_registry = ocp.handlers.DefaultCheckpointHandlerRegistry()
  for name, handler in compatibility_handlers.items():
    handler_registry.add(name, handler_compatibility.Args, handler)
  ckptr = ocp.AsyncCheckpointer(
      ocp.CompositeCheckpointHandler(handler_registry=handler_registry)
  )
  args = ocp.args.Composite(**{
      name: handler_compatibility.Args(checkpointable)
      for name, checkpointable in checkpointables.items()
  })
  return ckptr, args
