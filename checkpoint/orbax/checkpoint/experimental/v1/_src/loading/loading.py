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

"""Defines free-function interface for loading."""

import asyncio
import time
from typing import Any

from absl import logging
from orbax.checkpoint._src.logging import event_tracking
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
import orbax.checkpoint.experimental.v1._src.handlers.global_registration  # pylint: disable=unused-import
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout
from orbax.checkpoint.experimental.v1._src.layout import registry as layout_registry
from orbax.checkpoint.experimental.v1._src.loading import validation
from orbax.checkpoint.experimental.v1._src.metadata import types as metadata_types
from orbax.checkpoint.experimental.v1._src.path import types as path_types
from orbax.checkpoint.experimental.v1._src.synchronization import asyncio_utils
from orbax.checkpoint.experimental.v1._src.synchronization import multihost
from orbax.checkpoint.experimental.v1._src.synchronization import types as async_types
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types


PYTREE_CHECKPOINTABLE_KEY = checkpoint_layout.PYTREE_CHECKPOINTABLE_KEY
AbstractPyTree = tree_types.PyTreeOf[tree_types.AbstractLeafType]
CheckpointMetadata = metadata_types.CheckpointMetadata
PLACEHOLDER = ...


def _standardize_abstract_checkpointables(abstract_checkpointables):
  if isinstance(abstract_checkpointables, CheckpointMetadata):
    return abstract_checkpointables.metadata
  return abstract_checkpointables


def load_pytree(
    path: path_types.PathLike,
    abstract_pytree: (
        AbstractPyTree | CheckpointMetadata[AbstractPyTree] | None
    ) = None,
    *,
    checkpointable_name: str | None = PYTREE_CHECKPOINTABLE_KEY,
) -> tree_types.PyTreeOf[tree_types.LeafType]:
  """Loads a PyTree.

  Loads from a `PyTree` checkpoint. A `PyTree` checkpoint must be a path
  containing a subdirectory with the name provided by `checkpointable_name`,
  with default value `pytree`. See `checkpointable_name` for more details.

  The operation blocks until complete. For improved performance, consider using
  :py:func:`.load_pytree_async` instead.

  If `abstract_pytree` is not provided, the `PyTree` will be loaded exactly as
  saved.

  IMPORTANT: Loading is more brittle and error-prone when not providing
  `abstract_pytree`. Always provide `abstract_pytree` if possible. Note that
  you can always obtain the tree structure from a saved checkpoint using
  :py:func:`.pytree_metadata`.

  Providing the `abstract_pytree` guarantees two things:

  1. The restored tree will exactly match the structure of `abstract_pytree` (or
  raise an error if it is impossible to guarantee this). For example, if
  `abstract_pytree` is a custom object registered as a `PyTree`, the checkpoint
  will be restored as the same object, if possible.

  2. The leaves of the restored tree will be restored with the properties
  indicated by the abstract leaves. For example, if a leaf in `abstract_pytree`
  is a `jax.ShapeDtypeStruct`, the restored leaf will be a `jax.Array` with the
  same shape and `dtype`. Each `AbstractLeafType` has a corresponding `LeafType`
  that is restored.

  Args:
    path: The path to load the checkpoint from. This path must contain a
      subdirectory with name provided by `checkpointable_name`. See
      `checkpointable_name` for more details.
    abstract_pytree: Provides a tree structure for the checkpoint to be restored
      into. May be omitted to load exactly as saved, but this is much more
      brittle than providing the tree.
    checkpointable_name: The name of the checkpointable to load. Defaults to
      `pytree`. A subdirectory with this name must exist in `path`. If None,
      then path itself is expected to contain all files relevant for loading the
      PyTree, rather than any subdirectory. Such files include, for example,
      `manifest.ocdbt`, `_METADATA`, `ocdbt.process_X`.

  Returns:
    The restored `PyTree`.
  """
  start_time = time.time()
  asyncio_utils.maybe_apply_nest_asyncio()
  logging.info('Loading checkpoint from %s.', path)
  ctx = context_lib.get_context()
  path = ctx.file_options.path_class(path)
  layout, checkpointable_name = asyncio.run(
      layout_registry.get_checkpoint_layout_pytree(
          path, ctx.checkpoint_layout, checkpointable_name
      )
  )
  abstract_pytree = _standardize_abstract_checkpointables(abstract_pytree)

  return _load_impl_common(
      layout,
      lambda: layout.load_pytree(
          checkpointable_name,
          abstract_pytree,
      ),
      sync_key='load_pytree',
      start_time=start_time,
  )


def load_checkpointables(
    path: path_types.PathLike,
    abstract_checkpointables: (
        dict[str, Any] | CheckpointMetadata[dict[str, Any]] | None
    ) = None,
) -> dict[str, Any]:
  """Loads checkpointables.

  See documentation for :py:func:`.save_checkpointables` for more context on
  what a checkpointable is.

  This function can be used to load any checkpoint saved by
  :py:func:`.save_checkpointables` (or :py:func:`.save_pytree`). The path should
  contain a number of subdirectories - each of these represents the name of a
  checkpointable.

  The operation blocks until complete. For improved performance, consider using
  :py:func:`.load_checkpointables_async` instead.

  If `abstract_checkpointables` is not provided, the checkpointables will be
  loaded exactly as saved.

  IMPORTANT: Loading is more brittle and error-prone when not providing
  `abstract_checkpointables`. Always provide `abstract_checkpointables` if
  possible. Note that you can always obtain the information about the
  checkpointables using
  :py:func:`.checkpointables_metadata`.

  If `abstract_checkpointables` is provided, the value provided for each key
  is treated as the abstract type for the given checkpointable. For example, for
  a `PyTree` of `jax.Array`, the corresponding abstract checkpointable is a
  `PyTree` of `jax.ShapeDtypeStruct`. `None` is always a valid abstract
  checkpointable, which just indicates that the checkpointable should be loaded
  exactly as saved.

  The keys provided in `abstract_checkpointables` may be any subset of the
  checkpointables in the checkpoint. Any checkpointables names not provided in
  `abstract_checkpointables` will not be loaded.

  Args:
    path: The path to load the checkpoint from. This path must contain a
      subdirectory for each checkpointable.
    abstract_checkpointables: A dictionary of abstract checkpointables.
      Dictionary keys represent the names of the checkpointables, while the
      values are the abstract checkpointable objects themselves.

  Returns:
    A dictionary of checkpointables. Dictionary keys represent the names of the
    checkpointables, while the values are the checkpointable objects themselves.

  Raises:
    FileNotFoundError: If the checkpoint path does not exist.
  """
  start_time = time.time()
  asyncio_utils.maybe_apply_nest_asyncio()
  logging.info('Loading checkpoint from %s.', path)
  ctx = context_lib.get_context()
  path = ctx.file_options.path_class(path)
  layout = asyncio.run(
      layout_registry.get_checkpoint_layout(path, ctx.checkpoint_layout)
  )

  abstract_checkpointables = _standardize_abstract_checkpointables(
      abstract_checkpointables
  )
  validation.validate_abstract_checkpointables(abstract_checkpointables)

  return _load_impl_common(
      layout,
      lambda: layout.load_checkpointables(abstract_checkpointables),
      sync_key='load_checkpointables',
      start_time=start_time,
  )


def _load_impl_common(
    layout: checkpoint_layout.CheckpointLayout,
    load_awaitable_factory: Any,
    sync_key: str,
    start_time: float,
) -> dict[str, Any] | tree_types.PyTreeOf[tree_types.LeafType]:
  """Implementation of loading logic for both :py:func:`.load_checkpointables` and :py:func:`.load_pytree`.

  Args:
    layout: The layout to use for loading the checkpoint (Orbax, SafeTensors, or
      other).
    load_awaitable_factory: A factory function that returns an awaitable for
      loading the checkpoint based on either :py:func:`.load_checkpointables` or
      :py:func:`.load_pytree`.
    sync_key: The key to use for synchronization.
    start_time: The time when the loading process started.

  Returns:
    The loaded checkpointables or PyTree itself.
  """
  if not layout.path:
    raise ValueError('Path must not be None.')

  ctx = context_lib.get_context()

  async def _load() -> Any:
    load_awaitable = await load_awaitable_factory()
    result = await load_awaitable
    await multihost.sync_global_processes(
        multihost.unique_barrier_key(
            sync_key,
            prefix=ctx.multiprocessing_options.barrier_sync_key_prefix,
        ),
        operation_id=ctx.operation_id(),
        processes=ctx.multiprocessing_options.active_processes,
    )
    return result

  result = asyncio.run(_load())

  event_tracking.record_read_event(layout.path)

  duration_secs = time.time() - start_time
  logging.info(
      'Finished loading checkpoint in %.2f seconds from %s.',
      duration_secs,
      layout.path,
  )
  return result


def load_pytree_async(
    path: path_types.PathLike,
    abstract_pytree: (
        AbstractPyTree | CheckpointMetadata[AbstractPyTree] | None
    ) = None,
    *,
    checkpointable_name: str | None = PYTREE_CHECKPOINTABLE_KEY,
) -> async_types.AsyncResponse[tree_types.PyTreeOf[tree_types.LeafType]]:
  """Loads a PyTree asynchronously. Not yet implemented."""
  del path, abstract_pytree, checkpointable_name
  raise NotImplementedError('Asynchronous loading is not yet supported.')


def load_checkpointables_async(
    path: path_types.PathLike,
    abstract_checkpointables: (
        dict[str, Any] | CheckpointMetadata[dict[str, Any]] | None
    ) = None,
) -> async_types.AsyncResponse[dict[str, Any]]:
  """Loads checkpointables asynchronously. Not yet implemented."""
  del path, abstract_checkpointables
  raise NotImplementedError('Asynchronous loading is not yet supported.')
