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

"""Defines free-function interface for loading."""

import functools
import time
from typing import Any, Awaitable, Protocol

from absl import logging
from orbax.checkpoint._src import asyncio_utils
from orbax.checkpoint._src.logging import event_tracking
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
import orbax.checkpoint.experimental.v1._src.handlers.global_registration  # pylint: disable=unused-import
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout
from orbax.checkpoint.experimental.v1._src.layout import registry as layout_registry
from orbax.checkpoint.experimental.v1._src.loading import validation
from orbax.checkpoint.experimental.v1._src.metadata import types as metadata_types
from orbax.checkpoint.experimental.v1._src.path import types as path_types
from orbax.checkpoint.experimental.v1._src.synchronization import multihost
from orbax.checkpoint.experimental.v1._src.synchronization import types as async_types
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types



PYTREE_CHECKPOINTABLE_KEY = checkpoint_layout.PYTREE_CHECKPOINTABLE_KEY
AbstractPyTree = tree_types.PyTreeOf[tree_types.AbstractLeafType]
CheckpointMetadata = metadata_types.CheckpointMetadata
PLACEHOLDER = ...


class LoadFn(Protocol):
  """Protocol for a two-phase load function used in `_load_impl`.

  Is a callable that, when awaited, performs validation and setup, then
  resolves to a second awaitable for the background load operation (I/O).
  """

  async def __call__(self) -> Awaitable[Any]:
    ...


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

  This function must be called on all available controller processes.

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

  Example Usage::
    path = '/tmp/my_checkpoint'
    # Save a checkpoint
    pytree = {'a': jnp.arange(8), 'b': jnp.zeros(4)}
    ocp.save_pytree(path, pytree)

    # Load the checkpoint
    # Highly recommended to provide the abstract pytree (structure/shapes)
    abstract_pytree = jax.eval_shape(lambda: pytree)

    # Method A: Load using the abstract structure.
    # This automatically looks for the 'pytree' subdirectory inside 'path'.
    restored = ocp.load_pytree(path, abstract_pytree)

    # Method B: Infer structure from file. (Not recommended for production use)
    # cases or for complex trees.
    restored_inferred = ocp.load_pytree(path)

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
  logging.info('Loading checkpoint from %s.', path)
  ctx = context_lib.get_context()
  path = ctx.file_options.path_class(path)
  layout = asyncio_utils.run_sync(
      layout_registry.get_checkpoint_layout_pytree(
          path, ctx.checkpoint_layout, checkpointable_name
      )
  )
  abstract_pytree = _standardize_abstract_checkpointables(abstract_pytree)

  validation.validate_pytree_checkpointable_name(checkpointable_name)

  loaded_pytree = _load_impl(
      path,
      functools.partial(
          layout.load_pytree,
          path=path,
          checkpointable_name=checkpointable_name,
          abstract_pytree=abstract_pytree,
      ),
      start_time=start_time,
  )

  return loaded_pytree


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

  This function must be called on all available controller processes.

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

  Example Usage::
    path = '/tmp/my_checkpoint_step_100'

    # Save multiple components (checkpointables)
    params = {'w': jnp.ones((8, 8)), 'b': jnp.zeros(8)}
    opt_state = {'count': jnp.array(100)}

    # Setup Grain (Stateful Checkpointable)
    import grain
    dataset_iter = iter(
        grain.MapDataset.range(30)
        .batch(3)
        .map(lambda x: x.tolist())
    )

    ocp.save_checkpointables(path, {
        'model': params,
        'optimizer': opt_state,
        'dataset': dataset_iter,
    })

    # Load the checkpointables
    abstract_params = jax.eval_shape(lambda: params)
    abstract_opt = jax.eval_shape(lambda: opt_state)

    abstract_checkpointables = {
        'model': abstract_params,
        'optimizer': abstract_opt,
        # Dataset is restored statefully. An initialized object must be
        # passed, but its position will be set to the position recorded in the
        # checkpoint after restoring.
        'dataset': dataset_iter,
    }

    # Load all components
    restored = ocp.load_checkpointables(path, abstract_checkpointables)

    # Load only a subset
    restored_subset = ocp.load_checkpointables(
        path,
        {'model': abstract_params}
    )

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
  logging.info('Loading checkpoint from %s.', path)
  ctx = context_lib.get_context()
  path = ctx.file_options.path_class(path)
  layout = asyncio_utils.run_sync(
      layout_registry.get_checkpoint_layout(path, ctx.checkpoint_layout)
  )

  abstract_checkpointables = _standardize_abstract_checkpointables(
      abstract_checkpointables
  )
  validation.validate_abstract_checkpointables(abstract_checkpointables)

  if not hasattr(layout, 'load_checkpointables'):
    raise NotImplementedError(
        f'Layout {type(layout)} does not support loading checkpointables.'
    )

  return _load_impl(
      path,
      functools.partial(
          layout.load_checkpointables,
          path=path,
          abstract_checkpointables=abstract_checkpointables,
      ),
      start_time=start_time,
  )


def _load_impl(
    path: path_types.Path,
    load_fn: LoadFn,
    start_time: float,
) -> dict[str, Any] | tree_types.PyTreeOf[tree_types.LeafType]:
  """Implementation of loading logic for both :py:func:`.load_checkpointables` and :py:func:`.load_pytree`.

  Args:
    path: The path to the checkpoint.
    load_fn: A  function that returns an awaitable for loading the checkpoint
      based on either :py:func:`.load_checkpointables` or
      :py:func:`.load_pytree`.
    start_time: The time when the loading process started.

  Returns:
    The loaded checkpointables or PyTree itself.
  """
  if not path:
    raise ValueError('Path must not be None.')

  ctx = context_lib.get_context()

  async def _load() -> Any:
    load_awaitable = await load_fn()
    result = await load_awaitable
    await multihost.sync_global_processes(
        multihost.unique_barrier_key(
            '_load_impl',
            prefix=ctx.multiprocessing_options.barrier_sync_key_prefix,
        ),
        operation_id=ctx.operation_id(),
        processes=ctx.multiprocessing_options.active_processes,
    )
    return result

  result = asyncio_utils.run_sync(_load())

  event_tracking.record_read_event(path)

  duration_secs = time.time() - start_time
  logging.info(
      'Finished loading checkpoint in %.2f seconds from %s.',
      duration_secs,
      path,
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
