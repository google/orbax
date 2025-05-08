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

"""Defines free-function interface for loading."""

from typing import Any

from etils import epath
from orbax.checkpoint._src.checkpointers import async_checkpointer
from orbax.checkpoint._src.handlers import composite_checkpoint_handler
from orbax.checkpoint._src.handlers import handler_registration as legacy_handler_registration
from orbax.checkpoint._src.serialization import type_handlers
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.handlers import compatibility as handler_compatibility
from orbax.checkpoint.experimental.v1._src.handlers import composite_handler
import orbax.checkpoint.experimental.v1._src.handlers.global_registration  # pylint: disable=unused-import
from orbax.checkpoint.experimental.v1._src.metadata import types as metadata_types
from orbax.checkpoint.experimental.v1._src.path import format_utils
from orbax.checkpoint.experimental.v1._src.path import types as path_types
from orbax.checkpoint.experimental.v1._src.serialization import registration as serialization_registration
from orbax.checkpoint.experimental.v1._src.synchronization import types as async_types
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types


PYTREE_CHECKPOINTABLE_KEY = format_utils.PYTREE_CHECKPOINTABLE_KEY
AbstractPyTree = tree_types.PyTreeOf[tree_types.AbstractLeafType]
CheckpointMetadata = metadata_types.CheckpointMetadata
PLACEHOLDER = type_handlers.PLACEHOLDER


def _standardize_abstract_checkpointables(abstract_checkpointables):
  if isinstance(abstract_checkpointables, CheckpointMetadata):
    return abstract_checkpointables.metadata
  return abstract_checkpointables


def load_pytree(
    directory: path_types.PathLike,
    abstract_pytree: (
        AbstractPyTree | CheckpointMetadata[AbstractPyTree] | None
    ) = None,
    *,
    checkpointable_name: str | None = PYTREE_CHECKPOINTABLE_KEY,
) -> tree_types.PyTreeOf[tree_types.LeafType]:
  """Loads a PyTree.

  Loads from a PyTree checkpoint. A PyTree checkpoint must be a directory
  containing a subdirectory with name provided by `checkpointable_name`, with
  default value `pytree`. Please see `checkpointable_name` for more details.

  The operation blocks until complete. For improved performance, consider using
  `load_async` instead.

  If `abstract_pytree` is not provided, the PyTree will be loaded exactly as
  saved.

  IMPORTANT: Loading is more brittle and error-prone when not providing
  `abstract_pytree`. Always provide `abstract_pytree` if possible. Note that
  you can always obtain the tree structure from a saved checkpoint using
  `ocp.pytree_metadata`.

  Providing the `abstract_tree` guarantees two things:

  1. The restored tree will exactly match the structure of `abstract_pytree` (or
  raise an error if it is impossible to guarantee this). For example, if
  `abstract_pytree` is a custom object registered as a PyTree, the checkpoint
  will be restored as the same object, if possible.

  2. The leaves of the restored tree will be restored with the properties
  indicated by the abstract leaves. For example, if a leaf in `abstract_pytree`
  is a `jax.ShapeDtypeStruct`, the restored leaf will be a `jax.Array` with the
  same shape and dtype. Each `AbstractLeafType` has a corresponding `LeafType`
  that is restored.

  Args:
    directory: The directory to load the checkpoint from. This directory must
      contain a subdirectory with name provided by `checkpointable_name`. See
      `checkpointable_name` for more details.
    abstract_pytree: Provides a tree structure for the checkpoint to be restored
      into. May be omitted to load exactly as saved., but this is much more
      brittle than providing the tree.
    checkpointable_name: The name of the checkpointable to load. Defaults to
      `pytree`. A subdirectory with this name must exist in `directory`. If None
      then directory itself is expected to contain all files relevant for
      loading the PyTree, rather than any subdirectory. Such files include, for
      example, manifest.ocdbt, _METADATA, ocdbt.process_X.

  Returns:
    The restored PyTree.
  """
  format_utils.validate_pytree_checkpoint(
      directory, checkpointable_name=checkpointable_name
  )
  if checkpointable_name is None:  # directory is direct path to pytree ckpt.
    # TODO(niketkb): Refactor to load the pytree directly from the directory.

    # Workaround to load the PyTree by reusing the load_checkpointables
    # function:
    # Treat the directory as a checkpointable and its parent as the step dir.
    directory = epath.Path(directory)
    checkpointable_name = directory.name
    directory = directory.parent
  return load_checkpointables(
      directory,
      abstract_checkpointables={
          checkpointable_name: _standardize_abstract_checkpointables(
              abstract_pytree
          )
      },
  )[checkpointable_name]


def load_checkpointables(
    directory: path_types.PathLike,
    abstract_checkpointables: (
        dict[str, Any] | CheckpointMetadata[dict[str, Any]] | None
    ) = None,
) -> dict[str, Any]:
  """Loads checkpointables.

  See documentation for `save_checkpointables` for more context on what a
  "checkpointable" is.

  This function can be used to load any checkpoint saved by
  `save_checkpointables` (or `save_pytree`). The directory should contain a
  number of subdirectories - each of these represents the name of a
  checkpointable.

  If `abstract_checkpointables` is not provided, the checkpointables will be
  loaded exactly as saved.

  IMPORTANT: Loading is more brittle and error-prone when not providing
  `abstract_checkpointables`. Always provide `abstract_checkpointables` if
  possible. Note that you can always obtain the information about the
  checkpointables using `ocp.checkpointables_metadata`.

  If `abstract_checkpointables` is provided, the value provided for each key
  is treated as the abstract type for the given checkpointable. For example, for
  a PyTree of `jax.Array`, the corresponding abstract checkpointable is a PyTree
  of `jax.ShapeDtypeStruct`. `None` is always a valid abstract checkpointable,
  which just indicates that the checkpointable should be loaded exactly as
  saved.

  The keys provided in `abstract_checkpointables` may be any subset of the
  checkpointables in the checkpoint. Any checkpointables names not provided in
  `abstract_checkpointables` will not be loaded.

  Args:
    directory: The directory to load the checkpoint from. This directory must
      contain a subdirectory for each checkpointable.
    abstract_checkpointables: A dictionary of abstract checkpointables.
      Dictionary keys represent the names of the checkpointables, while the
      values are the abstract checkpointable objects themselves.

  Returns:
    A dictionary of checkpointables. Dictionary keys represent the names of the
    checkpointables, while the values are the checkpointable objects themselves.
  """
  directory = epath.Path(directory)
  format_utils.validate_checkpoint(directory)


  abstract_checkpointables = _standardize_abstract_checkpointables(
      abstract_checkpointables
  )
  ckptr, args = get_v0_checkpointer_and_args(
      directory,
      abstract_checkpointables,
      context=context_lib.get_context(),
  )
  restored = ckptr.restore(directory, args=args)
  return {k: v for k, v in zip(restored.keys(), restored.values())}


def load_pytree_async(
    directory: path_types.PathLike,
    abstract_pytree: (
        AbstractPyTree | CheckpointMetadata[AbstractPyTree] | None
    ) = None,
    *,
    checkpointable_name: str | None = PYTREE_CHECKPOINTABLE_KEY,
) -> async_types.AsyncResponse[tree_types.PyTreeOf[tree_types.LeafType]]:
  """Loads a PyTree asynchronously. Not yet implemented."""
  del directory, abstract_pytree, checkpointable_name
  raise NotImplementedError('Asynchronous loading is not yet supported.')


def load_checkpointables_async(
    directory: path_types.PathLike,
    abstract_checkpointables: (
        dict[str, Any] | CheckpointMetadata[dict[str, Any]] | None
    ) = None,
) -> async_types.AsyncResponse[dict[str, Any]]:
  """Loads a checkpointables asynchronously. Not yet implemented."""
  del directory, abstract_checkpointables
  raise NotImplementedError('Asynchronous loading is not yet supported.')


def get_v0_checkpointer_and_args(
    directory: path_types.Path,
    abstract_checkpointables: dict[str, Any] | None,
    *,
    context: context_lib.Context,
) -> tuple[
    async_checkpointer.AsyncCheckpointer,
    composite_checkpoint_handler.CompositeArgs,
]:
  """Construct V0 Checkpointer and Args for loading."""
  abstract_checkpointables = abstract_checkpointables or {}
  if (
      provided_reserved_keys := abstract_checkpointables.keys()
      & format_utils.RESERVED_CHECKPOINTABLE_KEYS
  ):
    raise ValueError(
        f'Provided reserved checkpointable keys: {provided_reserved_keys}.'
    )

  # pylint: disable=protected-access
  handlers = composite_handler.CompositeHandler(
      context.checkpointables_options.registry
  )._get_loadable_handlers(directory, abstract_checkpointables)
  # pylint: enable=protected-access
  if not abstract_checkpointables:
    abstract_checkpointables = {
        name: None
        for name in handlers.keys()
        if name not in format_utils.RESERVED_CHECKPOINTABLE_KEYS
        and (directory / name).exists()
    }

  compatibility_handlers = {
      name: handler_compatibility.get_compatibility_handler(handler)
      for name, handler in handlers.items()
  }
  legacy_handler_registry = (
      legacy_handler_registration.DefaultCheckpointHandlerRegistry()
  )
  for name, handler in compatibility_handlers.items():
    legacy_handler_registry.add(name, handler_compatibility.Args, handler)
  composite_options = composite_checkpoint_handler.CompositeOptions(
      async_options=context.async_options.v0(),
      file_options=context.file_options.v0(),
      multiprocessing_options=context.multiprocessing_options.v0(),
      temporary_path_class=context.file_options.temporary_path_class,
  )
  ckptr = async_checkpointer.AsyncCheckpointer(
      composite_checkpoint_handler.CompositeCheckpointHandler(
          handler_registry=legacy_handler_registry,
          composite_options=composite_options,
      ),
      async_options=context.async_options.v0(),
      multiprocessing_options=context.multiprocessing_options.v0(),
      file_options=context.file_options.v0(),
      temporary_path_class=context.file_options.temporary_path_class,
  )
  args = composite_checkpoint_handler.CompositeArgs(**{
      name: handler_compatibility.Args(checkpointable)
      for name, checkpointable in abstract_checkpointables.items()
  })
  return ckptr, args
