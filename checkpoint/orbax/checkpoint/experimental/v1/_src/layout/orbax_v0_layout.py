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

"""Defines `V0Layout`, a class to handle Orbax V0 checkpoint formats."""

import asyncio
import logging
from typing import Any, Awaitable

from orbax.checkpoint._src.path import async_path
from orbax.checkpoint._src.path import temporary_paths
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.handlers import registration
from orbax.checkpoint.experimental.v1._src.handlers import types as handler_types
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout
from orbax.checkpoint.experimental.v1._src.layout import orbax_layout
from orbax.checkpoint.experimental.v1._src.metadata import types as metadata_types
from orbax.checkpoint.experimental.v1._src.path import types as path_types
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types


InvalidLayoutError = checkpoint_layout.InvalidLayoutError
Path = path_types.Path
CheckpointLayout = checkpoint_layout.CheckpointLayout


def is_orbax_v0_checkpoint(path: path_types.PathLike) -> bool:
  """Determines if the given path is a Orbax checkpoint.

  Args:
    path: The path to the checkpoint directory.

  Returns:
    True if the path is a V1 Orbax checkpoint, False otherwise.
  """
  ctx = context_lib.get_context()
  path = ctx.file_options.path_class(path)
  try:
    asyncio.run(OrbaxV0Layout().validate(path))
    return True
  except InvalidLayoutError:
    return False


class OrbaxV0Layout(CheckpointLayout):
  """OrbaxV0Layout.

  This class handles Orbax V0 checkpoint formats. It inherits
  abstract methods from CheckpointLayout. It performs a few core functions:
    - Resolves handlers for saving and loading.
    - Saves and loads checkpointables to/from individual subdirectories by
    delegating to the resolved handlers.
  """

  def __init__(self):
    self._context = context_lib.get_context()
    self._handler_registry = registration.local_registry(
        self._context.checkpointables_options.registry,
        include_global_registry=False,
    )
    self._orbax_layout = orbax_layout.OrbaxLayout()

  async def _load_pytree_metadata(
      self,
      path: Path,
      metadata: orbax_layout.InternalCheckpointMetadata,
  ) -> metadata_types.CheckpointMetadata[dict[str, Any]]:
    handler_for_load = await _get_handler_for_direct_load_pytree(
        path,
        self._handler_registry,
        None,
    )

    if handler_for_load is None:
      raise ValueError(
          f"Failed to resolve handler for checkpointable '{path.name}'."
      )

    pytree_metadata = await handler_for_load.metadata(path)
    init_timestamp_nsecs = None
    commit_timestamp_nsecs = None
    custom_metadata = None
    if metadata:
      init_timestamp_nsecs = metadata.init_timestamp_nsecs
      commit_timestamp_nsecs = metadata.commit_timestamp_nsecs
      custom_metadata = metadata.custom_metadata

    return metadata_types.CheckpointMetadata(
        metadata=pytree_metadata,
        init_timestamp_nsecs=init_timestamp_nsecs,
        commit_timestamp_nsecs=commit_timestamp_nsecs,
        custom_metadata=custom_metadata,
    )

  async def metadata(
      self, path: Path
  ) -> metadata_types.CheckpointMetadata[dict[str, Any]]:
    """Returns the metadata describing the Orbax checkpoint.

    Args:
      path: The path to the checkpoint directory.

    Returns:
      The metadata describing the Orbax checkpoint.
    """
    checkpoint_metadata = await orbax_layout.read_checkpoint_metadata(path)
    # Delegate to OrbaxLayout if the checkpoint is a composite checkpoint.
    if isinstance(checkpoint_metadata.item_handlers, dict):
      return await self._orbax_layout.metadata(path)
    # Otherwise, load the metadata as a PyTree checkpoint.
    return await self._load_pytree_metadata(path, checkpoint_metadata)

  async def _validate(self, path: Path) -> None:
    """Validates a V0 checkpoint directory.

    Must be:
    - Existing
    - A directory.
    - Not a temporary path.
    - Must not have indicator file in parent directory.
    - AND
      - Has checkpoint metadata file
        OR
      - Has a valid checkpointable subdirectory under the current path

    Args:
      path: The path to the checkpoint directory.

    Raises:
      FileNotFoundError: If the path does not exist.
      NotADirectoryError: If the path is not a directory.
      ValueError: If the checkpoint is incomplete.
    """

    if not await async_path.exists(path):
      raise FileNotFoundError(f"Checkpoint path {path} does not exist.")

    if not await async_path.is_dir(path):
      raise NotADirectoryError(f"Checkpoint path {path} is not a directory.")

    if await temporary_paths.is_path_temporary(
        path,
        temporary_path_cls=self._context.file_options.temporary_path_class,
    ):
      raise ValueError(f"Found incomplete checkpoint at {path}.")

    if await async_path.exists(
        path.parent
    ) and await orbax_layout.has_indicator_file(path.parent):
      raise InvalidLayoutError(
          f"You are currently reading in checkpointable {path.name}, which is"
          " a subdirectory of a V1 Orbax checkpoint. Please consider loading"
          f" from {path.parent} instead."
      )

    if not await orbax_layout.has_checkpoint_metadata_file(path):
      if await orbax_layout.has_pytree_metadata_file(path):
        # If the directory has a pytree metadata file, it's a valid V0
        # top-level pytree checkpoint.
        return

      subpaths = await orbax_layout.get_subpaths(path)
      awaitables = [
          orbax_layout.has_checkpoint_metadata_file(subdir)
          for subdir in subpaths
      ]
      is_ckpt_list = await asyncio.gather(*awaitables)

      checkpoint_subdirectories = [
          subdir for subdir, is_ckpt in zip(subpaths, is_ckpt_list) if is_ckpt
      ]
      if checkpoint_subdirectories:
        raise InvalidLayoutError(
            "You are currently attempting to read a V0 checkpoint from a root"
            " directory, please consider loading one of the following"
            f" checkpoint subdirectories: {checkpoint_subdirectories}"
        )

  async def _validate_pytree(self, path: Path, checkpointable_name: str | None):
    """Validates that V0 checkpoint has pytree in 'checkpointable_name' subdir.

    Checks if checkpoint is either a top-level pytree checkpoint or has a
    checkpointable_name subdirectory containing a pytree checkpoint.

    Args:
      path: The path to the checkpoint directory.
      checkpointable_name: The name of the checkpointable to load. A
        subdirectory with this name must exist in `directory`. If None then
        `directory` is expected to contain the checkpoint directly. Defaults to
        `pytree`.

    Raises:
      FileNotFoundError: If the path does not exist, or if
      `checkpointable_name` pytree is not found in the directory
    """
    pytree_dir = (
        path if checkpointable_name is None else path / checkpointable_name
    )

    try:
      if not await async_path.exists(pytree_dir):
        raise FileNotFoundError
      elif not await orbax_layout.has_pytree_metadata_file(pytree_dir):
        raise FileNotFoundError
    except FileNotFoundError:
      if await orbax_layout.has_pytree_metadata_file(path):
        raise FileNotFoundError(
            "The checkpointable_name either does not exist or is missing"
            " Pytree checkpoint metadata. However, this current directory"
            " appears to be a valid Pytree checkpoint itself. Please consider"
            " loading with checkpointable_name=None."
        ) from None

      valid_pytree_checkpointable_names = (
          await orbax_layout.get_valid_pytree_names(path)
      )
      if valid_pytree_checkpointable_names:
        raise FileNotFoundError(
            "The checkpointable_name either does not exist or is missing Pytree"
            " checkpoint metadata. Please consider using one of the following"
            " valid pytree checkpointable names:"
            f" {valid_pytree_checkpointable_names}"
        ) from None
      raise FileNotFoundError(
          "The checkpointable_name either does not exist or is missing Pytree"
          " checkpoint metadata. There are no valid pytree checkpointables in"
          " this checkpoint"
      ) from None

    if not await orbax_layout.has_tensorstore_data_files(pytree_dir):
      logging.warning(
          "TensorStore data files not found in checkpoint path %s. This may be"
          " a sign of a malformed checkpoint, unless your checkpoint consists"
          " entirely of strings or other non-standard PyTree leaves.",
          path,
      )

  async def validate(self, path: Path) -> None:
    """Validates a V0 checkpoint directory.

    Args:
      path: The path to the checkpoint directory.

    Raises:
      InvalidLayoutError: If the path does not exist, or if the checkpoint is
        incomplete.
    """
    try:
      await self._validate(path)
    except BaseException as e:
      raise InvalidLayoutError(
          f"Failed to interpret path {path} as a V0 Orbax checkpoint."
          f" due to error encountered during validation: {e}"
      ) from e

  async def validate_pytree(
      self, path: Path, checkpointable_name: str | None
  ) -> None:
    """Validates the given path as a V0 PyTree checkpoint.

    Args:
      path: The path to the checkpoint directory.
      checkpointable_name: The name of the checkpointable to load. A
        subdirectory with this name must exist in `directory`. If None then
        `directory` is expected to contain the checkpoint directly. Defaults to
        `pytree`.

    Raises:
      InvalidLayoutError: If the path does not exist, or if the checkpoint is
        incomplete.
    """
    try:
      await self._validate_pytree(path, checkpointable_name)
    except BaseException as e:
      raise InvalidLayoutError(
          f"Failed to interpret path {path} as a V0 Orbax PyTree"
          f" checkpoint. Encountered error during validation: {e}"
      ) from e

  async def load_pytree(
      self,
      path: Path,
      checkpointable_name: str | None = None,
      abstract_pytree: (
          tree_types.PyTreeOf[tree_types.AbstractLeafType] | None
      ) = None,
  ) -> Awaitable[Any]:
    """Loads a V0 PyTree checkpoint.

    Attempts to load `checkpointable_name` pytree by finding its corresponding
    handler from the metadata. If `abstract_pytree` is provided, it attempts to
    load the checkpoint as a PyTree of the given abstract pytree.

    Args:
      path: The path to the checkpoint directory.
      checkpointable_name: The name of the pytree checkpointable to load.
      abstract_pytree: The abstract pytree to load.

    Returns:
      An awaitable containing the loaded PyTree.

    Raises:
      FileNotFoundError: If handler cannot be found for `checkpointable_name`.
    """
    if checkpointable_name is None:
      handler_for_load = await _get_handler_for_direct_load_pytree(
          path,
          self._handler_registry,
          abstract_pytree,
      )
      if handler_for_load is None:
        raise ValueError(
            f"Failed to resolve handler for checkpointable '{path.name}'."
        )
      result = await handler_for_load.load(
          path,
          abstract_pytree,
      )
      return result
    else:
      return await self._orbax_layout.load_pytree(
          path, checkpointable_name, abstract_pytree
      )

  async def load_checkpointables(
      self,
      path: Path,
      abstract_checkpointables: dict[str, Any] | None = None,
  ) -> Awaitable[dict[str, Any]]:
    """Loads checkpointables specified by `abstract_checkpointables`.

    Args:
      path: The path to the checkpoint directory.
      abstract_checkpointables: The abstract checkpointables to load.

    Returns:
      An awaitable containing the loaded checkpointables.
    """
    load_awaitable = await self._orbax_layout.load_checkpointables(
        path, abstract_checkpointables
    )
    return load_awaitable


async def _get_handler_for_direct_load_pytree(
    directory: path_types.Path,
    handler_registry: registration.CheckpointableHandlerRegistry,
    abstract_checkpointable: Any,
) -> handler_types.CheckpointableHandler  | None:
  """Returns a handler for direct load of a pytree checkpoint.

  1. Check if path name is explicitly registered in the handler registry.
  2. Check for handler_typestr in checkpoint metadata item_handlers.
  3. resolve_handler_for_load performs handler discovery based on 
  abstract_checkpointable type and handler_typestr.
  4. Default to pytree handler if no handler is resolved yet.

  Args:
    directory: The path to load the checkpoint metadata from.
    handler_registry: The handler registry to use for resolution.
    abstract_checkpointable: The abstract checkpointable to load.

  Returns:
    The handler for direct load of a pytree checkpoint.
  """
  checkpointable_name = directory.name
  # 1. Check if the handler is explicitly registered.
  resolved_handler = registration.get_registered_handler_by_name(
      handler_registry, checkpointable_name
  )
  if resolved_handler:
    return resolved_handler

  # 2. Find handler_typestr in checkpoint metadata item_handlers if checkpoint
  # metadata is in direct pytree format (str).
  metadata_handler_typestr = None
  checkpoint_metadata = await orbax_layout.read_checkpoint_metadata(directory)
  if isinstance(checkpoint_metadata.item_handlers, str):
    metadata_handler_typestr = checkpoint_metadata.item_handlers

  # 3. Resolve the handler using handler_typestr and
  # abstract_checkpointable type if either is specified.
  if abstract_checkpointable or metadata_handler_typestr:
    try:
      resolved_handler = registration.resolve_handler_for_load(
          handler_registry,
          abstract_checkpointable,
          name=checkpointable_name,
          handler_typestr=metadata_handler_typestr,
      )
    except registration.NoEntryError as e:
      logging.warning(
          "Failed to resolve handler for checkpointable %s: %s. Attempting to"
          " load using default pytree handler, otherwise defaulting to a"
          " None return value.",
          checkpointable_name,
          e,
      )
  else:
    logging.info(
        "No metadata present in checkpoint and no abstract checkpointable"
        " provided for checkpointable: \'%s\'. Attempting to load using"
        " pytree handler, otherwise defaulting to a None return value.",
        checkpointable_name,
    )

  # 4. If no handler is resolved yet, try to resolve using the default
  # pytree handler.
  if not resolved_handler:
    resolved_handler = registration.get_registered_handler_by_name(
        handler_registry, "pytree"
    )

  # 5. If no handler is resolved yet, default to None.
  return resolved_handler


