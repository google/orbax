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
from typing import Awaitable

from orbax.checkpoint._src import asyncio_utils
from orbax.checkpoint._src.path import async_path
from orbax.checkpoint._src.path import temporary_paths
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.handlers import registration
from orbax.checkpoint.experimental.v1._src.handlers import resolution as handler_resolution
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout
from orbax.checkpoint.experimental.v1._src.layout import orbax_layout
from orbax.checkpoint.experimental.v1._src.metadata import types as metadata_types
from orbax.checkpoint.experimental.v1._src.path import types as path_types


InvalidLayoutError = checkpoint_layout.InvalidLayoutError
Path = path_types.Path
CheckpointLayout = checkpoint_layout.CheckpointLayout
Checkpointable = checkpoint_layout.Checkpointable
AbstractCheckpointable = checkpoint_layout.AbstractCheckpointable


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
    asyncio_utils.run_sync(OrbaxV0Layout().validate_checkpointables(path))
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

  async def get_checkpointable_names(self, path: Path) -> list[str]:  # pyrefly: ignore[bad-override]
    """Returns candidate checkpointable names to use for loading.

    Checks all subdirectories and returns their names in an order that
    prioritizes the 'state' checkpointable name if present, and sorts the rest
    alphabetically.

    Args:
      path: The path to the checkpoint directory.

    Returns:
      A list of candidate checkpointable names to use for loading.
    """
    return await self._orbax_layout.get_checkpointable_names(path)

  async def _load_pytree_metadata(
      self,
      path: Path,
      metadata: handler_resolution.InternalCheckpointMetadata,
  ) -> metadata_types.CheckpointMetadata[AbstractCheckpointable]:
    handler_for_load = (
        await handler_resolution.get_handler_for_load_direct_pytree(
            path.name,
            self._handler_registry,
            None,
            metadata,
        )
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
        path=path,
        metadata=pytree_metadata,
        init_timestamp_nsecs=init_timestamp_nsecs,
        commit_timestamp_nsecs=commit_timestamp_nsecs,
        custom_metadata=custom_metadata,
    )

  async def _is_direct_checkpoint(self, path: Path) -> bool:
    """Returns True if `path` is a flat (direct) V0 checkpoint.

    A direct checkpoint stores a single PyTree at the root with no named
    checkpointable subdirectories. It is recognized either by checkpoint
    metadata whose `item_handlers` is a bare handler typestr (a string rather
    than a name->typestr dict), or by a PyTree metadata file at the root.

    Args:
      path: The path to the checkpoint directory.
    """
    checkpoint_metadata = await orbax_layout.read_checkpoint_metadata(path)
    return (  # pyrefly: ignore[bad-return]
        checkpoint_metadata
        and isinstance(checkpoint_metadata.item_handlers, str)
    ) or await orbax_layout.has_pytree_metadata_file(path)

  async def checkpointables_metadata(
      self, path: Path
  ) -> metadata_types.CheckpointMetadata[dict[str, AbstractCheckpointable]]:
    """Returns the metadata describing the composite Orbax checkpoint.

    Args:
      path: The path to the checkpoint directory.

    Returns:
      The metadata describing the Orbax checkpoint.

    Raises:
      InvalidLayoutError: If `path` is a flat (direct) checkpoint with no named
        checkpointables.
    """
    # A direct (flat) checkpoint has no named checkpointables; reject the
    # composite API and direct the caller to the flat API.
    if await self._is_direct_checkpoint(path):
      raise InvalidLayoutError(
          f"Checkpoint at {path} is a flat (direct) checkpoint with no"
          " named checkpointables. Use `ocp.metadata` / `ocp.load` instead."
      )
    # Otherwise this is a composite checkpoint; delegate to OrbaxLayout.
    return await self._orbax_layout.checkpointables_metadata(path)

  async def metadata(
      self, path: Path, checkpointable_name: str | None
  ) -> metadata_types.CheckpointMetadata[AbstractCheckpointable]:
    """Returns the metadata describing a single checkpointable in the V0 checkpoint."""
    checkpoint_metadata = await orbax_layout.read_checkpoint_metadata(path)
    if checkpointable_name is None:
      if not await self._is_direct_checkpoint(path):
        raise InvalidLayoutError(
            f"Checkpoint at {path} is a composite checkpoint with named"
            " checkpointables. Use `ocp.checkpointables_metadata` or"
            " `ocp.metadata(..., checkpointable_name='state')` instead of"
            " `ocp.metadata(..., checkpointable_name=None)`."
        )
      return await self._load_pytree_metadata(path, checkpoint_metadata)
    return await self._orbax_layout.metadata(path, checkpointable_name)

  async def _validate_checkpointables(self, path: Path) -> None:
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

    if await temporary_paths.is_path_temporary(path):
      raise ValueError(f"Found incomplete checkpoint at {path}.")

    if await orbax_layout.has_checkpoint_metadata_file(path.parent):
      raise InvalidLayoutError(
          f"The path ({path}) configured for loading appears to be a"
          " subdirectory of an Orbax checkpoint. Please try loading from the"
          f" parent directory: {path.parent} instead."
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
            "You are currently attempting to read an Orbax checkpoint from a"
            " root directory, please consider loading one of the following"
            f" checkpoint subdirectories: {checkpoint_subdirectories}"
        )

  async def _validate(self, path: Path, checkpointable_name: str | None):
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

  async def validate_checkpointables(self, path: Path) -> None:
    """Validates a V0 checkpoint directory.

    Args:
      path: The path to the checkpoint directory.

    Raises:
      InvalidLayoutError: If the path does not exist, or if the checkpoint is
        incomplete.
    """
    try:
      await self._validate_checkpointables(path)
    except (
        FileNotFoundError,
        NotADirectoryError,
        ValueError,
        InvalidLayoutError,
    ) as e:
      raise InvalidLayoutError(
          f"Failed to interpret path {path} as a V0 Orbax checkpoint."
      ) from e

  async def validate(self, path: Path, checkpointable_name: str | None) -> None:
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
      await self._validate(path, checkpointable_name)
    except (
        FileNotFoundError,
        NotADirectoryError,
        ValueError,
        InvalidLayoutError,
    ) as e:
      raise InvalidLayoutError(
          f"Failed to interpret path {path} as a V0 Orbax PyTree."
      ) from e

  async def load(
      self,
      path: Path,
      checkpointable_name: str | None = None,
      abstract_state: AbstractCheckpointable | None = None,
  ) -> Awaitable[Checkpointable]:
    """Loads a V0 PyTree checkpoint.

    Attempts to load `checkpointable_name` pytree by finding its corresponding
    handler from the metadata. If `abstract_state` is provided, it attempts to
    load the checkpoint as a PyTree of the given abstract pytree.

    Args:
      path: The path to the checkpoint directory.
      checkpointable_name: The name of the pytree checkpointable to load.
      abstract_state: The abstract pytree to load.

    Returns:
      An awaitable containing the loaded PyTree.

    Raises:
      FileNotFoundError: If handler cannot be found for `checkpointable_name`.
    """
    if checkpointable_name is None:
      checkpoint_metadata = await orbax_layout.read_checkpoint_metadata(path)
      if isinstance(checkpoint_metadata.item_handlers, dict):
        raise InvalidLayoutError(
            f"Checkpoint at {path} is a composite checkpoint with named"
            " checkpointables. Use `ocp.load_checkpointables` or"
            " `ocp.load(..., checkpointable_name='state')` instead of"
            " `ocp.load(..., checkpointable_name=None)`."
        )
      handler_for_load = (
          await handler_resolution.get_handler_for_load_direct_pytree(
              path.name,
              self._handler_registry,
              abstract_state,
              checkpoint_metadata,
          )
      )
      result = await handler_for_load.load(
          path,
          abstract_state,
      )
      return result
    else:
      return await self._orbax_layout.load(
          path, checkpointable_name, abstract_state
      )

  async def load_checkpointables(
      self,
      path: Path,
      abstract_checkpointables: dict[str, AbstractCheckpointable] | None = None,
  ) -> Awaitable[dict[str, Checkpointable]]:
    """Loads checkpointables specified by `abstract_checkpointables`.

    Args:
      path: The path to the checkpoint directory.
      abstract_checkpointables: The abstract checkpointables to load.

    Returns:
      An awaitable containing the loaded checkpointables.

    Raises:
      InvalidLayoutError: If `path` is a flat (direct) checkpoint with no named
        checkpointables.
    """
    if await self._is_direct_checkpoint(path):
      raise InvalidLayoutError(
          f"Checkpoint at {path} is a flat (direct checkpoint with no named"
          " checkpointables. Use `ocp.metadata` / `ocp.load` instead."
      )
    load_awaitable = await self._orbax_layout.load_checkpointables(
        path, abstract_checkpointables
    )
    return load_awaitable

  async def save_checkpointables(
      self,
      path: path_types.PathAwaitingCreation,
      *,
      checkpointables: dict[str, Checkpointable],
  ) -> Awaitable[None]:
    """Saves the checkpoint to the given directory."""
    raise NotImplementedError("Saving to Orbax V0 format is not supported.")
