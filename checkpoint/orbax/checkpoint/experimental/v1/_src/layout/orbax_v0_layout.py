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

"""Defines `V0Layout`, a class to handle Orbax V0 checkpoint formats."""

import logging
from typing import Any, Awaitable

from orbax.checkpoint._src.metadata import step_metadata_serialization
from orbax.checkpoint._src.path import async_path
from orbax.checkpoint._src.path import temporary_paths
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.handlers import composite_handler
from orbax.checkpoint.experimental.v1._src.handlers import pytree_handler
from orbax.checkpoint.experimental.v1._src.handlers import registration
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout
from orbax.checkpoint.experimental.v1._src.layout import orbax_layout
from orbax.checkpoint.experimental.v1._src.metadata import types as metadata_types
from orbax.checkpoint.experimental.v1._src.path import types as path_types


InvalidLayoutError = checkpoint_layout.InvalidLayoutError
Path = path_types.Path
CheckpointLayout = checkpoint_layout.CheckpointLayout


class OrbaxV0Layout(CheckpointLayout):
  """OrbaxV0Layout.

  This class handles Orbax V0 checkpoint formats. It inherits
  abstract methods from CheckpointLayout. It performs a few core functions:
    - Resolves handlers for saving and loading.
    - Saves and loads checkpointables to/from individual subdirectories by
    delegating to the resolved handlers.
  """

  def __init__(self, path: Path):
    self._context = context_lib.get_context()
    self._handler_registry = registration.local_registry(
        self._context.checkpointables_options.registry,
        include_global_registry=False,
    )
    self._composite_handler = composite_handler.CompositeHandler(
        self._handler_registry
    )
    self._path = path
    self._orbax_layout = orbax_layout.OrbaxLayout(path)

  @property
  def path(self) -> Path:
    return self._path

  async def _load_pytree_metadata(
      self,
  ) -> metadata_types.CheckpointMetadata[dict[str, Any]]:
    """Loads the metadata for a PyTree checkpoint."""
    handler = pytree_handler.PyTreeHandler(context=self._context)
    v0_metadata = await handler.metadata(self.path)

    init_timestamp_nsecs = None
    commit_timestamp_nsecs = None
    custom_metadata = None

    if await orbax_layout.has_checkpoint_metadata_file(self.path):
      checkpoint_metadata_path = self.path
    elif await orbax_layout.has_checkpoint_metadata_file(self.path.parent):
      checkpoint_metadata_path = self.path.parent
    else:
      checkpoint_metadata_path = None

    if checkpoint_metadata_path is not None:
      saved_metadata = step_metadata_serialization.get_step_metadata(
          checkpoint_metadata_path
      )
      init_timestamp_nsecs = saved_metadata.init_timestamp_nsecs
      commit_timestamp_nsecs = saved_metadata.commit_timestamp_nsecs
      custom_metadata = saved_metadata.custom_metadata

    return metadata_types.CheckpointMetadata(
        metadata=v0_metadata,
        init_timestamp_nsecs=init_timestamp_nsecs,
        commit_timestamp_nsecs=commit_timestamp_nsecs,
        custom_metadata=custom_metadata,
    )

  async def metadata(self) -> metadata_types.CheckpointMetadata[dict[str, Any]]:
    """Returns the metadata describing the Orbax checkpoint."""
    # If our directory has a PyTree saved directly to the root, we read it as
    # a PyTree checkpoint.
    if await orbax_layout.has_pytree_metadata_file(self.path):
      return await self._load_pytree_metadata()

    return await self._orbax_layout.metadata()

  async def _validate(self) -> None:
    """Validates a checkpoint directory.

    Must be:
    - Existing
    - A directory.
    - Not a temporary path.

    Raises:
      FileNotFoundError: If the path does not exist.
      NotADirectoryError: If the path is not a directory.
      ValueError: If the checkpoint is incomplete.
    """
    if not await async_path.exists(self.path):
      raise FileNotFoundError(f"Checkpoint path {self.path} does not exist.")

    if not await async_path.is_dir(self.path):
      raise NotADirectoryError(
          f"Checkpoint path {self.path} is not a directory."
      )

    if await temporary_paths.is_path_temporary(
        self.path,
        temporary_path_cls=self._context.file_options.temporary_path_class,
    ):
      raise ValueError(f"Found incomplete checkpoint at {self.path}.")

    if not await orbax_layout.has_checkpoint_metadata_file(self.path):
      if await orbax_layout.has_pytree_metadata_file(self.path):
        return

      checkpoint_subdirectories = []
      for subdir in await orbax_layout.get_subpaths(self.path):
        if await orbax_layout.has_checkpoint_metadata_file(subdir):
          checkpoint_subdirectories.append(subdir)
      if checkpoint_subdirectories:
        raise InvalidLayoutError(
            "You are currently attempting to read from a root directory, please"
            " consider loading one of the following checkpoint subdirectories: "
            f"{checkpoint_subdirectories}"
        )

      for subdir in await orbax_layout.get_subpaths(self.path):
        if await orbax_layout.has_pytree_metadata_file(subdir):
          return

      raise InvalidLayoutError(
          f"Checkpoint path {self.path} does not contain any valid V0"
          " checkpoint metadata."
      )

  async def _validate_pytree(self, checkpointable_name: str | None):
    """Validates a checkpoint path written by `ocp.save_pytree`.

    Args:
      checkpointable_name: The name of the checkpointable to load. A
        subdirectory with this name must exist in `directory`. If None then
        `directory` is expected to contain the checkpoint directly. Defaults to
        `pytree`.

    Raises:
      FileNotFoundError: If the path does not exist, or if `pytree` is not found
        in the directory
      ValueError: If the PyTree checkpoint is malformed.
    """
    pytree_dir = (
        self.path
        if checkpointable_name is None
        else self.path / checkpointable_name
    )

    if not await async_path.exists(
        pytree_dir
    ) or not await orbax_layout.has_pytree_metadata_file(pytree_dir):
      # 1. we should check other available subdirectories and see if any of them
      #   look like PyTree checkpoints, and instruct the user to consider
      #   whether they meant to specify any of those.
      pytree_checkpointable_names = []
      for subdir in await orbax_layout.get_subpaths(self.path):
        if await orbax_layout.has_pytree_metadata_file(subdir):
          pytree_checkpointable_names.append(subdir.name)
      # 2. Check checkpoint root directory if it is a PyTree checkpoint, suggest
      #   loading with checkpointable_name=None
      if await orbax_layout.has_pytree_metadata_file(self.path):
        pytree_checkpointable_names.append(None)

      if pytree_checkpointable_names:
        raise FileNotFoundError(
            "checkpointable_name either does not exist or is missing Pytree"
            " checkpoint metadataPlease consider using one of the following"
            " valid pytree checkpointable_names:"
            f" {pytree_checkpointable_names}"
        )
      raise FileNotFoundError(
          "checkpointable_name either does not exist or is missing Pytree"
          " checkpoint metadata There are no valid pytree checkpointables in"
          " this checkpoint"
      )

    if not await orbax_layout.has_tensorstore_data_files(pytree_dir):
      logging.warning(
          "TensorStore data files not found in checkpoint path %s. This may be"
          " a sign of a malformed checkpoint, unless your checkpoint consists"
          " entirely of strings or other non-standard PyTree leaves.",
          self.path,
      )

  async def validate(self) -> None:
    """Validates a V0 checkpoint directory."""
    try:
      await self._validate()
    except BaseException as e:
      raise InvalidLayoutError(
          f"Failed to interpret path {self._path} as an Orbax V0 checkpoint."
      ) from e

  async def validate_pytree(self, checkpointable_name: str | None) -> None:
    """Validates the given path as a V0 PyTree checkpoint."""
    try:
      await self._validate_pytree(checkpointable_name)
    except BaseException as e:
      raise InvalidLayoutError(
          f"Failed to interpret path {self._path} as an Orbax V0 PyTree"
          " checkpoint."
      ) from e

  async def load_pytree(
      self,
      checkpointable_name: str | None = None,
      abstract_pytree: Any | None = None,
  ) -> Awaitable[Any]:
    # Determine the directory, either root or checkpointable.
    pytree_dir = (
        self.path
        if checkpointable_name is None
        else self.path / checkpointable_name
    )

    handler = pytree_handler.PyTreeHandler(context=self._context)

    load_awaitable = await handler.load(pytree_dir, abstract_pytree)
    return load_awaitable

  async def load_checkpointables(
      self,
      abstract_checkpointables: dict[str, Any] | None = None,
  ) -> Awaitable[dict[str, Any]]:
    load_awaitable = await self._orbax_layout.load_checkpointables(
        abstract_checkpointables
    )
    return load_awaitable
