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

import asyncio
from typing import Any, Awaitable

from absl import logging
from orbax.checkpoint._src.metadata import checkpoint as checkpoint_metadata
from orbax.checkpoint._src.path import async_path
from orbax.checkpoint._src.path import temporary_paths
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.handlers import composite_handler
from orbax.checkpoint.experimental.v1._src.handlers import registration
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout
from orbax.checkpoint.experimental.v1._src.loading import v0_compatibility
from orbax.checkpoint.experimental.v1._src.metadata import types as metadata_types
from orbax.checkpoint.experimental.v1._src.path import types as path_types


InvalidLayoutError = checkpoint_layout.InvalidLayoutError
Path = path_types.Path
CheckpointLayout = checkpoint_layout.CheckpointLayout

PYTREE_METADATA_FILE = "_METADATA"
ORBAX_CHECKPOINT_INDICATOR_FILE = "orbax.checkpoint"
_OCDBT_MANIFEST_FILE = "ocdbt.manifest"
_ZARRAY_FILE = ".zarray"


async def _has_pytree_metadata_file(path: Path) -> bool:
  return await async_path.exists(path / PYTREE_METADATA_FILE)


async def _has_ocdbt_manifest_file(path: Path) -> bool:
  return await async_path.exists(path / _OCDBT_MANIFEST_FILE)


async def _has_zarray_files(path: Path) -> bool:
  paths = list(await async_path.iterdir(path))
  awaitables = [async_path.exists(p / _ZARRAY_FILE) for p in paths]
  return any(await asyncio.gather(*awaitables))


async def _has_tensorstore_data_files(path: Path) -> bool:
  return await _has_ocdbt_manifest_file(path) or await _has_zarray_files(path)


class V0Layout(CheckpointLayout):
  """V0Layout.

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

  @property
  def path(self) -> Path:
    return self._path

  async def metadata(self) -> metadata_types.CheckpointMetadata[dict[str, Any]]:
    # Uses the v0 checkpointer to get v0 StepMetadata
    checkpointer, _ = v0_compatibility.get_v0_checkpointer_and_args(
        self._path, None, context=context_lib.get_context()
    )
    step_metadata = checkpointer.metadata(self._path)

    item_metadata = {k: v for k, v in step_metadata.item_metadata.items()}
    # Exclude `metrics` if present.
    item_metadata.pop("metrics", None)

    return metadata_types.CheckpointMetadata[dict[str, Any]](
        metadata=item_metadata,
        init_timestamp_nsecs=step_metadata.init_timestamp_nsecs,
        commit_timestamp_nsecs=step_metadata.commit_timestamp_nsecs,
        custom_metadata=step_metadata.custom_metadata,
    )

  async def _validate(self) -> None:
    """Validates a V0 checkpoint directory."""
    if not await async_path.exists(self.path):
      raise FileNotFoundError(f"Checkpoint path {self.path} does not exist.")
    if not await async_path.is_dir(self.path):
      raise NotADirectoryError(
          f"Checkpoint path {self.path} is not a directory."
      )
    context = context_lib.get_context()
    if await temporary_paths.is_path_temporary(
        self.path,
        temporary_path_cls=context.file_options.temporary_path_class,
    ):
      raise ValueError(f"Found incomplete checkpoint at {self.path}.")

    # Path points to a single step checkpoint with valid checkpoint metadata.
    if await async_path.exists(
        checkpoint_metadata.step_metadata_file_path(self.path)
    ):
      return

    # The path itself points to a PyTree checkpointable.
    if await async_path.exists(self.path / PYTREE_METADATA_FILE):
      return

    subpaths = await async_path.iterdir(self.path)
    # The path points to a directory containing at least one PyTree
    # checkpointable.
    for subpath in subpaths:
      if await async_path.is_dir(subpath) and await async_path.exists(
          subpath / PYTREE_METADATA_FILE
      ):
        return

    raise ValueError(
        f"Checkpoint path {self.path} is not a valid V0 checkpoint."
    )

  async def _validate_pytree(self, checkpointable_name: str | None) -> None:
    """Validates the given path as a V0 PyTree checkpoint."""
    pytree_dir = (
        self.path
        if checkpointable_name is None
        else self.path / checkpointable_name
    )

    if checkpointable_name is not None and not await async_path.exists(
        pytree_dir
    ):
      raise FileNotFoundError(
          f"Checkpoint path {self.path} must contain a subdirectory named"
          f' "{checkpointable_name}".'
      )

    if not await _has_pytree_metadata_file(pytree_dir):
      # TODO(angelmau): Add following details to the error message:
      # 1. we should check other available subdirectories and see if any of them
      #   look like PyTree checkpoints, and instruct the user to consider
      #   whether they meant to specify any of those.
      # 2. we need to check the directory - if it contains PyTree files, suggest
      #   loading with checkpointable_name=None
      raise FileNotFoundError(
          f"Checkpoint path {self.path} does not contain a PyTree metadata"
          " file."
      )

    if not await _has_tensorstore_data_files(pytree_dir):
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

  async def load(
      self,
      abstract_checkpointables: dict[str, Any] | None = None,
  ) -> Awaitable[dict[str, Any]]:
    load_awaitable = await self._composite_handler.load(
        self._path, abstract_checkpointables
    )
    return load_awaitable
