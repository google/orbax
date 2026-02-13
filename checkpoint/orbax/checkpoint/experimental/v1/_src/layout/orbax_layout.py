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

"""Defines `OrbaxLayout`, a class to handle Orbax checkpoint formats."""

import asyncio
import enum
from typing import Any, Awaitable

from absl import logging
from orbax.checkpoint._src.metadata import checkpoint as checkpoint_metadata
from orbax.checkpoint._src.metadata import step_metadata_serialization
from orbax.checkpoint._src.path import async_path
from orbax.checkpoint._src.path import temporary_paths
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.handlers import composite_handler
from orbax.checkpoint.experimental.v1._src.handlers import pytree_handler
from orbax.checkpoint.experimental.v1._src.handlers import registration
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout
from orbax.checkpoint.experimental.v1._src.loading import v0_compatibility
from orbax.checkpoint.experimental.v1._src.metadata import serialization as metadata_serialization
from orbax.checkpoint.experimental.v1._src.metadata import types as metadata_types
from orbax.checkpoint.experimental.v1._src.path import types as path_types
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types


class CheckpointVersion(enum.Enum):
  V0 = 0
  V1 = 1


InvalidLayoutError = checkpoint_layout.InvalidLayoutError
CompositeHandler = composite_handler.CompositeHandler
Path = path_types.Path
CheckpointLayout = checkpoint_layout.CheckpointLayout

PYTREE_METADATA_FILE = "_METADATA"
ORBAX_CHECKPOINT_INDICATOR_FILE = "orbax.checkpoint"

_OCDBT_MANIFEST_FILE = "ocdbt.manifest"
_ZARRAY_FILE = ".zarray"


_V0_ERROR_MESSAGE = (
    "If your checkpoint was saved with the Orbax V0 API, please follow the"
    " instructions at"
    " https://orbax.readthedocs.io/en/latest/guides/checkpoint/v1/orbax_v0_to_v1_migration.html"
    " to load it with the Orbax V1 API."
)
_GENERAL_ERROR_MESSAGE = (
    " Note that a valid checkpoint path should always contain a file named"
    f" '{ORBAX_CHECKPOINT_INDICATOR_FILE}' (unless it was saved with the V0"
    f" API). {_V0_ERROR_MESSAGE}"
)


def checkpoint_version(path: path_types.PathLike) -> CheckpointVersion:
  """Returns the checkpoint version of the given path."""
  if (path / ORBAX_CHECKPOINT_INDICATOR_FILE).exists():
    return CheckpointVersion.V1
  else:
    return CheckpointVersion.V0


async def _subpaths(directory: Path) -> list[Path]:
  """Returns subdirectories up to a limit."""
  return list(await async_path.iterdir(directory))


def is_orbax_checkpoint(path: path_types.PathLike) -> bool:
  """Determines if the given path is an Orbax checkpoint.

  Args:
    path: The path to the checkpoint directory.

  Returns:
    True if the path is an Orbax checkpoint, False otherwise.
  """
  ctx = context_lib.get_context()
  path = ctx.file_options.path_class(path)
  try:
    asyncio.run(OrbaxLayout().validate(path))
    return True
  except InvalidLayoutError:
    return False


async def _has_ocdbt_manifest_file(path: Path) -> bool:
  return await async_path.exists(path / _OCDBT_MANIFEST_FILE)


async def _has_zarray_files(path: Path) -> bool:
  paths = list(await async_path.iterdir(path))
  awaitables = [async_path.exists(p / _ZARRAY_FILE) for p in paths]
  return any(await asyncio.gather(*awaitables))


async def has_tensorstore_data_files(path: Path) -> bool:
  return await _has_ocdbt_manifest_file(path) or await _has_zarray_files(path)


async def has_pytree_metadata_file(path: Path) -> bool:
  return await async_path.exists(path / PYTREE_METADATA_FILE)


async def has_indicator_file(path: Path) -> bool:
  """Checks if the indicator file exists in the given path."""
  return await async_path.exists(path / ORBAX_CHECKPOINT_INDICATOR_FILE)


class OrbaxLayout(CheckpointLayout):
  """OrbaxLayout.

  This class defines a class to handle Orbax checkpoint formats. It inherits
  abstract methods from :py:class:`~.CheckpointLayout`.
  It performs a few core functions:
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
    self._composite_handler = CompositeHandler(self._handler_registry)
    self._metadata_store = checkpoint_metadata.metadata_store(enable_write=True)

  async def metadata(
      self, path: Path
  ) -> metadata_types.CheckpointMetadata[dict[str, Any]]:
    """Returns the metadata describing the Orbax checkpoint."""
    # Uses the v0 checkpointer to get v0 StepMetadata
    checkpointer, _ = v0_compatibility.get_v0_checkpointer_and_args(
        path, None, context=context_lib.get_context()
    )
    step_metadata = checkpointer.metadata(path)

    item_metadata = {k: v for k, v in step_metadata.item_metadata.items()}
    # Exclude `metrics` if present. This is relevant only for
    # `training.Checkpointer`, and is separately added to the
    # `training.CheckpointMetadata` object.
    item_metadata.pop("metrics", None)

    return metadata_types.CheckpointMetadata[dict[str, Any]](
        metadata=item_metadata,
        init_timestamp_nsecs=step_metadata.init_timestamp_nsecs,
        commit_timestamp_nsecs=step_metadata.commit_timestamp_nsecs,
        custom_metadata=step_metadata.custom_metadata,
    )

  async def _validate_pytree(self, path: Path, checkpointable_name: str | None):
    """Validates a checkpoint path written by `ocp.save_pytree`.

    Args:
      path: The path to the checkpoint directory.
      checkpointable_name: The name of the checkpointable to load. A
        subdirectory with this name must exist in `directory`. If None then
        `directory` is expected to contain the checkpoint directly. Defaults to
        `pytree`.

    Raises:
      FileNotFoundError: If the path does not exist, or if `pytree` is not found
        in the directory
      ValueError: If the PyTree checkpoint is malformed.
    """
    # TODO(b/476156780): Remove v0 logic from V1 OrbaxLayout

    # If it's a V1 checkpoint, it's not valid for the PyTree to be saved
    # directly to the checkpoint directory.
    if (
        checkpoint_version(path) == CheckpointVersion.V1
        and checkpointable_name is None
    ):
      raise FileNotFoundError(
          "Cannot load a V1 checkpoint directly as a PyTree checkpointable."
      )

    # Determine the directory, either root or checkpointable.
    pytree_dir = (
        path if checkpointable_name is None else path / checkpointable_name
    )

    # Check if the directory exists and has PyTree metadata.
    if not await async_path.exists(
        pytree_dir
    ) or not await has_pytree_metadata_file(pytree_dir):
      # 1. we should check other available subdirectories and see if any of them
      #   look like PyTree checkpoints, and instruct the user to consider
      #   whether they meant to specify any of those.

      pytree_checkpointable_names = []
      for subdir in await _subpaths(path):
        if await has_pytree_metadata_file(subdir):
          pytree_checkpointable_names.append(subdir.name)
      # 2. Check checkpoint root directory if it is a PyTree checkpoint, suggest
      #   loading with checkpointable_name=None
      if await has_pytree_metadata_file(path):
        pytree_checkpointable_names.append(None)

      if pytree_checkpointable_names:
        raise FileNotFoundError(
            "checkpointable_name either does not exist or is missing Pytree"
            " checkpoint metadata. Please consider using one of the following"
            " valid pytree checkpointable_names:"
            f" {pytree_checkpointable_names}"
        )
      raise FileNotFoundError(
          "checkpointable_name either does not exist or is missing Pytree"
          " checkpoint metadata. There are no valid pytree checkpointables in"
          " this checkpoint"
      )

    if not await has_tensorstore_data_files(pytree_dir):
      logging.warning(
          "TensorStore data files not found in checkpoint path %s. This may be"
          " a sign of a malformed checkpoint, unless your checkpoint consists"
          " entirely of strings or other non-standard PyTree leaves.",
          path,
      )

  async def _validate(self, path: Path):
    """Validates a checkpoint directory.

    Must be:
    - Existing
    - A directory.
    - Not a temporary path.
    - OR
      - Has orbax.checkpoint indicator file.
      - OR
        - Has _CHECKPOINT_METADATA file.
        - A subdirectory has _METADATA file (PyTree checkpoint).

    Args:
      path: The path to the checkpoint directory.

    Raises:
      FileNotFoundError: If the path does not exist.
      NotADirectoryError: If the path is not a directory.
      ValueError: If the checkpoint is incomplete.
    """
    # TODO(b/476156780): Remove v0 logic from V1 OrbaxLayout
    if not await async_path.exists(path):
      raise FileNotFoundError(f"Checkpoint path {path} does not exist.")

    if not await async_path.is_dir(path):
      raise NotADirectoryError(f"Checkpoint path {path} is not a directory.")

    if await temporary_paths.is_path_temporary(
        path,
        temporary_path_cls=self._context.file_options.temporary_path_class,
    ):
      raise ValueError(f"Found incomplete checkpoint at {path}.")

    subpaths = await _subpaths(path)

    # Pass validation immediately if the indicator file is present.
    if ORBAX_CHECKPOINT_INDICATOR_FILE in [p.name for p in subpaths]:
      return

    # Path points to a checkpoint with valid metadata.
    if await async_path.exists(
        metadata_serialization.checkpoint_metadata_file_path(path)
    ):
      return

    # The path itself points to a PyTree checkpointable.
    if await has_pytree_metadata_file(path):
      return
    # The path points to a directory containing at least one PyTree
    # checkpointable.
    for subpath in subpaths:
      if await async_path.is_dir(subpath) and await has_pytree_metadata_file(
          subpath
      ):
        return

    raise FileNotFoundError(
        f"Checkpoint path {path} could not be identified as a valid Orbax"
        " checkpoint. The path must conform to one of the following"
        " conditions:\n  - Contain the indicator file"
        f" {ORBAX_CHECKPOINT_INDICATOR_FILE}. This should be true of all"
        " checkpoints saved with the Orbax V1 API. If not present, the"
        " checkpoint may have been saved with the V0 API.\n  - Contain the"
        " _CHECKPOINT_METADATA file.\n  - Point directly to a PyTree"
        " checkpointable (contain _METADATA file).\n  - Contain a subdirectory"
        " which is a PyTree checkpointable (contain _METADATA file).\n"
    )

  async def validate(self, path: Path):
    try:
      await self._validate(path)
    except BaseException as e:
      raise InvalidLayoutError(
          f"Failed to interpret path {path} as an Orbax checkpoint."
          f" {_GENERAL_ERROR_MESSAGE}"
      ) from e

  async def validate_pytree(
      self, path: Path, checkpointable_name: str | None
  ) -> None:
    """Validates the given path as a PyTree checkpoint."""
    try:
      await self._validate_pytree(path, checkpointable_name)
    except BaseException as e:
      raise InvalidLayoutError(
          f"Failed to interpret path {path} as an Orbax PyTree"
          f" checkpoint. {_GENERAL_ERROR_MESSAGE}"
      ) from e

  def _get_typestr(
      self, path: Path, checkpointable_name: str | None
  ) -> str | None:
    """Gets the typestr for the given path, falling back to parent if needed."""
    # TODO(b/476156780): Remove complex V0 handler resolution logic out of V1
    # OrbaxLayout and re-evaulate implementation

    # Attempt to get typestr from the step metadata file in the current
    # checkpoint path.
    metadata_path = checkpoint_metadata.step_metadata_file_path(path)
    if metadata_path.exists():
      serialized = self._metadata_store.read(metadata_path)
      if serialized:
        metadata = step_metadata_serialization.deserialize(serialized or {})
        # If checkpoint is V0 and pytree is saved directly to checkpoint,
        # we expect a single string type for a PyTree in the metadata.
        if checkpointable_name is None:
          if isinstance(metadata.item_handlers, str):
            return metadata.item_handlers
        else:
          if isinstance(metadata.item_handlers, dict):
            return metadata.item_handlers.get(checkpointable_name)

    # For pytree checkpointable directory, if direct path didn't yield a typestr
    # we try the parent path.
    if checkpointable_name is None:
      parent_metadata_path = checkpoint_metadata.step_metadata_file_path(
          path.parent
      )
      if parent_metadata_path.exists():
        serialized = self._metadata_store.read(parent_metadata_path)
        if serialized:
          metadata = step_metadata_serialization.deserialize(serialized or {})
          if isinstance(metadata.item_handlers, dict):
            return metadata.item_handlers.get(path.name)
      return None

  async def load_pytree(
      self,
      path: Path,
      checkpointable_name: str | None = None,
      abstract_pytree: (
          tree_types.PyTreeOf[tree_types.AbstractLeafType] | None
      ) = None,
  ) -> Awaitable[Any]:
    typestr = self._get_typestr(path, checkpointable_name)
    name_for_registration = checkpointable_name or path.name

    if typestr:
      handler = registration.resolve_handler_for_load(
          self._handler_registry,
          abstract_pytree,
          name=name_for_registration,
          handler_typestr=typestr,
      )
    # TODO(b/476156780): Remove from V1 OrbaxLayout and re-evaulate resolution
    # logic

    # If missing _CHECKPOINT_METADATA and its a V0 pytree checkpoint, check
    # if it has _METADATA
    elif checkpointable_name is None and await has_pytree_metadata_file(path):
      handler = pytree_handler.PyTreeHandler(context=self._context)
    else:
      raise ValueError(
          "Could not find handler information for the given checkpointable"
          f" name: {checkpointable_name} in path: {path}."
      )

    pytree_dir = (
        path if checkpointable_name is None else path / checkpointable_name
    )
    load_awaitable = await handler.load(pytree_dir, abstract_pytree)
    return load_awaitable

  async def load_checkpointables(
      self,
      path: Path,
      abstract_checkpointables: dict[str, Any] | None = None,
  ) -> Awaitable[dict[str, Any]]:
    load_awaitable = await self._composite_handler.load(
        path, abstract_checkpointables
    )
    return load_awaitable

  async def save(
      self,
      path: path_types.PathAwaitingCreation,
      *,
      checkpointables: dict[str, Any],
  ) -> Awaitable[None]:
    """Saves the checkpoint to the given directory."""
    save_awaitable = await self._composite_handler.save(
        path, checkpointables
    )
    return save_awaitable
