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
from orbax.checkpoint._src import asyncio_utils
from orbax.checkpoint._src.metadata import step_metadata_serialization
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.path import async_path
from orbax.checkpoint._src.path import temporary_paths
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.handlers import registration
from orbax.checkpoint.experimental.v1._src.handlers import resolution as handler_resolution
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout
from orbax.checkpoint.experimental.v1._src.metadata import serialization as metadata_serialization
from orbax.checkpoint.experimental.v1._src.metadata import types as metadata_types
from orbax.checkpoint.experimental.v1._src.path import types as path_types
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types


class CheckpointVersion(enum.Enum):
  V0 = 0
  V1 = 1


InvalidLayoutError = checkpoint_layout.InvalidLayoutError
Path = path_types.Path
CheckpointLayout = checkpoint_layout.CheckpointLayout
InternalCheckpointMetadata = (
    step_metadata_serialization.InternalCheckpointMetadata
)

PYTREE_METADATA_FILE = "_METADATA"
ORBAX_CHECKPOINT_INDICATOR_FILE = "orbax.checkpoint"
CHECKPOINT_METADATA = "_CHECKPOINT_METADATA"

_OCDBT_MANIFEST_FILE = "ocdbt.manifest"
_ZARRAY_FILE = ".zarray"


async def checkpoint_version(path: path_types.PathLike) -> CheckpointVersion:
  """Returns the checkpoint version of the given path."""
  if await has_indicator_file(path):
    return CheckpointVersion.V1
  else:
    return CheckpointVersion.V0


async def get_subpaths(directory: Path) -> list[Path]:
  """Returns subdirectories up to a limit."""
  return list(await async_path.iterdir(directory))


def is_orbax_v1_checkpoint(path: path_types.PathLike) -> bool:
  """Determines if the given path is a Orbax checkpoint.

  Args:
    path: The path to the checkpoint directory.

  Returns:
    True if the path is a V1 Orbax checkpoint, False otherwise.
  """

  ctx = context_lib.get_context()
  path = ctx.file_options.path_class(path)
  try:
    asyncio_utils.run_sync(OrbaxLayout().validate(path))
    return True
  except InvalidLayoutError:
    return False


async def _has_ocdbt_manifest_file(path: Path) -> bool:
  return await async_path.exists(path / _OCDBT_MANIFEST_FILE)


async def _has_zarray_files(path: Path) -> bool:
  paths = await get_subpaths(path)
  return any(
      await asyncio.gather(
          *[async_path.exists(p / _ZARRAY_FILE) for p in paths]
      )
  )


async def has_tensorstore_data_files(path: Path) -> bool:
  return await _has_ocdbt_manifest_file(path) or await _has_zarray_files(path)


async def has_pytree_metadata_file(path: Path) -> bool:
  return await async_path.exists(path / PYTREE_METADATA_FILE)


async def has_indicator_file(path: Path) -> bool:
  """Checks if the indicator file exists in the given path."""
  return await async_path.exists(path / ORBAX_CHECKPOINT_INDICATOR_FILE)


async def has_checkpoint_metadata_file(path: Path) -> bool:
  return await async_path.exists(path / CHECKPOINT_METADATA)


async def get_valid_pytree_names(path: Path) -> list[str]:
  subpaths = await get_subpaths(path)
  is_pytree_checkpoints = await asyncio.gather(
      *[has_pytree_metadata_file(s) for s in subpaths]
  )
  return [
      subdir.name
      for subdir, is_pytree in zip(subpaths, is_pytree_checkpoints)
      if is_pytree
  ]


async def _existing_checkpointable_names(
    directory: path_types.Path,
) -> list[str]:
  subpaths = await get_subpaths(directory)
  is_dir_checks = await asyncio.gather(
      *[async_path.is_dir(p) for p in subpaths]
  )
  return [
      subdir.name for subdir, is_dir in zip(subpaths, is_dir_checks) if is_dir
  ]


async def _create_orbax_identifier_file(
    directory: path_types.PathAwaitingCreation, primary_host: int | None
):
  """Creates a file called `orbax.checkpoint` for easy identification."""
  directory = await directory.await_creation()
  if multihost.is_primary_host(primary_host):
    # We allow the indicator file to already exist, in case we are performing
    # partial saving to a checkpoint in which the indicator is already present.
    await async_path.touch(
        directory / ORBAX_CHECKPOINT_INDICATOR_FILE, exist_ok=True
    )


async def _read_checkpoint_metadata(
    directory: path_types.Path,
) -> InternalCheckpointMetadata:
  """Returns the step metadata for a given path."""
  serialized_metadata = (
      await metadata_serialization.read(
          metadata_serialization.checkpoint_metadata_file_path(directory)
      )
      or {}
  )
  return InternalCheckpointMetadata.deserialize(serialized_metadata)


class OrbaxLayout(CheckpointLayout):
  """OrbaxLayout.

  This class defines a class to handle Orbax checkpoint formats. It inherits
  abstract methods from :py:class:`~.CheckpointLayout`.
  It performs a few core functions:
    - Validates the checkpoint directory.
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

  async def metadata(
      self, path: Path
  ) -> metadata_types.CheckpointMetadata[dict[str, Any]]:
    """Returns the metadata describing the Orbax checkpoint."""
    checkpoint_metadata = await _read_checkpoint_metadata(
        path
    )
    handlers_for_load = await handler_resolution.get_handlers_for_load(
        path, self._handler_registry, {}, checkpoint_metadata
    )
    existing_checkpointable_names = await _existing_checkpointable_names(path)
    abstract_checkpointables = {
        name: None
        for name in handlers_for_load.keys()
        if name in existing_checkpointable_names
    }
    if any(
        name not in existing_checkpointable_names
        for name in abstract_checkpointables.keys()
    ):
      raise KeyError(
          "Inferred checkpointables from metadata:"
          f" {abstract_checkpointables.keys()} for loading were not found in"
          " the checkpoint. Available checkpointables:"
          f" {existing_checkpointable_names}"
      )

    # Default to none for all existing checkpointable names, for
    # subdirectories that we are unable to find a handler for and load.
    item_metadata = {name: None for name in existing_checkpointable_names}
    for checkpointable_name in abstract_checkpointables.keys():
      handler = handlers_for_load[checkpointable_name]
      item_metadata[checkpointable_name] = await handler.metadata(
          path / checkpointable_name
      )
    # Exclude `metrics` if present. This is relevant only for
    # `training.Checkpointer`, and is separately added to the
    # `training.CheckpointMetadata` object.
    item_metadata.pop("metrics", None)

    return metadata_types.CheckpointMetadata[dict[str, Any]](
        metadata=item_metadata,
        init_timestamp_nsecs=checkpoint_metadata.init_timestamp_nsecs,
        commit_timestamp_nsecs=checkpoint_metadata.commit_timestamp_nsecs,
        custom_metadata=checkpoint_metadata.custom_metadata,
    )

  async def _validate_pytree(self, path: Path, checkpointable_name: str | None):
    """Validates checkpoint written by `save_pytree` or `save_checkpointables`.

    Validates that checkpointable_name is a Pytree checkpoint by verifying its
    path contains the required metadata files.

    Args:
      path: The path to the checkpoint directory.
      checkpointable_name: The name of the checkpointable to load. For Orbax V1,
        a subdirectory with this name must exist in `directory`.

    Raises:
      FileNotFoundError: If the path does not exist, or if
        `checkpointable_name` pytree is not found in the directory
      ValueError: If the PyTree checkpoint is malformed or user passed invalid
        `checkpointable_name`.
    """
    if checkpointable_name is None:
      raise ValueError(
          f"Attempting to load V1 checkpoint at {path} with"
          " `checkpointable_name=None`. This is only supported for legacy V0"
          " checkpoints. Please specify the name of the checkpointable to load."
          " Otherwise, omit `checkpointable_name` to load default 'pytree'"
          " checkpointable."
      )

    pytree_dir = path / checkpointable_name

    try:
      if not await async_path.exists(pytree_dir):
        raise FileNotFoundError
      elif not await has_pytree_metadata_file(pytree_dir):
        raise FileNotFoundError
    except FileNotFoundError:
      valid_pytree_checkpointable_names = await get_valid_pytree_names(path)
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

    if not await has_tensorstore_data_files(pytree_dir):
      logging.warning(
          "TensorStore data files not found in checkpoint path %s. This may be"
          " a sign of a malformed checkpoint, unless your checkpoint consists"
          " entirely of strings or other non-standard PyTree leaves.",
          path,
      )

  async def _validate(self, path: Path):
    """Validates a checkpoint directory to be a V1 Orbax checkpoint.

    Must fulfill all of the following:
    - Existing
    - A directory
    - Not a temporary path
    - Has orbax.checkpoint indicator file

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

    if not await has_checkpoint_metadata_file(path):
      raise FileNotFoundError(
          f"Checkpoint path {path} could not be identified as a valid Orbax"
          " V1 checkpoint. It is missing the checkpoint metadata file"
          f" '{CHECKPOINT_METADATA}'."
      )

    # Pass validation immediately if the indicator file is present.
    if await has_indicator_file(path):
      return
    raise FileNotFoundError(
        f"Checkpoint path {path} could not be identified as a valid Orbax"
        " V1 checkpoint. It is missing the indicator file"
        f" '{ORBAX_CHECKPOINT_INDICATOR_FILE}'."
    )

  async def validate(self, path: Path):
    """Validates the given path as a V1 Orbax checkpoint."""
    try:
      await self._validate(path)
    except BaseException as e:
      raise InvalidLayoutError(
          f"Failed to interpret path {path} as a V1 Orbax checkpoint."
          f" due to error encountered during validation: {e}"
      ) from e

  async def validate_pytree(
      self, path: Path, checkpointable_name: str | None
  ) -> None:
    """Validates the given path as a V1 PyTree checkpoint."""
    try:
      await self._validate_pytree(path, checkpointable_name)
    except BaseException as e:
      raise InvalidLayoutError(
          f"Failed to interpret path {path} as a V1 Orbax PyTree"
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
    """Loads pytree specified by `checkpointable_name`.

    Args:
      path: The path to the checkpoint.
      checkpointable_name: The name of the pytree checkpointable to load.
      abstract_pytree: The abstract pytree to load.

    Returns:
      An awaitable containing the loaded pytree.
    """
    load_awaitable = await self.load_checkpointables(
        path, {checkpointable_name: abstract_pytree}
    )
    return load_awaitable

  async def load_checkpointables(
      self,
      path: Path,
      abstract_checkpointables: dict[str, Any] | None = None,
  ) -> Awaitable[dict[str, Any]]:
    """Loads checkpointables specified by `abstract_checkpointables`.

    Args:
      path: The path to the checkpoint.
      abstract_checkpointables: The abstract checkpointables to load.

    Returns:
      An awaitable containing the loaded checkpointables.

    Raises:
      KeyError: If any of the specified checkpointable names are not found in
      the checkpoint.
    """
    abstract_checkpointables = abstract_checkpointables or {}
    checkpoint_metadata = await _read_checkpoint_metadata(
        path
    )
    handlers_for_load = await handler_resolution.get_handlers_for_load(
        path,
        self._handler_registry,
        abstract_checkpointables,
        checkpoint_metadata,
    )
    existing_checkpointable_names = await _existing_checkpointable_names(path)
    if not abstract_checkpointables:
      abstract_checkpointables = {
          name: None
          for name in handlers_for_load.keys()
          if name not in checkpoint_layout.RESERVED_CHECKPOINTABLE_KEYS
          and name in existing_checkpointable_names
      }
    if any(
        name not in existing_checkpointable_names
        for name in abstract_checkpointables.keys()
    ):
      raise KeyError(
          f"Requested checkpointables: {abstract_checkpointables.keys()} for"
          " loading were not found in the checkpoint. Available"
          f" checkpointables: {existing_checkpointable_names}"
      )

    load_ops = []
    for (
        checkpointable_name,
        abstract_checkpointable,
    ) in abstract_checkpointables.items():
      handler = handlers_for_load[checkpointable_name]
      load_ops.append(
          handler.load(
              path / checkpointable_name,
              abstract_checkpointable,
          )
      )
    load_awaitables = await asyncio.gather(*load_ops)

    async def _run_background() -> dict[str, Any]:
      loaded_checkpointables = []
      # TODO(b/398249409) Cannot use asyncio.gather because asyncio.run
      # is used in underlying implementation.
      for a in load_awaitables:
        loaded = await a
        loaded_checkpointables.append(loaded)
      return {
          checkpointable_name: loaded
          for checkpointable_name, loaded in zip(
              abstract_checkpointables.keys(),
              loaded_checkpointables,
          )
      }

    return _run_background()

  async def save(
      self,
      path: path_types.PathAwaitingCreation,
      *,
      checkpointables: dict[str, Any],
  ) -> Awaitable[None]:
    """Saves the checkpoint to the given directory.

    The subdirectories are expected to already exist.

    Args:
      path: The directory to save the checkpointables to. The checkpointables
        subdirectories exist under this directory.
      checkpointables: A mapping from checkpointable name to checkpointable.

    Returns:
      An awaitable that represents a background save operation.
    """
    context = context_lib.get_context()
    handlers_for_save = handler_resolution.get_handlers_for_save(
        self._handler_registry, checkpointables
    )
    save_ops = []
    for checkpointable_name, checkpointable in checkpointables.items():
      save_ops.append(
          handlers_for_save[checkpointable_name].save(
              path / checkpointable_name, checkpointable
          )
      )
    save_awaitables = await asyncio.gather(*save_ops)

    async def _run_background():
      await asyncio.gather(
          *save_awaitables,
          _create_orbax_identifier_file(
              path, context.multiprocessing_options.primary_host
          ),
      )

    return _run_background()
