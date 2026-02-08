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

from typing import Any, Awaitable

from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.handlers import composite_handler
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

  def __init__(self):
    self._context = context_lib.get_context()
    self._handler_registry = registration.local_registry(
        self._context.checkpointables_options.registry,
        include_global_registry=False,
    )
    self._composite_handler = composite_handler.CompositeHandler(
        self._handler_registry
    )
    self._orbax_layout = orbax_layout.OrbaxLayout()

  async def metadata(
      self, path: Path
  ) -> metadata_types.CheckpointMetadata[dict[str, Any]]:
    """Returns the metadata describing the Orbax checkpoint.

    Args:
      path: The path to the checkpoint directory.

    Returns:
      The metadata describing the Orbax checkpoint.
    """
    return await self._orbax_layout.metadata(path)

  async def _validate(self, path: Path) -> None:
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
    await self._orbax_layout.validate(path)

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
    await self._orbax_layout.validate_pytree(path, checkpointable_name)

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
          f"Failed to interpret path {path} as an Orbax V0 checkpoint."
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
          f"Failed to interpret path {path} as an Orbax V0 PyTree checkpoint."
      ) from e

  async def load(
      self,
      path: Path,
      abstract_checkpointables: dict[str, Any] | None = None,
  ) -> Awaitable[dict[str, Any]]:
    load_awaitable = await self._composite_handler.load(
        path, abstract_checkpointables
    )
    return load_awaitable
