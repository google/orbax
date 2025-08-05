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

"""Defines `CheckpointLayout`, a broader protocol used to save and load different checkpoint formats."""

from typing import Any, Awaitable, Protocol
from orbax.checkpoint.experimental.v1._src.metadata import types as metadata_types
from orbax.checkpoint.experimental.v1._src.path import types

Path = types.Path


class InvalidLayoutError(ValueError):
  """Raised when the checkpoint layout is invalid."""


class CheckpointLayout(Protocol):
  """CheckpointLayout.

  This class defines a protocol for different checkpoint formats. It is a helper
  component for `save_checkpointables` and
  `load_checkpointables`. It supports
  alternative checkpoint formats. It performs a few core functions:
    - Resolves handlers for saving and loading.
    - Saves and loads checkpointables to/from individual subdirectories by
    delegating to the resolved handlers.
  """

  @property
  def path(self) -> Path:
    """Returns the path of the checkpoint."""
    ...

  async def metadata(self) -> metadata_types.CheckpointMetadata[dict[str, Any]]:
    """Returns the metadata of the checkpoint.

    Returns:
      A dictionary of metadata. Dictionary keys represent the names of the
      checkpointables, while the values are the metadata objects themselves.
    """
    ...

  def validate(self) -> None:
    """Validates the path, determining if it conforms to this instance.

    Returns:
      None.

    Raises:
      InvalidLayoutError: If the path does not conform to this instance.
    """
    ...

  def validate_pytree(self, checkpointable_name: str | None) -> None:
    """Validates the path as a PyTree checkpoint.

    Args:
      checkpointable_name: The name of the checkpointable to load.

    Returns:
      None.

    Raises:
      InvalidLayoutError: If the path does not conform to this instance
        as a PyTree checkpoint.
    """
    ...

  async def load(
      self,
      abstract_checkpointables: dict[str, Any] | None = None,
  ) -> Awaitable[dict[str, Any]]:
    """Loads the checkpoint from the given directory.

    Args:
      abstract_checkpointables: A dictionary of abstract checkpointables.
        Dictionary keys represent the names of the checkpointables, while the
        values are the abstract checkpointable objects themselves.

    Returns:
      An awaitable dictionary of checkpointables. Dictionary keys represent the
      names of
      the checkpointables, while the values are the checkpointable objects
      themselves.
    """
    ...
