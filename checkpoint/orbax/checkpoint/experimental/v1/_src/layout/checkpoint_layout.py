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

"""Defines `CheckpointLayout`, a broader protocol used to save and load different checkpoint formats."""

import abc
from typing import Awaitable, Protocol
from orbax.checkpoint.experimental.v1._src.handlers import types as handler_types
from orbax.checkpoint.experimental.v1._src.metadata import types as metadata_types
from orbax.checkpoint.experimental.v1._src.path import types as path_types

Path = path_types.Path


### Constants shared by all layouts. ###

STATE_CHECKPOINTABLE_KEY = "state"
EMPTY_CHECKPOINTABLE_KEY = ""
AUTO_CHECKPOINTABLE_KEY = "AUTO"

METRICS_CHECKPOINTABLE_KEY = "metrics"

RESERVED_CHECKPOINTABLE_KEYS = frozenset({
    METRICS_CHECKPOINTABLE_KEY,
    AUTO_CHECKPOINTABLE_KEY,
})

Checkpointable = handler_types.Checkpointable
AbstractCheckpointable = handler_types.AbstractCheckpointable


class InvalidLayoutError(ValueError):
  """Raised when the checkpoint layout is invalid."""


class CheckpointLayout(Protocol):
  """CheckpointLayout.

  This class defines a protocol for different checkpoint formats. It is a helper
  component for :py:func:`~.v1.save_checkpointables` and
  :py:func:`~.v1.load_checkpointables`.
  It supports alternative checkpoint formats. It performs a few core functions:
    - Resolves handlers for saving and loading.
    - Saves and loads checkpointables to/from individual subdirectories by
    delegating to the resolved handlers.
  """

  @abc.abstractmethod
  async def get_checkpointable_names(self, path: Path) -> list[str | None]:
    """Returns a list of candidate checkpointable names to use for loading.

    Attempts to resolve checkpointable names for by inspecting the checkpoint
    format and finding appropriate checkpointable names to use. The result may
    be an empty list.

    Args:
      path: The path to the checkpoint.

    Returns:
      A list of checkpointable names, ordered by priority / importance.
    """
    ...

  @abc.abstractmethod
  async def checkpointables_metadata(
      self, path: Path
  ) -> metadata_types.CheckpointMetadata[
      dict[str, handler_types.AbstractCheckpointable]
  ]:
    """Returns the metadata of the checkpoint.

    Args:
      path: The path to the checkpoint.

    Returns:
      A dictionary of metadata. Dictionary keys represent the names of the
      checkpointables, while the values are the metadata objects themselves.
    """

    ...

  @abc.abstractmethod
  async def metadata(
      self, path: Path, checkpointable_name: str | None
  ) -> metadata_types.CheckpointMetadata[AbstractCheckpointable]:
    """Returns the metadata of a single checkpointable in the checkpoint.

    Args:
      path: The path to the checkpoint.
      checkpointable_name: The name of the checkpointable to inspect.

    Returns:
      The metadata structure for the given checkpointable.
    """
    ...

  @abc.abstractmethod
  async def validate_checkpointables(self, path: Path) -> None:
    """Validates the path, determining if it conforms to this instance.

    Args:
      path: The path to the checkpoint.

    Raises:
      InvalidLayoutError: If the path does not conform to this layout.
    """
    ...

  @abc.abstractmethod
  async def validate(self, path: Path, checkpointable_name: str | None) -> None:
    """Validates the path as a PyTree checkpoint.

    Args:
      path: The path to the checkpoint.
      checkpointable_name: The name of the checkpointable to load. as a PyTree
        checkpoint.
    """
    ...

  @abc.abstractmethod
  async def load(
      self,
      path: Path,
      checkpointable_name: str | None = None,
      abstract_state: AbstractCheckpointable | None = None,
  ) -> Awaitable[Checkpointable]:
    """Loads a PyTree state from the checkpoint.

    Args:
      path: The path to the checkpoint.
      checkpointable_name: The name of the checkpointable to load.
      abstract_state: The abstract PyTree structure.

    Returns:
      An awaitable PyTree.
    """
    ...

  @abc.abstractmethod
  async def load_checkpointables(
      self,
      path: Path,
      abstract_checkpointables: dict[str, AbstractCheckpointable] | None = None,
  ) -> Awaitable[dict[str, Checkpointable]]:
    """Loads checkpointables specified by `abstract_checkpointables`.

    Args:
      path: The path to the checkpoint.
      abstract_checkpointables: The abstract structures to load.

    Returns:
      An awaitable containing a dictionary mapping checkpointable names to
      loaded checkpointable values.
    """
    ...

  @abc.abstractmethod
  async def save_checkpointables(
      self,
      path: path_types.PathAwaitingCreation,
      *,
      checkpointables: dict[str, Checkpointable],
  ) -> Awaitable[None]:
    """Saves the checkpoint to the given directory.

    Args:
      path: The directory to save the checkpoint to.
      checkpointables: A dictionary of checkpointables to save. Dictionary keys
        represent the names of the checkpointables, while the values are the
        checkpointable objects themselves.

    Returns:
      An awaitable that completes when the save operation is finished.
    """
    ...
