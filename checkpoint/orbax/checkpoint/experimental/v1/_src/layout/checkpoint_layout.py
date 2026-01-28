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

import abc
from typing import Any, Awaitable, Protocol
from orbax.checkpoint.experimental.v1._src.metadata import types as metadata_types
from orbax.checkpoint.experimental.v1._src.path import types

Path = types.Path


### Constants shared by all layouts. ###

PYTREE_CHECKPOINTABLE_KEY = "pytree"

METRICS_CHECKPOINTABLE_KEY = "metrics"

RESERVED_CHECKPOINTABLE_KEYS = frozenset({
    METRICS_CHECKPOINTABLE_KEY,
})


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

  async def metadata(
      self, path: Path
  ) -> metadata_types.CheckpointMetadata[dict[str, Any]]:
    """Returns the metadata of the checkpoint.

    Args:
      path: The path to the checkpoint.

    Returns:
      A dictionary of metadata. Dictionary keys represent the names of the
      checkpointables, while the values are the metadata objects themselves.
    """

    ...

  @abc.abstractmethod
  async def validate(self, path: Path) -> None:
    """Validates the path, determining if it conforms to this instance.

    Args:
      path: The path to the checkpoint.

    Raises:
      InvalidLayoutError: If the path does not conform to this layout.
    """
    ...

  @abc.abstractmethod
  async def validate_pytree(
      self, path: Path, checkpointable_name: str | None
  ) -> None:
    """Validates the path as a PyTree checkpoint.

    Args:
      path: The path to the checkpoint.
      checkpointable_name: The name of the checkpointable to load. as a PyTree
        checkpoint.
    """
    ...

  async def load(
      self,
      path: Path,
      abstract_checkpointables: dict[str, Any] | None = None,
  ) -> Awaitable[dict[str, Any]]:
    """Loads the checkpoint from the given directory.

    Args:
      path: The path to the checkpoint.
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

  async def save(
      self,
      path: types.PathAwaitingCreation,
      *,
      checkpointables: dict[str, Any],
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
