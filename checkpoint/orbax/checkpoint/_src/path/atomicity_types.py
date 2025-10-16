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

"""Defines the types and Protocols for creating and finalizing temporary paths.

Chiefly defines the TemporaryPath Protocol for creating and finalizing temporary
paths, which allow other implementations.
"""

from __future__ import annotations

import abc
from etils import epath
from orbax.checkpoint import options as options_lib
from orbax.checkpoint._src.metadata import checkpoint as checkpoint_metadata



TMP_DIR_SUFFIX = '.orbax-checkpoint-tmp'
COMMIT_SUCCESS_FILE = 'commit_success.txt'


class ValidationError(ValueError):
  """Raised when a TemporaryPath or its final path is invalid."""


class TemporaryPath(abc.ABC):
  """Class that represents a temporary path.

  Importantly, the temporary path always has a corresponding finalized path, and
  is primarily constructed from this path. The class contains logic to create
  the temporary path, and to finalize it into the final path.

  `from_final` should be called from all hosts to construct the temporary path
  instance from the given final path.
  `create` and `finalize` must be called only on the primary host.
  """

  @classmethod
  @abc.abstractmethod
  async def validate(
      cls,
      temporary_path: epath.Path,
  ):
    """Validates the temporary path or raises a ValidationError."""
    ...

  @classmethod
  @abc.abstractmethod
  async def validate_final(
      cls,
      final_path: epath.Path,
  ):
    """Validates the final path or raises a ValidationError."""
    ...

  @classmethod
  @abc.abstractmethod
  def from_temporary(
      cls,
      temporary_path: epath.Path,
      *,
      file_options: options_lib.FileOptions | None = None,
      use_snapshot: bool | None = None,
  ) -> TemporaryPath:
    """Creates a TemporaryPath from a temporary path."""
    ...

  @classmethod
  @abc.abstractmethod
  def from_final(
      cls,
      final_path: epath.Path,
      *,
      checkpoint_metadata_store: (
          checkpoint_metadata.MetadataStore | None
      ) = None,
      file_options: options_lib.FileOptions | None = None,
      use_snapshot: bool | None = None,
  ) -> TemporaryPath:
    """Creates a TemporaryPath from a final path."""
    ...

  @abc.abstractmethod
  def get(self) -> epath.Path:
    """Constructs the temporary path without actually creating it."""
    ...

  @abc.abstractmethod
  def get_final(self) -> epath.Path:
    """Returns the final path without creating it."""
    ...

  @abc.abstractmethod
  async def create(self) -> epath.Path:
    """Creates the temporary path on disk.

    NOTE: This method should be only called on the primary host.

    Returns:
      The created temporary path.
    """
    ...

  @abc.abstractmethod
  async def finalize(
      self,
  ) -> None:
    """Finalizes the temporary path into the final path.

    NOTE: This method should be only called on the primary host.

    This function is called from a background thread.

    """
    ...
