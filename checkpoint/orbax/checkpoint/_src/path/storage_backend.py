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

"""Checkpoint storage interface and base implementations.

This module defines the abstract StorageBackend interface for managing
checkpoint paths across different file systems. Base implementations for GCS
and local file systems are provided here
"""

import abc
import dataclasses
import enum

from absl import logging
from etils import epath
from orbax.checkpoint._src.path import atomicity_types
from orbax.checkpoint._src.path import step as step_lib


@dataclasses.dataclass(frozen=True)
class CheckpointPathMetadata:
  """Internal representation of checkpoint path metadata.

  Attributes:
    path: The file system path of the checkpoint.
    status: The status of the checkpoint.
    version: The version of the checkpoint with an index and step number. (e.g.
      '1.step_1')
    tags: A list of tags associated with the checkpoint. May not be available in
      all backend implementations; for unsupported backends this field will be
      `None`.
  """

  class Status(enum.Enum):
    COMMITTED = 1
    UNCOMMITTED = 2

  path: str
  status: Status
  version: str | None
  tags: set[str] | None = None


@dataclasses.dataclass(frozen=True)
class CheckpointFilter:
  """Criteria for filtering checkpoints.

  TODO: b/466312058 This class will contain fields for filtering checkpoints by
  various criteria.
  """


@dataclasses.dataclass(frozen=True)
class CheckpointReadOptions:
  """Options for reading checkpoints.

  Attributes:
    filter: Optional filter criteria for selecting checkpoints.
    enable_strong_reads: If True, enables strong read consistency when querying
      checkpoints. This may have performance implications but ensures the most
      up-to-date results.
  """

  filter: CheckpointFilter | None = None
  enable_strong_reads: bool = False


class StorageBackend[MetadataT: step_lib.Metadata](abc.ABC):
  """An abstract base class for a storage backend.

  This class defines a common interface for managing checkpoint paths in
  different file systems.
  """

  @abc.abstractmethod
  def list_checkpoints(
      self,
      base_path: str | epath.PathLike,
  ) -> list[CheckpointPathMetadata]:
    """Lists checkpoints for a given base path and version pattern."""
    raise NotImplementedError('Subclasses must provide implementation')

  @abc.abstractmethod
  def list_metadata(
      self,
      *,
      base_path: epath.PathLike,
      step_prefix: str,
      include_uncommitted: bool,
      step: int | None = None,
      index: int | None = None,
      glob_pattern: str | None = None,
      options: CheckpointReadOptions | None = None,
  ) -> list[MetadataT]:
    """Low-level method to return metadata based on steps, index and glob.

    Args:
      base_path: Base path of a checkpoint sequence.
      step_prefix: Prefix of the step name. Usually `step`.
      include_uncommitted: Some backends distinguish commited from uncommited
        checkpoints. If supported, this flag controls whether we skip or include
        uncommitted checkpoints.
      step: If provided, filters the results by step.
      index: If provided, filters the results by index.
      glob_pattern: A glob pattern to filter the results by, based on the
        convention on file-naming. Glob can be e.g. '*.step_*' and these are not
        regex.
      options: Options for reading checkpoints.
    """
    raise NotImplementedError('Subclasses must provide implementation')

  @abc.abstractmethod
  def get_temporary_path_class(self) -> type[atomicity_types.TemporaryPath]:
    """Returns a TemporaryPath class for the storage backend."""
    raise NotImplementedError('Subclasses must provide implementation')

  @abc.abstractmethod
  def delete_checkpoint(
      self,
      checkpoint_path: str | epath.PathLike,
  ) -> None:
    """Deletes a checkpoint from the storage backend."""
    raise NotImplementedError('Subclasses must provide implementation')


class GCSStorageBackend(StorageBackend):
  """A StorageBackend implementation for GCS (Google Cloud Storage).

  # TODO(b/425293362): Implement this class.
  """

  def get_temporary_path_class(self) -> type[atomicity_types.TemporaryPath]:
    """Returns the final checkpoint path directly."""
    raise NotImplementedError(
        'get_temporary_path_class is not yet implemented for GCSStorageBackend.'
    )

  def list_checkpoints(
      self, base_path: str | epath.PathLike
  ) -> list[CheckpointPathMetadata]:
    """Lists checkpoints for a given base path and version pattern."""
    raise NotImplementedError(
        'list_checkpoints is not yet implemented for GCSStorageBackend.'
    )

  def list_metadata(
      self,
      base_path: epath.PathLike,
      step_prefix: str,
      *,
      include_uncommitted: bool,
      step: int | None = None,
      index: int | None = None,
      glob_pattern: str | None = None,
      options: CheckpointReadOptions | None = None,
  ) -> list[step_lib.Metadata]:
    """Returns a list of IndexPrefixMetadata for a given path."""
    raise NotImplementedError(
        'list_metadata is not yet implemented for LocalStorageBackend.'
    )

  def delete_checkpoint(
      self,
      checkpoint_path: str | epath.PathLike,
  ) -> None:
    """Deletes the checkpoint at the given path."""
    raise NotImplementedError(
        'delete_checkpoint is not yet implemented for GCSStorageBackend.'
    )


class LocalStorageBackend(StorageBackend):
  """A LocalStorageBackend implementation for local file systems.

  # TODO(b/425293362): Implement this class.
  """

  def get_temporary_path_class(self) -> type[atomicity_types.TemporaryPath]:
    """Returns the final checkpoint path directly."""
    raise NotImplementedError(
        'get_temporary_path_class is not yet implemented for'
        ' LocalStorageBackend.'
    )

  def list_checkpoints(
      self,
      base_path: str | epath.PathLike,
  ) -> list[CheckpointPathMetadata]:
    """Lists checkpoints for a given base path and version pattern."""
    raise NotImplementedError(
        'list_checkpoints is not yet implemented for LocalStorageBackend.'
    )

  def list_metadata(
      self,
      base_path: epath.PathLike,
      step_prefix: str,
      *,
      include_uncommitted: bool,
      step: int | None = None,
      index: int | None = None,
      glob_pattern: str | None = None,
      options: CheckpointReadOptions | None = None,
  ) -> list[step_lib.Metadata]:
    """Returns a list of IndexPrefixMetadata for a given path."""
    raise NotImplementedError(
        'list_metadata is not yet implemented for LocalStorageBackend.'
    )

  def delete_checkpoint(
      self,
      checkpoint_path: str | epath.PathLike,
  ) -> None:
    """Deletes the checkpoint at the given path."""
    try:
      epath.Path(checkpoint_path).rmtree()
      logging.info('Removed old checkpoint (%s)', checkpoint_path)
    except OSError:
      logging.exception('Failed to remove checkpoint (%s)', checkpoint_path)
