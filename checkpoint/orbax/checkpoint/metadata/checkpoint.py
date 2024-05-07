# Copyright 2024 The Orbax Authors.
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

"""Manages metadata of checkpoints at step level (not item level)."""

from __future__ import annotations
import dataclasses
import json
import threading
from typing import Any, Optional, Protocol
from absl import logging
from etils import epath

_METADATA_FILENAME = '_CHECKPOINT_METADATA'


def _metadata_file_path(path: epath.PathLike) -> epath.Path:
  """Returns the path to metadata file for a given checkpoint directory."""
  return epath.Path(path) / _METADATA_FILENAME


@dataclasses.dataclass
class CheckpointMetadata:
  """Metadata of a checkpoint at step level (not per item).

  NOTE: Internal class. Please reach out to Orbax team if you want to use it in
  your codebase.

  Attributes:
    init_timestamp_nsecs: timestamp when uncommitted checkpoint was initialized.
      Specified as nano seconds since epoch. default=None.
    commit_timestamp_nsecs: commit timestamp of a checkpoint, specified as nano
      seconds since epoch. default=None.
  """

  init_timestamp_nsecs: Optional[int] = None
  commit_timestamp_nsecs: Optional[int] = None

  @classmethod
  def from_dict(cls, dict_data: Any) -> CheckpointMetadata:
    validated_dict = {}
    if 'init_timestamp_nsecs' in dict_data:
      validated_dict['init_timestamp_nsecs'] = dict_data['init_timestamp_nsecs']
    if 'commit_timestamp_nsecs' in dict_data:
      validated_dict['commit_timestamp_nsecs'] = dict_data[
          'commit_timestamp_nsecs'
      ]
    return CheckpointMetadata(**validated_dict)


class CheckpointMetadataStore(Protocol):
  """Manages storage of `CheckpointMetadata`."""

  def write(
      self,
      checkpoint_path: epath.PathLike,
      checkpoint_metadata: CheckpointMetadata,
  ) -> None:
    """[Over]Writes `checkpoint_metadata` to `checkpoint_path`/*metadata_file*."""
    ...

  def read(
      self, checkpoint_path: epath.PathLike
  ) -> Optional[CheckpointMetadata]:
    """Reads `checkpoint_path`/*metadata_file* and returns `CheckpointMetadata`."""
    ...

  def update(
      self,
      checkpoint_path: epath.PathLike,
      **kwargs,
  ) -> None:
    """Safely updates CheckpointMetadata at `checkpoint_path`/*metadata_file*.

    If no updatable CheckpointMetadata is found at
    `checkpoint_path`/*metadata_file*, then it creates a new one with `kwargs`
    attributes.

    Args:
      checkpoint_path: path to checkpoint dir (step dir).
      **kwargs: Attributes of CheckpointMetadata is kwargs format.
    """
    ...


@dataclasses.dataclass(frozen=True)
class _CheckpointMetadataStore(CheckpointMetadataStore):
  """Internal impl to manage storage of `CheckpointMetadata`.

  Write operations are thread safe: within a process multiple threads write
  without corrupting data.

  NOTE: Write operations are not guaranteed to be safe across processes. But it
  should be okay as writes are expected to be called from just one jax process.

  Read operations are inherently thread safe and *process safe* too.

  Attributes:
    enable_write: if True then write operations are allowed, otherwise write
      operations are **no op**. Read operations are always allowed.
  """

  enable_write: bool
  # TODO(niketkb): Support locking per checkpoint path.
  _write_lock: threading.RLock = threading.RLock()

  def write(
      self,
      checkpoint_path: epath.PathLike,
      checkpoint_metadata: CheckpointMetadata,
  ) -> None:
    if not self.enable_write:
      return
    with self._write_lock:
      checkpoint_path = epath.Path(checkpoint_path)
      if not checkpoint_path.exists():
        raise ValueError(f'Checkpoint path does not exist: {checkpoint_path}')
      _metadata_file_path(checkpoint_path).write_text(
          json.dumps(dataclasses.asdict(checkpoint_metadata))
      )
      logging.info(
          'Wrote CheckpointMetadata=%s to %s',
          checkpoint_metadata,
          checkpoint_path,
      )

  def read(
      self, checkpoint_path: epath.PathLike
  ) -> Optional[CheckpointMetadata]:
    metadata_file = _metadata_file_path(checkpoint_path)
    if not metadata_file.exists():
      logging.warning(
          'CheckpointMetadata file does not exist: %s', metadata_file
      )
      return None
    data = json.loads(metadata_file.read_text())
    result = CheckpointMetadata.from_dict(data)
    logging.info('Read CheckpointMetadata=%s from %s', result, checkpoint_path)
    return result

  def update(
      self,
      checkpoint_path: epath.PathLike,
      **kwargs,
  ) -> None:
    if not self.enable_write:
      return
    with self._write_lock:
      metadata = self.read(checkpoint_path) or CheckpointMetadata()
      updated = dataclasses.replace(metadata, **kwargs)
      self.write(checkpoint_path, updated)
      logging.info(
          'Updated CheckpointMetadata=%s to %s', updated, checkpoint_path
      )


_CHECKPOINT_METADATA_STORE_FOR_WRITES = _CheckpointMetadataStore(
    enable_write=True
)
_CHECKPOINT_METADATA_STORE_FOR_READS = _CheckpointMetadataStore(
    enable_write=False
)


def checkpoint_metadata_store(*, enable_write: bool) -> CheckpointMetadataStore:
  """Returns `CheckpointMetadataStore` instance based on `enable_write` value.

  Write operations are thread safe: within a process multiple threads write
  without corrupting data.

  NOTE: Write operations are not guaranteed to be safe across processes. But it
  should be okay as writes are expected to be called from just one jax process.

  Read operations are inherently thread safe and *process safe* too.

  Args:
    enable_write: if True then write operations are allowed, otherwise write
      operations are **no op**. Read operations are always allowed.
  """
  if enable_write:
    return _CHECKPOINT_METADATA_STORE_FOR_WRITES
  return _CHECKPOINT_METADATA_STORE_FOR_READS
