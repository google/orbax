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
import concurrent.futures
import dataclasses
import json
import threading
from typing import Any, Optional, Protocol
from absl import logging
from etils import epath

_METADATA_FILENAME = '_CHECKPOINT_METADATA'


def metadata_file_path(path: epath.PathLike) -> epath.Path:
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

  def is_blocking_writer(self) -> bool:
    """Returns True if the store performs blocking writes, otherwise False."""
    ...

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

  def wait_until_finished(self) -> None:
    """Waits for completion of non blocking writes if applicable."""
    ...

  def close(self) -> None:
    """Closes the store after cleaning up resources if any."""
    ...


class _CheckpointMetadataStoreImpl(CheckpointMetadataStore):
  """Basic internal reusable impl of `CheckpointMetadata` storage.

  It is neither thread safe, nor does it check for read/write capabilities.
  """

  def is_blocking_writer(self) -> bool:
    return True

  def write(
      self,
      checkpoint_path: epath.PathLike,
      checkpoint_metadata: CheckpointMetadata,
  ) -> None:
    checkpoint_path = epath.Path(checkpoint_path)
    if not checkpoint_path.exists():
      raise ValueError(f'Checkpoint path does not exist: {checkpoint_path}')
    json_data = json.dumps(dataclasses.asdict(checkpoint_metadata))
    bytes_written = metadata_file_path(checkpoint_path).write_text(json_data)
    if bytes_written == 0:
      raise ValueError(
          f'Failed to write CheckpointMetadata={checkpoint_metadata},'
          f' json={json_data} to {checkpoint_path}'
      )
    logging.log_every_n(
        logging.INFO,
        'Wrote CheckpointMetadata=%s, json=%s to %s',
        100,
        checkpoint_metadata,
        json_data,
        checkpoint_path,
    )

  def read(
      self, checkpoint_path: epath.PathLike
  ) -> Optional[CheckpointMetadata]:
    metadata_file = metadata_file_path(checkpoint_path)
    if not metadata_file.exists():
      logging.warning(
          'CheckpointMetadata file does not exist: %s', metadata_file
      )
      return None
    try:
      raw_data = metadata_file.read_text()
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.error(
          'Failed to read CheckpointMetadata file: %s, error: %s',
          metadata_file,
          e,
      )
      return None
    try:
      json_data = json.loads(raw_data)
    except json.decoder.JSONDecodeError as e:
      # TODO(b/340287956): Found empty metadata files, how is it possible.
      logging.error(
          'Failed to json parse CheckpointMetadata file: %s, file content: %s,'
          ' error: %s',
          metadata_file,
          raw_data,
          e,
      )
      return None
    result = CheckpointMetadata.from_dict(json_data)
    logging.log_every_n(
        logging.INFO,
        'Read CheckpointMetadata=%s from %s',
        500,
        result,
        checkpoint_path,
    )
    return result

  def update(
      self,
      checkpoint_path: epath.PathLike,
      **kwargs,
  ) -> None:
    metadata = self.read(checkpoint_path) or CheckpointMetadata()
    updated = dataclasses.replace(metadata, **kwargs)
    self.write(checkpoint_path, updated)
    logging.log_every_n(
        logging.INFO,
        'Updated CheckpointMetadata=%s to %s',
        100,
        updated,
        checkpoint_path,
    )


@dataclasses.dataclass
class _BlockingCheckpointMetadataStore(CheckpointMetadataStore):
  """Manages storage of `CheckpointMetadata` with blocking writes.

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
  _write_lock: threading.RLock = dataclasses.field(init=False)
  _store_impl: _CheckpointMetadataStoreImpl = dataclasses.field(init=False)

  def __post_init__(self):
    self._store_impl = _CheckpointMetadataStoreImpl()
    if self.enable_write:
      self._write_lock = threading.RLock()

  def is_blocking_writer(self) -> bool:
    return True

  def write(
      self,
      checkpoint_path: epath.PathLike,
      checkpoint_metadata: CheckpointMetadata,
  ) -> None:
    if not self.enable_write:
      return
    with self._write_lock:
      self._store_impl.write(checkpoint_path, checkpoint_metadata)

  def read(
      self, checkpoint_path: epath.PathLike
  ) -> Optional[CheckpointMetadata]:
    return self._store_impl.read(checkpoint_path)

  def update(
      self,
      checkpoint_path: epath.PathLike,
      **kwargs,
  ) -> None:
    if not self.enable_write:
      return
    with self._write_lock:
      self._store_impl.update(checkpoint_path, **kwargs)


@dataclasses.dataclass
class _NonBlockingCheckpointMetadataStore(CheckpointMetadataStore):
  """Manages storage of `CheckpointMetadata` with non blocking writes.

  By default it behaves like a read only `CheckpointMetadataStore`. But the same
  instance is reused if user requests for a write-enabled instance in the same
  process.

  The writes are non blocking. Read responses don't reflect in progress writes.
  """

  enable_write: bool
  _write_lock: threading.RLock = dataclasses.field(init=False)
  _store_impl: _CheckpointMetadataStoreImpl = dataclasses.field(init=False)
  # We need to make sure that only one thread writes/updates to a given path.
  # A single threaded executor is a simple solution. We can improve it by
  # introducing a multi threaded executor but setting up tasks such that all
  # tasks for a path is handled by the same thread.
  _single_thread_executor: concurrent.futures.ThreadPoolExecutor = (
      dataclasses.field(init=False)
  )
  # List of futures associated with write/update calls. It gets reset after each
  # `wait_until_finished()` call.
  _write_futures: list[concurrent.futures.Future[None]] = dataclasses.field(
      init=False
  )

  def __post_init__(self):
    self._store_impl = _CheckpointMetadataStoreImpl()
    if self.enable_write:
      self._write_lock = threading.RLock()
      self._single_thread_executor = concurrent.futures.ThreadPoolExecutor(
          max_workers=1, thread_name_prefix='metadata_store'
      )
      self._write_futures = []

  def is_blocking_writer(self) -> bool:
    return False

  def _add_to_write_futures(
      self, future: concurrent.futures.Future[None]
  ) -> None:
    """Adds `future` to the `_write_futures` list while removing `done` futures."""
    if not self.enable_write:
      return
    with self._write_lock:
      self._write_futures.append(future)

  def _write_and_log(
      self,
      checkpoint_path: epath.PathLike,
      checkpoint_metadata: CheckpointMetadata,
  ) -> None:
    """Writes `checkpoint_metadata` and logs error if any."""
    try:
      self._store_impl.write(checkpoint_path, checkpoint_metadata)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.exception(
          'Failed to write metadata=%s path=%s: %s',
          checkpoint_metadata,
          checkpoint_path,
          e,
      )
      raise

  def write(
      self,
      checkpoint_path: epath.PathLike,
      checkpoint_metadata: CheckpointMetadata,
  ) -> None:
    """[Over]Writes `checkpoint_metadata` in non blocking manner."""
    if not self.enable_write:
      logging.warning(
          'Write requested but enable_write=false, checkpoint_metadata=%s'
          ' checkpoint_path=%s',
          checkpoint_metadata,
          checkpoint_path,
      )
      return
    with self._write_lock:
      future = self._single_thread_executor.submit(
          self._write_and_log, checkpoint_path, checkpoint_metadata
      )
      self._add_to_write_futures(future)

  def read(
      self, checkpoint_path: epath.PathLike
  ) -> Optional[CheckpointMetadata]:
    """Reads `checkpoint_path`/*metadata_file* and returns `CheckpointMetadata`."""
    return self._store_impl.read(checkpoint_path)

  def _update_and_log(self, checkpoint_path: epath.PathLike, **kwargs) -> None:
    """Updates checkpoint metadata attributes and logs error if any."""
    try:
      self._store_impl.update(checkpoint_path, **kwargs)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.exception(
          'Failed to update metadata=%s path=%s: %s',
          kwargs,
          checkpoint_path,
          e,
      )
      raise

  def _validate_kwargs(self, **kwargs) -> None:
    _ = CheckpointMetadata(**kwargs)

  def update(
      self,
      checkpoint_path: epath.PathLike,
      **kwargs,
  ) -> None:
    """Updates CheckpointMetadata in non blocking manner.

    If no updatable CheckpointMetadata is found at
    `checkpoint_path`/*metadata_file*, then it creates a new one with `kwargs`
    attributes.

    Args:
      checkpoint_path: path to checkpoint dir (step dir).
      **kwargs: Attributes of CheckpointMetadata is kwargs format.
    """
    if not self.enable_write:
      logging.warning(
          'Update requested but enable_write=false, kwargs=%s'
          ' checkpoint_path=%s',
          kwargs,
          checkpoint_path,
      )
      return
    with self._write_lock:
      self._validate_kwargs(**kwargs)
      future = self._single_thread_executor.submit(
          self._update_and_log, checkpoint_path, **kwargs
      )
      self._add_to_write_futures(future)

  def wait_until_finished(self) -> None:
    """Waits for completion of writes."""
    if not self.enable_write:
      return
    with self._write_lock:
      while self._write_futures:
        future = self._write_futures.pop(0)
        future.result()

  def close(self) -> None:
    """Closes the store after cleaning up resources if any."""
    if self.enable_write:
      with self._write_lock:
        self._single_thread_executor.shutdown()
        logging.info('Closing %s', self)


_CHECKPOINT_METADATA_STORE_FOR_WRITES = _BlockingCheckpointMetadataStore(
    enable_write=True
)
_CHECKPOINT_METADATA_STORE_FOR_READS = _BlockingCheckpointMetadataStore(
    enable_write=False
)
_CHECKPOINT_METADATA_STORE_NON_BLOCKING_FOR_READS = (
    _NonBlockingCheckpointMetadataStore(enable_write=False)
)


def checkpoint_metadata_store(
    *,
    enable_write: bool,
    blocking_write: bool = False,
) -> CheckpointMetadataStore:
  """Returns `CheckpointMetadataStore` instance based on `enable_write` value.

  Write operations are thread safe: within a process multiple threads write
  without corrupting data.

  NOTE: Write operations are not guaranteed to be safe across processes. But it
  should be okay as writes are expected to be called from just one jax process.

  Read operations are inherently thread safe and *process safe* too.

  NOTE: `CheckpointMetadataStore` instance created with `enable_write=True` and
  `blocking_write=False` must be closed with `.close()` to release thread
  resources. Prefer to reuse an instance created for this scenario.

  Args:
    enable_write: if True then write operations are allowed, otherwise write
      operations are **no op**. Read operations are always allowed.
    blocking_write: if True then write operations are blocking, otherwise non
      blocking. Read responses don't reflect in progress writes.
  """
  if not blocking_write:
    if enable_write:
      return _NonBlockingCheckpointMetadataStore(enable_write=True)
    return _CHECKPOINT_METADATA_STORE_NON_BLOCKING_FOR_READS

  if enable_write:
    return _CHECKPOINT_METADATA_STORE_FOR_WRITES
  return _CHECKPOINT_METADATA_STORE_FOR_READS
