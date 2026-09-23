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

"""Recovery metadata for deletion of a named checkpointable.

Whole-checkpoint deletion does not create recovery records. Records for partial
operations live inside the checkpoint, outside the directory being removed.
"""

import asyncio
from collections.abc import Collection
import json
import os
import pathlib
import tempfile
from typing import Any
from urllib import parse

from absl import logging
from orbax.checkpoint._src.path import async_path
from orbax.checkpoint._src.path import gcs_utils
from orbax.checkpoint.experimental.v1._src.deletion import errors
from orbax.checkpoint.experimental.v1._src.deletion import validation
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout
from orbax.checkpoint.experimental.v1._src.metadata import serialization as metadata_serialization
from orbax.checkpoint.experimental.v1._src.path import types as path_types

RECORD_FILENAME = checkpoint_layout.CHECKPOINT_DELETION_KEY


DeletionInProgressError = errors.DeletionInProgressError
DeletionRecoveryError = errors.DeletionRecoveryError


async def _read_json(path: path_types.Path) -> dict[str, Any]:
  """Reads metadata without converting malformed documents into empty metadata.

  Args:
    path: The path to the metadata file.

  Returns:
    The metadata dictionary.

  Raises:
    ValueError: If the metadata file does not contain a dictionary.
  """
  result = json.loads(await async_path.read_text(path))
  if not isinstance(result, dict):
    raise ValueError(f'Expected a metadata dictionary at {path}.')
  return result


async def read_checkpoint_metadata(path: path_types.Path) -> dict[str, Any]:
  """Reads checkpoint metadata, propagating I/O and JSON errors."""
  return await _read_json(
      metadata_serialization.checkpoint_metadata_file_path(path)
  )


def updated_metadata(original: dict[str, Any], name: str) -> dict[str, Any]:
  """Returns a copy with the selected handler removed, preserving other fields.

  Args:
    original: The original checkpoint metadata dictionary.
    name: The name of the checkpointable to remove.

  Returns:
    A copy of the original metadata with the selected handler removed.

  Raises:
    ValueError: If the original metadata is not a dictionary or if the
      handler metadata for the given checkpointable is not found.
  """
  if not isinstance(original, dict):
    raise ValueError('Original checkpoint metadata must be a dictionary.')
  handlers = original.get('item_handlers')
  if not isinstance(handlers, dict) or name not in handlers:
    raise ValueError(f'No handler metadata for checkpointable {name!r}.')
  return {
      **original,
      'item_handlers': {
          key: value for key, value in handlers.items() if key != name
      },
  }


async def read_record(path: path_types.Path) -> dict[str, Any] | None:
  """Validates the record and its metadata, or returns None if absent.

  Both reads and deletion retries use this check so they agree on whether an
  interrupted operation still describes the checkpoint.

  Args:
    path: The path to the checkpoint directory.

  Returns:
    The deletion record if it exists and is valid, otherwise None.

  Raises:
    DeletionRecoveryError: If the deletion record is invalid.
  """
  record_path = path / RECORD_FILENAME
  if not await async_path.exists(record_path):
    return None
  try:
    record = await _read_json(record_path)
  except FileNotFoundError:
    # The worker may have finished cleanup after the existence check.
    return None
  except (ValueError, TypeError, IsADirectoryError) as e:
    raise DeletionRecoveryError(
        f'Invalid deletion record at {record_path}: {e}'
    ) from e
  try:
    validation.validate_checkpointable_name(record['checkpointable_name'])
    if (
        not isinstance(record.get('operation_id'), str)
        or len(record['operation_id']) != 32
        or any(c not in '0123456789abcdef' for c in record['operation_id'])
    ):
      raise ValueError('Missing operation id.')
    if record['updated_metadata'] != updated_metadata(
        record['original_metadata'], record['checkpointable_name']
    ):
      raise ValueError('Invalid replacement metadata.')
    if record.get('destination') is not None and not isinstance(
        record['destination'], str
    ):
      raise ValueError('Invalid relocation destination.')
    current = await read_checkpoint_metadata(path)
    if current not in (record['original_metadata'], record['updated_metadata']):
      raise ValueError(
          'Checkpoint metadata conflicts with the deletion record.'
      )
    return record
  except (
      ValueError,
      TypeError,
      KeyError,
      FileNotFoundError,
      IsADirectoryError,
  ) as e:
    raise DeletionRecoveryError(
        f'Invalid deletion record at {record_path}: {e}'
    ) from e


async def ensure_available(path: path_types.Path) -> None:
  """Rejects mutations of a checkpoint with an unfinished partial deletion.

  Args:
    path: The path to the checkpoint directory.

  Raises:
    DeletionInProgressError: If the checkpoint has an unfinished
      deletion.
  """
  if await read_record(path) is not None:
    raise DeletionInProgressError(
        f'Checkpoint at {path} has an unfinished deletion. Retry that deletion'
        ' before modifying the checkpoint.'
    )


async def check_read(
    path: path_types.Path, requested_read_names: Collection[str] | None
) -> str | None:
  """Returns the logically deleted name without changing any files.

  None means discovery: warn that the item is excluded. Explicit requests for
  that item fail. An empty collection validates the record without warning,
  for callers that have not yet resolved the requested checkpointable names.

  Args:
    path: The path to the checkpoint directory.
    requested_read_names: The checkpointables the caller wants to read.
      None requests discovery; an empty collection only validates the record.

  Returns:
    The name logically excluded by pending deletion, or None if the deletion
    record was not found.

  Raises:
    DeletionInProgressError: If the read request includes the
      checkpointable being deleted.
    DeletionRecoveryError: If the deletion record is invalid.
  """
  record = await read_record(path)
  if record is None:
    return None
  deleting_name = record['checkpointable_name']
  if requested_read_names is None:
    logging.warning(
        'Excluding checkpointable %r at %s: deletion was started but cleanup '
        'is incomplete. Retry deletion of this checkpointable to finish.',
        deleting_name,
        path,
    )
  elif deleting_name in requested_read_names:
    raise DeletionInProgressError(
        f'Checkpointable {deleting_name!r} at {path} is pending deletion '
        'and cannot be read. Retry deletion of this checkpointable to '
        'finish cleanup.'
    )
  return deleting_name


def _local_path(path: path_types.Path) -> pathlib.Path:
  if parse.urlparse(str(path)).scheme:
    raise NotImplementedError(
        f'Atomic deletion metadata updates are unsupported at {path}.'
    )
  return pathlib.Path(str(path))


def check_atomic_write_support(path: path_types.Path) -> None:
  """Checks the publication backend before creating partial-deletion state."""
  if not gcs_utils.is_gcs_path(path):
    _local_path(path)


async def write_json(
    path: path_types.Path,
    value: dict[str, Any],
    *,
    exclusive: bool = False,
    permission_mode: int | None = None,
) -> None:
  """Atomically publishes a complete JSON document.

  For local files, fsync precedes rename (or an exclusive hard-link
  publication). GCS uploads commit one object; exclusive publication uses a
  generation match. Other schemes require an explicit implementation rather than
  a text overwrite.

  Args:
    path: The path to the metadata file.
    value: The metadata dictionary to write.
    exclusive: Whether to fail if the path already exists.
    permission_mode: The permission mode to use for the metadata file if it does
      not exist.

  Raises:
    NotImplementedError: If the publication backend is not supported.
    FileExistsError: If the path already exists and exclusive is True.
  """
  # Keep the atomic sequence together in one worker, including flush/cleanup.
  await asyncio.to_thread(
      _write_json,
      path,
      value,
      exclusive=exclusive,
      permission_mode=permission_mode,
  )


def _write_json(
    path: path_types.Path,
    value: dict[str, Any],
    *,
    exclusive: bool,
    permission_mode: int | None,
) -> None:
  """Blocking implementation of `write_json`; runs in a worker thread."""
  data = json.dumps(value, sort_keys=True)
  if gcs_utils.is_gcs_path(path):
    bucket, blob_path = gcs_utils.split_gcs_path(path)
    blob = gcs_utils.get_bucket(bucket).blob(blob_path)
    kwargs = {'if_generation_match': 0} if exclusive else {}
    blob.upload_from_string(data, content_type='application/json', **kwargs)
    return
  local = _local_path(path)
  descriptor, temporary = tempfile.mkstemp(
      prefix='.orbax-delete-', dir=local.parent
  )
  try:
    with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
      if local.exists():
        os.fchmod(stream.fileno(), local.stat().st_mode & 0o777)
      elif permission_mode is not None:
        os.fchmod(stream.fileno(), permission_mode)
      stream.write(data)
      stream.flush()
      os.fsync(stream.fileno())
    if exclusive:
      os.link(temporary, local)
    else:
      os.replace(temporary, local)
    _sync_directory(path.parent)
  finally:
    pathlib.Path(temporary).unlink(missing_ok=True)


async def sync_directory(path: path_types.Path) -> None:
  """Flushes directory changes without blocking the event loop."""
  await asyncio.to_thread(_sync_directory, path)


def _sync_directory(path: path_types.Path) -> None:
  """Flushes local directory changes; GCS object operations commit on return."""
  if gcs_utils.is_gcs_path(path):
    return
  descriptor = os.open(_local_path(path), os.O_RDONLY)
  try:
    os.fsync(descriptor)
  finally:
    os.close(descriptor)
