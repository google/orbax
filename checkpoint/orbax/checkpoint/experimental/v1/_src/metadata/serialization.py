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

"""Metadata serialization functions."""

import json
from absl import logging
from orbax.checkpoint._src.path import async_path
from orbax.checkpoint.experimental.v1._src.metadata import types as metadata_types
from orbax.checkpoint.experimental.v1._src.path import types as path_types

SerializedMetadata = metadata_types.SerializedMetadata
_CHECKPOINT_METADATA_FILENAME = '_CHECKPOINT_METADATA'


def checkpoint_metadata_file_path(path: path_types.Path) -> path_types.Path:
  """The path to step metadata file for a given checkpoint directory."""
  return path / _CHECKPOINT_METADATA_FILENAME


async def write(metadata_file: path_types.Path, metadata: SerializedMetadata):
  """Writes metadata to a file.

  This function should typically be called within a background thread. It is
  expected that the metadata is written in only one place and never updated.

  Args:
    metadata_file: The file to write metadata to.
    metadata: The metadata to write.

  Raises:
    FileNotFoundError: If the parent directory of the metadata file does not
      exist.
    ValueError: If the metadata file cannot be written to.
  """
  if not await async_path.exists(metadata_file.parent):
    raise FileNotFoundError(
        f'Metadata path does not exist: {metadata_file.parent}'
    )
  if not isinstance(metadata, dict):
    raise TypeError(
        f'Metadata must be an instance of `dict`, got {metadata}.'
    )
  json_data = json.dumps(metadata)
  bytes_written = await async_path.write_text(metadata_file, json_data)
  if bytes_written == 0:
    raise ValueError(
        f'Failed to write Metadata={metadata},'
        f' json={json_data} to {metadata_file}'
    )


async def read(metadata_file: path_types.Path) -> SerializedMetadata | None:
  """Reads metadata from a file.

  Like `serialize`, it is expected that when called, this function is a unique
  reader of the metadata file.

  Args:
    metadata_file: The file to read metadata from.

  Returns:
    The metadata read from the file, or None if the file does not exist or
    cannot be read.
  """
  if not await async_path.exists(metadata_file):
    logging.warning(
        'Metadata file does not exist: %s', metadata_file
    )
    return None
  try:
    raw_data = await async_path.read_text(metadata_file)
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.error(
        'Failed to read Metadata file: %s, error: %s',
        metadata_file,
        e,
    )
    return None
  try:
    json_data = json.loads(raw_data)
  except json.decoder.JSONDecodeError as e:
    # TODO(b/340287956): Found empty metadata files, how is it possible.
    logging.error(
        'Failed to json parse Metadata file: %s, '
        'file content: %s, '
        'error: %s',
        metadata_file,
        raw_data,
        e,
    )
    return None
  return json_data
