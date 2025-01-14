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

"""Storage for `array_metadata.ArrayMetadata` (not value.ArrayMetadata)."""

import json
import threading
from typing import Any, Iterator, List, Sequence
from absl import logging
from etils import epath
from orbax.checkpoint._src.metadata import array_metadata as array_metadata_lib
from orbax.checkpoint._src.multihost import multihost


class PathResolver:
  """Resolves paths for the ArrayMetadata store read and write."""

  _metadata_subdir = 'array_metadatas'

  def _file_name(self, process_index: int | str) -> str:
    return f'process_{process_index}'

  def get_process_index(self, file_path: epath.Path) -> int:
    """Returns the process index from the file path."""
    process_index = file_path.name.removeprefix('process_')
    if process_index.isdigit():
      return int(process_index)
    raise ValueError(
        f'Invalid ArrayMetadata file path: {file_path}; expected file name'
        ' to start with "process_"'
    )

  def get_write_file_path(
      self, checkpoint_dir: epath.Path, process_index: int
  ) -> epath.Path:
    """Returns the file path to write."""
    return (
        checkpoint_dir / self._metadata_subdir / self._file_name(process_index)
    )

  def get_read_file_paths(
      self, checkpoint_dir: epath.Path, process_index: int | None = None
  ) -> Iterator[epath.Path] | epath.Path | None:
    """Returns the file paths to read.

    Args:
      checkpoint_dir: The base path containing metadata for each process.
      process_index: The process index to read. If None, then read all processes
        under `checkpoint_dir`.

    Returns:
      Iterator of file paths to read if `process_index` is None. A file path to
      read if `process_index` is not None. None if `process_index` is not None
      but metadata file does not exist.
    """
    if process_index is None:
      file_name_pattern = self._file_name('*')
      return checkpoint_dir.glob(f'{self._metadata_subdir}/{file_name_pattern}')
    file_path = (
        checkpoint_dir / self._metadata_subdir / self._file_name(process_index)
    )
    if file_path.exists():
      return file_path
    return None


class SerDeserializer:
  """Serializes and deserializes `tensorstore_utils.ArrayMetadata`."""

  def _to_dict(
      self, array_metadata: array_metadata_lib.ArrayMetadata
  ) -> dict[str, Any]:
    """Converts `array_metadata` to a dictionary."""
    return {
        'array_metadata': {
            'param_name': array_metadata.param_name,
            'write_shape': array_metadata.write_shape,
            'chunk_shape': array_metadata.chunk_shape,
        }
    }

  def _from_dict(self, obj: dict[str, Any]) -> Any:
    """Converts a json object to `SerializedArrayMetadata` or `obj`."""
    if 'array_metadata' in obj:
      array_metadata = obj['array_metadata']
      return array_metadata_lib.SerializedArrayMetadata(
          param_name=array_metadata['param_name'],
          write_shape=tuple(array_metadata['write_shape']),
          chunk_shape=tuple(array_metadata['chunk_shape']),
      )
    return obj

  def serialize(
      self, array_metadatas: Sequence[array_metadata_lib.ArrayMetadata]
  ) -> str:
    """Serializes `array_metadatas` to string."""
    obj = {
        'array_metadatas': [
            self._to_dict(array_metadata) for array_metadata in array_metadatas
        ]
    }
    return json.dumps(obj)

  def deserialize(
      self, serialized: str
  ) -> List[array_metadata_lib.SerializedArrayMetadata]:
    """Deserializes `serialized` to `tensorstore_utils.ArrayMetadata`."""
    obj = json.loads(serialized, object_hook=self._from_dict)
    return obj['array_metadatas']


class Store:
  """Storage for `tensorstore_utils.ArrayMetadata` (not value.ArrayMetadata)."""

  def __init__(
      self,
      path_resolver: PathResolver = PathResolver(),
      ser_deser: SerDeserializer = SerDeserializer(),
  ):
    self._path_resolver = path_resolver
    self._ser_deser = ser_deser

  async def write(
      self,
      checkpoint_dir: epath.Path,
      array_metadatas: Sequence[array_metadata_lib.ArrayMetadata],
      process_index: int,
  ) -> None:
    """Writes `array_metadatas` to a file under `checkpoint_dir`.

    See `PathResolver.get_write_file_path()` for the file path resolution.

    Args:
      checkpoint_dir: The base path containing metadata for each process.
      array_metadatas: The sequence of metadata to write.
      process_index: The Jax process index used to resolve the file path.
    """
    file_path = self._path_resolver.get_write_file_path(
        checkpoint_dir, process_index
    )
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(self._ser_deser.serialize(array_metadatas))
    logging.info(
        '[process=%s][thread=%s] Wrote %d tensorstore_utils.ArrayMetadata'
        ' to %s',
        multihost.process_index(),
        threading.current_thread().name,
        len(array_metadatas),
        file_path,
    )

  def read(
      self,
      checkpoint_dir: epath.Path,
      process_index: int | None = None,
  ) -> (
      dict[int, List[array_metadata_lib.SerializedArrayMetadata]]
      | List[array_metadata_lib.SerializedArrayMetadata]
      | None
  ):
    """Reads `SerializedArrayMetadata` from storage under `checkpoint_dir`.

    Args:
      checkpoint_dir: The base path containing metadata for each process.
      process_index: The process index to read. If None, then read all processes
        under `checkpoint_dir`.

    Returns:
      A dictionary of process index to list of metadata if `process_index`
      is None. A list of metadata if `process_index` is not None. None if
      metadata does not exist.
    """
    if not checkpoint_dir.exists():
      raise ValueError(
          f'Checkpoint directory does not exist: {checkpoint_dir}.'
      )
    file_paths = self._path_resolver.get_read_file_paths(
        checkpoint_dir, process_index
    )
    if file_paths is None:
      logging.warning(
          '[process=%s][thread=%s] No metadata found for process_index=%s,'
          ' checkpoint_dir=%s. Please ignore if input checkpoint does not'
          ' contain any jax.Array.',
          multihost.process_index(),
          threading.current_thread().name,
          process_index,
          checkpoint_dir,
      )
      return None
    if isinstance(file_paths, epath.Path):
      return self._ser_deser.deserialize(file_paths.read_text())
    result = {
        self._path_resolver.get_process_index(
            file_path
        ): self._ser_deser.deserialize(file_path.read_text())
        for file_path in file_paths
    }
    if not result:
      logging.warning(
          '[process=%s][thread=%s] No metadata found for any process_index,'
          ' checkpoint_dir=%s. Please ignore if input checkpoint does not'
          ' contain any jax.Array.',
          multihost.process_index(),
          threading.current_thread().name,
          checkpoint_dir,
      )
      return None
    return result
