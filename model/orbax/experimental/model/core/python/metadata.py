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

"""Model version metadata and its serialization."""

import dataclasses
from orbax.experimental.model.core.python import file_utils


@dataclasses.dataclass
class ModelVersion:
  """Model version metadata."""

  _VERSION_KEY = 'version'
  _MIME_TYPE_KEY = 'mime_type'
  _MANIFEST_FILE_PATH_KEY = 'manifest_file_path'

  version: str
  mime_type: str
  manifest_file_path: str

  def save(self, path: str) -> None:
    """Saves the model version metadata to a file."""
    with file_utils.open_file(path, 'w') as f:
      f.write(f'{self._MANIFEST_FILE_PATH_KEY}: "{self.manifest_file_path}"\n')
      f.write(f'{self._VERSION_KEY}: "{self.version}"\n')
      f.write(f'{self._MIME_TYPE_KEY}: "{self.mime_type}"\n')

  @classmethod
  def load(cls, path: str) -> 'ModelVersion':
    """Loads the model version metadata from a file."""

    version = ''
    mime_type = ''
    manifest_file_path = ''

    with file_utils.open_file(path, 'r') as f:
      for line in f:
        line = line.strip()
        if not line:
          continue
        if ':' not in line:
          raise ValueError(f'Malformed line: {line}')

        key, value = line.split(':', 1)
        key = key.strip()
        value = value.strip()

        if not value.startswith('"') or not value.endswith('"'):
          raise ValueError('All values must be double-quoted')

        value = value[1:-1]
        if key == cls._MANIFEST_FILE_PATH_KEY:
          manifest_file_path = value
        elif key == cls._VERSION_KEY:
          version = value
        elif key == cls._MIME_TYPE_KEY:
          mime_type = value
        else:
          raise ValueError(f'Unknown key: {key}')

      if not version:
        raise ValueError('Version is empty')
      if not mime_type:
        raise ValueError('MIME type is empty')
      if not manifest_file_path:
        raise ValueError('Manifest file path is empty')

      return cls(
          version=version,
          mime_type=mime_type,
          manifest_file_path=manifest_file_path,
      )
