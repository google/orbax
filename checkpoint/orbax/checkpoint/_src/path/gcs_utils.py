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

"""Utils for interacting with GCS paths."""

import functools
import os
from urllib import parse
from absl import logging
from etils import epath


_GCS_PATH_PREFIX = ('gs://',)


def is_gcs_path(path: epath.Path) -> bool:
  return path.as_posix().startswith(_GCS_PATH_PREFIX)


def to_gcsfuse_path(path: epath.PathLike) -> str:
  """Converts a GCS path to a gcsfuse path string.

  GCSfuse paths start with /gcs/ and are accessible via File API when gcsfuse
  is enabled.

  Args:
    path: A GCS path which can be a string or epath.Path.

  Returns:
    A gcsfuse path string starting with /gcs/.

  Raises:
    ValueError: If path is not a GCS path.
  """
  path_str = str(path)
  if path_str.startswith('gs://'):
    return '/gcs/' + path_str[5:]
  elif path_str.startswith('/gcs/'):
    return path_str
  else:
    raise ValueError(f'Path is not a GCS path: {path}')


def parse_gcs_path(path: epath.PathLike) -> tuple[str, str]:
  parsed = parse.urlparse(str(path))
  assert parsed.scheme == 'gs', f'Unsupported scheme for GCS: {parsed.scheme}'
  # Strip the leading slash from the path.
  standardized_path = parsed.path
  if standardized_path.startswith('/'):
    standardized_path = standardized_path[1:]
  # Add a trailing slash if it's missing.
  if not standardized_path.endswith('/'):
    standardized_path = standardized_path + '/'
  return parsed.netloc, standardized_path


@functools.lru_cache(maxsize=32)
def get_bucket(bucket_name: str):
  # pylint: disable=g-import-not-at-top
  from google.cloud import storage  # pytype: disable=import-error

  client = storage.Client()
  return client.get_bucket(bucket_name)


def is_hierarchical_namespace_enabled(path: epath.PathLike) -> bool:
  """Return whether hierarchical namespace is enabled."""
  parsed = parse.urlparse(str(path))
  if parsed.scheme != 'gs':
    return False
  bucket_name, _ = parse_gcs_path(path)
  bucket = get_bucket(bucket_name)
  return (
      hasattr(bucket, 'hierarchical_namespace_enabled')
      and bucket.hierarchical_namespace_enabled
  )


def cleanup_hns_folders(path: epath.Path) -> None:
  """For a hierarchical namespace bucket, delete empty folders recursively."""
  # pylint: disable=g-import-not-at-top
  from google.cloud import storage_control_v2  # pytype: disable=import-error

  bucket, prefix = parse_gcs_path(path)

  client = storage_control_v2.StorageControlClient()
  project_path = client.common_project_path('_')
  bucket_path = f'{project_path}/buckets/{bucket}'
  folders = set(
      # Format: "projects/{project}/buckets/{bucket}/folders/{folder}"
      folder.name
      for folder in client.list_folders(
          request=storage_control_v2.ListFoldersRequest(
              parent=bucket_path, prefix=prefix.strip('/') + '/'
          )
      )
  )

  while folders:
    parents = set(os.path.dirname(x.rstrip('/')) + '/' for x in folders)
    leaves = folders - parents
    requests = [storage_control_v2.DeleteFolderRequest(name=f) for f in leaves]
    for req in requests:
      client.delete_folder(request=req)
    folders = folders - leaves
    logging.vlog(
        1,
        'Deleted %s folders, %s remaining. [%s][%s]',
        len(leaves),
        len(folders),
        bucket,
        prefix,
    )


def rmtree(path: epath.Path) -> None:
  """Deletes a GCS path, performing HNS folder cleanup if necessary.

  Args:
    path: the global path to delete, must be a GCS path.

  Raises:
    ValueError: if path is not a GCS path.
  """
  if not is_gcs_path(path):
    raise ValueError(f'Path is not a GCS path: {path}')

  path.rmtree()

  # For HNS, clean up the remaining empty directory structure.
  if is_hierarchical_namespace_enabled(path):
    cleanup_hns_folders(path)
