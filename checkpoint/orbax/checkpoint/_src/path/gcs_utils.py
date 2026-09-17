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
import pathlib
from urllib import parse

from absl import logging
from etils import epath

_GCS_PATH_PREFIX = ('gs://',)

_MOUNTS_FILE = '/proc/mounts'
_GCSFUSE_FSTYPES = ('fuse.gcsfuse', 'gcsfuse')
# Octal escapes used by the kernel for whitespace in mount table fields.
_MOUNT_POINT_ESCAPES = (
    ('\\040', ' '),
    ('\\011', '\t'),
    ('\\012', '\n'),
    ('\\134', '\\'),
)


def is_gcs_path(path: pathlib.PurePosixPath) -> bool:
  return path.as_posix().startswith(_GCS_PATH_PREFIX)


def _unescape_mount_point(mount_point: str) -> str:
  for escape, char in _MOUNT_POINT_ESCAPES:
    mount_point = mount_point.replace(escape, char)
  return mount_point


@functools.lru_cache(maxsize=1)
def _mount_table() -> tuple[tuple[str, str], ...]:
  """Returns (mount_point, fstype) pairs, or () if the table is unreadable."""
  try:
    with open(_MOUNTS_FILE, 'rt') as f:
      lines = f.read().splitlines()
  except OSError:
    return ()
  table = []
  for line in lines:
    fields = line.split()
    if len(fields) >= 3:
      table.append((_unescape_mount_point(fields[1]), fields[2]))
  return tuple(table)


def is_gcsfuse_path(path: epath.PathLike) -> bool:
  """Returns whether `path` resides on a GCSFuse mount.

  The system mount table is read once per process and cached, so mounts
  established after the first call are not detected.

  Args:
    path: A local filesystem path. URI-style paths (e.g. `gs://...`) are never
      considered GCSFuse paths.

  Returns:
    True if the deepest mount containing `path` is a GCSFuse filesystem.
  """
  path_str = os.fspath(path)
  if '://' in path_str:
    return False
  resolved = os.path.realpath(path_str)
  best_mount_point = ''
  best_fstype = ''
  for mount_point, fstype in _mount_table():
    if resolved == mount_point or resolved.startswith(
        mount_point.rstrip('/') + '/'
    ):
      if len(mount_point) > len(best_mount_point):
        best_mount_point = mount_point
        best_fstype = fstype
  return best_fstype in _GCSFUSE_FSTYPES


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
