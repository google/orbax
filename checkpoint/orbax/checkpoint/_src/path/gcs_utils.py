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
from typing import Any
from urllib import parse
from absl import logging
from etils import epath

_GCS_PATH_PREFIX = ('gs://',)


def is_gcs_path(path: pathlib.PurePosixPath) -> bool:
  return path.as_posix().startswith(_GCS_PATH_PREFIX)


def parse_gcs_path(
    path: epath.PathLike, add_trailing_slash: bool = True
) -> tuple[str, str]:
  """Parses a GCS path into a bucket name and a path within the bucket."""
  path_str = str(path)
  for prefix in _GCS_PATH_PREFIX:
    if prefix != 'gs://' and path_str.startswith(prefix):
      path_str = 'gs://' + path_str.removeprefix(prefix)
      break
  parsed = parse.urlparse(path_str)
  assert parsed.scheme == 'gs', f'Unsupported scheme for GCS: {parsed.scheme}'
  if not parsed.netloc:
    raise ValueError(
        'The GCS path should contain the bucket name and the '
        f'file path inside the bucket. Got: {path}'
    )
  # Strip the leading slash from the path.
  standardized_path = parsed.path
  if standardized_path.startswith('/'):
    standardized_path = standardized_path[1:]
  # Add a trailing slash if it's missing.
  if add_trailing_slash and not standardized_path.endswith('/'):
    standardized_path = standardized_path + '/'
  return parsed.netloc, standardized_path


def get_kvstore_for_gcs(ckpt_path: str) -> dict[str, Any]:
  """Constructs a TensorStore kvstore spec for a GCS path."""
  gcs_bucket, path_without_bucket = parse_gcs_path(
      ckpt_path, add_trailing_slash=False
  )
  # TODO(b/518937340): Consider enabling gcs_grpc by default.
  # TODO(b/518937340): Migrate TENSORSTORE_GCS_BACKEND flag to `Context`.
  gcs_backend = os.environ.get('TENSORSTORE_GCS_BACKEND', 'gcs')
  logging.vlog(
      1, 'Using GCS backend (TENSORSTORE_GCS_BACKEND): %s', gcs_backend
  )
  return {
      'driver': gcs_backend,
      'bucket': gcs_bucket,
      'path': path_without_bucket,
  }


@functools.lru_cache(maxsize=32)
def get_bucket(bucket_name: str):
  # pylint: disable=g-import-not-at-top
  from google.cloud import storage

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
  from google.cloud import storage_control_v2  # pyrefly: ignore[missing-module-attribute]

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


def rmtree(path: epath.Path, *, missing_ok: bool = False) -> None:
  """Deletes a GCS path, performing HNS folder cleanup if necessary.

  Args:
    path: the global path to delete, must be a GCS path.
    missing_ok: Whether to ignore if the path does not exist.

  Raises:
    ValueError: if path is not a GCS path.
    FileNotFoundError: if path does not exist and missing_ok is False.
  """
  if not is_gcs_path(path):
    raise ValueError(f'Path is not a GCS path: {path}')

  try:
    path.rmtree()
  except FileNotFoundError:
    if not missing_ok:
      raise
    return

  # For HNS, clean up the remaining empty directory structure.
  if is_hierarchical_namespace_enabled(path):
    cleanup_hns_folders(path)
