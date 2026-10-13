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


def is_gcs_path(path: pathlib.PurePosixPath) -> bool:
  return path.as_posix().startswith(_GCS_PATH_PREFIX)


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


def split_gcs_path(path: epath.PathLike) -> tuple[str, str]:
  """Splits a GCS path into (bucket_name, relative_blob_path).

  `parse_gcs_path` is intentionally not reused: it asserts a `gs` scheme,
  appends a trailing slash to the returned object path, and does not understand
  non-URL GCS mount points. Blob names here are passed directly to the storage
  client and must not gain a trailing slash.

  Args:
    path: The GCS path to split.

  Returns:
    A tuple of (bucket_name, relative_blob_path).
  """
  path_str = str(path)
  if path_str.startswith('gs://'):
    parsed = parse.urlparse(path_str)
    return parsed.netloc, parsed.path.lstrip('/')
  for prefix in _GCS_PATH_PREFIX:
    if path_str.startswith(prefix):
      parts = path_str[len(prefix) :].split('/', 1)
      bucket = parts[0]
      blob_path = parts[1] if len(parts) > 1 else ''
      return bucket, blob_path
  parsed = parse.urlparse(path_str)
  return parsed.netloc, parsed.path.lstrip('/')


def gcs_bucket_root(path: epath.PathLike) -> str:
  """Returns the bucket root of a GCS path, preserving its access prefix.

  `is_gcs_path` accepts `gs://` URLs as well as non-URL GCS mount points. Paths
  derived from the source (such as a relocation destination) must keep the same
  prefix so that they are reached through the same filesystem as the source.

  Args:
    path: The GCS path.

  Returns:
    The prefix and bucket, e.g. `gs://bucket` or `/gcs/bucket`.

  Raises:
    ValueError: If `path` is not a recognized GCS path.
  """
  path_str = str(path)
  bucket, _ = split_gcs_path(path)
  if bucket:
    for prefix in _GCS_PATH_PREFIX:
      if path_str.startswith(prefix):
        return f'{prefix}{bucket}'
  raise ValueError(f'Could not parse GCS bucket from path: {path}.')


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

    # When having directory placeholder, rmtree will fail to
    # delete the files inside the directory, so do another round of deletion.
    # TODO: b/570544727 - resolve double rmtree required on GCS.
    path.rmtree(missing_ok=True)
  except FileNotFoundError:
    if not missing_ok:
      raise
    return

  # For HNS, clean up the remaining empty directory structure.
  if is_hierarchical_namespace_enabled(path):
    cleanup_hns_folders(path)
