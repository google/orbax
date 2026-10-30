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
from typing import Any, NamedTuple
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
  path_str = os.fspath(path)
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


def get_kvstore_for_gcs(ckpt_path: epath.PathLike) -> dict[str, Any]:
  """Constructs a TensorStore kvstore spec for a GCS path."""
  gcs_bucket, path_without_bucket = parse_gcs_path(
      ckpt_path, add_trailing_slash=False
  )
  # TODO(b/518937340): Consider enabling gcs_grpc by default.
  # TODO(b/518937340): Migrate TENSORSTORE_GCS_BACKEND flag to `Context`.
  gcs_backend = os.environ.get('TENSORSTORE_GCS_BACKEND')
  if gcs_backend is None:
    if os.fspath(ckpt_path).startswith('gs://') and is_zonal_bucket(gcs_bucket):
      gcs_backend = 'gcs_grpc'
    else:
      gcs_backend = 'gcs'
  logging.vlog(
      1, 'Using GCS backend (TENSORSTORE_GCS_BACKEND): %s', gcs_backend
  )
  return {
      'driver': gcs_backend,
      'bucket': gcs_bucket,
      'path': path_without_bucket,
  }


@functools.lru_cache(maxsize=1)
def _get_storage_client() -> Any:
  """Returns a cached google.cloud.storage.Client instance."""
  # pylint: disable=g-import-not-at-top
  from google.cloud import storage

  return storage.Client()


@functools.lru_cache(maxsize=1)
def _get_storage_control_client() -> Any | None:
  """Returns a cached StorageControlClient.

  Returns:
    A cached StorageControlClient instance, or None if the
    google.cloud.storage_control_v2 package cannot be imported.
  """
  try:
    # pylint: disable=g-import-not-at-top
    # pyrefly: ignore[missing-module-attribute]
    from google.cloud import storage_control_v2
  except ImportError:
    return None
  return storage_control_v2.StorageControlClient()


@functools.lru_cache(maxsize=32)
def get_bucket(bucket_name: str) -> Any:
  """Returns the GCS bucket for the given bucket name."""
  # pylint: disable=g-import-not-at-top
  from google.api_core import exceptions as api_exceptions
  # pylint: enable=g-import-not-at-top

  client = _get_storage_client()
  try:
    return client.get_bucket(bucket_name)
  except api_exceptions.Forbidden:
    return client.bucket(bucket_name)


class _BucketInfo(NamedTuple):
  is_zonal: bool
  is_hns: bool


@functools.lru_cache(maxsize=32)
def _get_bucket_info(bucket_name: str) -> _BucketInfo:
  """Returns _BucketInfo(is_zonal, is_hns) for the given bucket name."""
  if not bucket_name:
    return _BucketInfo(False, False)
  # pylint: disable=g-import-not-at-top
  from google.api_core import exceptions as api_exceptions
  from google.auth import exceptions as auth_exceptions
  # pylint: enable=g-import-not-at-top
  try:
    if (client := _get_storage_control_client()) is not None:
      layout = client.get_storage_layout(
          name=f'projects/_/buckets/{bucket_name}/storageLayout'
      )
      hns = getattr(layout, 'hierarchical_namespace', None)
      return _BucketInfo(
          getattr(layout, 'location_type', None) == 'zone',
          bool(getattr(hns, 'enabled', False)),
      )
  except api_exceptions.GoogleAPIError as e:
    logging.vlog(1, 'Failed to check storage layout %s: %s', bucket_name, e)
  except (auth_exceptions.GoogleAuthError, OSError, ValueError) as e:
    logging.vlog(1, 'Failed to check storage layout %s: %s', bucket_name, e)
    return _BucketInfo(False, False)
  try:
    bucket = get_bucket(bucket_name)
    return _BucketInfo(
        getattr(bucket, 'location_type', None) == 'zone'
        or getattr(bucket, 'storage_class', None) == 'RAPID',
        bool(getattr(bucket, 'hierarchical_namespace_enabled', False)),
    )
  except (
      ImportError,
      api_exceptions.GoogleAPIError,
      auth_exceptions.GoogleAuthError,
      OSError,
      ValueError,
  ) as e:
    logging.vlog(1, 'Failed to check bucket %s: %s', bucket_name, e)
    return _BucketInfo(False, False)


def is_zonal_bucket(bucket_name: str) -> bool:
  """Returns True if the GCS bucket is a zonal Rapid bucket."""
  return _get_bucket_info(bucket_name).is_zonal


def is_hierarchical_namespace_enabled(path: epath.PathLike) -> bool:
  """Return whether hierarchical namespace is enabled."""
  if parse.urlparse(os.fspath(path)).scheme != 'gs':
    return False
  bucket_name, _ = parse_gcs_path(path)
  return _get_bucket_info(bucket_name).is_hns


def cleanup_hns_folders(path: epath.Path) -> None:
  """For a hierarchical namespace bucket, delete empty folders recursively."""
  # pylint: disable=g-import-not-at-top
  from google.cloud import storage_control_v2  # pyrefly: ignore[missing-module-attribute]

  bucket, prefix = parse_gcs_path(path)

  client = (
      _get_storage_control_client() or storage_control_v2.StorageControlClient()
  )
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
