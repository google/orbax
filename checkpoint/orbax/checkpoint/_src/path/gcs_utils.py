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

"""Utils for interacting with GCS paths."""

import functools
import os
from urllib import parse
from etils import epath


_GCS_PATH_PREFIX = ('gs://',)


def is_gcs_path(path: epath.Path) -> bool:
  return path.as_posix().startswith(_GCS_PATH_PREFIX)


def parse_gcs_path(
    path: epath.PathLike, add_trailing_slash: bool = True
) -> tuple[str, str]:
  """Parses a GCS path into bucket and path within the bucket.

  Args:
    path: The GCS path to parse (e.g., "gs://my-bucket/path/to/object").
    add_trailing_slash: Whether to ensure the returned path has a trailing
      slash.

  Returns:
    A tuple containing the bucket name and the path within the bucket.
  """
  parsed = parse.urlparse(str(path))
  assert parsed.scheme == 'gs', f'Unsupported scheme for GCS: {parsed.scheme}'
  # Strip the leading slash from the path.
  standardized_path = parsed.path
  if standardized_path.startswith('/'):
    standardized_path = standardized_path[1:]
  # Add a trailing slash if it's missing.
  if add_trailing_slash and not standardized_path.endswith('/'):
    standardized_path = standardized_path + '/'
  return parsed.netloc, standardized_path


def get_kvstore_for_gcs(ckpt_path: str):
  """Constructs a TensorStore kvstore spec for a GCS path.

  Args:
    ckpt_path: A GCS path of the form gs://<bucket>/<path>.

  Returns:
    A dictionary containing the TensorStore kvstore spec.

  Raises:
    ValueError: if ckpt_path is not a valid GCS path.
  """
  gcs_bucket, path_without_bucket = parse_gcs_path(
      ckpt_path, add_trailing_slash=False
  )
  # TODO(stoelinga): Switch to gcs_grpc by default.
  # gcs_grpc performs roughly twice as fast as gcs backend.
  gcs_backend = os.environ.get('TENSORSTORE_GCS_BACKEND', 'gcs')
  spec = {'driver': gcs_backend, 'bucket': gcs_bucket}
  if path_without_bucket:
    spec['path'] = path_without_bucket
  return spec


@functools.lru_cache(maxsize=32)
def get_bucket(bucket_name: str):
  # pylint: disable=g-import-not-at-top
  from google.cloud import storage  # pytype: disable=import-error

  client = storage.Client()
  return client.get_bucket(bucket_name)


def is_hierarchical_namespace_enabled(path: epath.PathLike) -> bool:
  """Return whether hierarchical namespace is enabled."""
  bucket_name, _ = parse_gcs_path(path)
  bucket = get_bucket(bucket_name)
  return bucket.hierarchical_namespace_enabled
