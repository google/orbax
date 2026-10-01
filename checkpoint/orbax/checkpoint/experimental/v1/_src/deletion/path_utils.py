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

"""Path operations used by free-function and partial deletion.

Managed whole-step deletion uses the manager's existing v0 deleter instead.
"""

import os
import pathlib
from urllib import parse

from orbax.checkpoint._src.path import deleter
from orbax.checkpoint._src.path import gcs_utils
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout
from orbax.checkpoint.experimental.v1._src.metadata import serialization as metadata_serialization
from orbax.checkpoint.experimental.v1._src.path import types as path_types

# Non-URL mount prefixes under which a GCS bucket may be reached.
_GCS_MOUNT_PREFIXES = ('/gcs/',)


def validate_checkpointable_name(name: str) -> None:
  """Requires an explicit user name identifying one child directory.

  Loading's name validator permits AUTO and legacy unnamed pytrees. Deletion
  must select a literal directory; reserved names come from the shared layout.

  Args:
    name: The name of the checkpointable.

  Raises:
    TypeError: If `name` is not a string.
    ValueError: If `name` is an invalid checkpointable name.
  """
  if not isinstance(name, str):
    raise TypeError('checkpointable_name must be a string or None.')
  if (
      not name
      or name in ('.', '..')
      or name in checkpoint_layout.RESERVED_CHECKPOINTABLE_KEYS
      or name == metadata_serialization.CHECKPOINT_METADATA_FILENAME
      or any(c in name for c in ('/', '\\', '\x00'))
  ):
    raise ValueError(f'Invalid checkpointable name: {name!r}.')


def check_path(path: path_types.Path) -> None:
  """Rejects storage roots and local symlink targets."""
  parsed = parse.urlparse(str(path))
  if not path.name or not parsed.path.strip('/') or str(path) in ('.', '..'):
    raise ValueError(f'Cannot delete a storage root: {path}.')
  if not parsed.scheme and pathlib.Path(str(path)).is_symlink():
    raise ValueError(f'Cannot delete a checkpoint through a symlink: {path}.')


def validate_options(context: context_lib.Context) -> None:
  """Rejects relocation settings that would escape the checkpoint parent.

  Only traversal is checked. The v0 backend joins these values onto a directory
  without inspecting them, so anything that stays relative is accepted here too;
  narrowing the character set further would reject configurations that v0
  already supports.

  Args:
    context: The current context.

  Raises:
    ValueError: If the deletion options are invalid.
  """
  options = context.deletion_options
  full_path = options.gcs_deletion_options.todelete_full_path
  if full_path is not None and (
      not full_path
      or full_path.startswith('/')
      or '..' in full_path.split('/')
      or '://' in full_path
  ):
    raise ValueError('todelete_full_path must be a relative bucket path.')


def check_destination(
    target: path_types.Path, checkpoint_root: path_types.Path
) -> None:
  """Rejects destinations that land inside the checkpoint being deleted.

  Literal containment is checked first so the common misconfiguration reports
  the configured path. Local destinations are then compared after resolution,
  because a symlinked ancestor can redirect an apparently external destination
  back into the checkpoint. A symlink pointing somewhere else entirely is a
  valid administrative setup and is allowed.

  Args:
    target: The relocation destination.
    checkpoint_root: The root of the checkpoint.

  Raises:
    ValueError: If the destination is inside the checkpoint.
  """
  if target == checkpoint_root or checkpoint_root in target.parents:
    raise ValueError('The deletion destination must be outside the checkpoint.')
  if parse.urlparse(str(target)).scheme:
    return  # Remote schemes have no symlinks to resolve.
  root = os.path.realpath(str(checkpoint_root))
  resolved = os.path.realpath(str(target))
  if resolved == root or resolved.startswith(root + os.sep):
    raise ValueError(
        f'The deletion destination resolves inside the checkpoint: {target}.'
    )


def split_gcs_path(path: path_types.Path) -> tuple[str, str]:
  """Splits a GCS path into (bucket_name, relative_blob_path).

  `gcs_utils.parse_gcs_path` is intentionally not reused: it asserts a `gs`
  scheme, appends a trailing slash to the returned object path, and does not
  understand the `_GCS_MOUNT_PREFIXES` mount points. Blob names here are
  passed directly to the storage client and must not gain a trailing slash.

  Args:
    path: The GCS path to split.

  Returns:
    A tuple of (bucket_name, relative_blob_path).
  """
  path_str = str(path)
  if path_str.startswith('gs://'):
    parsed = parse.urlparse(path_str)
    return parsed.netloc, parsed.path.lstrip('/')
  for prefix in _GCS_MOUNT_PREFIXES:
    if path_str.startswith(prefix):
      parts = path_str[len(prefix):].split('/', 1)
      bucket = parts[0]
      blob_path = parts[1] if len(parts) > 1 else ''
      return bucket, blob_path
  parsed = parse.urlparse(path_str)
  return parsed.netloc, parsed.path.lstrip('/')


def destination(
    path: path_types.Path,
    context: context_lib.Context,
    operation_id: str,
    *,
    checkpoint_root: path_types.Path | None = None,
    whole_relocation_name: str | None = None,
) -> path_types.Path | None:
  """Resolves relocation without creating directories or moving the target."""
  validate_options(context)
  root = checkpoint_root if checkpoint_root is not None else path
  options = context.deletion_options
  trash = options.gcs_deletion_options.todelete_full_path
  if gcs_utils.is_gcs_path(path):
    # Like the v0 deleter, GCS ignores the local todelete_subdir option.
    if trash is None:
      return None
    bucket, _ = split_gcs_path(path)
    parent = context.file_options.path_class(f'gs://{bucket}/{trash}')
    target = parent / f'{path.name}-{operation_id}'
    check_destination(target, root)
    return target
  if trash is not None:
    raise NotImplementedError('todelete_full_path requires a GCS checkpoint.')
  return None


def check_recovery_destination(
    target: path_types.Path,
    root: path_types.Path,
    name: str,
    operation_id: str,
) -> None:
  """Validates a recorded item destination independently of current options."""
  if gcs_utils.is_gcs_path(root):
    if not gcs_utils.is_gcs_path(target):
      raise ValueError(
          'Recovery destination must use the same storage backend.'
      )
    root_bucket, _ = split_gcs_path(root)
    target_bucket, _ = split_gcs_path(target)
    if root_bucket != target_bucket:
      raise ValueError(
          'Recovery destination must use the same storage backend.'
      )
    expected_name = f'{name}-{operation_id}'
  else:
    source_url = parse.urlparse(str(root))
    target_url = parse.urlparse(str(target))
    if (source_url.scheme, source_url.netloc) != (
        target_url.scheme,
        target_url.netloc,
    ):
      raise ValueError(
          'Recovery destination must use the same storage backend.'
      )
    expected_name = f'{root.name}-{name}-{operation_id}'
    if root.parent not in target.parents or target.parent == root.parent:
      raise ValueError(
          'Recovery destination must be under the checkpoint parent.'
      )
  if target.name != expected_name or '..' in target.parts:
    raise ValueError('Invalid recovery destination name.')
  check_destination(target, root)


def remove(path: path_types.Path) -> None:
  """Uses the same physical deletion implementation as the v0 step deleter."""
  deleter.PathDeleter(path.parent).delete(path)


def delete_path(
    path: path_types.Path,
    target: path_types.Path | None = None,
    *,
    keep_markers_until_last: bool = False,
) -> None:
  """Deletes or relocates exactly the supplied path without replacing a target."""
  if target is not None:
    if not path.exists():
      if target.exists():
        return  # Resume a previously completed relocation.
      raise FileNotFoundError(
          f'Neither relocation source nor destination exists: {path}.'
      )
    if target.exists():
      raise FileExistsError(f'Deletion destination already exists: {target}.')
    deleter.PathDeleter(path.parent).delete(path, destination=target)
    return
  if keep_markers_until_last:
    # Directory rmtree may remove metadata before failing on a payload. Keep
    # identification files for retries, without depending on their contents.
    markers = {
        checkpoint_layout.ORBAX_CHECKPOINT_INDICATOR_FILE,
        checkpoint_layout.PYTREE_METADATA_FILE,
        metadata_serialization.checkpoint_metadata_file_path(path).name,
    }
    for child in path.iterdir():
      if child.name not in markers:
        if child.is_dir():
          remove(child)
        else:
          child.unlink()
  remove(path)
