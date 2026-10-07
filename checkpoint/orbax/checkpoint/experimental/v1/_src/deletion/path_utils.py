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

import asyncio
import os
from urllib import parse

from orbax.checkpoint._src.path import async_path
from orbax.checkpoint._src.path import deleter
from orbax.checkpoint._src.path import gcs_utils
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout
from orbax.checkpoint.experimental.v1._src.metadata import serialization as metadata_serialization
from orbax.checkpoint.experimental.v1._src.path import types as path_types


async def check_path(path: path_types.Path) -> None:
  """Rejects storage roots and local symlink targets."""
  parsed = parse.urlparse(str(path))
  if not path.name or not parsed.path.strip('/') or str(path) in ('.', '..'):
    raise ValueError(f'Cannot delete a storage root: {path}.')
  if not parsed.scheme and await async_path.is_link(path):
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


async def check_destination(
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
  root, resolved = await asyncio.gather(
      asyncio.to_thread(os.path.realpath, str(checkpoint_root)),
      asyncio.to_thread(os.path.realpath, str(target)),
  )
  if resolved == root or resolved.startswith(root + os.sep):
    raise ValueError(
        f'The deletion destination resolves inside the checkpoint: {target}.'
    )


async def destination(
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
    # Keep the source's prefix (gs:// or a mount point) so the destination is
    # reached through the same filesystem as the checkpoint being moved.
    parent = context.file_options.path_class(
        f'{gcs_utils.gcs_bucket_root(path)}/{trash}'
    )
    target = parent / f'{path.name}-{operation_id}'
    await check_destination(target, root)
    return target
  if trash is not None:
    raise NotImplementedError('todelete_full_path requires a GCS checkpoint.')
  return None


async def check_recovery_destination(
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
    root_bucket, _ = gcs_utils.split_gcs_path(root)
    target_bucket, _ = gcs_utils.split_gcs_path(target)
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
  await check_destination(target, root)


async def remove(path: path_types.Path) -> None:
  """Uses the same physical deletion implementation as the v0 step deleter."""
  await asyncio.to_thread(deleter.PathDeleter(path.parent).delete, path)


async def delete_path(
    path: path_types.Path,
    target: path_types.Path | None = None,
    *,
    keep_markers_until_last: bool = False,
) -> None:
  """Deletes or relocates exactly the supplied path without replacing a target."""
  if target is not None:
    if not await async_path.exists(path):
      if await async_path.exists(target):
        return  # Resume a previously completed relocation.
      raise FileNotFoundError(
          f'Neither relocation source nor destination exists: {path}.'
      )
    if await async_path.exists(target):
      raise FileExistsError(f'Deletion destination already exists: {target}.')
    await asyncio.to_thread(
        deleter.PathDeleter(path.parent).delete, path, destination=target
    )
    return
  if keep_markers_until_last:
    # Directory rmtree may remove metadata before failing on a payload. Keep
    # identification files for retries, without depending on their contents.
    markers = {
        checkpoint_layout.ORBAX_CHECKPOINT_INDICATOR_FILE,
        checkpoint_layout.PYTREE_METADATA_FILE,
        metadata_serialization.CHECKPOINT_METADATA_FILENAME,
    }
    children = await asyncio.to_thread(lambda: list(path.iterdir()))
    for child in children:
      if child.name not in markers:
        if await async_path.is_dir(child):
          await remove(child)
        else:
          await async_path.unlink(child)
  await remove(path)
