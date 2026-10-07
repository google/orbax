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

"""Validates Orbax deletion targets without loading checkpoint payloads."""

import asyncio
from typing import Any

from orbax.checkpoint._src.path import async_path
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.context import options
from orbax.checkpoint.experimental.v1._src.deletion import path_utils
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout
from orbax.checkpoint.experimental.v1._src.metadata import serialization as metadata_serialization
from orbax.checkpoint.experimental.v1._src.path import types as path_types

InvalidLayoutError = checkpoint_layout.InvalidLayoutError


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


def validate_arguments(
    checkpointable_name: str | None, missing_ok: bool
) -> None:
  if checkpointable_name is not None:
    validate_checkpointable_name(checkpointable_name)
  if not isinstance(missing_ok, bool):
    raise TypeError('missing_ok must be a bool.')


def validate_step(step: int) -> None:
  """Rejects booleans explicitly because bool is a subclass of int."""
  if not isinstance(step, int) or isinstance(step, bool):
    raise TypeError('step must be an explicit Python int.')
  if step < 0:
    raise ValueError('step must be nonnegative.')


async def validate_root(
    path: path_types.Path,
    *,
    context: context_lib.Context,
    managed: bool = False,
) -> None:
  """Validates the deletion boundary, including incomplete known step paths.

  A manager already resolves a step within its run. Free functions additionally
  require a format marker (or an empty root left by interrupted cleanup).
  Metadata contents need not be readable to delete the whole checkpoint.

  Args:
    path: The path to validate.
    context: The current context.
    managed: Whether the deletion is managed by a Checkpointer.

  Raises:
    NotADirectoryError: If the path is not a directory.
    NotImplementedError: If the checkpoint layout is not supported.
    InvalidLayoutError: If the path is not a valid Orbax checkpoint.
  """
  if not await async_path.is_dir(path):
    raise NotADirectoryError(f'Checkpoint path is not a directory: {path}.')
  if context.checkpoint_layout not in (
      options.CheckpointLayout.ORBAX,
      options.CheckpointLayout.AUTO,
  ):
    raise NotImplementedError(
        'Deletion currently supports Orbax checkpoint layouts.'
    )
  if managed:
    return
  parent_markers = (
      checkpoint_layout.ORBAX_CHECKPOINT_INDICATOR_FILE,
      metadata_serialization.CHECKPOINT_METADATA_FILENAME,
      checkpoint_layout.CHECKPOINT_DELETION_KEY,
  )
  parent_exists = await asyncio.gather(
      *(async_path.exists(path.parent / name) for name in parent_markers)
  )
  if any(parent_exists):
    raise InvalidLayoutError(
        f'{path} is a checkpointable directory. Delete from its checkpoint root'
        ' using checkpointable_name instead.'
    )
  entries = await asyncio.to_thread(lambda: list(path.iterdir()))
  if not entries:
    return
  markers = (
      checkpoint_layout.ORBAX_CHECKPOINT_INDICATOR_FILE,
      metadata_serialization.CHECKPOINT_METADATA_FILENAME,
      checkpoint_layout.PYTREE_METADATA_FILE,
  )
  marker_exists = await asyncio.gather(
      *(async_path.is_file(path / name) for name in markers)
  )
  if not any(marker_exists):
    raise InvalidLayoutError(f'Not an Orbax checkpoint root: {path}.')


async def prepare_item(path: path_types.Path, name: str) -> dict[str, Any]:
  """Prepares a partial metadata update for a self-contained composite item."""
  # Metadata uses the pure name validator above; defer this reverse dependency.
  from orbax.checkpoint.experimental.v1._src.deletion import metadata  # pylint: disable=g-import-not-at-top # pyrefly: ignore[missing-module-attribute]

  metadata.check_atomic_write_support(path)
  original = await metadata.read_checkpoint_metadata(path)
  handlers = original.get('item_handlers')
  if not isinstance(handlers, dict):
    raise InvalidLayoutError(
        f'Checkpoint at {path} has no named checkpointables.'
    )
  item = path / name
  await path_utils.check_path(item)
  if name not in handlers or not await async_path.is_dir(item):
    raise FileNotFoundError(
        f'Checkpointable {name!r} does not exist at {path}.'
    )
  remaining_names = [
      key
      for key in handlers
      if key != name
      and key not in checkpoint_layout.RESERVED_CHECKPOINTABLE_KEYS
  ]
  remaining = await asyncio.gather(
      *(async_path.is_dir(path / key) for key in remaining_names)
  )
  if not any(remaining):
    raise ValueError(
        'Cannot delete the final checkpointable; delete the whole checkpoint'
        ' instead.'
    )
  return original
