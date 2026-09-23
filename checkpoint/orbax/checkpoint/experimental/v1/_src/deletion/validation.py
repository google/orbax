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

""""Validates Orbax deletion targets without loading checkpoint payloads."""

from typing import Any

from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.context import options
from orbax.checkpoint.experimental.v1._src.deletion import metadata
from orbax.checkpoint.experimental.v1._src.deletion import path_utils
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout
from orbax.checkpoint.experimental.v1._src.metadata import serialization as metadata_serialization
from orbax.checkpoint.experimental.v1._src.path import types as path_types

InvalidLayoutError = checkpoint_layout.InvalidLayoutError


def validate_arguments(
    checkpointable_name: str | None, missing_ok: bool
) -> None:
  if checkpointable_name is not None:
    path_utils.validate_checkpointable_name(checkpointable_name)
  if not isinstance(missing_ok, bool):
    raise TypeError('missing_ok must be a bool.')


def validate_step(step: int) -> None:
  if not isinstance(step, int) or isinstance(step, bool):
    raise TypeError('step must be an explicit Python int.')
  if step < 0:
    raise ValueError('step must be nonnegative.')


def validate_root(
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
  if not path.is_dir():
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
  if any(
      (path.parent / name).exists()
      for name in (
          checkpoint_layout.ORBAX_CHECKPOINT_INDICATOR_FILE,
          metadata_serialization.checkpoint_metadata_file_path(path).name,
          metadata.RECORD_FILENAME,
      )
  ):
    raise InvalidLayoutError(
        f'{path} is a checkpointable directory. Delete from its checkpoint root'
        ' using checkpointable_name instead.'
    )
  entries = list(path.iterdir())
  if not entries:
    return
  if not any(
      (path / name).is_file()
      for name in (
          checkpoint_layout.ORBAX_CHECKPOINT_INDICATOR_FILE,
          metadata_serialization.checkpoint_metadata_file_path(path).name,
          checkpoint_layout.PYTREE_METADATA_FILE,
      )
  ):
    raise InvalidLayoutError(f'Not an Orbax checkpoint root: {path}.')


def prepare_item(path: path_types.Path, name: str) -> dict[str, Any]:
  """Prepares a partial metadata update for a self-contained composite item."""
  metadata.check_atomic_write_support(path)
  original = metadata.read_json(
      metadata_serialization.checkpoint_metadata_file_path(path)
  )
  handlers = original.get('item_handlers')
  if not isinstance(handlers, dict):
    raise InvalidLayoutError(
        f'Checkpoint at {path} has no named checkpointables.'
    )
  item = path / name
  path_utils.check_path(item)
  if name not in handlers or not item.is_dir():
    raise FileNotFoundError(
        f'Checkpointable {name!r} does not exist at {path}.'
    )
  remaining = [
      key
      for key in handlers
      if key != name
      and key not in checkpoint_layout.RESERVED_CHECKPOINTABLE_KEYS
      and (path / key).is_dir()
  ]
  if not remaining:
    raise ValueError(
        'Cannot delete the final checkpointable; delete the whole checkpoint'
        ' instead.'
    )
  return original
