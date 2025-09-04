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

"""Utilities for parsing and validating checkpoint paths.

TODO(b/396190818): The validation functions in this module deserve to be
refactored to allow for greater flexibility, less specifically tied to one
implementation. At the moment they are tailored specifically to `save_pytree` /
`save_checkpointables`.
"""

import itertools

from absl import logging
from etils import epath
from orbax.checkpoint._src.metadata import checkpoint as checkpoint_metadata
from orbax.checkpoint._src.path import format_utils
from orbax.checkpoint._src.path import step as v0_step_lib
from orbax.checkpoint.experimental.v1._src.path import types as path_types


PYTREE_CHECKPOINTABLE_KEY = 'pytree'

METRICS_CHECKPOINTABLE_KEY = 'metrics'

RESERVED_CHECKPOINTABLE_KEYS = frozenset({
    METRICS_CHECKPOINTABLE_KEY,
})


def subdirs(directory: path_types.Path, *, limit: int = 3) -> list[str]:
  return list(
      itertools.islice(
          (subdir.name for subdir in directory.iterdir() if subdir.is_dir()),
          limit,
      )
  )


def validate_checkpoint_directory(path: path_types.PathLike):
  """Validates a checkpoint directory.

  Args:
    path: The path to the checkpoint directory.

  Raises:
    FileNotFoundError: If the path does not exist.
    NotADirectoryError: If the path is not a directory.
    ValueError: If the checkpoint is incomplete.
  """
  path = epath.Path(path)
  if not path.exists():
    raise FileNotFoundError(f'Checkpoint path {path} does not exist.')
  if not path.is_dir():
    raise NotADirectoryError(f'Checkpoint path {path} is not a directory.')
  if v0_step_lib.is_path_temporary(path):
    raise ValueError(f'Found incomplete checkpoint at {path}.')


def validate_checkpoint_metadata(path: path_types.PathLike):
  """Validates the checkpoint-level metadata (_CHECKPOINT_METADATA)."""
  metadata_store = checkpoint_metadata.metadata_store(enable_write=False)
  # Path points to a single step checkpoint with valid metadata.
  checkpoint_metadata_path = checkpoint_metadata.step_metadata_file_path(path)
  if not checkpoint_metadata_path.exists():
    raise FileNotFoundError(
        f'Checkpoint path {path} does not contain a valid metadata file:'
        f' {checkpoint_metadata_path.name}. Please ensure the path specified'
        ' for loading points to a valid Orbax checkpoint, indicated by the'
        ' presence of the metadata file.'
    )
  # TODO(niketkb): This check can be removed because the caller will anyway read
  # the metadata file. We are reading the file twice.
  if metadata_store.read(checkpoint_metadata_path) is None:
    raise ValueError(
        f'Failed to read valid metadata for checkpoint path {path}.'
    )


def validate_pytree_checkpoint(
    path: path_types.PathLike,
    *,
    checkpointable_name: str | None = PYTREE_CHECKPOINTABLE_KEY,
):
  """Validates a checkpoint path written by `ocp.save_pytree`.

  Args:
    path: The path to the checkpoint directory.
    checkpointable_name: The name of the checkpointable to load. A subdirectory
      with this name must exist in `directory`. If None then `directory` is
      expected to contain the checkpoint directly. Defaults to `pytree`.

  Raises:
    FileNotFoundError: If the path does not exist, or if `pytree` is not found
      in the directory
    ValueError: If the PyTree checkpoint is malformed.
  """
  path = epath.Path(path)
  pytree_dir = (
      path if checkpointable_name is None else path / checkpointable_name
  )
  if checkpointable_name is not None and not pytree_dir.exists():
    raise FileNotFoundError(
        f'Checkpoint path {path} must contain a subdirectory named'
        f' "{checkpointable_name}". Found subdirectories: {subdirs(path)}.'
        ' Please try inspecting the checkpointable metadata using'
        ' `ocp.checkpointables_metadata()` or try loading the checkpoint using'
        ' `ocp.load_checkpointables()`.'
    )
  if not format_utils._has_pytree_metadata_file(  # pylint: disable=protected-access
      pytree_dir
  ):
    # TODO(niketkb): Add following details to the error message:
    # 1. we should check other available subdirectories and see if any of them
    #   look like PyTree checkpoints, and instruct the user to consider
    #   whether they meant to specify any of those.
    # 2. we need to check the directory - if it contains PyTree files, suggest
    #   loading with checkpointable_name=None
    raise FileNotFoundError(
        f'Checkpoint path {path} does not contain a PyTree metadata file.'
    )
  if not format_utils._has_tensorstore_data_files(  # pylint: disable=protected-access
      pytree_dir
  ):
    logging.warning(
        'TensorStore data files not found in checkpoint path %s. This may be a'
        ' sign of a malformed checkpoint, unless your checkpoint consists'
        ' entirely of strings or other non-standard PyTree leaves.',
        path,
    )
