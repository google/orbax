# Copyright 2024 The Orbax Authors.
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

from absl import logging
from etils import epath
from orbax.checkpoint._src.metadata import checkpoint as checkpoint_metadata
from orbax.checkpoint._src.path import format_utils
from orbax.checkpoint.experimental.v1._src.path import types as path_types

PYTREE_CHECKPOINTABLE_KEY = 'pytree'

RESERVED_CHECKPOINTABLE_KEYS = frozenset({
})


def validate_pytree_checkpoint(path: path_types.PathLike):
  """Validates a checkpoint path written by `ocp.save_pytree`.

  Args:
    path: The path to the checkpoint directory.

  Raises:
    FileNotFoundError: If the path does not exist, or if `pytree` is not found
      in the directory
    NotADirectoryError: If the path is not a directory.
    ValueError: If the PyTree checkpoint is malformed or metadata cannot be
      read.
  """
  path = epath.Path(path)
  if not path.exists():
    raise FileNotFoundError(f'Checkpoint path {path} does not exist.')
  if not path.is_dir():
    raise NotADirectoryError(f'Checkpoint path {path} is not a directory.')
  metadata_store = checkpoint_metadata.metadata_store(enable_write=False)
  # Path points to a single step checkpoint with valid metadata.
  checkpoint_metadata_path = checkpoint_metadata.step_metadata_file_path(path)
  if not checkpoint_metadata_path.exists():
    raise FileNotFoundError(
        f'Checkpoint path {path} does not contain a valid metadata file.'
    )
  if metadata_store.read(checkpoint_metadata_path) is None:
    raise ValueError(
        f'Failed to read valid metadata for checkpoint path {path}.'
    )
  if not (path / PYTREE_CHECKPOINTABLE_KEY).exists():
    raise FileNotFoundError(
        f'Checkpoint path {path} does not contain a PyTree checkpointable'
        f' (called "{PYTREE_CHECKPOINTABLE_KEY}").'
    )
  if not format_utils._has_pytree_metadata_file(  # pylint: disable=protected-access
      path / PYTREE_CHECKPOINTABLE_KEY
  ):
    raise FileNotFoundError(
        f'Checkpoint path {path} does not contain a PyTree metadata file.'
    )
  if not format_utils._has_tensorstore_data_files(  # pylint: disable=protected-access
      path / PYTREE_CHECKPOINTABLE_KEY
  ):
    logging.warning(
        'TensorStore data files not found in checkpoint path %s. This may be a'
        ' sign of a malformed checkpoint, unless your checkpoint consists'
        ' entirely of strings or other non-standard PyTree leaves.',
        path,
    )
