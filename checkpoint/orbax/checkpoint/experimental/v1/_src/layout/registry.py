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

"""Registry for checkpoint layouts."""

from etils import epath
from orbax.checkpoint.experimental.v1._src.handlers import composite_handler
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout
from orbax.checkpoint.experimental.v1._src.layout import orbax_layout
from orbax.checkpoint.experimental.v1._src.layout import safetensors_layout
from orbax.checkpoint.experimental.v1._src.path import types as path_types

InvalidLayoutError = checkpoint_layout.InvalidLayoutError


def get_checkpoint_layout(
    path: path_types.PathLike,
) -> checkpoint_layout.CheckpointLayout:
  """Returns the checkpoint layout class for the given path.

  Args:
    path: The path to the checkpoint.

  Returns:
    The class of the matching CheckpointLayout.

  Raises:
    InvalidLayoutError: If the path is not a valid checkpoint for any registered
    layout, with details from each layout's validation attempt.
  """
  path = epath.Path(path)
  error_messages = {}

  try:
    orbax_layout.OrbaxLayout().validate(path)
    return orbax_layout.OrbaxLayout(path)
  except InvalidLayoutError as e:
    error_messages["orbax"] = e

  try:
    safetensors_layout.SafetensorsLayout().validate(path)
    return safetensors_layout.SafetensorsLayout(path)
  except InvalidLayoutError as e:
    error_messages["safetensors"] = e

  raise InvalidLayoutError(
      f"Could not recognize the checkpoint at {path} as a valid checkpoint."
      " Encountered the following error when interpreting as an Orbax"
      f" checkpoint: {error_messages['orbax']} If you are attempting to load an"
      " Orbax checkpoint, please verify that the path points to a valid Orbax"
      " checkpoint with `ocp.is_orbax_checkpoint(path)`. It may also be"
      " helpful to look for the presence of the"
      f" {composite_handler.ORBAX_CHECKPOINT_INDICATOR_FILE} file."
      "Alternatively attempting to interpret as a Safetensors checkpoint failed"
      f" with message: {error_messages['safetensors']}"
  )
