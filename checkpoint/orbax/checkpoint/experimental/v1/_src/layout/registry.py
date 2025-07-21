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
from orbax.checkpoint.experimental.v1._src.context import options as options_lib
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout
from orbax.checkpoint.experimental.v1._src.layout import orbax_layout
from orbax.checkpoint.experimental.v1._src.layout import safetensors_layout
from orbax.checkpoint.experimental.v1._src.path import types as path_types

InvalidLayoutError = checkpoint_layout.InvalidLayoutError


def get_checkpoint_layout(
    path: path_types.PathLike, layout_enum: options_lib.CheckpointLayout
) -> checkpoint_layout.CheckpointLayout:
  """Returns the checkpoint layout class for the given path.

  Args:
    path: The path to the checkpoint.
    layout_enum: The checkpoint layout to use.

  Returns:
    The class of the matching CheckpointLayout.

  Raises:
    InvalidLayoutError: If the path is not a valid checkpoint for any registered
    layout, with details from each layout's validation attempt.
  """
  path = epath.Path(path)

  match layout_enum:
    case options_lib.CheckpointLayout.ORBAX:
      layout_class = orbax_layout.OrbaxLayout
    case options_lib.CheckpointLayout.SAFETENSORS:
      layout_class = safetensors_layout.SafetensorsLayout
    case _:
      raise ValueError(f"Unsupported checkpoint layout: {layout_enum}")

  try:
    layout = layout_class(path)
    layout.validate()
    return layout
  except InvalidLayoutError as e:
    raise InvalidLayoutError(
        f"Could not recognize the checkpoint at {path} as a valid"
        f" {layout_enum.value} checkpoint. If you are trying to load a"
        " checkpoint that does not conform to the stadnard Orbax format, use"
        " `ocp.Context(layout=...)` to specify the expected checkpoint layout."
    ) from e
