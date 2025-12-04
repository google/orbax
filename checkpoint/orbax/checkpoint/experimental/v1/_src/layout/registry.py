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

from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.context import options as options_lib
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout
from orbax.checkpoint.experimental.v1._src.layout import orbax_layout
from orbax.checkpoint.experimental.v1._src.layout import orbax_v0_layout
from orbax.checkpoint.experimental.v1._src.layout import safetensors_layout
from orbax.checkpoint.experimental.v1._src.path import types as path_types

InvalidLayoutError = checkpoint_layout.InvalidLayoutError
CheckpointLayout = checkpoint_layout.CheckpointLayout
CheckpointLayoutEnum = options_lib.CheckpointLayout


async def _get_checkpoint_layout_class(
    path: path_types.PathLike,
    layout_enum: CheckpointLayoutEnum,
) -> type[CheckpointLayout]:
  """Returns the checkpoint layout class for the given layout enum."""
  match layout_enum:
    case CheckpointLayoutEnum.ORBAX:
      if (
          await orbax_layout.checkpoint_version(path)
          == orbax_layout.CheckpointVersion.V0
      ):
        return orbax_v0_layout.OrbaxV0Layout
      else:
        return orbax_layout.OrbaxLayout
    case CheckpointLayoutEnum.SAFETENSORS:
      return safetensors_layout.SafetensorsLayout
    case _:
      raise ValueError(f"Unsupported checkpoint layout: {layout_enum}")


async def get_checkpoint_layout(
    path: path_types.PathLike, layout_enum: CheckpointLayoutEnum
) -> CheckpointLayout:
  """Returns the checkpoint layout class for the given path and validates it.

  Args:
    path: The path to the checkpoint.
    layout_enum: The checkpoint layout to use.

  Returns:
    The class of the matching :py:class:`.CheckpointLayout`.

  Raises:
    InvalidLayoutError: If the path is not a valid checkpoint for any registered
    layout, with details from each layout's validation attempt.
  """
  ctx = context_lib.get_context()
  path = ctx.file_options.path_class(path)

  layout_class = await _get_checkpoint_layout_class(path, layout_enum)

  try:
    layout = layout_class(path)
    await layout.validate()
    return layout
  except InvalidLayoutError as e:
    raise InvalidLayoutError(
        f"Could not recognize the checkpoint at {path} as a valid"
        f" {layout_enum.value} checkpoint. If you are trying to load a"
        " checkpoint that does not conform to the standard Orbax format, use"
        " `ocp.Context(layout=...)` to specify the expected checkpoint layout."
    ) from e


async def get_checkpoint_layout_pytree(
    path: path_types.PathLike,
    layout_enum: CheckpointLayoutEnum,
    checkpointable_name: str | None = None,
) -> tuple[checkpoint_layout.CheckpointLayout, str | None]:
  """Returns the checkpoint layout and checkpointable name for the given path."""
  ctx = context_lib.get_context()
  path = ctx.file_options.path_class(path)

  layout = await get_checkpoint_layout(path, layout_enum)
  await layout.validate_pytree(checkpointable_name)
  return layout, checkpointable_name


async def validate_pytree_checkpoint_path(
    path: path_types.PathLike, layout_enum: CheckpointLayoutEnum
):
  layout_class = await _get_checkpoint_layout_class(path, layout_enum)
  if layout_class == orbax_layout.OrbaxLayout:
    raise AssertionError(
        "A V1 checkpoint was saved and user is attempting to load"
        f" {path} directly as a PyTree checkpointable, this is only"
        " supported for V0 checkpoints."
    )
