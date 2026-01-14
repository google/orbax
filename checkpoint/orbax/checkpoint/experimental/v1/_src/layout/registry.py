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
from orbax.checkpoint.experimental.v1._src.layout import safetensors_layout
from orbax.checkpoint.experimental.v1._src.layout import v0_layout
from orbax.checkpoint.experimental.v1._src.path import types as path_types

InvalidLayoutError = checkpoint_layout.InvalidLayoutError
CheckpointLayout = checkpoint_layout.CheckpointLayout
CheckpointLayoutEnum = options_lib.CheckpointLayout


async def get_checkpoint_layout(
    path: path_types.PathLike, layout_enum: CheckpointLayoutEnum
) -> CheckpointLayout:
  """Returns the checkpoint layout class for the given path.

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

  match layout_enum:
    case CheckpointLayoutEnum.ORBAX:
      # This allows us to restore a v0 checkpoint with its own layout.
      if _is_v0_checkpoint(path):
        layout_class = v0_layout.V0Layout
      else:
        layout_class = orbax_layout.OrbaxLayout
    case CheckpointLayoutEnum.SAFETENSORS:
      layout_class = safetensors_layout.SafetensorsLayout
    case _:
      raise ValueError(f"Unsupported checkpoint layout: {layout_enum}")

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
  layout = await get_checkpoint_layout(path, layout_enum)
  layout, checkpointable_name = await _try_resolve_pytree_checkpointable(
      layout, checkpointable_name
  )
  await layout.validate_pytree(checkpointable_name)
  return layout, checkpointable_name


async def _try_resolve_pytree_checkpointable(
    layout: CheckpointLayout,
    checkpointable_name: str | None,
) -> tuple[CheckpointLayout, str | None]:
  """Tries to resolve the `PyTree` checkpointable name for a given layout.

  Args:
    layout: The :py:class:`.CheckpointLayout` object.
    checkpointable_name: An optional name for the `PyTree` checkpointable.

  Returns:
    A tuple containing the (potentially updated)
    :py:class:`.CheckpointLayout`
    and the
    resolved checkpointable name.

  Raises:
    ValueError: If it's a V0 checkpoint and a `PyTree` checkpointable name
      cannot be resolved.
  """
  # Selected a specific name; use it.
  if checkpointable_name is not None:
    return layout, checkpointable_name
  # Not a v0 checkpoint; use the default name.
  if not _is_v0_checkpoint(layout.path):
    if checkpointable_name is None:
      raise ValueError(
          "Cannot extract pytree from top-level V1 checkpoint directory."
      )
    return layout, checkpointable_name
  # If it's a V0 checkpoint, we can try to resolve the checkpointable name from
  # the path.
  if not isinstance(layout, (orbax_layout.OrbaxLayout, v0_layout.V0Layout)):
    raise AssertionError(
        f"Expected an OrbaxLayout or V0Layout, but got a {type(layout)}."
    )

  # If the path itself is a V0 PyTree checkpoint (flat structure), we can
  # "zoom out" to the parent directory and treat the current directory as the
  # checkpointable.
  try:
    original_path = layout.path
    new_layout = v0_layout.V0Layout(original_path.parent)
    await new_layout.validate_pytree(original_path.name)
    return new_layout, original_path.name
  except checkpoint_layout.InvalidLayoutError:
    pass

  # It may be a V0 checkpoint containing a PyTree checkpointable. It
  # is possible for there to be multiple, but this would be unusual, and it is
  # fine to just return the first one.
  dir_names = [p.name for p in layout.path.iterdir() if p.is_dir()]
  for name in dir_names:
    try:
      await layout.validate_pytree(name)
    except checkpoint_layout.InvalidLayoutError:
      continue
    return layout, name
  raise checkpoint_layout.InvalidLayoutError(
      f"Detected an Orbax V0 checkpoint at {layout.path}, but failed to resolve"
      " a checkpointable name for the `PyTree` checkpointable. Found"
      f" subdirectory names: {dir_names}."
  )


def _is_v0_checkpoint(path: path_types.PathLike) -> bool:
  ctx = context_lib.get_context()
  path = ctx.file_options.path_class(path)
  return not (path / orbax_layout.ORBAX_CHECKPOINT_INDICATOR_FILE).exists()
