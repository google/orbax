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

"""Registry for checkpoint layouts."""

import asyncio
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.context import options as options_lib
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout
from orbax.checkpoint.experimental.v1._src.layout import numpy_layout
from orbax.checkpoint.experimental.v1._src.layout import orbax_layout
from orbax.checkpoint.experimental.v1._src.layout import orbax_v0_layout
from orbax.checkpoint.experimental.v1._src.layout import pytorch_layout
from orbax.checkpoint.experimental.v1._src.layout import safetensors_layout
from orbax.checkpoint.experimental.v1._src.path import types as path_types

InvalidLayoutError = checkpoint_layout.InvalidLayoutError
CheckpointLayout = checkpoint_layout.CheckpointLayout
CheckpointLayoutEnum = options_lib.CheckpointLayout

ORBAX_LAYOUT_CLASSES = [
    orbax_layout.OrbaxLayout,
    orbax_v0_layout.OrbaxV0Layout,
]


async def _is_orbax_checkpoint_async(path: path_types.PathLike) -> bool:
  ctx = context_lib.get_context()
  path = ctx.file_options.path_class(path)

  tasks = []
  for layout_cls in ORBAX_LAYOUT_CLASSES:
    tasks.append(layout_cls().validate(path))

  results = await asyncio.gather(*tasks, return_exceptions=True)
  return any(not isinstance(r, Exception) for r in results)


def is_orbax_checkpoint(path: path_types.PathLike) -> bool:
  """Returns True if the path is an Orbax checkpoint."""
  return asyncio.run(_is_orbax_checkpoint_async(path))


async def get_layout_class(
    layout_enum: CheckpointLayoutEnum, path: path_types.PathLike | None = None
) -> type[CheckpointLayout]:
  """Returns the layout class for the given layout enum."""
  match layout_enum:
    case CheckpointLayoutEnum.ORBAX:
      if path is None or (
          await orbax_layout.checkpoint_version(path)
          == orbax_layout.CheckpointVersion.V1
      ):
        return orbax_layout.OrbaxLayout
      else:
        return orbax_v0_layout.OrbaxV0Layout
    case CheckpointLayoutEnum.SAFETENSORS:
      return safetensors_layout.SafetensorsLayout
    case CheckpointLayoutEnum.NUMPY:
      return numpy_layout.NumpyLayout
    case CheckpointLayoutEnum.PYTORCH:
      return pytorch_layout.PyTorchLayout
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

  layout_class = await get_layout_class(layout_enum, path)

  try:
    layout = layout_class()
    await layout.validate(path)
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
) -> CheckpointLayout:
  """Validates pytree checkpoint and returns the layout for the given path."""
  ctx = context_lib.get_context()
  path = ctx.file_options.path_class(path)

  layout = await get_checkpoint_layout(path, layout_enum)
  await layout.validate_pytree(path, checkpointable_name)
  return layout
