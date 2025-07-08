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

from typing import Type
from etils import epath
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout
from orbax.checkpoint.experimental.v1._src.layout import orbax_layout
from orbax.checkpoint.experimental.v1._src.layout import safetensors_layout
from orbax.checkpoint.experimental.v1._src.path import types as path_types


def get_checkpoint_layout(
    path: path_types.PathLike,
) -> checkpoint_layout.CheckpointLayout:
  """Returns the checkpoint layout for the given path."""
  path = epath.Path(path)

  orbax_layout_obj = orbax_layout.OrbaxLayout()
  if orbax_layout_obj.validate(path):
    return orbax_layout.OrbaxLayout()

  safetensors_layout_obj = safetensors_layout.SafetensorsLayout()
  if safetensors_layout_obj.validate(path):
    return safetensors_layout.SafetensorsLayout()

  raise ValueError(f"Can't determine format from path:: {path}")


def get_registered_layouts() -> list[Type[checkpoint_layout.CheckpointLayout]]:
  """Returns a list of registered layouts."""
  return [
      orbax_layout.OrbaxLayout,
      safetensors_layout.SafetensorsLayout,
  ]
