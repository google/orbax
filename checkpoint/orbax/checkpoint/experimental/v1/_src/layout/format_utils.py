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

"""Utilities for validating checkpoint formats."""

from etils import epath
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout
from orbax.checkpoint.experimental.v1._src.layout import orbax_layout
from orbax.checkpoint.experimental.v1._src.path import types as path_types

InvalidLayoutError = checkpoint_layout.InvalidLayoutError


def is_orbax_checkpoint(path: path_types.PathLike) -> bool:
  """Determines if the given path is an Orbax checkpoint.

  Args:
    path: The path to the checkpoint directory.

  Returns:
    True if the path is an Orbax checkpoint, False otherwise.
  """
  path = epath.Path(path)
  try:
    orbax_layout.OrbaxLayout(path).validate()
    return True
  except InvalidLayoutError:
    return False
