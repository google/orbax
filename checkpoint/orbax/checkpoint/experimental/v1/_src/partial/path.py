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

"""Utility functions for partial saving paths."""

from etils import epath
from orbax.checkpoint.experimental.v1._src.path import types as path_types

PARTIAL_SAVE_SUFFIX = '.partial_save'


def is_partial_save_path(
    path: path_types.PathLike, allow_tmp_dir: bool = False
) -> bool:
  path = epath.Path(path)
  if allow_tmp_dir:
    return PARTIAL_SAVE_SUFFIX in path.name
  else:
    return path.name.endswith(PARTIAL_SAVE_SUFFIX)


def add_partial_save_suffix(path: path_types.PathLike) -> path_types.Path:
  path = epath.Path(path)
  return path.parent / (path.name + PARTIAL_SAVE_SUFFIX)


def remove_partial_save_suffix(path: path_types.PathLike) -> path_types.Path:
  path = epath.Path(path)
  return path.parent / path.name.removesuffix(PARTIAL_SAVE_SUFFIX)
