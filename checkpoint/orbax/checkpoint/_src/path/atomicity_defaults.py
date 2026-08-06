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

"""Utilities for interacting with the atomicity module.

Mainly provides helper functions like `get_default_temporary_path_class` that
help interacting with the `atomicity` module, but can have larger dependencies
than we would want to introduce into the base `atomicity` module.
"""

from absl import logging
from etils import epath
from orbax.checkpoint import options as options_lib
from orbax.checkpoint._src.path import atomicity
from orbax.checkpoint._src.path import atomicity_types
from orbax.checkpoint._src.path import gcs_utils
from orbax.checkpoint._src.path import utils


_ATOMICITY_MODE_TO_PATH_CLASS: dict[
    options_lib.AtomicityMode, type[atomicity_types.TemporaryPath]
] = {
    options_lib.AtomicityMode.COMMIT_FILE: atomicity.CommitFileTemporaryPath,
    options_lib.AtomicityMode.ATOMIC_RENAME: (
        atomicity.AtomicRenameTemporaryPath
    ),
}


def _resolve_temporary_path_class_from_options(
    atomicity_options: options_lib.AtomicityOptions | None,
) -> type[atomicity_types.TemporaryPath] | None:
  """Resolves explicit temporary path class from atomicity_options.

  Args:
    atomicity_options: Optional AtomicityOptions specifying atomicity mode.

  Returns:
    Concrete TemporaryPath class if mode is explicit, or None if mode is AUTO.

  Raises:
    ValueError: If atomicity_mode is not supported.
  """
  if (
      atomicity_options is None
      or atomicity_options.mode == options_lib.AtomicityMode.AUTO
  ):
    return None
  cls = _ATOMICITY_MODE_TO_PATH_CLASS.get(atomicity_options.mode)
  if cls is not None:
    return cls
  raise ValueError(f'Unsupported atomicity_mode: {atomicity_options.mode}.')


def get_item_default_temporary_path_class(
    path: epath.Path,
    *,
    atomicity_options: options_lib.AtomicityOptions | None = None,
) -> type[atomicity_types.TemporaryPath]:
  """Returns the default temporary path class for a given sub-item path.

  Args:
    path: The sub-item path.
    atomicity_options: Optional atomicity options configuration.

  Returns:
    The default TemporaryPath class to use.
  """
  path_cls = _resolve_temporary_path_class_from_options(atomicity_options)
  if path_cls is not None:
    if (
        gcs_utils.is_gcs_path(path)
        and path_cls == atomicity.AtomicRenameTemporaryPath
    ):
      logging.warning(
          'AtomicRenameTemporaryPath can cause major performance issues for GCS'
          ' paths. '
      )

    return path_cls
  if gcs_utils.is_gcs_path(path):
    return atomicity.CommitFileTemporaryPath
  else:
    return atomicity.AtomicRenameTemporaryPath


def get_default_temporary_path_class(
    path: epath.Path,
    *,
    atomicity_options: options_lib.AtomicityOptions | None = None,
) -> type[atomicity_types.TemporaryPath]:
  """Returns the default temporary path class for a given checkpoint path.

  Args:
    path: The root checkpoint directory path.
    atomicity_options: Optional atomicity options configuration.

  Returns:
    The default TemporaryPath class for the given path.
  """
  path_cls = _resolve_temporary_path_class_from_options(atomicity_options)
  if path_cls is not None:
    if (
        gcs_utils.is_gcs_path(path)
        and path_cls == atomicity.AtomicRenameTemporaryPath
    ):
      logging.warning(
          'AtomicRenameTemporaryPath can cause major performance issues for GCS'
          ' paths. '
      )

    return path_cls
  if gcs_utils.is_gcs_path(path):
    return atomicity.CommitFileTemporaryPath
  else:
    return atomicity.AtomicRenameTemporaryPath
