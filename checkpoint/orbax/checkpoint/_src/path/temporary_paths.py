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

"""Utilities for working with TemporaryPath objects.

These are specifically oriented toward identifying and manipulating temporary
or finalized paths.
"""

import asyncio
from typing import Iterable, Type

from absl import logging
from etils import epath
from orbax.checkpoint import options as options_lib
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.path import async_path
from orbax.checkpoint._src.path import atomicity_defaults
from orbax.checkpoint._src.path import atomicity_types


ValidationError = atomicity_types.ValidationError

TMP_DIR_SUFFIX = atomicity_types.TMP_DIR_SUFFIX


async def is_path_temporary(
    path: epath.PathLike,
    *,
    temporary_path_cls: Type[atomicity_types.TemporaryPath] | None = None,
) -> bool:
  """Determines if the given path represents a temporary checkpoint.

  Path takes the form::

    path/to/my/dir/<name>.orbax-checkpoint-tmp-<timestamp>/  # not finalized
    path/to/my/dir/<name>/  # finalized

  Alternatively::

    gs://path/to/my/dir/<name>/  # finalized
      commit_success.txt
      ...
    gs://<path/to/my/dir/<name>/  # not finalized
      ...

  Args:
    path: Directory.
    temporary_path_cls: The TemporaryPath class to use for validation.

  Returns:
    True if the checkpoint is a recognized temporary checkpoint.

  Raises:
    Validation error if the provided path cannot be recognized.
  """
  path = epath.Path(path)
  temporary_path_cls = (
      temporary_path_cls
      or atomicity_defaults.get_default_temporary_path_class(path)
  )
  try:
    await temporary_path_cls.validate(path)
    return True
  except ValidationError as e:
    logging.warning(
        'Path %s could not be identified as a temporary checkpoint path using'
        ' %s. Got error: %s',
        path,
        temporary_path_cls,
        e,
    )
    return False


async def is_path_finalized(
    path: epath.PathLike,
    *,
    temporary_path_cls: Type[atomicity_types.TemporaryPath] | None = None,
) -> bool:
  """Determines if the given path represents a finalized checkpoint.

  Path takes the form::

    path/to/my/dir/<name>.orbax-checkpoint-tmp-<timestamp>/  # not finalized
    path/to/my/dir/<name>/  # finalized

  Alternatively::

    gs://path/to/my/dir/<name>/  # finalized
      commit_success.txt
      ...
    gs://<path/to/my/dir/<name>/  # not finalized
      ...

  Args:
    path: Directory.
    temporary_path_cls: The TemporaryPath class to use for validation.

  Returns:
    True if the checkpoint is finalized.

  Raises:
    Validation error if the provided path cannot be recognized.
  """
  path = epath.Path(path)
  temporary_path_cls = (
      temporary_path_cls
      or atomicity_defaults.get_default_temporary_path_class(path)
  )
  try:
    await temporary_path_cls.validate_final(path)
    return True
  except ValidationError as e:
    logging.warning(
        'Path %s could not be identified as a finalized checkpoint path using'
        ' %s. Got error: %s',
        path,
        temporary_path_cls,
        e,
    )
    return False


async def all_temporary_paths(
    root_directory: epath.PathLike,
    *,
    temporary_path_cls: Type[atomicity_types.TemporaryPath] | None = None,
) -> Iterable[atomicity_types.TemporaryPath]:
  """Returns a list of tmp checkpoint dir names in `root_directory`."""
  root_directory = epath.Path(root_directory)
  if not await async_path.exists(root_directory):
    return []

  def _build_temporary_path(path: epath.Path) -> atomicity_types.TemporaryPath:
    path_cls = (
        temporary_path_cls
        or atomicity_defaults.get_default_temporary_path_class(path)
    )
    return path_cls.from_temporary(path)

  return [
      _build_temporary_path(p)
      for p in await async_path.iterdir(root_directory)
      if await is_path_temporary(p, temporary_path_cls=temporary_path_cls)
  ]


async def cleanup_temporary_paths(
    directory: epath.PathLike,
    *,
    multiprocessing_options: options_lib.MultiprocessingOptions | None = None,
    temporary_path_cls: Type[atomicity_types.TemporaryPath] | None = None,
):
  """Cleanup steps in `directory` with tmp files, as these are not finalized."""
  directory = epath.Path(directory)
  multiprocessing_options = (
      multiprocessing_options or options_lib.MultiprocessingOptions()
  )
  logging.info('Cleaning up existing temporary directories at %s.', directory)
  if multihost.is_primary_host(multiprocessing_options.primary_host):
    tmp_paths = await all_temporary_paths(
        directory, temporary_path_cls=temporary_path_cls
    )
    await asyncio.gather(
        *[async_path.rmtree(tmp_path.get()) for tmp_path in tmp_paths]
    )
  multihost.sync_global_processes(
      multihost.unique_barrier_key(
          'cleanup_tmp_dirs',
          prefix=multiprocessing_options.barrier_sync_key_prefix,
      ),
      timeout=multihost.coordination_timeout(),
      processes=multiprocessing_options.active_processes,
  )
