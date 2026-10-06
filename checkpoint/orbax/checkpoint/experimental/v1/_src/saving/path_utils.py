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

"""Internal utilities for path handling in saving."""

from absl import logging
from orbax.checkpoint._src.path import async_path
from orbax.checkpoint._src.path import atomicity_defaults
from orbax.checkpoint._src.path import atomicity_types
from orbax.checkpoint._src.path.snapshot import snapshot as snapshot_lib
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.path import types as path_types
from orbax.checkpoint.experimental.v1._src.synchronization import multihost


def get_temporary_path(
    path: path_types.Path,
    *,
    context: context_lib.Context,
    snapshot_type: snapshot_lib.SnapshotType | None = None,
) -> atomicity_types.TemporaryPath:
  """Gets a :py:class:`~.atomicity_types.TemporaryPath` for the given path.

  Args:
    path: The final path to use for the checkpoint.
    context: The Orbax context.
    snapshot_type: The type of snapshot to use for the temporary path.

  Returns:
    A TemporaryPath for the given path.
  """
  temporary_path_cls = atomicity_defaults.get_default_temporary_path_class(
      path,
      atomicity_options=context.atomicity.v0(),
  )
  tmpdir = temporary_path_cls.from_final(
      path,
      # Ensure metadata store is NOT passed, to prevent separate metadata
      # writing.
      checkpoint_metadata_store=None,
      file_options=context.file_options.v0(),
      snapshot_type=snapshot_type,
  )
  return tmpdir


async def maybe_overwrite_existing(
    path: path_types.Path,
    *,
    overwrite: bool,
    primary_host: int | None = 0,
) -> None:
  """Checks if `path` exists on primary host and removes or raises.

  Args:
    path: The path to check and potentially remove.
    overwrite: Whether to overwrite the path if it exists.
    primary_host: The primary host index, or None if all hosts are primary.

  Raises:
    ValueError: If `path` exists and `overwrite` is False.
  """
  if not multihost.is_primary_host(primary_host):
    return
  if not await async_path.exists(path):
    return
  if overwrite:
    logging.info(
        '[process=%s] Specified `overwrite`: removing existing path.',
        multihost.process_index(),
    )
    await async_path.rmtree(path)
  else:
    raise ValueError(f'Destination {path} already exists.')
