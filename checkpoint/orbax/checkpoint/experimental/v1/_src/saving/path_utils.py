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

"""Internal utilities for path handling in saving."""

from absl import logging
from orbax.checkpoint._src.path import async_path
from orbax.checkpoint._src.path import atomicity_defaults
from orbax.checkpoint._src.path import atomicity_types
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.path import types as path_types
from orbax.checkpoint.experimental.v1._src.synchronization import multihost



def get_temporary_path(
    path: path_types.Path,
    *,
    context: context_lib.Context,
    use_snapshot: bool | None = None,
) -> atomicity_types.TemporaryPath:
  """Gets a TemporaryPath for the given path."""
  temporary_path_class = (
      context.file_options.temporary_path_class
      or atomicity_defaults.get_default_temporary_path_class(path)
  )
  tmpdir = temporary_path_class.from_final(
      path,
      # Ensure metadata store is NOT passed, to prevent separate metadata
      # writing.
      checkpoint_metadata_store=None,
      file_options=context.file_options.v0(),
      use_snapshot=use_snapshot,
  )
  return tmpdir


async def remove_existing_path(
    path: path_types.Path,
    *,
    context: context_lib.Context,
):
  if multihost.is_primary_host(context.multiprocessing_options.primary_host):
    logging.info(
        '[process=%s] Specified `overwrite`: removing existing path.',
        multihost.process_index(),
    )
    await async_path.rmtree(path)
  await multihost.sync_global_processes(
      multihost.unique_barrier_key(
          'save_checkpointables_async:rmtree',
          prefix=context.multiprocessing_options.barrier_sync_key_prefix,
      ),
      processes=context.multiprocessing_options.active_processes,
  )


async def maybe_overwrite_existing(
    path: path_types.Path,
    *,
    overwrite: bool,
    context: context_lib.Context,
) -> None:
  """Checks if the path exists and overwrites it if necessary."""
  if await async_path.exists(path):
    if overwrite:
      await remove_existing_path(path, context=context)
    else:
      raise ValueError(f'Destination {path} already exists.')
