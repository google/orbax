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

"""Internal utilities for saving whole and partial checkpoints."""

from typing import Any
import uuid

from absl import logging
from orbax.checkpoint._src.path import atomicity_defaults
from orbax.checkpoint._src.path import atomicity_types
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.path import async_path
from orbax.checkpoint.experimental.v1._src.path import format_utils
from orbax.checkpoint.experimental.v1._src.path import types as path_types
from orbax.checkpoint.experimental.v1._src.synchronization import multihost
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types



def get_temporary_path(
    path: path_types.Path, *, context: context_lib.Context
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
      multiprocessing_options=context.multiprocessing_options.v0(),
      file_options=context.file_options.v0(),
  )
  return tmpdir


async def remove_existing_path(
    path: path_types.Path,
    *,
    context: context_lib.Context,
):
  """Removes the existing path if it exists."""
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


def add_internal_checkpointables(
    checkpointables: dict[str, Any],
    *,
    context: context_lib.Context,
    metrics: tree_types.JsonType | None = None,
) -> dict[str, Any]:
  """Adds descriptor to checkpointables if enabled."""
  # Global registration ties metrics key to JsonHandler.
  if metrics:
    checkpointables[format_utils.METRICS_CHECKPOINTABLE_KEY] = metrics
  return checkpointables
