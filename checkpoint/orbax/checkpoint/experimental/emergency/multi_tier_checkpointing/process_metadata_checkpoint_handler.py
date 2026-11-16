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

"""ProcessMetadataCheckpointHandler class.

Implementation of CheckpointHandler interface.
"""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Callable, Coroutine, List, Tuple

from absl import logging
from etils import epath
import jax
from orbax.checkpoint import checkpoint_args
from orbax.checkpoint import options as options_lib
from orbax.checkpoint._src import asyncio_utils
from orbax.checkpoint._src.futures import future
from orbax.checkpoint._src.handlers import async_checkpoint_handler
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint.experimental.emergency import mesh_consistency

# Empty marker file that a step saved with `backup=True` carries directly in
# its step directory; Multi-Tier Checkpointing backs such a step up regardless
# of its backup interval.
BACKUP_REQUESTED_FILE = '_BACKUP_REQUESTED'


CheckpointArgs = checkpoint_args.CheckpointArgs
PreviousDistributedToDeviceIds = List[List[int]]
PreviousDeviceIds = List[int]
register_with_handler = checkpoint_args.register_with_handler


async def _save_and_request_backup(
    save: Coroutine[None, None, None], directory: epath.Path
) -> None:
  """Runs the process metadata `save` and then marks the step for backup.

  The marker belongs directly in the step directory, i.e. the parent of this
  item's directory. The process metadata item is saved on every process for
  every step, so it is a convenient place to write the marker while the step is
  still being assembled. Writing it after `save` keeps it behind the same
  directory-creation signal, since directories may be created asynchronously.

  Args:
    save: The coroutine saving the process metadata item.
    directory: The process metadata item directory.
  """
  await save
  marker = directory.parent / BACKUP_REQUESTED_FILE
  logging.info('Requesting a backup of %s via %s', directory.parent, marker)
  await asyncio.to_thread(marker.touch)


class ProcessMetadataCheckpointHandler(
    async_checkpoint_handler.AsyncCheckpointHandler
):
  """Saves process metadata."""

  def __init__(
      self,
      multiprocessing_options: options_lib.MultiprocessingOptions = (
          options_lib.MultiprocessingOptions()
      ),
      distributed_to_device_ids_fn: Callable[[], List[List[int]]] | None = None,
  ) -> None:
    """Initializes ProcessMetadataCheckpointHandler."""
    self._distributed_to_device_ids_fn = (
        distributed_to_device_ids_fn or multihost.distributed_to_device_ids
    )

  async def async_save(
      self,
      directory: epath.Path,
      args: ProcessMetadataSaveArgs,
  ) -> List[future.Future] | None:
    """Saves the given process metadata.

    Args:
      directory: save location directory.
      args: ProcessMetadataSaveArgs (see below).

    Returns:
      A list of commit futures.
    """
    logging.vlog(2, 'Saving process metadata to %s', directory)
    save = mesh_consistency.save_process_metadata(
        directory,
        args.global_mesh,
        self._distributed_to_device_ids_fn(),
    )
    if args.backup_requested:
      save = _save_and_request_backup(save, directory)
    return [
        future.CommitFutureAwaitingContractedSignals(
            save,
            name='process_metadata_ch_save',
        )
    ]

  def save(
      self,
      directory: epath.Path,
      args: ProcessMetadataSaveArgs,
  ) -> None:
    async def async_save(
        directory: epath.Path,
        args: ProcessMetadataSaveArgs,
    ) -> None:
      commit_futures = await self.async_save(directory, args)
      if commit_futures:
        for f in commit_futures:
          f.result()

    asyncio_utils.run_sync(async_save(directory, args))

  def restore(
      self,
      directory: epath.Path,
      args: ProcessMetadataRestoreArgs,
  ) -> Tuple[PreviousDistributedToDeviceIds, PreviousDeviceIds]:
    """Restores metadata from directory.

    Args:
      directory: restore location directory.
      args: ProcessMetadataRestoreArgs (see below).

    Returns:
      A tuple of previous distributed to device ids and previous device ids.
    """
    logging.vlog(2, 'Restoring process metadata from %s', directory)
    return mesh_consistency.read_process_metadata(directory)


@register_with_handler(ProcessMetadataCheckpointHandler, for_save=True)
@dataclasses.dataclass
class ProcessMetadataSaveArgs(CheckpointArgs):
  """Parameters for saving process metadata.

  Attributes:
    global_mesh: the global mesh.
    backup_requested: whether to mark the step for backup by Multi-Tier
      Checkpointing (see `BACKUP_REQUESTED_FILE`).
  """

  global_mesh: jax.sharding.Mesh
  backup_requested: bool = False


@register_with_handler(ProcessMetadataCheckpointHandler, for_restore=True)
@dataclasses.dataclass
class ProcessMetadataRestoreArgs(CheckpointArgs):
  """Parameters for restoring process metadata."""
