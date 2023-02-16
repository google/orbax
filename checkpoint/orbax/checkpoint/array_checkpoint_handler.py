# Copyright 2022 The Orbax Authors.
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

"""ArrayCheckpointHandler for saving and restoring individual arrays/scalars."""

import asyncio
from typing import List, Optional, Union

from etils import epath
import jax
import numpy as np
from orbax.checkpoint import aggregate_handlers
from orbax.checkpoint import async_checkpoint_handler
from orbax.checkpoint import future
from orbax.checkpoint import type_handlers
from orbax.checkpoint import utils

ArrayType = Union[int, float, np.number, np.ndarray, jax.Array]


class ArrayCheckpointHandler(async_checkpoint_handler.AsyncCheckpointHandler):
  """Handles saving and restoring individual arrays and scalars."""

  def __init__(self, checkpoint_name: Optional[str] = None):
    """Initializes the handler.

    Args:
      checkpoint_name: Provides a name for the directory under which Tensorstore
        files will be saved. Defaults to 'checkpoint'.
    """
    if not checkpoint_name:
      checkpoint_name = 'checkpoint'
    self._checkpoint_name = checkpoint_name

  def _is_supported_type(self, item: ArrayType) -> bool:
    return isinstance(item, (np.ndarray, jax.Array)) or utils.is_scalar(item)

  async def async_save(
      self,
      directory: epath.Path,
      item: ArrayType,
      save_args: Optional[type_handlers.SaveArgs] = None,
  ) -> Optional[List[future.Future]]:
    """Saves an object asynchronously.

    Args:
      directory: Folder in which to save.
      item: An array or scalar object.
      save_args: A single SaveArgs object specifying save information.

    Returns:
      A list of commit futures which can be run to complete the save.
    """
    if not self._is_supported_type(item):
      raise TypeError(f'Unsupported type: {type(item)}.')

    if save_args and save_args.aggregate:
      aggregate_handler = aggregate_handlers.get_aggregate_handler()
      await aggregate_handler.serialize(
          directory / self._checkpoint_name, item
      )
      return []

    info = type_handlers.ParamInfo(
        name=self._checkpoint_name,
        path=directory / self._checkpoint_name,
        aggregate=False,
    )
    type_handler = type_handlers.get_type_handler(type(item))
    return await type_handler.serialize(item, info, args=save_args)

  def save(self, directory: epath.Path, item: ArrayType, *args, **kwargs):
    """Saves the provided item.

    Blocks until both copy and commit complete.

    See async_save.

    Args:
      directory: the directory to save to.
      item: the item to be saved.
      *args: additional arguments for save.
      **kwargs: additional arguments for save.
    """

    async def async_save(*args, **kwargs):
      commit_futures = await self.async_save(*args, **kwargs)  # pytype: disable=bad-return-type
      # Futures are already running, so sequential waiting is equivalent to
      # concurrent waiting.
      for f in commit_futures:
        f.result()  # Block on result.

    asyncio.run(async_save(directory, item, *args, **kwargs))
    utils.sync_global_devices('ArrayCheckpointHandler:save')

  def restore(
      self,
      directory: epath.Path,
      item: Optional[ArrayType] = None,
      restore_args: Optional[type_handlers.RestoreArgs] = None,
  ) -> ArrayType:
    """Restores an object.

    Args:
      directory: folder from which to read.
      item: unused.
      restore_args: a single RestoreArgs object specifying relevant information.
        Must specify restore_type to indicate the desired restoration type. If
        the restore_type is jax.Array, RestoreArgs should be ArrayRestoreArgs.

    Returns:
      The restored object.
    """
    del item
    if restore_args is None:
      restore_args = type_handlers.RestoreArgs()
    if restore_args.lazy:
      raise ValueError(
          'Lazy restoration not supported for ArrayCheckpointHandler.'
      )

    if (directory / self._checkpoint_name).is_file():
      aggregate_handler = aggregate_handlers.get_aggregate_handler()
      result = aggregate_handler.deserialize(directory / self._checkpoint_name)
      if not self._is_supported_type(result):
        raise TypeError(f'Unsupported type: {type(result)}.')
    else:
      info = type_handlers.ParamInfo(
          name=self._checkpoint_name,
          path=directory / self._checkpoint_name,
          aggregate=False,
      )
      type_handler = type_handlers.get_type_handler(restore_args.restore_type)
      result = asyncio.run(type_handler.deserialize(info, args=restore_args))

    utils.sync_global_devices('ArrayCheckpointHandler:restore')
    return result

  def structure(self, directory: epath.Path) -> ArrayType:
    raise NotImplementedError(
        'Unsupported method `structure` for ArrayCheckpointHandler.'
    )
