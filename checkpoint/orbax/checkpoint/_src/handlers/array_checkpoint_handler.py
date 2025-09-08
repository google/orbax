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

"""ArrayCheckpointHandler for saving and restoring individual arrays/scalars."""

from __future__ import annotations

import dataclasses
from typing import List, Optional, Union

from etils import epath
import jax
import numpy as np
from orbax.checkpoint import aggregate_handlers
from orbax.checkpoint import checkpoint_args
from orbax.checkpoint import utils
from orbax.checkpoint._src import asyncio_utils
from orbax.checkpoint._src.futures import future
from orbax.checkpoint._src.handlers import async_checkpoint_handler
from orbax.checkpoint._src.handlers import base_pytree_checkpoint_handler
from orbax.checkpoint._src.serialization import type_handlers


CheckpointArgs = checkpoint_args.CheckpointArgs
register_with_handler = checkpoint_args.register_with_handler
BasePyTreeCheckpointHandler = (
    base_pytree_checkpoint_handler.BasePyTreeCheckpointHandler
)
BasePyTreeSaveArgs = base_pytree_checkpoint_handler.BasePyTreeSaveArgs
BasePyTreeRestoreArgs = base_pytree_checkpoint_handler.BasePyTreeRestoreArgs

ArrayType = Union[int, float, np.number, np.ndarray, jax.Array]

_ELEMENT_KEY = 'ELEMENT'
# OCDBT has no real benefit for a single array.
_USE_OCDBT_FOR_SAVE = False
PYTREE_METADATA_FILE = base_pytree_checkpoint_handler.PYTREE_METADATA_FILE


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
    self._aggregate_handler = aggregate_handlers.MsgpackHandler()
    self._base_pytree_checkpoint_handler = BasePyTreeCheckpointHandler(
        use_ocdbt=_USE_OCDBT_FOR_SAVE
    )

  def _is_supported_type(self, item: ArrayType) -> bool:
    return isinstance(item, (np.ndarray, jax.Array)) or utils.is_scalar(item)

  async def async_save(
      self,
      directory: epath.Path,
      item: Optional[ArrayType] = None,
      save_args: Optional[type_handlers.SaveArgs] = None,
      args: Optional[ArraySaveArgs] = None,
  ) -> Optional[List[future.Future]]:
    """Saves an object asynchronously.

    Args:
      directory: Folder in which to save.
      item: Deprecated, use `args`.
      save_args: Deprecated, use `args`.
      args: An ocp.array_checkpoint_handler.ArraySaveArgs (see below).

    Returns:
      A list of commit futures which can be run to complete the save.
    """
    if args is not None:
      item = args.item
      save_args = args.save_args

    if not self._is_supported_type(item):
      raise TypeError(f'Unsupported type: {type(item)}.')

    if save_args is None:
      save_args = type_handlers.SaveArgs()

    pytree_args = BasePyTreeSaveArgs(
        item={self._checkpoint_name: item},
        save_args={self._checkpoint_name: save_args},
    )
    return await self._base_pytree_checkpoint_handler.async_save(
        directory, args=pytree_args
    )

  def save(self, directory: epath.Path, *args, **kwargs):
    """Saves an array synchronously."""

    async def async_save():
      commit_futures = await self.async_save(directory, *args, **kwargs)  # pytype: disable=bad-return-type
      # Futures are already running, so sequential waiting is equivalent to
      # concurrent waiting.
      if commit_futures:  # May be None.
        for f in commit_futures:
          f.result()  # Block on result.

    asyncio_utils.run_sync(async_save())

  def restore(
      self,
      directory: epath.Path,
      item: Optional[ArrayType] = None,
      restore_args: Optional[type_handlers.RestoreArgs] = None,
      args: Optional[ArrayRestoreArgs] = None,
  ) -> ArrayType:
    """Restores an object.

    Args:
      directory: folder from which to read.
      item: Deprecated, use `args`.
      restore_args: Deprecated, use `args`.
      args: An ocp.array_checkpoint_handler.ArrayRestoreArgs object (see below).

    Returns:
      The restored object.
    """
    if args is None:
      args = ArrayRestoreArgs(item=item, restore_args=restore_args)

    if (directory / PYTREE_METADATA_FILE).exists():
      pytree_args = BasePyTreeRestoreArgs(
          {self._checkpoint_name: args.item} if args.item is not None else None,
          restore_args={self._checkpoint_name: args.restore_args},
      )
      return self._base_pytree_checkpoint_handler.restore(
          directory, args=pytree_args
      )[self._checkpoint_name]

    # TODO(nikhilbansall): Remove this logic once support for legacy
    # checkpoints lacking PYTREE_METADATA_FILE is no longer needed.
    restore_args = args.restore_args or type_handlers.RestoreArgs()

    checkpoint_path = directory / self._checkpoint_name
    info = type_handlers.ParamInfo(
        name=self._checkpoint_name,
        path=checkpoint_path,
        parent_dir=directory,
        skip_deserialize=False,
        is_ocdbt_checkpoint=type_handlers.is_ocdbt_checkpoint(directory),
    )
    restore_type = restore_args.restore_type
    if restore_type is None:
      restore_type = type_handlers.default_restore_type(restore_args)
    type_handler = type_handlers.get_type_handler(restore_type)
    result = asyncio_utils.run_sync(
        type_handler.deserialize([info], args=[restore_args])
    )[0]

    return result

  def finalize(self, directory: epath.Path):
    self._base_pytree_checkpoint_handler.finalize(directory)

  def close(self):
    """See superclass documentation."""
    self._aggregate_handler.close()


@register_with_handler(ArrayCheckpointHandler, for_save=True)
@dataclasses.dataclass
class ArraySaveArgs(CheckpointArgs):
  """Parameters for saving an array or scalar.

  Attributes:
    item (required): an array or scalar object.
    save_args: a `ocp.SaveArgs` object specifying save options.
  """

  item: ArrayType
  save_args: Optional[type_handlers.SaveArgs] = None


@register_with_handler(ArrayCheckpointHandler, for_restore=True)
@dataclasses.dataclass
class ArrayRestoreArgs(CheckpointArgs):
  """Array restore args.

  Attributes:
    item: unused, but provided as an option for legacy-compatibility reasons.
    restore_args: a `ocp.RestoreArgs` object specifying restore options.
  """

  item: Optional[ArrayType] = None
  restore_args: Optional[type_handlers.RestoreArgs] = None
