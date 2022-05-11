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

"""Checkpointer interface."""

import abc
import asyncio
from typing import Any, Optional
from jax.experimental import multihost_utils


class Checkpointer(abc.ABC):
  """An interface providing save/restore methods used on a savable item.

  Item may be a PyTree, Dataset, or any other supported object.
  """

  def save(self, directory: str, item: Any, *args, **kwargs):
    """Saves the provided item.

    See async_save.

    Args:
      directory: the directory to save to.
      item: the item to be saved.
      *args: additional arguments for save.
      **kwargs: additional arguments for save.
    """
    asyncio.run(self.async_save(directory, item, *args, **kwargs))
    multihost_utils.sync_global_devices('Checkpointer:save')

  @abc.abstractmethod
  async def async_save(self, directory: str, item: Any):
    """Saves the provided item asynchronously.

    Args:
      directory: the directory to save to.
      item: the item to be saved.
    """
    pass

  def restore(self,
              directory: str,
              item: Optional[Any] = None,
              **kwargs) -> Any:
    """Restores the provided item synchronously.

    See async_restore.

    Args:
      directory: the directory to restore from.
      item: an item with the same structure as that to be restored.
      **kwargs: additional arguments for restore.

    Returns:
      The restored item.
    """
    result = asyncio.run(self.async_restore(directory, item, **kwargs))
    multihost_utils.sync_global_devices('Checkpointer:restore')
    return result

  @abc.abstractmethod
  async def async_restore(self,
                          directory: str,
                          item: Optional[Any] = None) -> Any:
    """Restores the provided item asynchronously.

    Note that `item` may be supplied as a way to provide the structure of the
    item to be restored, and thereby to provide a contract between the caller
    and the saved file. However, `item` is optional, since not all Checkpointer
    implementations will require it as an argument.

    Args:
      directory: the directory to restore from.
      item: an item with the same structure as that to be restored.

    Returns:
      The restored item.
    """
    pass
