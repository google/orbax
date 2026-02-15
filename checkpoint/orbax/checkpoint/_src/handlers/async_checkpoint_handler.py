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

"""AsyncCheckpointHandler interface."""

import abc
import asyncio
from typing import List, Optional

from etils import epath
from orbax.checkpoint._src.futures import future
from orbax.checkpoint._src.handlers import checkpoint_handler


class AsyncCheckpointHandler(checkpoint_handler.CheckpointHandler):
  """An interface providing async methods that can be used with CheckpointHandler.

  This is the legacy interface that expects resolved paths. For handlers that
  support async directory creation via futures, use
  FutureAwareAsyncCheckpointHandler.
  """

  @abc.abstractmethod
  async def async_save(
      self, directory: epath.Path, *args, **kwargs
  ) -> Optional[List[future.Future]]:
    """Constructs a save operation.

    Synchronously awaits a copy of the item, before returning commit futures
    necessary to save the item.

    Note: Any operations on directory should be done by using
    `future.CommitFutureAwaitingContractedSignals` to wait for directories to be
    created.

    Args:
      directory: the directory to save to.
      *args: additional arguments for save.
      **kwargs: additional arguments for save.
    """
    pass


class FutureAwareAsyncCheckpointHandler(AsyncCheckpointHandler):
  """Extended interface supporting async directory creation via futures.

  This interface allows handlers to accept Future[epath.Path] as the directory
  parameter, enabling directory creation to overlap with pre-serialization work.
  This is particularly beneficial for storage systems like TFHub where directory
  allocation can take several seconds.
  """

  @abc.abstractmethod
  async def async_save(  # pytype: disable=signature-mismatch
      self, directory: 'asyncio.Future[epath.Path]', *args, **kwargs
  ) -> Optional[List[future.Future]]:
    """Constructs a save operation with support for future directories.

    Synchronously awaits a copy of the item, before returning commit futures
    necessary to save the item.

    Implementations should delay awaiting the directory future until the
    resolved
    path is actually needed, allowing async directory creation to overlap with
    pre-serialization work (e.g., batching param infos, preparing save args).

    Args:
      directory: An asyncio.Future that resolves to the directory path to save
        to. This enables async directory creation to overlap with
        pre-serialization work.
      *args: additional arguments for save.
      **kwargs: additional arguments for save.

    Returns:
      Optional list of commit futures for background operations.
    """
    pass
