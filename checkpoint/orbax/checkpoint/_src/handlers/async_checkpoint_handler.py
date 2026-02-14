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
from typing import List, Optional

from etils import epath
from orbax.checkpoint._src.futures import future
from orbax.checkpoint._src.handlers import checkpoint_handler
from orbax.checkpoint.experimental.v1._src.path import types as path_types


class AsyncCheckpointHandler(checkpoint_handler.CheckpointHandler):
  """An interface providing async methods used with AsyncCheckpointer."""

  @abc.abstractmethod
  async def async_save(
      self,
      directory: epath.Path,
      *args,
      **kwargs,
  ) -> Optional[List[future.Future]]:
    """Saves the given item to the provided directory.

    Args:
      directory: the directory to save to.
      *args: additional arguments for save.
      **kwargs: additional arguments for save.

    Returns:
      A list of commit futures which can be awaited upon to complete the save
      operation.
    """
    pass


class DeferredPathAsyncCheckpointHandler(AsyncCheckpointHandler):
  """Handler interface that receives Path or PathAwaitingCreation.

  This interface extends AsyncCheckpointHandler with an async_save method that
  accepts either an epath.Path or PathAwaitingCreation, allowing handlers to
  work with deferred paths (e.g., TFHub) where the actual path is allocated
  asynchronously.

  Handlers implementing this interface can:
  1. Receive a deferred path representation before the path is allocated
  2. Wait for STEP_DIRECTORY_CREATION signal inside their CommitFuture
  3. Access the path via await_creation() or .path after the signal
  """

  @abc.abstractmethod
  async def async_save(
      self,
      directory: epath.Path | path_types.PathAwaitingCreation,
      *args,
      **kwargs,
  ) -> Optional[List[future.Future]]:
    """Constructs a save operation with support for deferred paths.

    This method accepts an epath.Path or PathAwaitingCreation.
    When a deferred path is passed, handler coroutines should wait for the
    STEP_DIRECTORY_CREATION signal before accessing the path.

    Args:
      directory: The directory to save to. May be an epath.Path or
        PathAwaitingCreation. For deferred paths, await_creation() or signal
        ordering ensures the path is available.
      *args: additional arguments for save.
      **kwargs: additional arguments for save.

    Returns:
      A list of futures that will commit the data when awaited.
    """

    pass
