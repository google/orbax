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

"""Utilities for processing paths in asynchronous contexts."""

from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Iterable, Sequence

from absl import logging
import jax
from orbax.checkpoint._src.futures import future
from orbax.checkpoint._src.futures import synchronization
from orbax.checkpoint._src.path import async_path
from orbax.checkpoint._src.path import atomicity_types
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.path import types
from orbax.checkpoint.experimental.v1._src.synchronization import multihost

Path = types.Path
PathLike = types.PathLike
TemporaryPath = atomicity_types.TemporaryPath


async def _create_paths(
    tmp_path: TemporaryPath,
    *,
    subdirectories: Iterable[str],
    context: context_lib.Context,
    operation_id: str,
    completion_signals: Sequence[synchronization.HandlerAwaitableSignal],
):
  """Creates :py:class:`.`TemporaryPath` and subdirectories."""
  active_processes = context.multiprocessing_options.active_processes
  primary_host = context.multiprocessing_options.primary_host
  barrier_sync_key_prefix = (
      context.multiprocessing_options.barrier_sync_key_prefix
  )
  if multihost.is_primary_host(primary_host):
    start = time.time()
    path = await tmp_path.create()
    # subdirectory assumed to not have any nesting.
    subdir_ops = [
        async_path.mkdir(path / subdirectory, parents=False, exist_ok=False)
        for subdirectory in subdirectories
    ]
    await asyncio.gather(*subdir_ops)
    directory_creation_secs = time.time() - start
    jax.monitoring.record_event_duration_secs(
        '/jax/orbax/write/directory_creation_secs',
        directory_creation_secs,
    )
    jax.monitoring.record_event_duration_secs(
        '/jax/orbax/write/async_directory_creation_secs',
        directory_creation_secs,
    )
    logging.vlog(
        1,
        'Asynchronous directory creation took %s seconds',
        directory_creation_secs,
    )
    future.set_signals(completion_signals, operation_id=operation_id)
  await multihost.sync_global_processes(
      multihost.unique_barrier_key(
          'create_directory:post',
          prefix=barrier_sync_key_prefix,
      ),
      operation_id=operation_id,
      timeout=multihost.coordination_timeout(),
      processes=active_processes,
  )


class _SubdirectoryAwaitingCreation(types.PathAwaitingCreation):
  """A :py:class:`.`PathAwaitingCreation` that is a subdirectory of a base path.

  It expects to receive a base :py:class:`.PathAwaitingCreation` during
  initialization, and
  will wait for that path to be created before allowing access to the
  subdirectory. Any further subdirectory appending will still reference the
  same base path.
  """

  def __init__(self, path: Path, base_path: types.PathAwaitingCreation):
    self._base_path = base_path
    self._path = path

  def __truediv__(self, other: PathLike) -> types.PathAwaitingCreation:
    if not isinstance(other, PathLike):
      raise TypeError(f'Expected PathLike, got {type(other)}.')
    return _SubdirectoryAwaitingCreation(
        self.path / other,
        self._base_path,
    )

  @property
  def path(self) -> Path:
    return self._path

  async def await_creation(self) -> Path:
    """Waits for the directory to be created."""
    await self._base_path.await_creation()
    return self._path


class PathAwaitingCreation(types.PathAwaitingCreation):
  """Implementation of :py:class:`.PathAwaitingCreation` that creates paths asynchronously.

  This implementation also includes a `create` API that allows a caller to
  create the paths immediately. The `await_creation` API may also trigger a
  creation if it has not already started. This allows implementers of
  custom handlers to trigger themselves without running into hangs.

  Creation is carried out uniquely. Repeated calls to `create` will NOT create
  any additional directories. For example, using::

    p = PathAwaitingCreation.build(path, ['a', 'b'])
    new_p = p / 'c'

  will not create a subdirectory named 'c'. Only directories `path` and `path/a`
  and `path/b` will be created.

  For async directory creations, `create` should be called from a background
  thread. We allow the background thread to initiate the mkdir operation
  directly, instead of initiating in the main thread and carrying it over to the
  background thread.
  COST:
   - Introduces a small amount of slowdown, but the overall impact should be
     very marginal given that it would affect only the background thread.
  BENEFIT:
   - Simplifies the logic involved in allowing control of the async operations
     to be transferred between threads.
  We should not go all the way to this proposal's logical conclusion and just
  require individual handlers to create their own directories. Remember that
  directories must be created in a centralized place to avoid duplicate
  requests, which creates additional QPS burden on the filesystem.
  """

  def __init__(
      self,
      path: Path,
      awaitable: Awaitable[None],
  ):
    self._path = path
    self._awaitable = awaitable
    self._creation_completed = False
    self._lock = asyncio.Lock()

  def __truediv__(self, other: PathLike) -> types.PathAwaitingCreation:
    """Creates a new :py:class:`.PathAwaitingCreation` that appends to the current path.

    The path returned will reference the original "base" object, and
    `await_creation` will wait for the base path to be created. Subdirectories
    of the base path appended in this way (unless they were already created by
    the `create` operation) will not be automatically created, and the user will
    need to create them after calling `await_creation`.

    Args:
      other: The subdirectory to create.

    Returns:
      A new :py:class:`.PathAwaitingCreation` that appends to the current path.
    """
    if not isinstance(other, PathLike):
      raise TypeError(f'Expected PathLike, got {type(other)}.')
    return _SubdirectoryAwaitingCreation(
        self.path / other,
        self,
    )

  @property
  def path(self) -> Path:
    return self._path

  async def create(self) -> Path:
    """Creates the directory if it has not already been created.

    Use a lock to ensure that only one process starts the creation.
    Any other processes will wait at the lock if a creation is already in
    progress, then return immediately since `_creation_completed` will have
    already been set.

    Returns:
      The path that was created.
    """
    async with self._lock:
      if self._creation_completed:
        return self._path
      self._creation_completed = True
      await self._awaitable
      # Any operations after this might cause deadlock if self._awaitable raises
      # an exception.
    return self._path

  async def await_creation(self) -> Path:
    """Waits for the directory to be created.

    Creation will be triggered if it has not already started.
    Returns:
      The path that was created.
    """
    await self.create()  # This is a no-op if already created.
    return self._path

  @classmethod
  def build(
      cls,
      path: TemporaryPath,
      subdirectories: Iterable[str],
  ) -> PathAwaitingCreation:
    # TODO(b/407609827): V0 TypeHandler implementations, which are still used on
    # the saving path, do not have knowledge of the `PathAwaitingCreation`, and
    # instead rely on signals. We will need to continue using signals for now,
    # until `LeafHandler` implementations can be updated.
    completion_signals = [
        synchronization.HandlerAwaitableSignal.STEP_DIRECTORY_CREATION,
        synchronization.HandlerAwaitableSignal.ITEM_DIRECTORY_CREATION,
    ]
    future.AwaitableSignalsContract.add_to_awaitable_signals_contract(
        completion_signals
    )
    awaitable = _create_paths(
        path,
        subdirectories=subdirectories,
        context=context_lib.get_context(),
        operation_id=context_lib.get_context().operation_id(),
        completion_signals=completion_signals,
    )
    return cls(path.get(), awaitable)
