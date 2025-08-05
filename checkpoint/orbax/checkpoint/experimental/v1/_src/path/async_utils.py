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

"""Utilities for processing paths in asynchronous contexts."""

import asyncio
from typing import Iterable
from orbax.checkpoint._src.futures import future
from orbax.checkpoint._src.futures import synchronization
from orbax.checkpoint._src.path import atomicity
from orbax.checkpoint._src.path import atomicity_types
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.path import types

Path = types.Path
PathLike = types.PathLike
PathAwaitingCreation = types.PathAwaitingCreation


class _PathAwaitingCreation(PathAwaitingCreation):
  """Implementation of `PathAwaitingCreation` that wraps an awaitable."""

  def __init__(
      self, directory: Path, f: future.Future
  ):
    self._directory = directory
    self._f = f

  def __truediv__(
      self, other: PathAwaitingCreation | PathLike
  ) -> PathAwaitingCreation:
    if isinstance(other, PathAwaitingCreation):
      other = other.path
    return _PathAwaitingCreation(self.path / other, self._f)

  @property
  def path(self) -> Path:
    return self._directory

  async def await_creation(self) -> Path:
    await asyncio.to_thread(self._f.result)
    return self._directory


def start_async_mkdir(
    path: atomicity_types.TemporaryPath,
    subdirectories: Iterable[str] = (),
    operation_id: str | None = None,
) -> PathAwaitingCreation:
  """Starts async directory creation on a TemporaryPath.

  The mkdir operation is started immediately in a background thread on `path`.
  Creation is also started for any provided subdirectories. A
  `PathAwaitingCreation` object is returned.

  Subsequent operations on the returned object will NOT create any additional
  directories. For example, using::

    p = start_async_mkdir(path, ['a', 'b'])
    new_p = p / 'c'

  will not create a subdirectory named 'c'. Only directories `path` and `path/a`
  and `path/b` will be created.

  Args:
    path: The path to create. May be an instance of `TemporaryPath`.
    subdirectories: A sequence of subdirectories to create under `path`.
    operation_id: The operation id to use for the barrier keys. If None, the
      current operation id is used.

  Returns:
    A PathAwaitingCreation object.
  """
  context = context_lib.get_context()

  # TODO(b/407609827): V0 TypeHandler implementations, which are still used on
  # the saving path, do not have knowledge of the `PathAwaitingCreation`, and
  # instead rely on signals. We will need to continue using signals for now,
  # until `LeafHandler` implementations can be updated.
  completion_signals = [
      synchronization.HandlerAwaitableSignal.STEP_DIRECTORY_CREATION,
      synchronization.HandlerAwaitableSignal.ITEM_DIRECTORY_CREATION,
  ]
  f = atomicity.create_all_async(
      [path],
      completion_signals=completion_signals,
      multiprocessing_options=context.multiprocessing_options.v0(),
      subdirectories=[name for name in subdirectories],
      operation_id=operation_id,
  )
  return _PathAwaitingCreation(path.get(), f)
