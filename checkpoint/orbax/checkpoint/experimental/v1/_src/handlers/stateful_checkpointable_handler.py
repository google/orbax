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

"""StatefulCheckpointableHandler class."""

from typing import Any, Awaitable, Generic
from orbax.checkpoint.experimental.v1._src.handlers import types as handler_types
from orbax.checkpoint.experimental.v1._src.path import types as path_types

T = handler_types.T


class StatefulCheckpointableHandler(
    handler_types.CheckpointableHandler[T, T],
    Generic[T],
):
  """Serializes/deserializes a Checkpointable."""

  async def save(
      self,
      directory: path_types.PathAwaitingCreation,
      checkpointable: handler_types.StatefulCheckpointable[T],
  ) -> Awaitable[None]:
    return await checkpointable.save(directory)

  async def load(
      self,
      directory: path_types.Path,
      abstract_checkpointable: (
          handler_types.StatefulCheckpointable[T] | None
      ) = None,
  ) -> Awaitable[T]:
    if abstract_checkpointable is None:
      raise ValueError(
          'To restore a `StatefulCheckpointable`, you must pass an instance of'
          ' the object.'
      )

    # Returns Awaitable[None]
    background_load = await abstract_checkpointable.load(directory)

    async def _background_load() -> T:
      await background_load
      # After loading, `abstract_checkpointable` (actually just a concrete
      # checkpointable) should be populated with the loaded data.
      return abstract_checkpointable

    return _background_load()

  async def metadata(self, directory: path_types.Path) -> T:
    raise NotImplementedError(
        'Metadata retrieval is not supported for objects implementing'
        ' `StatefulCheckpointable`.'
    )

  def is_handleable(self, checkpointable: Any) -> bool:
    return isinstance(checkpointable, handler_types.StatefulCheckpointable)

  def is_abstract_handleable(self, abstract_checkpointable: Any) -> bool | None:
    return isinstance(
        abstract_checkpointable, handler_types.StatefulCheckpointable
    )
