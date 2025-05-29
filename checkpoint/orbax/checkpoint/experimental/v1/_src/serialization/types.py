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

"""Define types for `LeafHandler`."""

import dataclasses
from typing import Awaitable, Generic, Protocol, Sequence, TypeVar
from orbax.checkpoint._src.serialization import serialization as serialization_v0
from orbax.checkpoint._src.tree import utils as tree_utils
from orbax.checkpoint.experimental.v1._src.path import types as path_types
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types
import tensorstore as ts

Leaf = TypeVar('Leaf')
AbstractLeaf = TypeVar('AbstractLeaf')


@dataclasses.dataclass
class SerializationParam(Generic[Leaf]):
  keypath: tree_types.PyTreeKeyPath
  value: Leaf

  @property
  def name(self) -> str:
    return tree_utils.param_name_from_keypath(self.keypath)


@dataclasses.dataclass
class SerializationContext:
  parent_dir: path_types.PathAwaitingCreation
  ts_context: ts.Context | None = None
  byte_limiter: serialization_v0.LimitInFlightBytes | None = None


@dataclasses.dataclass
class DeserializationParam(Generic[AbstractLeaf]):
  keypath: tree_types.PyTreeKeyPath
  value: AbstractLeaf | None = None

  @property
  def name(self) -> str:
    return tree_utils.param_name_from_keypath(self.keypath)


@dataclasses.dataclass
class DeserializationContext:
  parent_dir: path_types.Path
  ocdbt_checkpoint: bool
  zarr3_checkpoint: bool
  ts_context: ts.Context | None = None
  byte_limiter: serialization_v0.LimitInFlightBytes | None = None


class LeafHandler(Protocol[Leaf, AbstractLeaf]):
  """Interface for reading and writing a PyTree leaf."""

  async def serialize(
      self,
      params: Sequence[SerializationParam[Leaf]],
      serialization_context: SerializationContext,
  ) -> Awaitable[None]:
    """Writes the specified leaves of a checkpointable to a storage location.

    This method is responsible for copying the leaf data from a remote device in
    a synchronous fashion (if applicable).  The whole operation will be started
    in background as soon as the awaitable is returned.  The returned awaitable
    can be awaited to confirm the completion of the final commit operation to a
    storage location.

    The function can be used in a multihost setting, but should not implement
    extra logic to ensure atomicity.

    Note: For function authors, within this function, make sure the destination
    directory is fully created by waiting for `await
    serialization_context.parent_dir.await_creation()` before performing any
    storage write operations.

    Args:
      params: a sequence of SerializationParam per leaf.
      serialization_context: SerializationContext for the leaf handler.

    Returns:
      A awaitable which can be awaited to complete the save operation.
    """
    ...

  async def deserialize(
      self,
      params: Sequence[DeserializationParam[AbstractLeaf]],
      deserialization_context: DeserializationContext,
  ) -> Awaitable[Sequence[Leaf]]:
    """Returns sequence of leaves from a stored checkpointable location.

    This method initiates the background copying of leaf data from a storage
    location to remote devices. The returned awaitable allows you to await it to
    confirm the completion of this data transfer.

    Args:
      params: sequence of DeserializationParam per leaf.
      deserialization_context: DeserializationContext for the leaf handler.

    Returns:
      A awaitable which can be awaited to complete the load operation and obtain
      a squence of leaves.
    """
    ...

  async def metadata(
      self,
      params: Sequence[DeserializationParam[None]],
      deserialization_context: DeserializationContext,
  ) -> Sequence[AbstractLeaf]:
    """Returns a squence of AbstractLeaf from a stored checkpointable location.

    Args:
      params: sequence of DeserializationParam[None] per leaf, the keypath is
        essential to look up the leaf metadata.
      deserialization_context: DeserializationContext for the leaf handler.

    Returns:
      Sequence of AbstractLeaf for each provided DeserializationParam.
    """
    ...
