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
from typing import Any, Awaitable, Generic, Protocol, Sequence, Tuple, Type, TypeVar

from orbax.checkpoint._src.serialization import serialization as serialization_v0
from orbax.checkpoint._src.tree import utils as tree_utils
from orbax.checkpoint.experimental.v1._src.path import types as path_types
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types
import tensorstore as ts

Leaf = TypeVar('Leaf')
AbstractLeaf = TypeVar('AbstractLeaf')

PLACEHOLDER = ...


def is_placeholder(value: Any) -> bool:
  return value is PLACEHOLDER


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


LeafHandlerRegistryItem = Tuple[
    Type[Leaf], Type[AbstractLeaf], Type[LeafHandler[Leaf, AbstractLeaf]]
]


class LeafHandlerRegistry(Protocol):
  """A Protocol for a LeafHandlerRegistry.

  This protocol defines the interface for a leaf handler registry. It acts as a
  lookup service, associating specific Leaf or AbstractLeaf types with their
  corresponding leaf handlers. It can be accessed through the module function
  get/set/is_handable/is_abstract_handlable.
  """

  def get(self, leaf_type: Type[Leaf]) -> Type[LeafHandler[Leaf, Any]]:
    """Returns the handler type registered for a given Leaf type.

    Args:
      leaf_type: The type to get the handler for.

    Returns:
      The handler type registered for the given Leaf type.

    Raises:
      ValueError: If the leaf_type is not registered.
    """
    ...

  def get_abstract(
      self, abstract_type: Type[AbstractLeaf]
  ) -> Type[LeafHandler[Any, AbstractLeaf]]:
    """Returns the handler type registered for a given abstract type.

    Args:
      abstract_type: The abstract type to get the handler for.

    Returns:
      The handler type registered for the given abstract type.

    Raises:
      ValueError: If the abstract_type is not registered.
    """
    ...

  def get_all(self) -> Sequence[LeafHandlerRegistryItem]:
    """Returns all registered handlers. Useful to examine what is registered.

    Returns:
      A sequence of tuples containing the type, abstract type, and handler type
      for corresponding registered handler.
    """
    ...

  def add(
      self,
      leaf_type: Type[Leaf],
      abstract_type: Type[AbstractLeaf],
      handler_type: Type[LeafHandler[Leaf, AbstractLeaf]],
      override: bool = False,
  ):
    """Registers the handler_type for a leaf_type and abstract_type pair.

    If there is already a registered handler_type for the a leaf_type, its
    coressponding abstract_type and handler_type will be overridden when
    `override` is True. If the abstract_type has already associated with another
    leaf_type, an error will be raised if `override` is False.


    Args:
      leaf_type: The type to register the handler for.
      abstract_type: The abstract type to register the handler for.
      handler_type: The handler to register.
      override: Whether to override the handler if it already exists.
    """
    ...

  def is_handleable(self, leaf_type: Type[Any]) -> bool:
    """Returns True if the leaf_type is handleable by any registered handler."""
    ...

  def is_abstract_handleable(self, abstract_type: Type[Any]) -> bool:
    """Returns True if the abstract_type is handlable by any registered handler."""
    ...
