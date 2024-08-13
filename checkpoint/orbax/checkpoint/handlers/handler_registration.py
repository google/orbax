# Copyright 2024 The Orbax Authors.
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

"""Registry for `CheckpointHandler`s.

TODO: b/357913996 - Add proper user-facing documentation.
"""

from typing import MutableMapping, Optional, Protocol, Type, Union

from orbax.checkpoint import checkpoint_args
from orbax.checkpoint import checkpoint_handler

CheckpointArgs = checkpoint_args.CheckpointArgs
CheckpointHandler = checkpoint_handler.CheckpointHandler
HandlerRegistryMapping = MutableMapping[
    tuple[Optional[str], type[CheckpointArgs]], CheckpointHandler
]


class CheckpointHandlerRegistry(Protocol):
  """Protocol for `CheckpointHandlerRegistry`."""

  def add(
      self,
      item: Optional[str],
      args: Union[Type[CheckpointArgs], CheckpointArgs],
      handler: Union[CheckpointHandler, Type[CheckpointHandler]],
      **kwargs,
  ):
    ...

  def get(
      self,
      item: Optional[str],
      args: Union[Type[CheckpointArgs], CheckpointArgs],
  ) -> CheckpointHandler:
    ...

  def has(
      self,
      item: Optional[str],
      args: Union[Type[CheckpointArgs], CheckpointArgs],
  ) -> bool:
    ...

  def get_all_entries(
      self,
  ) -> HandlerRegistryMapping:
    ...


class AlreadyExistsError(ValueError):
  """Raised when an entry already exists in the registry."""


class NoEntryError(KeyError):
  """Raised when no entry exists in the registry."""


def _get_args_type(
    args: Union[Type[CheckpointArgs], CheckpointArgs],
) -> type[CheckpointArgs]:
  if isinstance(args, type):
    return args
  else:
    return type(args)


class DefaultCheckpointHandlerRegistry(CheckpointHandlerRegistry):
  """Default implementation of `CheckpointHandlerRegistry`."""

  def __init__(
      self, other_registry: Optional[CheckpointHandlerRegistry] = None
  ):
    self._registry: HandlerRegistryMapping = {}

    # Initialize the registry with entries from other registry.
    if other_registry:
      for (
          item,
          args_type,
      ), handler in other_registry.get_all_entries().items():
        self.add(item, args_type, handler)

  def add(
      self,
      item: Optional[str],
      args: Union[Type[CheckpointArgs], CheckpointArgs],
      handler: Union[CheckpointHandler, Type[CheckpointHandler]],
  ):
    """Adds an entry to the registry.

    Args:
      item: The item name. If None, the entry will be added as a general `arg`
        entry.
      args: The args type to be added to registry. If a concerete type is
        provided, the type will be added to the registry.
      handler: The handler. If a type is provided, an instance of the type will
        be added to the registry.

    Raises:
      AlreadyExistsError: If an entry for the given item and args type already
        exists in the registry.
    """
    args_type = _get_args_type(args)

    if self.has(item, args_type):
      raise AlreadyExistsError(
          f'Entry for item={item} and args_type={args_type} already'
          ' exists in the registry.'
      )
    else:
      handler_instance = handler() if isinstance(handler, type) else handler
      self._registry[(item, args_type)] = handler_instance

  def get(
      self,
      item: Optional[str],
      args: Union[Type[CheckpointArgs], CheckpointArgs],
  ) -> CheckpointHandler:
    """Returns the handler for the given item and args type.

    Args:
      item: The item name. If None, the entry will be added as a general
        `args_type` entry.
      args: The args type to get the handler for. If a concerete type is 
        provided, the type will be used to get the handler.

    If item the item has not been registered, the general `args_type` entry will
    be returned if it exists.

    Raises:
      NoEntryError: If no entry for the given item and args type exists in the
        registry.
    """
    args_type = _get_args_type(args)

    if self.has(item, args_type):
      return self._registry[(item, args_type)]

    # Fall back to general `args_type` if there is no entry for the given item
    # in the registry.
    if item is not None:
      if (None, args_type) in self._registry:
        return self.get(None, args_type)

    raise NoEntryError(
        f'No entry for item={item} and args_ty={args_type} in the registry.'
    )

  def has(
      self,
      item: Optional[str],
      args: Union[Type[CheckpointArgs], CheckpointArgs],
  ) -> bool:
    """Returns whether an entry for the given item and args type exists in the registry.

    Args:
      item: The item name or None.
      args: The args type to check for. If a concrete type is provided, the
        type will be used to check for the entry.

    Does not check for fall back to general `args_type` entry.
    """
    args_type = _get_args_type(args)
    return (
        item,
        args_type,
    ) in self._registry

  def get_all_entries(
      self,
  ) -> HandlerRegistryMapping:
    """Returns all entries in the registry."""
    return self._registry
