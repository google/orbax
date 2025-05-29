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

"""Registry for `CheckpointableHandler`.

A `CheckpointableHandler` defines logic needed to save and restore a
"checkpointable" object. Once defined, the handler must be registered either
globally or locally.

To register the handler globally, use `register_handler`. Alternatively,
create a local registry confined to a specific scope by using
`local_registry` (note that globally-registered handlers are included in this
registry by default).

For example::

  ocp.handlers.register_handler(FooHandler)

  registry = ocp.handlers.local_registry()
  # `registry` already contains FooHandler.
  registry.add(BarHandler)
  # Scope this handler specifically to checkpointables named 'baz'.
  registry.add(BazHandler, 'baz')

  checkpointables_options = ocp.options.CheckpointablesOptions(
      registry=registry
  )
  with ocp.Context(checkpointables_options=checkpointables_options):
    ocp.save_checkpointables(...)

If a registered handler is scoped to a specific name (e.g.
`registry.add(BazHandler, 'baz')`), then this handler will always be
prioritized for saving or loading the checkpointable with that name, even if
the handler is not capable of saving/loading the checkpointable.

In the most common case, where a handler is not scoped to a specific name,
a given checkpointable (or abstract_checkpointable) will be resolved to a
handler returning True for `is_handleable` (or `is_abstract_handleable`),
respectively. If multiple handlers are usable, the first usable handler will be
returned. When loading, the handler type used for saving will be recorded in
the metadata, and will be used to resolve the handler, if a corresponding
handler is present in the registry.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, Sequence, Type, TypeVar

from absl import logging
from orbax.checkpoint.experimental.v1._src.handlers import types as handler_types

CheckpointableHandler = handler_types.CheckpointableHandler
RegistryEntry = tuple[Type[CheckpointableHandler], str | None]


def add_all(
    registry: CheckpointableHandlerRegistry,
    other_registry: CheckpointableHandlerRegistry,
) -> CheckpointableHandlerRegistry:
  """Adds all entries from `other_registry` to `registry`."""
  for handler, checkpointable in other_registry.get_all_entries():
    registry.add(handler, checkpointable)
  return registry


class CheckpointableHandlerRegistry(Protocol):
  """A registry defining a mapping from name to CheckpointableHandler.

  See module docstring for usage details.
  """

  def add(
      self,
      handler_type: Type[CheckpointableHandler],
      checkpointable: str | None = None,
  ) -> CheckpointableHandlerRegistry:
    """Adds an entry to the registry."""
    ...

  def get(
      self,
      checkpointable: str,
  ) -> Type[CheckpointableHandler]:
    """Gets the type of a `CheckpointableHandler` from the registry."""
    ...

  def has(
      self,
      checkpointable: str,
  ) -> bool:
    """Checks if an entry exists in the registry."""
    ...

  def get_all_entries(
      self,
  ) -> Sequence[RegistryEntry]:
    ...


class AlreadyExistsError(ValueError):
  """Raised when an entry already exists in the registry."""


class NoEntryError(KeyError):
  """Raised when no entry exists in the registry."""


class _DefaultCheckpointableHandlerRegistry(CheckpointableHandlerRegistry):
  """Default implementation of `CheckpointableHandlerRegistry`."""

  def __init__(
      self, other_registry: CheckpointableHandlerRegistry | None = None
  ):
    self._registry: list[RegistryEntry] = []

    # Initialize the registry with entries from other registry.
    if other_registry:
      add_all(self, other_registry)

  def add(
      self,
      handler_type: Type[CheckpointableHandler],
      checkpointable: str | None = None,
  ) -> CheckpointableHandlerRegistry:
    """Adds an entry to the registry.

    Args:
      handler_type: The handler type.
      checkpointable: The checkpointable name. If not-None, the registered
        handler will be scoped to that specific name. Otherwise, the handler
        will be available for any checkpointable name.

    Returns:
      The registry itself.

    Raises:
      AlreadyExistsError: If an entry for the given checkpointable name or
        handler type already exists in the registry.
      ValueError: If the handler is not default-constructible.
    """
    if not isinstance(handler_type, type):
      raise ValueError(
          f'The `handler_type` must be a type, but got {handler_type}.'
      )
    registered_handler_types = [
        handler_type for handler_type, _ in self.get_all_entries()
    ]
    if checkpointable:
      if self.has(checkpointable):
        raise AlreadyExistsError(
            f'Entry for checkpointable={checkpointable} already'
            ' exists in the registry.'
        )
    elif handler_type in registered_handler_types:
      raise AlreadyExistsError(
          f'Handler type {handler_type} already exists in the registry.'
      )
    self._registry.append((handler_type, checkpointable))
    return self

  def get(
      self,
      checkpointable: str,
  ) -> Type[CheckpointableHandler]:
    """Returns the handler for the given checkpointable name.

    Args:
      checkpointable: The checkpointable name.

    Returns:
      The handler for the given checkpointable name.

    Raises:
      NoEntryError: If no entry for the given checkpointable name exists in the
      registry.
    """
    for handler, checkpointable_name in self._registry:
      if checkpointable == checkpointable_name:
        return handler

    raise NoEntryError(
        f'No entry for checkpointable={checkpointable} in the registry.'
    )

  def has(
      self,
      checkpointable: str,
  ) -> bool:
    """Returns whether an entry for the given checkpointable name exists.

    Args:
      checkpointable: A checkpointable name.

    Returns:
      True if an entry for the given checkpointable name exists, False
      otherwise.
    """
    return any(
        checkpointable_name == checkpointable
        for _, checkpointable_name in self._registry
    )

  def get_all_entries(
      self,
  ) -> Sequence[RegistryEntry]:
    """Returns all entries in the registry."""
    return self._registry

  def __repr__(self):
    return f'_DefaultCheckpointableHandlerRegistry({self.get_all_entries()})'

  def __str__(self):
    return f'_DefaultCheckpointableHandlerRegistry({self.get_all_entries()})'


class ReadOnlyCheckpointableHandlerRegistry(CheckpointableHandlerRegistry):
  """Read-only implementation of `CheckpointableHandlerRegistry`."""

  def __init__(self, registry: CheckpointableHandlerRegistry):
    self._registry = registry

  def add(
      self,
      handler_type: Type[CheckpointableHandler],
      checkpointable: str | None = None,
  ) -> CheckpointableHandlerRegistry:
    raise NotImplementedError('Adding not implemented for read-only registry.')

  def get(
      self,
      checkpointable: str,
  ) -> Type[CheckpointableHandler]:
    return self._registry.get(checkpointable)

  def has(
      self,
      checkpointable: str,
  ) -> bool:
    return self._registry.has(checkpointable)

  def get_all_entries(
      self,
  ) -> Sequence[RegistryEntry]:
    return self._registry.get_all_entries()

  def __repr__(self):
    return f'ReadOnlyCheckpointableHandlerRegistry({self.get_all_entries()})'

  def __str__(self):
    return f'ReadOnlyCheckpointableHandlerRegistry({self.get_all_entries()})'


_GLOBAL_REGISTRY = _DefaultCheckpointableHandlerRegistry()


def global_registry() -> CheckpointableHandlerRegistry:
  """Returns the global registry."""
  return _GLOBAL_REGISTRY


def local_registry(
    other_registry: CheckpointableHandlerRegistry | None = None,
    *,
    include_global_registry: bool = True,
) -> CheckpointableHandlerRegistry:
  """Creates a local registry.

  Args:
    other_registry: An optional registry of handlers to include in the returned
      registry.
    include_global_registry: If true, includes globally-registered handlers in
      the returned registry by default.

  Returns:
    A local registry.
  """
  registry = _DefaultCheckpointableHandlerRegistry()
  if include_global_registry:
    registry = add_all(registry, global_registry())
  if other_registry:
    registry = add_all(registry, other_registry)
  return registry


CheckpointableHandlerType = TypeVar(
    'CheckpointableHandlerType', bound=Type[CheckpointableHandler]
)


def register_handler(
    cls: CheckpointableHandlerType,
) -> CheckpointableHandlerType:
  """Registers a :py:class:`.CheckpointableHandler` globally.

  The order in which handlers are registered matters. If multiple handlers
  could potentially be used to save or load, the one added most recently will be
  used.

  Usage::

    @ocp.handlers.register_handler
    class FooHandler(ocp.handlers.CheckpointableHandler[Foo, AbstractFoo]):
      ...

    ### OR ###

    ocp.handlers.register_handler(FooHandler)

  Args:
    cls: The handler class.

  Returns:
    The handler class.
  """
  _GLOBAL_REGISTRY.add(cls)
  return cls


def _construct_handler_instance(
    name: str,
    handler_type: Type[CheckpointableHandler],
) -> CheckpointableHandler:
  """Attempts to default-construct a handler type if possible."""
  assert isinstance(handler_type, type)
  try:
    return handler_type()
  except TypeError as e:
    raise ValueError(
        'The `CheckpointableHandler` resolved for'
        f' checkpointable={name} could not be'
        ' default-constructed. Please ensure the object is'
        ' default-constructible or provide a concrete instance.'
    ) from e


def _get_possible_handlers(
    registry: CheckpointableHandlerRegistry,
    is_handleable_fn: Callable[[CheckpointableHandler, Any], bool],
    checkpointable: Any | None,
    name: str,
) -> Sequence[CheckpointableHandler]:
  """Raises a NoEntryError if no possible handlers are found."""
  registry_entries = [
      (
          _construct_handler_instance(checkpointable_name, handler),
          checkpointable_name,
      )
      for handler, checkpointable_name in registry.get_all_entries()
  ]
  if checkpointable is None:
    # All handlers are potentially usable if checkpointable is not provided.
    possible_handlers = [
        handler
        for handler, checkpointable_name in registry_entries
        if checkpointable_name is None
    ]
  else:
    possible_handlers = [
        handler
        for handler, checkpointable_name in registry_entries
        if checkpointable_name is None
        and is_handleable_fn(handler, checkpointable)
    ]
  if not possible_handlers:
    available_handlers = [
        handler_type for handler_type, _ in registry.get_all_entries()
    ]
    error_msg = (
        f'Could not identify a valid handler for the checkpointable: "{name}"'
        f' and checkpointable type={type(checkpointable)}. Make sure to'
        ' register a `CheckpointableHandler` for the object using'
        ' `register_handler`, or by specifying a local registry'
        ' (`CheckpointablesOptions`). If a handler is already registered,'
        ' ensure that `is_handleable` correctly identifies the object as'
        f' handleable. The available handlers are: {available_handlers}'
    )
    raise NoEntryError(error_msg)
  return possible_handlers


def resolve_handler_for_save(
    registry: CheckpointableHandlerRegistry,
    checkpointable: Any,
    *,
    name: str,
) -> CheckpointableHandler:
  """Resolves a CheckpointableHandler for saving.

    1. If a name matching the provided checkpointable name is explicitly
       registered, return the corresponding handler.
    2. Resolve based on the `checkpointable` (using
      `CheckpointableHandler.is_handleable`).
    3. If multiple handlers are usable, return the *last* usable handler. This
       allows us to resolve the most recently-registered handler.

  Args:
    registry: The CheckpointableHandlerRegistry to search.
    checkpointable: A checkpointable to resolve.
    name: The name of the checkpointable.

  Raises:
    NoEntryError: If no compatible `CheckpointableHandler` can be found.

  Returns:
    A CheckpointableHandler instance.
  """
  # If explicitly registered, use that first.
  if registry.has(name):
    return _construct_handler_instance(name, registry.get(name))

  if checkpointable is None:
    raise ValueError('checkpointable must not be None for saving.')

  def is_handleable_fn(handler: CheckpointableHandler, ckpt: Any) -> bool:
    return handler.is_handleable(ckpt)

  possible_handlers = _get_possible_handlers(
      registry, is_handleable_fn, checkpointable, name
  )

  # Prefer the first handler in the absence of any other information.
  return possible_handlers[-1]


def resolve_handler_for_load(
    registry: CheckpointableHandlerRegistry,
    abstract_checkpointable: Any | None,
    *,
    name: str,
    handler_typestr: str,
) -> CheckpointableHandler:
  """Resolves a CheckpointableHandler for loading.

    1. If name is explicitly registered, return the handler.
    2. Resolve based on the `abstract_checkpointable` (using
      `CheckpointableHandler.is_abstract_handleable`).
    3. If `abstract_checkpointable` is None or not provided, all registered
      handlers not scoped to a specific item name are potentially usable.
    4. If multiple handlers are usable, return the handler with the matching
      typestr. If no matching typestr is found, then the handler used for saving
      may not be available now.
    4. Return the *last* usable handler. This allows us to resolve the most
       recently-registered handler.

  Raises:
    NoEntryError: If no compatible `CheckpointableHandler` can be found.

  Args:
    registry: The CheckpointableHandlerRegistry to search.
    abstract_checkpointable: An abstract checkpointable to resolve.
    name: The name of the checkpointable.
    handler_typestr: A CheckpointableHandler typestr to guide resolution.

  Returns:
    A CheckpointableHandler instance.
  """
  # If explicitly registered, use that first.
  if registry.has(name):
    return _construct_handler_instance(name, registry.get(name))

  def is_handleable_fn(
      handler: CheckpointableHandler, ckpt: Any
  ) -> bool | None:
    return handler.is_abstract_handleable(ckpt)

  possible_handlers = _get_possible_handlers(
      registry, is_handleable_fn, abstract_checkpointable, name
  )
  possible_handler_typestrs = [
      handler_types.typestr(type(handler)) for handler in possible_handlers
  ]

  try:
    idx = possible_handler_typestrs.index(handler_typestr)
    return possible_handlers[idx]
  except ValueError:
    logging.warning(
        'No handler found for typestr %s. The checkpointable may be restored'
        ' with different handler logic than was used for saving.',
        handler_typestr,
    )

  # Prefer the first handler in the absence of any other information.
  return possible_handlers[-1]
