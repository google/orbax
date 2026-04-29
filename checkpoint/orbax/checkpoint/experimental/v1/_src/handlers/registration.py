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

"""Registry for :py:class:`~.v1.handlers.CheckpointableHandler`.

A :py:class:`~.v1.handlers.CheckpointableHandler` defines logic needed to save
and restore a "checkpointable" object. Once defined, the handler must be
registered either globally or locally.

To register the handler globally, use
:py:func:`~.v1._src.handlers.registration.register_handler`.
Alternatively,
create a local registry confined to a specific scope by using
:py:func:`~.v1._src.handlers.registration.local_registry`
(note that globally-registered handlers are included in this
registry by default).

For example::

  ocp.handlers.register_handler(FooHandler)

  registry = ocp.handlers.local_registry()
  # `registry` already contains FooHandler.
  registry.add(BarHandler)
  # Scope this handler specifically to checkpointables named 'baz'.
  registry.add(BazHandler, 'baz')
  # Secondary typestrs provide a way to map legacy handler typestr identifiers
  # to a new v1 handler class.
  registry.add(BazHandler, secondary_typestrs=['OldBazHandlerTypestr'])

  checkpointables_options = ocp.options.CheckpointablesOptions(
      registry=registry
  )
  with ocp.Context(checkpointables_options=checkpointables_options):
    ocp.save_checkpointables(...)

Handler resolution for saving/loading follows this logic:

    1. If a registered handler is scoped to a specific name
      (e.g. `registry.add(BazHandler, 'baz')`), then this handler will always
      be prioritized for saving or loading the checkpointable with that name,
      even if the handler is not capable of saving/loading the checkpointable.
    2. In the absence of an explicit name match, the registry filters for
      handlers returning `True` for `is_handleable` (during save) or
      `is_abstract_handleable` (during load).
    3. [Pertains to loading only] The handler type used for saving will be
      recorded in the metadata, and will be used to resolve the handler, if a
      corresponding handler is present in the registry. If not, scan the
      secondary typestrs of registered handlers for a match.
    4. If no metadata match is found (or during saving), the most recently
       registered capable handler is returned.
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
  for handler, name in other_registry.get_all_entries():
    registry.add(
        handler,
        checkpointable_name=name,
        secondary_typestrs=other_registry.get_secondary_typestrs(
            handler
        ),
    )
  return registry


class CheckpointableHandlerRegistry(Protocol):
  """A registry for :py:class:`~.v1.handlers.CheckpointableHandler` instances.

  This protocol defines the core interface for adding, retrieving, and checking
  for the existence of handlers that manage the saving and loading of specific
  checkpointable types within the Orbax framework.

  As a `Protocol`, it serves as a structural type definition. Any class that
  implements these four methods (`add`, `get`, `has`, and `get_all_entries`)
  with the correct signatures is considered a valid registry by static type
  checkers, without needing to explicitly inherit from this class.

  Example:
    Implementing a custom registry that fulfills this protocol. Note that
    explicit inheritance is not required for type checkers to recognize it::

      from typing import Type, Sequence, Tuple, Optional
      from orbax.checkpoint.v1 import handlers

      class MyCustomRegistry:
        def __init__(self) -> None:
          self._entries: list[
              Tuple[Type[handlers.CheckpointableHandler], Optional[str]]
          ] = []

        def add(
            self,
            handler_type: Type[handlers.CheckpointableHandler],
            checkpointable: Optional[str] = None,
        ) -> 'MyCustomRegistry':
          self._entries.append((handler_type, checkpointable))
          return self

        def get(
            self, checkpointable: str
        ) -> Type[handlers.CheckpointableHandler]:
          for h_type, name in self._entries:
            if name == checkpointable:
              return h_type
          raise KeyError(f'Not found: {checkpointable}')

        def has(self, checkpointable: str) -> bool:
          return any(name == checkpointable for _, name in self._entries)

        def get_all_entries(
            self,
        ) -> Sequence[
            Tuple[Type[handlers.CheckpointableHandler], Optional[str]]
        ]:
          return self._entries

  Methods:
    add(handler_type, checkpointable=None): Adds an entry to the registry.
      Returns the registry instance to allow method chaining.
    get(checkpointable): Gets the type of a `CheckpointableHandler` from the
      registry by its associated checkpointable name.
    has(checkpointable): Checks if an entry exists in the registry for the given
      checkpointable name. Returns True if it exists, False otherwise.
    get_all_entries(): Returns a sequence of all registered entries as
      (handler_type, name) tuples.
  """

  def add(
      self,
      handler_type: Type[CheckpointableHandler],
      *,
      checkpointable_name: str | None = None,
      secondary_typestrs: Sequence[str] | None = None,
  ) -> CheckpointableHandlerRegistry:
    """Adds an entry to the registry."""
    ...

  def get(
      self,
      checkpointable_name: str,
  ) -> Type[CheckpointableHandler]:
    """Gets the type of a :py:class:`~.v1.handlers.CheckpointableHandler` from the registry."""
    ...

  def has(
      self,
      checkpointable_name: str,
  ) -> bool:
    """Checks if an entry exists in the registry."""
    ...

  def get_all_entries(
      self,
  ) -> Sequence[RegistryEntry]:
    ...

  def get_secondary_typestrs(
      self,
      handler_type: Type[CheckpointableHandler],
  ) -> Sequence[str]:
    """Returns all secondary typestrs associated with the given handler type."""
    ...


class AlreadyExistsError(ValueError):
  """An entry already exists in the registry."""


class NoEntryError(KeyError):
  """No entry exists in the registry."""


class NotHandleableError(ValueError):
  """A checkpointable is not handleable by a handler."""


class _DefaultCheckpointableHandlerRegistry(CheckpointableHandlerRegistry):
  """Default implementation of :py:class:`~.v1.handlers.registration.CheckpointableHandlerRegistry`."""

  def __init__(
      self, other_registry: CheckpointableHandlerRegistry | None = None
  ):
    self._registry: list[RegistryEntry] = []
    self._secondary_typestrs: dict[
        Type[CheckpointableHandler], Sequence[str]
    ] = {}

    # Initialize the registry with entries from other registry.
    if other_registry:
      add_all(self, other_registry)

  def add(
      self,
      handler_type: Type[CheckpointableHandler],
      *,
      checkpointable_name: str | None = None,
      secondary_typestrs: Sequence[str] | None = None,
  ) -> CheckpointableHandlerRegistry:
    """Adds an entry to the registry.

    Adds a primary handler_type to the registry with an optional checkpointable
    name and an optional sequence of secondary typestrs that can be used to
    identify the handler.

    Note: We only guarantee unique handler type entries in the registry and do
    not explicitly prevent a primary handler type from being registered and its
    typestr being used as a secondary_typestr for itself or another handler.

    Args:
      handler_type: The handler type.
      checkpointable_name: The checkpointable name. If not-None, the registered
        handler will be scoped to that specific name. Otherwise, the handler
        will be available for any checkpointable name.
      secondary_typestrs: A sequence of alternate typestrs that serve as
        secondary identifiers for the handler.

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
    if checkpointable_name:
      if self.has(checkpointable_name):
        raise AlreadyExistsError(
            f'Entry for checkpointable_name={checkpointable_name} already'
            ' exists in the registry.'
        )
    elif handler_type in registered_handler_types:
      raise AlreadyExistsError(
          f'Handler type {handler_type} already exists in the registry.'
      )
    self._registry.append((handler_type, checkpointable_name))
    if secondary_typestrs is not None:
      self._secondary_typestrs[handler_type] = secondary_typestrs
    return self

  def get(
      self,
      checkpointable_name: str,
  ) -> Type[CheckpointableHandler]:
    """Returns the handler for the given checkpointable name.

    Args:
      checkpointable_name: The checkpointable name.

    Returns:
      The handler for the given checkpointable name.

    Raises:
      NoEntryError: If no entry for the given checkpointable name exists in the
      registry.
    """
    for handler, name in self._registry:
      if checkpointable_name == name:
        return handler

    raise NoEntryError(
        f'No entry for checkpointable_name={checkpointable_name} in the'
        ' registry.'
    )

  def has(
      self,
      checkpointable_name: str,
  ) -> bool:
    """Returns whether an entry for the given checkpointable name exists.

    Args:
      checkpointable_name: A checkpointable name.

    Returns:
      True if an entry for the given checkpointable name exists, False
      otherwise.
    """
    return any(
        name == checkpointable_name
        for _, name in self._registry
    )

  def get_all_entries(
      self,
  ) -> Sequence[RegistryEntry]:
    """Returns all entries in the registry."""
    return self._registry

  def get_secondary_typestrs(
      self,
      handler_type: Type[CheckpointableHandler],
  ) -> Sequence[str]:
    """Returns all secondary typestrs associated with the given handler type."""
    return self._secondary_typestrs.get(handler_type, [])

  def __repr__(self):
    return f'_DefaultCheckpointableHandlerRegistry({self.get_all_entries()})'

  def __str__(self):
    return f'_DefaultCheckpointableHandlerRegistry({self.get_all_entries()})'


class ReadOnlyCheckpointableHandlerRegistry(CheckpointableHandlerRegistry):
  """Read-only implementation of :py:class:`~.v1.handlers.registration.CheckpointableHandlerRegistry`."""

  def __init__(self, registry: CheckpointableHandlerRegistry):
    self._registry = registry

  def add(
      self,
      handler_type: Type[CheckpointableHandler],
      *,
      checkpointable_name: str | None = None,
      secondary_typestrs: Sequence[str] | None = None,
  ) -> CheckpointableHandlerRegistry:
    raise NotImplementedError('Adding not implemented for read-only registry.')

  def get(
      self,
      checkpointable_name: str,
  ) -> Type[CheckpointableHandler]:
    return self._registry.get(checkpointable_name)

  def has(
      self,
      checkpointable_name: str,
  ) -> bool:
    return self._registry.has(checkpointable_name)

  def get_all_entries(
      self,
  ) -> Sequence[RegistryEntry]:
    return self._registry.get_all_entries()

  def get_secondary_typestrs(
      self,
      handler_type: Type[CheckpointableHandler],
  ) -> Sequence[str]:
    return self._registry.get_secondary_typestrs(handler_type)

  def __repr__(self):
    return f'ReadOnlyCheckpointableHandlerRegistry({self.get_all_entries()})'

  def __str__(self):
    return f'ReadOnlyCheckpointableHandlerRegistry({self.get_all_entries()})'


_GLOBAL_REGISTRY = _DefaultCheckpointableHandlerRegistry()


def global_registry() -> CheckpointableHandlerRegistry:
  """Returns the global registry.

  The global registry serves as the default, singleton storage for all
  handlers registered throughout the application's lifecycle via
  `register_handler`.

  Example:
    Retrieve the global registry to inspect available handlers::

      from orbax.checkpoint.v1 import handlers

      # Fetch the singleton global registry
      registry = handlers.global_registry()

      # Check if a specific handler name is registered globally
      is_registered = registry.has("my_custom_model_handler")

  Returns:
    CheckpointableHandlerRegistry: The global singleton registry instance.
  """
  return _GLOBAL_REGISTRY


def local_registry(
    other_registry: CheckpointableHandlerRegistry | None = None,
    *,
    include_global_registry: bool = True,
) -> CheckpointableHandlerRegistry:
  """Creates a local registry.

  This function builds a new registry by optionally combining the existing
  global registry with a provided custom registry. It is highly useful for
  overriding handlers for a specific checkpointer operation without mutating
  the global state.

  Example:
    Create a registry with custom handlers, potentially including global ones::

      from orbax.checkpoint.v1 import handlers

      class MyHandler(handlers.CheckpointableHandler):
        pass

      # Create a registry and add a handler. By default, it includes
      # globally-registered handlers.
      my_registry = handlers.local_registry()
      my_registry.add(MyHandler)

      # To start with an empty registry, use:
      # my_registry = handlers.local_registry(include_global_registry=False)

  Args:
    other_registry: An optional registry of handlers to include in the returned
      registry.
    include_global_registry: If True, includes globally-registered handlers in
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
    *,
    checkpointable_name: str | None = None,
    secondary_typestrs: Sequence[str] | None = None,
) -> CheckpointableHandlerType:
  """Registers a :py:class:`~.v1.handlers.CheckpointableHandler` globally.

  The order in which handlers are registered strictly matters. If multiple
  handlers could potentially be used to save or load an object (i.e., are
  capable of handling the checkpointable according to `is_handleable`/
  `is_abstract_handleable` for `save`/`load`, respectively), the framework
  resolves them in Last-In, First-Out (LIFO) order. This means the handler
  added most recently will be selected.

  Example:
    Registering a custom handler using a direct function call.
    Note the import path from the v1 namespace::

      from orbax.checkpoint.v1 import handlers

      class BarHandler(handlers.CheckpointableHandler):
        pass

      handlers.register_handler(BarHandler)

  Args:
    cls: The handler class to register globally.
    checkpointable_name: The checkpointable name. If not-None, the registered
      handler will be scoped to that specific name. Otherwise, the handler
      will be available for any checkpointable name.
    secondary_typestrs: A sequence of alternate handler typestrs that serve as
      secondary identifiers for the handler.

  Returns:
    The handler class.
  """
  _GLOBAL_REGISTRY.add(
      cls,
      checkpointable_name=checkpointable_name,
      secondary_typestrs=secondary_typestrs,
  )
  return cls


def _construct_handler_instance(
    name: str | None,
    handler_type: Type[CheckpointableHandler],
) -> CheckpointableHandler:
  """Attempts to default-construct a handler type if possible."""
  assert isinstance(handler_type, type)
  try:
    return handler_type()
  except TypeError as e:
    raise ValueError(
        'The :py:class:`~.v1.handlers.CheckpointableHandler`'
        f' resolved for checkpointable={name} could not be default-constructed.'
        ' Please ensure the object is default-constructible or provide a'
        ' concrete instance.'
    ) from e


def _get_possible_handlers(
    registry: CheckpointableHandlerRegistry,
    is_handleable: Callable[[CheckpointableHandler, Any], bool],
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
        and is_handleable(handler, checkpointable)
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


def get_registered_handler_by_name(
    registry: CheckpointableHandlerRegistry,
    name: str,
) -> CheckpointableHandler | None:
  """Returns the handler for the given name if registered."""
  if registry.has(name):
    return _construct_handler_instance(name, registry.get(name))
  return None


def resolve_handler_for_save(
    registry: CheckpointableHandlerRegistry,
    checkpointable: Any,
    *,
    name: str,
) -> CheckpointableHandler:
  """Resolves a :py:class:`~.v1.handlers.CheckpointableHandler` for saving.

    1. If a name matching the provided checkpointable name is explicitly
       registered, return the corresponding handler.
    2. Resolve based on the `checkpointable` (using
      :py:meth:`~.v1._src.handlers.types.CheckpointableHandler.is_handleable`).
    3. If multiple handlers are usable, return the *last* usable handler. This
       allows us to resolve the most recently-registered handler.

  Args:
    registry: The
      :py:class:`~.v1.handlers.registration.CheckpointableHandlerRegistry` to
      search.
    checkpointable: A checkpointable to resolve.
    name: The name of the checkpointable.

  Returns:
    A :py:class:`~.v1.handlers.CheckpointableHandler` instance.

  Raises:
    ValueError: If the checkpointable is None.
    NoEntryError: If no compatible
      :py:class:`~.v1.handlers.CheckpointableHandler` can be found.
  """
  # If explicitly registered, use that first.
  if registry.has(name):
    return _construct_handler_instance(name, registry.get(name))

  if checkpointable is None:
    raise ValueError('checkpointable must not be None for saving.')

  def is_handleable(handler: CheckpointableHandler, ckpt: Any) -> bool:
    return handler.is_handleable(ckpt)

  possible_handlers = _get_possible_handlers(
      registry, is_handleable, checkpointable, name
  )

  # Prefer the last handler in the absence of any other information.
  return possible_handlers[-1]


def resolve_handler_for_load(
    registry: CheckpointableHandlerRegistry,
    abstract_checkpointable: Any | None,
    *,
    name: str,
    handler_typestr: str | None = None,
) -> CheckpointableHandler:
  """Resolves a :py:class:`~.v1.handlers.CheckpointableHandler` for loading.

    1. If name is explicitly registered, return the handler.
    2. Resolve based on the `abstract_checkpointable` (using
      :py:meth:`~.v1._src.handlers.types.CheckpointableHandler.is_abstract_handleable`).
    3. If `abstract_checkpointable` is None or not provided, all registered
      handlers not scoped to a specific item name are potentially usable.
    4. If multiple handlers are usable, return the handler with the matching
      typestr. If no matching typestr is found, then the handler used for saving
      may not be available now.
    5. Return the *last* usable handler. This allows us to resolve the most
      recently-registered handler, unless abstract_checkpointable is None, in
      which case raise a NoEntryError.

  Args:
    registry: The
      :py:class:`~.v1.handlers.registration.CheckpointableHandlerRegistry` to
      search.
    abstract_checkpointable: An abstract checkpointable to resolve.
    name: The name of the checkpointable.
    handler_typestr: A :py:class:`~.v1.handlers.CheckpointableHandler` typestr
      to guide resolution. We allow a None value for handler_typestr as its
      possible to find the last registered handler given a specified
      abstract_checkpointable.

  Returns:
    A :py:class:`~.v1.handlers.CheckpointableHandler` instance.

  Raises:
    NoEntryError: If no compatible
    :py:class:`~.v1.handlers.CheckpointableHandler`
    can be found.
  """
  # If explicitly registered, use that first.
  if registry.has(name):
    return _construct_handler_instance(name, registry.get(name))

  def is_handleable(handler: CheckpointableHandler, ckpt: Any) -> bool | None:
    return handler.is_abstract_handleable(ckpt)

  possible_handlers = _get_possible_handlers(
      registry, is_handleable, abstract_checkpointable, name
  )
  possible_handler_typestrs = [
      handler_types.typestr(type(handler)) for handler in possible_handlers
  ]

  if handler_typestr:
    if handler_typestr in possible_handler_typestrs:
      idx = possible_handler_typestrs.index(handler_typestr)
      return possible_handlers[idx]
    # Attempt to find a handler with a matching secondary typestr.
    for i in reversed(range(len(possible_handlers))):
      if handler_typestr in registry.get_secondary_typestrs(
          type(possible_handlers[i])
      ):
        return possible_handlers[i]
    logging.warning(
        'No handler found for typestr %s (or its converted form). The '
        'checkpointable may be restored with different handler logic '
        'than was used for saving.',
        handler_typestr,
    )

  if abstract_checkpointable:
    # Prefer the last handler in the absence of any other information.
    return possible_handlers[-1]

  raise NoEntryError(
      f'No entry for checkpointable={name} in the registry, using'
      f' handler_typestr={handler_typestr} and'
      f' abstract_checkpointable={abstract_checkpointable}. Registry contents:'
      f' {registry.get_all_entries()}'
  )
