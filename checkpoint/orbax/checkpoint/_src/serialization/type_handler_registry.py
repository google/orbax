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

"""Handles registration of TypeHandlers."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from absl import logging
import jax
import numpy as np
from orbax.checkpoint._src.metadata import array_metadata_store as array_metadata_store_lib
from orbax.checkpoint._src.metadata import empty_values
from orbax.checkpoint._src.metadata import pytree_metadata_options as pytree_metadata_options_lib
from orbax.checkpoint._src.serialization import type_handlers
from orbax.checkpoint._src.serialization import types


def get_param_typestr(
    value: Any,
    registry: types.TypeHandlerRegistry,
    pytree_metadata_options: pytree_metadata_options_lib.PyTreeMetadataOptions,
) -> str:
  """Retrieves the typestr for a given value."""
  if empty_values.is_supported_empty_value(value, pytree_metadata_options):
    typestr = empty_values.get_empty_value_typestr(
        value, pytree_metadata_options
    )
  else:
    try:
      handler = registry.get(type(value))
      typestr = handler.typestr()
    except ValueError:
      # Not an error because users' training states often have a bunch of
      # random unserializable objects in them (empty states, optimizer
      # objects, etc.). An error occurring due to a missing TypeHandler
      # will be surfaced elsewhere.
      typestr = empty_values.RESTORE_TYPE_NONE
  return typestr


class _TypeHandlerRegistryImpl(types.TypeHandlerRegistry):
  """The implementation for TypeHandlerRegistry."""

  def __init__(self, *handlers: Tuple[Any, types.TypeHandler]):
    """Create a type registry.

    Args:
      *handlers: an optional list of handlers to initialize with.
    """
    self._type_registry: List[
        Tuple[Callable[[Any], bool], types.TypeHandler]
    ] = []
    self._typestr_registry: Dict[str, types.TypeHandler] = {}
    if handlers:
      for ty, h in handlers:
        self.add(ty, h, override=True, ignore_warnings=True)

  def add(
      self,
      ty: Any,
      handler: types.TypeHandler,
      func: Optional[Callable[[Any], bool]] = None,
      override: bool = False,
      ignore_warnings: bool = False,
  ):
    if func is None:
      func = lambda t: issubclass(t, ty)

    existing_handler_idx = None
    for i, (f, _) in enumerate(self._type_registry):
      if f(ty):
        existing_handler_idx = i
        # Ignore the possibility for subsequent matches, as these will not be
        # used anyway.
        break

    if existing_handler_idx is None:
      if handler.typestr() in self._typestr_registry:
        if override:
          if not ignore_warnings:
            logging.warning(
                'Type handler registry overriding type "%s" collision on %s',
                ty,
                handler.typestr(),
            )
        else:
          raise ValueError(
              f'Type "{ty}" has a `typestr` ("{handler.typestr()}") which'
              ' collides with that of an existing TypeHandler.'
          )
      self._type_registry.append((func, handler))
      self._typestr_registry[handler.typestr()] = handler
    elif override:
      if not ignore_warnings:
        logging.warning(
            'Type handler registry type "%s" overriding %s',
            ty,
            handler.typestr(),
        )
      self._type_registry[existing_handler_idx] = (func, handler)
      self._typestr_registry[handler.typestr()] = handler
    else:
      raise ValueError(f'A TypeHandler for "{ty}" is already registered.')

  def get(self, ty: Any) -> types.TypeHandler:
    if isinstance(ty, str):
      if ty in self._typestr_registry:
        return self._typestr_registry[ty]
    else:
      for func, handler in self._type_registry:
        if func(ty):
          return handler
    raise ValueError(f'Unknown type: "{ty}". Must register a TypeHandler.')

  def has(self, ty: Any) -> bool:
    try:
      self.get(ty)
      return True
    except ValueError:
      return False


def create_type_handler_registry(
    *handlers: Tuple[Any, types.TypeHandler]
) -> types.TypeHandlerRegistry:
  """Create a type registry.

  Args:
    *handlers: optional pairs of `(<type>, <handler>)` to initialize the
      registry with.

  Returns:
    A TypeHandlerRegistry instance with only the specified handlers.
  """
  return _TypeHandlerRegistryImpl(*handlers)


_DEFAULT_TYPE_HANDLERS = tuple([
    (int, type_handlers.ScalarHandler()),
    (float, type_handlers.ScalarHandler()),
    (bytes, type_handlers.ScalarHandler()),
    (np.number, type_handlers.ScalarHandler()),
    (np.ndarray, type_handlers.NumpyHandler()),
    (
        jax.Array,
        type_handlers.ArrayHandler(
            array_metadata_store=array_metadata_store_lib.Store()
        ),
    ),
    (str, type_handlers.StringHandler()),
    (type(type_handlers.PLACEHOLDER), type_handlers.PlaceholderHandler()),
])


GLOBAL_TYPE_HANDLER_REGISTRY = create_type_handler_registry(
    *_DEFAULT_TYPE_HANDLERS
)


def register_type_handler(
    ty: Any,
    handler: types.TypeHandler,
    func: Optional[Callable[[Any], bool]] = None,
    override: bool = False,
):
  """Registers a type for serialization/deserialization with a given handler.

  Note that it is possible for a type to match multiple different entries in
  the registry, each with a different handler. In this case, only the first
  match is used.

  Args:
    ty: A type to register.
    handler: a TypeHandler capable of reading and writing parameters of type
      `ty`.
    func: A function that accepts a type and returns True if the type should be
      handled by the provided TypeHandler. If this parameter is not specified,
      defaults to `lambda t: issubclass(t, ty)`.
    override: if True, will override an existing mapping of type to handler.

  Raises:
    ValueError if a type is already registered and override is False.
  """
  GLOBAL_TYPE_HANDLER_REGISTRY.add(ty, handler, func, override)


def get_type_handler(ty: Any) -> types.TypeHandler:
  """Returns the handler registered for a given type, if available."""
  return GLOBAL_TYPE_HANDLER_REGISTRY.get(ty)


def has_type_handler(ty: Any) -> bool:
  """Returns if there is a handler registered for a given type."""
  return GLOBAL_TYPE_HANDLER_REGISTRY.has(ty)


def supported_types() -> list[Any]:
  """Returns the default list of supported types."""
  return [ty for ty, _ in _DEFAULT_TYPE_HANDLERS]


def register_standard_handlers_with_options(**kwargs):
  """Re-registers a select set of handlers with the given options.

  This is intended to override options en masse for the standard numeric
  TypeHandlers and their corresponding types (scalars, numpy arrays and
  jax.Arrays).

  Args:
    **kwargs: keyword arguments to pass to each of the standard handlers.
  """
  # TODO(b/314258967): clean those up.
  del kwargs['use_ocdbt'], kwargs['ts_context']
  GLOBAL_TYPE_HANDLER_REGISTRY.add(
      int, type_handlers.ScalarHandler(**kwargs), override=True
  )
  GLOBAL_TYPE_HANDLER_REGISTRY.add(
      float, type_handlers.ScalarHandler(**kwargs), override=True
  )
  GLOBAL_TYPE_HANDLER_REGISTRY.add(
      bytes, type_handlers.ScalarHandler(**kwargs), override=True
  )
  GLOBAL_TYPE_HANDLER_REGISTRY.add(
      np.number, type_handlers.ScalarHandler(**kwargs), override=True
  )
  GLOBAL_TYPE_HANDLER_REGISTRY.add(
      np.ndarray, type_handlers.NumpyHandler(**kwargs), override=True
  )
  GLOBAL_TYPE_HANDLER_REGISTRY.add(
      jax.Array, type_handlers.ArrayHandler(**kwargs), override=True
  )
