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

"""Leaf Handler Registry."""

from typing import Any, Dict, Sequence, Tuple, Type

from absl import logging
import jax
import numpy as np
from orbax.checkpoint.experimental.v1._src.serialization import array_leaf_handler
from orbax.checkpoint.experimental.v1._src.serialization import numpy_leaf_handler
from orbax.checkpoint.experimental.v1._src.serialization import protocol_utils
from orbax.checkpoint.experimental.v1._src.serialization import scalar_leaf_handler
from orbax.checkpoint.experimental.v1._src.serialization import string_leaf_handler
from orbax.checkpoint.experimental.v1._src.serialization import types
import typing_extensions


# The standard type and abstract type to handler mapping.
# The type to abstract type pairs are well defined standard and users should
# rarely need to override the pair.
STANDARD_TYPE_AND_ABSTRACT_TYPE_TO_HANDLER = {
    (
        jax.Array,
        types.AbstractShardedArray,
    ): array_leaf_handler.ArrayLeafHandler,
    (
        np.ndarray,
        types.AbstractArray,
    ): numpy_leaf_handler.NumpyLeafHandler,
    (
        int,
        int,
    ): scalar_leaf_handler.ScalarLeafHandler,
    (
        float,
        float,
    ): scalar_leaf_handler.ScalarLeafHandler,
    (
        bytes,
        bytes,
    ): scalar_leaf_handler.ScalarLeafHandler,
    (
        str,
        str,
    ): string_leaf_handler.StringLeafHandler,
}


class BaseLeafHandlerRegistry:
  """Base Leaf Handler Registry implements the LeafHandlerRegistry Protocol.

  This registry maps concrete PyTree leaf types and abstract leaf types to
  their corresponding :py:class:`~.v1.types.LeafHandler` implementations.
  It providesdynamic handler resolution, meaning if a specific leaf type is not
  directly registered, the registry will attempt to resolve a handler registered
  to one of its base classes or protocols.

  Example:
    Registering and retrieving a custom handler for a leaf type::

      registry = BaseLeafHandlerRegistry()
      registry.add(
          leaf_type=np.ndarray,
          abstract_type=core.AbstractArray,
          handler_type=ArrayHandler
      )

      # Retrieves ArrayHandler directly
      handler = registry.get(np.ndarray)

      # Also resolves to ArrayHandler due to `issubclass` matching
      class CustomArray(np.ndarray): pass
      sub_handler = registry.get(CustomArray)
  """

  def __init__(self):
    self._leaf_type_registry: Dict[
        Type[Any], Type[types.LeafHandler[Any, Any]]
    ] = {}
    self._abstract_type_registry: Dict[
        Type[Any], Type[types.LeafHandler[Any, Any]]
    ] = {}

    # for easy look up for replacement
    self._handler_to_types: Dict[
        Type[types.LeafHandler[Any, Any]], Tuple[Type[Any], Type[Any]]
    ] = {}
    self._secondary_typestrs: Dict[
        Type[types.LeafHandler[Any, Any]], Sequence[str]
    ] = {}

  def _try_get(
      self, leaf_type: Type[types.Leaf]
  ) -> Type[types.LeafHandler[types.Leaf, Any]] | None:
    """Returns the handler registered for a given type, if available."""
    for registered_ty, handler_type in self._leaf_type_registry.items():
      if issubclass(leaf_type, registered_ty):
        return handler_type

    # no handler found
    return None

  def get(
      self, leaf_type: Type[types.Leaf]
  ) -> Type[types.LeafHandler[types.Leaf, Any]]:
    if (handler_type := self._try_get(leaf_type)) is None:
      raise ValueError(
          f'Unknown Leaf type: "{leaf_type}". Must register it with'
          ' LeafHandlerRegistry.'
      )

    return handler_type

  def _try_get_abstract(
      self,
      abstract_type: Type[types.AbstractLeaf],
  ) -> Type[types.LeafHandler[Any, types.AbstractLeaf]] | None:
    """Returns the handler registered for a given abstract type, if available."""
    for (
        registered_abstract_ty,
        handler_type,
    ) in self._abstract_type_registry.items():
      if typing_extensions.is_protocol(registered_abstract_ty):  # pytype: disable=not-supported-yet
        if protocol_utils.is_subclass_protocol(
            cls=abstract_type, protocol=registered_abstract_ty
        ):
          return handler_type
      elif issubclass(abstract_type, registered_abstract_ty):
        return handler_type

    # no handler found
    return None

  def get_abstract(
      self,
      abstract_type: Type[types.AbstractLeaf],
  ) -> Type[types.LeafHandler[Any, types.AbstractLeaf]]:
    if (handler_type := self._try_get_abstract(abstract_type)) is None:
      raise ValueError(
          f'Unknown AbstractLeaf type: "{abstract_type}". Must register it with'
          ' LeafHandlerRegistry.'
      )

    return handler_type

  def get_all(
      self,
  ) -> Sequence[types.LeafHandlerRegistryItem]:
    """Returns all registered handlers.

    Returns:
      A sequence of tuples, where each tuple contains the registered concrete
      leaf type, the abstract leaf type, and the corresponding handler type.
    """
    return [
        (
            leaf_type,
            abstract_type,
            handler_type,
        )
        for (leaf_type, handler_type), abstract_type in zip(
            self._leaf_type_registry.items(), self._abstract_type_registry
        )
    ]

  def add(
      self,
      leaf_type: Type[types.Leaf],
      abstract_type: Type[types.AbstractLeaf],
      handler_type: Type[types.LeafHandler[types.Leaf, types.AbstractLeaf]],
      override: bool = False,
      secondary_typestrs: Sequence[str] | None = None,
  ):
    """Adds a handler_type for a given leaf_type and abstract_type pair.

    Args:
      leaf_type: The concrete PyTree leaf type to register.
      abstract_type: The abstract representation of the leaf type.
      handler_type: The `LeafHandler` class responsible for this leaf type.
      override: If True, overwrites existing registrations for the given
        leaf or abstract types. Defaults to False.
      secondary_typestrs: A sequence of alternate handler typestrs that serve as
        secondary identifiers for the handler.

    Raises:
      ValueError: If the `leaf_type` or `abstract_type` is already registered
        and `override` is False. Also raised if the `abstract_type` is already
        registered with a fundamentally different handler type.
    """
    current_handler_type = self._try_get(leaf_type)
    current_abstract_handle_type = self._try_get_abstract(abstract_type)

    if not override and (current_handler_type or current_abstract_handle_type):
      raise ValueError(
          f'Leaf_type[{leaf_type}] or abstract_type[{abstract_type}] has'
          f' already registered, current_handler: {current_handler_type}, '
          f'current_abstract_handle_type: {current_abstract_handle_type}'
      )

    logging.vlog(
        1,
        'add: leaf_type[%s], abstract_type[%s], handler_type[%s],'
        ' current_handler[%s], current_abstract_handle_type[%s]',
        leaf_type,
        abstract_type,
        handler_type,
        current_handler_type,
        current_abstract_handle_type,
    )

    if current_handler_type and (
        current_abstract_handle_type
        and current_handler_type != current_abstract_handle_type
    ):
      raise ValueError(
          f'Abstract_type[{abstract_type}] has already registered with a'
          ' different type.'
      )
    elif current_handler_type and not current_abstract_handle_type:
      # need to remove the previous abstract type
      _, old_abstract_ty = self._handler_to_types.pop(current_handler_type)
      self._abstract_type_registry.pop(old_abstract_ty)

    # new type and abstract type pair
    self._leaf_type_registry[leaf_type] = handler_type
    self._abstract_type_registry[abstract_type] = handler_type
    self._handler_to_types[handler_type] = (leaf_type, abstract_type)
    # Allows for multiple handlers to be associated with the same leaf_type and
    # abstract_type pair, typically for backward compatibility.
    if secondary_typestrs is not None:
      self._secondary_typestrs[handler_type] = (
          secondary_typestrs
      )

  def is_handleable(self, leaf_type: Type[Any]) -> bool:
    """Returns True if the type is handleable.

    This checks if the provided concrete leaf type, or any of its base classes,
    has a specifically registered handler in the registry.

    Args:
      leaf_type: The concrete PyTree leaf type to check.

    Returns:
      True if a valid handler exists for the type, False otherwise.
    """
    return self._try_get(leaf_type) is not None

  def is_abstract_handleable(self, abstract_type: Type[Any]) -> bool:
    """Returns True if the abstract type is handlable.

    This checks if the provided abstract leaf type, or any of its matching base
    classes or protocols, has a registered handler in the registry.

    Args:
      abstract_type: The abstract PyTree leaf type to check.

    Returns:
      True if a valid handler exists for the abstract type, False otherwise.
    """
    return self._try_get_abstract(abstract_type) is not None

  def get_secondary_typestrs(
      self, handler_type: Type[types.LeafHandler[Any, Any]]
  ) -> Sequence[str]:
    return self._secondary_typestrs.get(handler_type, [])


class StandardLeafHandlerRegistry(BaseLeafHandlerRegistry):
  """Default Leaf Handler Registry.

  This registry is designed as the default implementation of
  LeafHandlerRegistry.  It also registers the handlers for all the standard
  types including jax.Array, np.ndarray, int, float, bytes, and np.number.

  Example:
    Instantiate the registry and immediately use it for standard types::

      from orbax.checkpoint.v1.serialization import
      StandardLeafHandlerRegistry
      import numpy as np

      registry = StandardLeafHandlerRegistry()

      # True, because numpy arrays are registered by default
      is_supported = registry.is_handleable(np.ndarray)
  """

  def __init__(self):
    super().__init__()
    for (
        ty,
        abstract_ty,
    ), handler_class in STANDARD_TYPE_AND_ABSTRACT_TYPE_TO_HANDLER.items():
      self.add(ty, abstract_ty, handler_class)
