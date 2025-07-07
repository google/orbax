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

"""Leaf Handler Registry."""

from typing import Any, Dict, Sequence, Tuple, Type

from absl import logging
import jax
import numpy as np
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.serialization import array_leaf_handler
from orbax.checkpoint.experimental.v1._src.serialization import numpy_leaf_handler
from orbax.checkpoint.experimental.v1._src.serialization import protocol_utils
from orbax.checkpoint.experimental.v1._src.serialization import scalar_leaf_handler
from orbax.checkpoint.experimental.v1._src.serialization import types
import typing_extensions

# The standard type and abstract type to handler mapping.
# The type to abstract type pairs are well defined standard and users should
# rarely need to override the pair.
STANDARD_TYPE_AND_ABSTRACT_TYPE_TO_HANDLER = {
    (
        jax.Array,
        array_leaf_handler.AbstractArray,
    ): array_leaf_handler.ArrayLeafHandler,
    (
        np.ndarray,
        numpy_leaf_handler.AbstractNumpy,
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
        np.number,
        np.number,
    ): scalar_leaf_handler.ScalarLeafHandler,
}


class BaseLeafHandlerRegistry:
  """Base Leaf Handler Registry that implements the LeafHandlerRegistry Protocol."""

  def __init__(self, context: context_lib.Context | None = None):
    self._type_registry: Dict[Type[Any], types.LeafHandler[Any, Any]] = {}
    self._abstract_type_registry: Dict[
        Type[Any], types.LeafHandler[Any, Any]
    ] = {}

    # for easy look up for replacement
    self._handler_to_types: Dict[
        types.LeafHandler[Any, Any], Tuple[Type[Any], Type[Any]]
    ] = {}
    self._context = context_lib.get_context(context)

  def _try_get(
      self, ty: Type[types.Leaf]
  ) -> types.LeafHandler[types.Leaf, Any] | None:
    """Returns the handler registered for a given type, if available."""
    for registered_ty, handler in self._type_registry.items():
      if issubclass(ty, registered_ty):
        return handler

    # no handler found
    return None

  def get(
      self, ty: Type[types.Leaf]
  ) -> types.LeafHandler[types.Leaf, Any] | None:
    if (handler := self._try_get(ty)) is None:
      raise ValueError(
          f'Unknown Leaf type: "{ty}". Must register it with'
          ' LeafHandlerRegistry.'
      )

    return handler

  def _try_get_abstract(
      self, abstract_ty: Type[types.AbstractLeaf]
  ) -> types.LeafHandler[Any, types.AbstractLeaf] | None:
    """Returns the handler registered for a given abstract type, if available."""
    for registered_abstract_ty, handler in self._abstract_type_registry.items():
      if typing_extensions.is_protocol(registered_abstract_ty):  # pytype: disable=not-supported-yet
        if protocol_utils.is_subclass_protocol(
            cls=abstract_ty, protocol=registered_abstract_ty
        ):
          return handler
      elif issubclass(abstract_ty, registered_abstract_ty):
        return handler

    # no handler found
    return None

  def get_abstract(
      self, abstract_ty: Type[types.AbstractLeaf]
  ) -> types.LeafHandler[Any, types.AbstractLeaf]:
    if (handler := self._try_get_abstract(abstract_ty)) is None:
      raise ValueError(
          f'Unknown AbstractLeaf type: "{abstract_ty}". Must register it with'
          ' LeafHandlerRegistry.'
      )

    return handler

  def get_all(
      self,
  ) -> Sequence[types.LeafHandlerRegistryItem]:
    """Returns all registered handlers."""
    ret = []

    for (ty, handler), abstract_ty in zip(
        self._type_registry.items(), self._abstract_type_registry
    ):
      ret.append((ty, abstract_ty, handler))
    return ret

  def add(
      self,
      ty: Type[types.Leaf],
      abstract_ty: Type[types.AbstractLeaf],
      handler: types.LeafHandler[types.Leaf, types.AbstractLeaf],
      override: bool = False,
  ):
    """Adds a handler for a given type and abstract type pair."""
    current_handler = self._try_get(ty)
    handler_abstract_ty = self._try_get_abstract(abstract_ty)

    if not override and (current_handler or handler_abstract_ty):
      raise ValueError(
          f'Type[{ty}] or Abstract_ty[{abstract_ty}] has already registered.'
      )

    logging.vlog(
        1,
        'add: ty[%s], abstract_ty[%s], current_handler[%s],'
        ' handler_abstract_ty[%s]',
        ty,
        abstract_ty,
        current_handler,
        handler_abstract_ty,
    )

    if current_handler and (
        handler_abstract_ty and current_handler != handler_abstract_ty
    ):
      raise ValueError(
          f'Abstract_ty[{abstract_ty}] has already registered with a'
          ' different type.'
      )
    elif current_handler and not handler_abstract_ty:
      # need to remove the previous abstract type
      _, old_abstract_ty = self._handler_to_types.pop(current_handler)
      self._abstract_type_registry.pop(old_abstract_ty)

    # new type and abstract type pair
    self._type_registry[ty] = handler
    self._abstract_type_registry[abstract_ty] = handler
    self._handler_to_types[handler] = (ty, abstract_ty)

  def is_handleable(self, ty: Type[Any]) -> bool:
    """Returns True if the type is handleable."""
    return self.get(ty) is not None

  def is_abstract_handleable(self, abstract_ty: Type[Any]) -> bool:
    """Returns True if the abstract type is handlable."""
    return self.get_abstract(abstract_ty) is not None


class StandardLeafHandlerRegistry(BaseLeafHandlerRegistry):
  """Default Leaf Handler Registry.

  This registry is designed as the default implementation of
  LeafHandlerRegistry. It also registers the handlers for all the standard types
  including jax.Array, np.ndarray, int, float, bytes, and np.number.
  """

  def __init__(self, context: context_lib.Context | None = None):
    super().__init__(context)

    for (
        ty,
        abstract_ty,
    ), handler_class in STANDARD_TYPE_AND_ABSTRACT_TYPE_TO_HANDLER.items():
      self.add(ty, abstract_ty, handler_class(context=context))
