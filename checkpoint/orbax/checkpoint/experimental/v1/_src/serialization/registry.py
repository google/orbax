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

import dataclasses
from typing import Any, Sequence, Type

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


# The standard type, abstract type, and optional typestrs to handler mapping.
# The type to abstract type pairs are well defined standard and users should
# rarely need to override the pair.
STANDARD_TYPE_AND_ABSTRACT_TYPE_AND_TYPESTR_TO_HANDLER = [
    (
        jax.Array,
        types.AbstractShardedArray,
        ['jax.Array'],
        array_leaf_handler.ArrayLeafHandler,
    ),
    (
        np.ndarray,
        types.AbstractArray,
        ['np.ndarray'],
        numpy_leaf_handler.NumpyLeafHandler,
    ),
    (
        int,
        int,
        ['scalar'],
        scalar_leaf_handler.ScalarLeafHandler,
    ),
    (
        float,
        float,
        ['scalar'],
        scalar_leaf_handler.ScalarLeafHandler,
    ),
    (
        bytes,
        bytes,
        ['scalar'],
        scalar_leaf_handler.ScalarLeafHandler,
    ),
    (
        str,
        str,
        ['string'],
        string_leaf_handler.StringLeafHandler,
    ),
]


@dataclasses.dataclass
class _Registration:
  leaf_type: Type[Any]
  abstract_type: Type[Any]
  handler_type: Type[types.LeafHandler[Any, Any]]
  secondary_typestrs: Sequence[str] | None


class BaseLeafHandlerRegistry:
  """Base Leaf Handler Registry that implements the LeafHandlerRegistry Protocol."""

  def __init__(self):
    # Flat history for exact pairing and get_all().
    self._entries: list[_Registration] = []

    # Sorted [Generic -> Specific] pairs of: (leaf_type, handler_type)
    self._leaf_type_registry: list[
        tuple[Type[Any], Type[types.LeafHandler[Any, Any]]]
    ] = []

    # Sorted [Generic -> Specific] pairs of: (abstract_type, handler_type)
    self._abstract_type_registry: list[
        tuple[Type[Any], Type[types.LeafHandler[Any, Any]]]
    ] = []

  def _is_abstract_subprotocol(
      self, type_a: Type[Any], type_b: Type[Any]
  ) -> bool:
    """Checks if 'type_a' is a subclass or sub-protocol of 'type_b'."""
    try:
      if typing_extensions.is_protocol(type_b):   # pytype: disable=not-supported-yet
        return protocol_utils.is_subclass_protocol(
            cls=type_a, protocol=type_b
        )
      return issubclass(type_a, type_b)
    except TypeError:
      return False

  def _try_get(
      self, leaf_type: Type[types.Leaf]
  ) -> Type[types.LeafHandler[types.Leaf, Any]] | None:
    """Returns the handler last registered for a given type, if available."""
    for registered_leaf_ty, handler_type in reversed(self._leaf_type_registry):
      if issubclass(leaf_type, registered_leaf_ty):
        return handler_type
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
    """Returns the handler last registered for a given abstract type, if available."""
    for registered_abstract_ty, handler_type in reversed(
        self._abstract_type_registry
    ):
      if self._is_abstract_subprotocol(abstract_type, registered_abstract_ty):
        return handler_type
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
    """Returns all registered handlers."""
    return [
        (
            entry.leaf_type,
            entry.abstract_type,
            entry.handler_type,
        )
        for entry in self._entries
    ]

  def get_leaf_type_handler_pairs(
      self,
  ) -> Sequence[tuple[Type[Any], Type[types.LeafHandler[Any, Any]]]]:
    """Returns the leaf type registry."""
    return self._leaf_type_registry

  def get_abstract_type_handler_pairs(
      self,
  ) -> Sequence[tuple[Type[Any], Type[types.LeafHandler[Any, Any]]]]:
    """Returns the abstract type registry."""
    return self._abstract_type_registry

  def add(
      self,
      leaf_type: Type[types.Leaf],
      abstract_type: Type[types.AbstractLeaf],
      handler_type: Type[types.LeafHandler[types.Leaf, types.AbstractLeaf]],
      override: bool = False,
      secondary_typestrs: Sequence[str] | None = None,
  ):
    """Registers a `handler_type` for a given `leaf_type` and `abstract_type` pair.

    The registry automatically maintains a [Generic -> Specific] hierarchy for 
    both leaf and abstract types to ensure correct resolution. 

    A conflict occurs if the exact `leaf_type` is already registered, or if the 
    `abstract_type` is already mapped to a different handler. Set
    `override=True` to automatically remove conflicting entries and force the
    new registration.

    Args:
      leaf_type: The concrete type to register the handler for.
      abstract_type: The abstract type to register the handler for.
      handler_type: The handler class to register.
      override: If True, bypasses conflict errors and replaces existing
        conflicting entries.
      secondary_typestrs: Optional alternate identifiers for the handler.

    Raises:
      ValueError: If a duplicate `leaf_type` or conflicting `abstract_type`
        mapping exists and `override` is False.
    """
    if not override:
      for e in self._entries:
        if e.leaf_type == leaf_type:
          raise ValueError(
              f'leaf_type [{leaf_type}] is already handled by '
              f'{e.handler_type}. Use override=True to replace its entry. '
              f'Registry: {self._entries}'
          )
        if e.abstract_type == abstract_type and e.handler_type != handler_type:
          raise ValueError(
              f'abstract_type[{abstract_type}] is already handled by '
              f'{e.handler_type}. Use override=True to replace all '
              f'conflicting entries. Registry: {self._entries}'
          )

    # Handle overrides cleanly across all tracking lists
    if override:
      new_entries = []
      to_remove_leaves = []
      to_remove_abstracts = []

      for e in self._entries:
        is_conflict = (e.leaf_type == leaf_type) or (
            e.abstract_type == abstract_type and e.handler_type != handler_type
        )
        if is_conflict:
          # Track the tuples we need to delete out of the sorted registries
          to_remove_leaves.append((e.leaf_type, e.handler_type))
          to_remove_abstracts.append((e.abstract_type, e.handler_type))
          logging.vlog(
              1,
              'clearing conflicting entry: leaf_type[%s], abstract_type[%s],'
              ' handler_type[%s] during override.',
              e.leaf_type, e.abstract_type, e.handler_type,
          )
        else:
          new_entries.append(e)

      self._entries = new_entries
      # Remove all conflicting entries from the sorted registries.
      for item in to_remove_leaves:
        if item in self._leaf_type_registry:
          self._leaf_type_registry.remove(item)
      for item in to_remove_abstracts:
        if item in self._abstract_type_registry:
          self._abstract_type_registry.remove(item)

    new_reg = _Registration(
        leaf_type, abstract_type, handler_type, secondary_typestrs
    )
    self._entries.append(new_reg)

    # Insert into leaf registry (Tuple: leaf_type, handler_type)
    leaf_insert_idx = len(self._leaf_type_registry)
    for i, (reg_leaf_ty, _) in enumerate(self._leaf_type_registry):
      try:
        if issubclass(reg_leaf_ty, leaf_type) and reg_leaf_ty != leaf_type:
          leaf_insert_idx = i
          break
      except TypeError:
        pass
    self._leaf_type_registry.insert(leaf_insert_idx, (leaf_type, handler_type))

    # Insert into abstract registry (Tuple: abstract_type, handler_type)
    abstract_insert_idx = len(self._abstract_type_registry)
    for i, (reg_abstract_ty, _) in enumerate(self._abstract_type_registry):
      if (
          self._is_abstract_subprotocol(reg_abstract_ty, abstract_type)
          and reg_abstract_ty != abstract_type
      ):
        abstract_insert_idx = i
        break
    self._abstract_type_registry.insert(
        abstract_insert_idx, (abstract_type, handler_type)
    )

  def is_handleable(self, leaf_type: Type[Any]) -> bool:
    """Returns True if the type is handleable."""
    return self._try_get(leaf_type) is not None

  def is_abstract_handleable(self, abstract_type: Type[Any]) -> bool:
    """Returns True if the abstract type is handlable."""
    return self._try_get_abstract(abstract_type) is not None

  def get_secondary_typestrs(
      self, handler_type: Type[types.LeafHandler[Any, Any]]
  ) -> Sequence[str]:
    for entry in self._entries:
      if entry.handler_type == handler_type:
        return entry.secondary_typestrs or []
    return []


class StandardLeafHandlerRegistry(BaseLeafHandlerRegistry):
  """Default Leaf Handler Registry.

  This registry is designed as the default implementation of
  LeafHandlerRegistry.  It also registers the handlers for all the standard
  types including jax.Array, np.ndarray, int, float, bytes, and np.number.
  """

  def __init__(self):
    super().__init__()
    for (
        ty,
        abstract_ty,
        typestrs,
        handler_class,
    ) in STANDARD_TYPE_AND_ABSTRACT_TYPE_AND_TYPESTR_TO_HANDLER:
      self.add(ty, abstract_ty, handler_class, secondary_typestrs=typestrs)
