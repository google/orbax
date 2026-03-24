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

from collections.abc import Sequence
import dataclasses
from typing import Any

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
  """A registration entry for a LeafHandler.

  Attributes:
    leaf_type: The concrete PyTree leaf type.
    abstract_type: The abstract representation of the leaf type.
    handler_type: The LeafHandler class.
    secondary_typestrs: Optional alternate identifiers for the handler.
    leaf_specificity_score: Specificity score for the leaf type. Higher value
      means more specific type relative to other leaf types which it is a
      subclass of. This determines which handler we resolve to during
      save/load operations.
    abstract_specificity_score: Specificity score for the abstract type. Higher
      value means more specific type relative to other abstract types which it
      is a subprotocol/subclass of. This determines which handler we resolve to
      during save/load operations.
  """

  leaf_type: type[Any]
  abstract_type: type[Any]
  handler_type: type[types.LeafHandler[Any, Any]]
  secondary_typestrs: Sequence[str] | None
  leaf_specificity_score: int
  abstract_specificity_score: int


def _is_abstract_subprotocol(
    type_a: type[Any], type_b: type[Any]
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
    # Sorted [Generic -> Specific] primarily by leaf_specificity_score.
    self._entries: list[_Registration] = []

  def _try_get(
      self, leaf_type: type[types.Leaf]
  ) -> type[types.LeafHandler[types.Leaf, Any]] | None:
    """Returns the most specific handler for a given type, if available."""
    # self._entries is sorted Generic -> Specific by leaf_specificity_score.
    # Iterating reversed checks the most specific handlers first.
    for entry in reversed(self._entries):
      try:
        if issubclass(leaf_type, entry.leaf_type):
          return entry.handler_type
      except TypeError:
        pass
    return None

  def get(
      self, leaf_type: type[types.Leaf]
  ) -> type[types.LeafHandler[types.Leaf, Any]]:
    if (handler_type := self._try_get(leaf_type)) is None:
      raise ValueError(
          f'Unknown Leaf type: "{leaf_type!r}". Must register it with'
          ' LeafHandlerRegistry.'
      )
    return handler_type

  def _try_get_abstract(
      self,
      abstract_type: type[types.AbstractLeaf],
  ) -> type[types.LeafHandler[Any, types.AbstractLeaf]] | None:
    """Returns the most specific handler for a given abstract type."""
    # Sort ascending by abstract_specificity_score (lowest to highest).
    sorted_entries = sorted(
        self._entries,
        key=lambda e: e.abstract_specificity_score
    )
    # Iterating reversed checks the most specific handlers first.
    for entry in reversed(sorted_entries):
      if _is_abstract_subprotocol(abstract_type, entry.abstract_type):
        return entry.handler_type
    return None

  def get_abstract(
      self,
      abstract_type: type[types.AbstractLeaf],
  ) -> type[types.LeafHandler[Any, types.AbstractLeaf]]:
    if (handler_type := self._try_get_abstract(abstract_type)) is None:
      raise ValueError(
          f'Unknown AbstractLeaf type: "{abstract_type!r}". Must register it'
          ' with LeafHandlerRegistry.'
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
            entry.leaf_type,
            entry.abstract_type,
            entry.handler_type,
        )
        for entry in self._entries
    ]

  def add(
      self,
      leaf_type: type[types.Leaf],
      abstract_type: type[types.AbstractLeaf],
      handler_type: type[types.LeafHandler[types.Leaf, types.AbstractLeaf]],
      override: bool = False,
      secondary_typestrs: Sequence[str] | None = None,
  ):
    """Registers a `handler_type` for a `leaf_type` and `abstract_type` pair.

    The registry automatically maintains a [Generic -> Specific] hierarchy for
    both leaf and abstract types using dynamic topological priorities to ensure
    correct resolution. We maintain and recalculate these specificity scores to
    ensure that the most specific handler is chosen during resolution.

    A conflict occurs if the exact `leaf_type` is already registered, or if the
    `abstract_type` is already mapped to a different handler. Set
    `override=True` to automatically remove conflicting entries and force the
    new registration.

    Args:
      leaf_type: The concrete PyTree leaf type to register.
      abstract_type: The abstract representation of the leaf type.
      handler_type: The `LeafHandler` class responsible for this leaf type.
      override: If True, overwrites existing registrations for the given
        leaf or abstract types. Defaults to False.
      secondary_typestrs: A sequence of alternate handler typestrs that serve as
        secondary identifiers for the handler.

    Raises:
      ValueError: If a duplicate `leaf_type` or conflicting `abstract_type`
        mapping exists and `override` is False.
    """

    # Check for exact duplicate registration
    for e in self._entries:
      if (
          e.leaf_type == leaf_type
          and e.abstract_type == abstract_type
          and e.handler_type == handler_type
      ):
        if override and e.secondary_typestrs != secondary_typestrs:
          logging.info(
              'Updating secondary_typestrs for existing registration: %s -> %s',
              e.secondary_typestrs,
              secondary_typestrs,
          )
          e.secondary_typestrs = secondary_typestrs
          return
        logging.info(
            'Registration already exists for leaf_type[%s], '
            'abstract_type[%s], handler_type[%s]. Skipping.',
            leaf_type,
            abstract_type,
            handler_type,
        )
        return

    if override:
      # Filter out conflicting entries if override is True.
      new_entries = []
      for e in self._entries:
        is_conflict = (e.leaf_type == leaf_type) or (
            e.abstract_type == abstract_type and e.handler_type != handler_type
        )
        if is_conflict:
          logging.warning(
              'clearing conflicting entry: leaf_type[%s], abstract_type[%s]'
              ' handler_type[%s] during override.',
              e.leaf_type,
              e.abstract_type,
              e.handler_type,
          )
        else:
          new_entries.append(e)
      self._entries = new_entries
    else:
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

    # Append the new entry with default priorities
    new_reg = _Registration(
        leaf_type,
        abstract_type,
        handler_type,
        secondary_typestrs,
        leaf_specificity_score=0,
        abstract_specificity_score=0,
    )
    self._entries.append(new_reg)
    # Recalculate specificity scores for all entries since new entry was added
    # and may change the specificity scores of existing entries.
    self._recalculate_specificity_scores()

    # Sort the single source of truth [Generic -> Specific] based on leaf type
    # primarily, and abstract type secondarily.
    self._entries.sort(
        key=lambda x: (
            x.leaf_specificity_score,
            x.abstract_specificity_score,
            x.handler_type.__name__,
        )
    )

  def _recalculate_specificity_scores(self) -> None:
    """Recalculates specificity scores and sorts the registry."""
    for target_entry in self._entries:
      leaf_count = 0
      abstract_count = 0
      for other_entry in self._entries:
        # Count how many leaf types this target is a subclass of.
        try:
          if (
              target_entry.leaf_type != other_entry.leaf_type and
              issubclass(target_entry.leaf_type, other_entry.leaf_type)
          ):
            leaf_count += 1
        except TypeError:
          pass
        # Count how many abstract types this target is a subprotocol of.
        if (
            target_entry.abstract_type != other_entry.abstract_type and
            _is_abstract_subprotocol(
                target_entry.abstract_type, other_entry.abstract_type
            )
        ):
          abstract_count += 1
      target_entry.leaf_specificity_score = leaf_count
      target_entry.abstract_specificity_score = abstract_count

  def is_handleable(self, leaf_type: type[Any]) -> bool:
    """Returns True if the type is handleable.

    This checks if the provided concrete leaf type, or any of its base classes,
    has a specifically registered handler in the registry.

    Args:
      leaf_type: The concrete PyTree leaf type to check.

    Returns:
      True if a valid handler exists for the type, False otherwise.
    """
    return self._try_get(leaf_type) is not None

  def is_abstract_handleable(self, abstract_type: type[Any]) -> bool:
    """Returns True if the abstract type is handleable.

    This checks if the provided abstract leaf type, or any of its matching base
    classes or protocols, has a registered handler in the registry.

    Args:
      abstract_type: The abstract PyTree leaf type to check.

    Returns:
      True if a valid handler exists for the abstract type, False otherwise.
    """
    return self._try_get_abstract(abstract_type) is not None

  def get_secondary_typestrs(
      self, handler_type: type[types.LeafHandler[Any, Any]]
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
        typestrs,
        handler_class,
    ) in STANDARD_TYPE_AND_ABSTRACT_TYPE_AND_TYPESTR_TO_HANDLER:
      self.add(ty, abstract_ty, handler_class, secondary_typestrs=typestrs)
