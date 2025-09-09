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

"""Wrapper for :py:class:`serialization.LeafHandler`.

This :py:class:`CheckpointableHandler` is a wrapper for checkpointables where
support is already implemented at the PyTree leaf level.
"""

from typing import Any, Awaitable, TypeVar

import jax
import numpy as np
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.handlers import pytree_handler
from orbax.checkpoint.experimental.v1._src.handlers import types as handler_types
from orbax.checkpoint.experimental.v1._src.path import types as path_types
from orbax.checkpoint.experimental.v1._src.serialization import registry
from orbax.checkpoint.experimental.v1._src.serialization import types as serialization_types


Leaf = TypeVar('Leaf')
AbstractLeaf = TypeVar('AbstractLeaf')


class _LeafHandler(handler_types.CheckpointableHandler[Leaf, AbstractLeaf]):
  """Base class for handlers that operate on individual PyTree leaves.

  This handler wraps `PyTreeHandler` to provide support for checkpointables
  that are single leaves in a PyTree.
  """

  def __init__(self):
    self._context = context_lib.get_context()

  async def save(
      self, directory: path_types.PathAwaitingCreation, checkpointable: Leaf
  ) -> Awaitable[None]:
    return await pytree_handler.PyTreeHandler().save(
        directory, [checkpointable]
    )

  async def load(
      self,
      directory: path_types.Path,
      abstract_checkpointable: AbstractLeaf | None = None,
  ) -> Awaitable[Leaf]:
    if abstract_checkpointable is None:
      abstract_pytree = None
    else:
      abstract_pytree = [abstract_checkpointable]

    background_load = await pytree_handler.PyTreeHandler().load(
        directory, abstract_pytree
    )

    async def background_load_wrapper() -> Leaf:
      loaded_pytree = await background_load
      return loaded_pytree[0]

    return background_load_wrapper()

  async def metadata(self, directory: path_types.Path) -> AbstractLeaf:
    pytree_metadata = await pytree_handler.PyTreeHandler().metadata(directory)
    return pytree_metadata[0]

  def is_handleable(self, checkpointable: Any) -> bool:
    try:
      pytree_handler.PyTreeHandler().validate_leaves_handleable(
          [checkpointable]
      )
      return True
    except registry.UnregisteredTypeError:
      return False

  def is_abstract_handleable(self, abstract_checkpointable: Any) -> bool | None:
    try:
      pytree_handler.PyTreeHandler().validate_abstract_leaves_handleable(
          [abstract_checkpointable]
      )
      return True
    except registry.UnregisteredTypeError:
      return False


class ShardedArrayHandler(
    _LeafHandler[jax.Array, serialization_types.AbstractShardedArray]
):
  pass


class ArrayHandler(_LeafHandler[np.ndarray, serialization_types.AbstractArray]):
  pass


class StringHandler(_LeafHandler[str, serialization_types.AbstractString]):
  pass


class ScalarHandler(
    _LeafHandler[serialization_types.Scalar, serialization_types.AbstractScalar]
):
  pass
