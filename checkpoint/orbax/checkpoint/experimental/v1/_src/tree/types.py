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

"""Types for PyTree utilities."""

import typing
from typing import Any, Generic, TypeVar
import jax
import numpy as np
from orbax.checkpoint._src.tree import types

JsonType = types.JsonType

T = TypeVar("T")

if typing.TYPE_CHECKING:
  PyTreeOf = Any
  PyTree = Any

else:

  class PyTreeOf(Generic[T]):
    """A PyTree with leaf types of type `T`.

    Functionally this type is treated as `Any` since JAX PyTrees cannot be
    identified by static type checkers.

    See https://jax.readthedocs.io/en/latest/pytrees.html for information on
    PyTrees.

    At a very high level, a PyTree is a container-like object such as a dict,
    list, or `flax.struct.dataclass`.
    The elements of these containers can be traversed as
    a nested tree using `jax.tree.*` functions.

    In a checkpointing context, tree leaves are typically arrays or scalars.
    Even though arrays are logically lists, they are treated by JAX as leaf
    nodes.

    Note that all leaf nodes are definitionally PyTrees.
    """

    pass

  class PyTree:
    """A PyTree with any leaf type (:py:class:`~.PyTreeOf[Any]`)."""

    pass


PyTreeKey = types.PyTreeKey
PyTreeKeyPath = types.PyTreePath

ScalarType = int | float | bool
LeafType = jax.Array | np.ndarray | str | ScalarType | Any
AbstractLeafType = Any  # TODO(cpgaffney): Add a type for abstract leaves.

JsonType = types.JsonType

PLACEHOLDER = ...
