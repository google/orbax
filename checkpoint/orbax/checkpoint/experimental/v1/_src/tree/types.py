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
from orbax.checkpoint._src.tree import types as tree_types
from orbax.checkpoint.experimental.v1._src.arrays import types as array_types

T = TypeVar("T")

if typing.TYPE_CHECKING:
  PyTreeOf = Any
  PyTree = Any

else:

  class PyTreeOf(Generic[T]):
    """A PyTree with leaf types of type ``T``.

    Functionally this type is treated as ``Any`` since JAX PyTrees cannot be
    identified by static type checkers.

    See https://jax.readthedocs.io/en/latest/pytrees.html for information on
    PyTrees.

    At a very high level, a PyTree is a container-like object such as a dict,
    list, or ``flax.struct.dataclass``.
    The elements of these containers can be traversed as
    a nested tree using ``jax.tree.*`` functions.

    In a checkpointing context, tree leaves are typically arrays or scalars.
    Even though arrays are logically lists, they are treated by JAX as leaf
    nodes.

    Note that all leaf nodes are definitionally PyTrees.
    """

    pass

  class PyTree:
    r"""Alias for (:py:class:`~.PyTreeOf`\[``Any``\])."""

    pass


#: A single key entry within a JAX PyTree path (sequence index, dict key,
#: attribute name, or flattened index).
PyTreeKey = tree_types.PyTreeKey

#: A tuple of :py:data:`~.PyTreeKey` entries identifying a path from the root
#: of a PyTree to a leaf node.
PyTreeKeyPath = tree_types.PyTreePath

#: Union of standard concrete leaf types supported in a checkpointed PyTree.
Leaf = jax.Array | np.ndarray | array_types.Scalar | str

#: Union of abstract leaf descriptors (shapes/dtypes/shardings or scalar types)
#: used as targets when loading a PyTree checkpoint.
AbstractLeaf = (
    array_types.AbstractArray
    | array_types.AbstractShardedArray
    | array_types.AbstractScalar
    | str
)

#: Recursive JSON-serializable container type (lists and string-keyed dicts
#: of JSON primitives ``str | int | float | bool | None`` or nested containers).
JsonType = tree_types.JsonType

#: A JSON-serializable value (primitive ``str | int | float | bool | None`` or
#: nested :py:data:`~.JsonType` container).
JsonValue = tree_types.JsonValue

#: Sentinel value used in target abstract trees to omit a leaf when loading a
#: partial PyTree checkpoint.
PLACEHOLDER = ...
