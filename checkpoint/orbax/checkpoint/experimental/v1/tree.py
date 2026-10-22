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

"""Public symbols for tree module.

Standard supported leaf types are described by the table below.
See
https://orbax.readthedocs.io/en/latest/guides/checkpoint/v1/checkpointing_pytrees.html#standard-leaf-types
for more information.

.. list-table::
   :header-rows: 1

   * - ``Leaf`` Type
     - ``AbstractLeaf`` Type
     - Properties
   * - ``jax.Array``
     - ``ocp.v1.arrays.AbstractShardedArray`` (``jax.ShapeDtypeStruct``)
     - ``shape``, ``dtype``, ``sharding``
   * - ``np.ndarray``
     - ``ocp.v1.arrays.AbstractArray`` (``np.ndarray``)
     - ``shape``, ``dtype``
   * - ``int``
     - ``int``
     -
   * - ``float``
     - ``float``
     -
   * - ``np.number``
     - ``np.number``
     -
   * - ``bytes``
     - ``bytes``
     -
   * - ``bool``
     - ``bool``
     -
   * - ``str``
     - ``str``
     -
"""

# pylint: disable=g-importing-member, g-multiple-import, g-bad-import-order, unused-import

from orbax.checkpoint.experimental.v1._src.tree import types as _tree_types
from orbax.checkpoint.experimental.v1._src.tree.structure_utils import (
    merge_trees as merge,
)
from orbax.checkpoint.experimental.v1._src.tree.types import (
    PyTree,
    PyTreeOf,
)

#: A single key entry within a JAX PyTree path (sequence index, dict key,
#: attribute name, or flattened index).
PyTreeKey = _tree_types.PyTreeKey

#: A tuple of :py:data:`PyTreeKey` entries identifying a path from the root of a
#: PyTree to a leaf node.
PyTreeKeyPath = _tree_types.PyTreeKeyPath

#: Union of standard concrete leaf types supported in a checkpointed PyTree.
Leaf = _tree_types.Leaf

#: Union of abstract leaf descriptors (shapes/dtypes/shardings or scalar types)
#: used as targets when loading a PyTree checkpoint.
AbstractLeaf = _tree_types.AbstractLeaf

#: Recursive JSON-serializable container type (lists and string-keyed dicts of
#: JSON primitives ``str | int | float | bool | None`` or nested containers).
JsonType = _tree_types.JsonType
