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

| `Leaf` Type | `AbstractLeaf` Type | Properties |
:------- | :-------- | :-------- |
|`jax.Array`|`ocp.arrays.AbstractShardedArray` (`jax.ShapeDtypeStruct`)
|`shape`, `dtype`,
`sharding`|
|`np.ndarray`|`ocp.arrays.AbstractArray` (`np.ndarray`) |`shape`, `dtype`|
|`int`|`int`|  |
|`float`|`float`| |
|`bytes`|`bytes`| |
|`str`|`str`| |
"""

# pylint: disable=g-importing-member, g-multiple-import, g-bad-import-order, unused-import

from orbax.checkpoint.experimental.v1._src.tree.structure_utils import (
    merge_trees as merge,
)
from orbax.checkpoint.experimental.v1._src.tree.types import (
    PyTree,
    PyTreeOf,
    PyTreeKey,
    PyTreeKeyPath,
    JsonType,
)
from orbax.checkpoint.experimental.v1._src.tree.types import (
    Leaf,
    AbstractLeaf,
)
