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

"""Management of fragments of arrays.

A fragment is a lot like a shard but its shape is not constrained by any
relationship to a mesh of devices, or to other fragments.

For each type FS in {AbstractFragments, NpFragments and JaxFragments}, there are
class methods:
  - `all_of(x)`: gives an FS shaped like x, with a single fragment that spans
    its whole shape. If FS is concrete then the fragment value will be x. If
    FS is `AbstractFragments` then x may additionally be a
    `jax.ShapeDtypeStruct`.
  - `none_of(x)`: gives an FS shaped like x, with no fragments.
  - `addressable_shards_of(x)`: gives an FS shaped like x, with one fragment for
    each distinct addressable shard of x. If FS is `AbstractFragments` then x
    may additionally be a `jax.ShapeDtypeStruct`.
  - `of(x, *indices)`: gives an FS shaped like x, with one fragment for each
    index in `indices`. If FS is concrete then the fragment values will be
    slices of x. If FS is `AbstractFragments` then x may additionally be
    a `jax.ShapeDtypeStruct`.
  - `like(fragments, x)` takes an existing Fragments instance and returns a new
    instance of FS, with the same shape and dtype and fragment indices, but
    with the fragment values replaced by slices of x. If FS is
    `AbstractFragments` then x must be `None` and may be omitted.

Use `FS.of()` and friends to make instances out of arraylike things.
Use `FS.like()` to convert between different kinds of Fragments.
Use `{np, jnp}.asarray(fragments)` to make arraylike things out of (full)
Fragments.
"""

# pylint: disable=g-bad-import-order,g-importing-member,unused-import,unused-wildcard-import,wildcard-import
import sys
from orbax.checkpoint.v0._src.arrays import fragments as _mod
from orbax.checkpoint.v0._src.arrays.fragments import *
from orbax.checkpoint.v0._src.arrays.fragments import _ndarray_from_index
from orbax.checkpoint.v0._src.arrays.fragments import _index_from_ndarray
from orbax.checkpoint.v0._src.arrays.fragments import _qualified_name
from orbax.checkpoint.v0._src.arrays.fragments import _check_fragment_value_type
from orbax.checkpoint.v0._src.arrays.fragments import _GenericFragment
from orbax.checkpoint.v0._src.arrays.fragments import _ConcreteFragment
from orbax.checkpoint.v0._src.arrays.fragments import _F
from orbax.checkpoint.v0._src.arrays.fragments import _GenericFragments
from orbax.checkpoint.v0._src.arrays.fragments import _ConcreteFragments
from orbax.checkpoint.v0._src.arrays.fragments import _is_full
from orbax.checkpoint.v0._src.arrays.fragments import _gen_distinct_addressable_indices

sys.modules[__name__] = _mod
