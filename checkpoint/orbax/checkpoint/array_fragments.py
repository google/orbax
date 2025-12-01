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

"""Management of fragments of arrays.

A fragment is a lot like a shard but its shape is not constrained by any
relationship to a mesh of devices, or to other fragments.
"""

# pylint: disable=g-importing-member, g-multiple-import, unused-import
from orbax.checkpoint._src.arrays.fragments import (
    Fragment,
    Fragments,
    Index,
    NpIndex,
    Shape,
    addressable_shards,
    abstract_fragments,
    stack_fragments,
    validate_fragments_can_be_stacked,
)
# pylint: enable=g-importing-member, g-multiple-import, unused-import
