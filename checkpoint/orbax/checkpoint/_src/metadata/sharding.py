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

"""ShardingMetadata representing Sharding property."""

# pylint: disable=g-bad-import-order,g-importing-member,unused-import,unused-wildcard-import,wildcard-import
import sys
from orbax.checkpoint.v0._src.metadata import sharding as _mod
from orbax.checkpoint.v0._src.metadata.sharding import *
from orbax.checkpoint.v0._src.metadata.sharding import _AXIS_TYPE_MAP
from orbax.checkpoint.v0._src.metadata.sharding import _PARTITION_SPEC
from orbax.checkpoint.v0._src.metadata.sharding import _SHARDING
from orbax.checkpoint.v0._src.metadata.sharding import _SHARDING_TYPE
from orbax.checkpoint.v0._src.metadata.sharding import _DEVICE_STR
from orbax.checkpoint.v0._src.metadata.sharding import _MESH_AXES
from orbax.checkpoint.v0._src.metadata.sharding import _MESH_AXIS_TYPES
from orbax.checkpoint.v0._src.metadata.sharding import _MESH_SHAPE
from orbax.checkpoint.v0._src.metadata.sharding import _DEVICES_SHAPE
from orbax.checkpoint.v0._src.metadata.sharding import _DEVICE_MESH
from orbax.checkpoint.v0._src.metadata.sharding import _MEMORY_KIND
from orbax.checkpoint.v0._src.metadata.sharding import _ID

sys.modules[__name__] = _mod
