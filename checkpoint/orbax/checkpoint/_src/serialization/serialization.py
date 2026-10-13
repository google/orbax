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

"""Array serialization and deserialization."""

# pylint: disable=g-bad-import-order,g-importing-member,unused-import,unused-wildcard-import,wildcard-import
import sys
from orbax.checkpoint.v0._src.serialization import serialization as _mod
from orbax.checkpoint.v0._src.serialization.serialization import *
from orbax.checkpoint.v0._src.serialization.serialization import _REMOVED_VALUE
from orbax.checkpoint.v0._src.serialization.serialization import _CHECKPOINT_SUCCESS
from orbax.checkpoint.v0._src.serialization.serialization import _get_metadata
from orbax.checkpoint.v0._src.serialization.serialization import _spec_has_metadata
from orbax.checkpoint.v0._src.serialization.serialization import _read_shard
from orbax.checkpoint.v0._src.serialization.serialization import _read_array_index_and_device_put
from orbax.checkpoint.v0._src.serialization.serialization import _get_device_to_index_map
from orbax.checkpoint.v0._src.serialization.serialization import _read_and_create_array

sys.modules[__name__] = _mod
