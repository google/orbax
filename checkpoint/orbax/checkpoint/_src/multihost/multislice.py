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

"""Multislice utilities."""

# pylint: disable=g-bad-import-order,g-importing-member,unused-import,unused-wildcard-import,wildcard-import
import sys
from orbax.checkpoint.v0._src.multihost import multislice as _mod
from orbax.checkpoint.v0._src.multihost.multislice import *
from orbax.checkpoint.v0._src.multihost.multislice import _process_in_device_replica
from orbax.checkpoint.v0._src.multihost.multislice import _get_slice_shape
from orbax.checkpoint.v0._src.multihost.multislice import _globalized_abstract_array
from orbax.checkpoint.v0._src.multihost.multislice import _single_device_scope
from orbax.checkpoint.v0._src.multihost.multislice import _globalize_single_replica_arrays
from orbax.checkpoint.v0._src.multihost.multislice import _sum_over_replica_axis
from orbax.checkpoint.v0._src.multihost.multislice import _merge_program
from orbax.checkpoint.v0._src.multihost.multislice import _chunk_by_memory

sys.modules[__name__] = _mod
