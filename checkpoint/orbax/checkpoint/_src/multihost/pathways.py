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

"""Pathways-specific multihost utilities."""

# pylint: disable=g-bad-import-order,g-importing-member,unused-import,unused-wildcard-import,wildcard-import
import sys
from orbax.checkpoint.v0._src.multihost import pathways as _mod
from orbax.checkpoint.v0._src.multihost.pathways import *
from orbax.checkpoint.v0._src.multihost.pathways import _WARNED_REPR_PATTERNS
from orbax.checkpoint.v0._src.multihost.pathways import _extract_int_from_repr
from orbax.checkpoint.v0._src.multihost.pathways import _get_device_task_index
from orbax.checkpoint.v0._src.multihost.pathways import _get_device_slice_index
from orbax.checkpoint.v0._src.multihost.pathways import _get_device_worker_key
from orbax.checkpoint.v0._src.multihost.pathways import _get_worker_count_key

sys.modules[__name__] = _mod
