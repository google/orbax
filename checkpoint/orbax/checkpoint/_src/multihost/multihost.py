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

"""Orbax utils related to multihost_utils functionality."""

# pylint: disable=g-bad-import-order,g-importing-member,unused-import,unused-wildcard-import,wildcard-import
import sys
from orbax.checkpoint.v0._src.multihost import multihost as _mod
from orbax.checkpoint.v0._src.multihost.multihost import *
from orbax.checkpoint.v0._src.multihost.multihost import _DEFAULT_BARRIER_TIMEOUT
from orbax.checkpoint.v0._src.multihost.multihost import _TEST_CASE_INDEX
from orbax.checkpoint.v0._src.multihost.multihost import _DISTRIBUTED_TO_DEVICE_IDS
from orbax.checkpoint.v0._src.multihost.multihost import _RUNTIME_TO_DISTRIBUTED_ID
from orbax.checkpoint.v0._src.multihost.multihost import _PATHWAYS_COLOCATED_RUNTIME_ACTIVE
from orbax.checkpoint.v0._src.multihost.multihost import _unique_barrier_key
from orbax.checkpoint.v0._src.multihost.multihost import _maybe_log_reached_preemption

sys.modules[__name__] = _mod
