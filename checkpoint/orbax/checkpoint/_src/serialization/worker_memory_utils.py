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

"""Utilities for managing worker memory usage."""

# pylint: disable=g-bad-import-order,g-importing-member,unused-import,unused-wildcard-import,wildcard-import
import sys
from orbax.checkpoint.v0._src.serialization import worker_memory_utils as _mod
from orbax.checkpoint.v0._src.serialization.worker_memory_utils import *
from orbax.checkpoint.v0._src.serialization.worker_memory_utils import _device_to_worker_ids
from orbax.checkpoint.v0._src.serialization.worker_memory_utils import _estimate_worker_memory_usage
from orbax.checkpoint.v0._src.serialization.worker_memory_utils import _increment_worker_memory_usage
from orbax.checkpoint.v0._src.serialization.worker_memory_utils import _is_array_under_memory_budget
from orbax.checkpoint.v0._src.serialization.worker_memory_utils import _humanize_worker_memory_usage

sys.modules[__name__] = _mod
