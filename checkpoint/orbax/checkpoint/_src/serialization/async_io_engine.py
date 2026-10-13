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

"""AsyncIoEngine module.

This module encapsulates the concurrency and execution orchestration layers for
Orbax.
Its primary responsibility is managing how work is dispatched to the Python
`asyncio` event loop and thread pools.

Scope:
* `asyncio.gather` and future management.
* Concurrency gating (e.g., `ByteLimiter`, `MemoryRegulator`).
* Top-level I/O telemetry and performance logging (e.g., throughput
calculation).

Anti-Scope (What does NOT belong here):
* Storage Backend Logic: Low-level serialization drivers, TensorStore
bindings.
* PyTree Math: Structural diffing, tree traversal, and `ParamInfo`
generation.
* Metadata Persistence: File-system JSON writes and Descriptor
management.
"""

# pylint: disable=g-bad-import-order,g-importing-member,unused-import,unused-wildcard-import,wildcard-import
import sys
from orbax.checkpoint.v0._src.serialization import async_io_engine as _mod
from orbax.checkpoint.v0._src.serialization.async_io_engine import *
from orbax.checkpoint.v0._src.serialization.async_io_engine import _get_memory_size

sys.modules[__name__] = _mod
