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

"""Batched filesystem probing utilities for checkpoint inspection.

A shared path utility library providing low-latency directory indexing and
concurrent probe primitives, especially used by checkpoint layout and format
detection routines.

Probing individual marker files sequentially on distributed or cloud
filesystems incurs significant RPC round-trip latency. This module minimizes
filesystem round trips through two core primitives:

1. **Ask once, answer many:** A single non-recursive directory scan answers all
   marker existence checks directly under a directory with one round trip.
   `DirectoryIndex` snapshots that listing for fast membership lookups.
2. **Ask in parallel:** When multiple subdirectories or paths must be checked
   concurrently, `exists_many`, `is_dir_many`, and `index_directories` issue
   checks concurrently via asyncio rather than serially.

Note: Directory listings are strictly non-recursive to avoid traversing large
tensor or chunk subtrees.
"""

# pylint: disable=g-bad-import-order,g-importing-member,unused-import,unused-wildcard-import,wildcard-import
import sys
from orbax.checkpoint.v0._src.path import fs_probe as _mod
from orbax.checkpoint.v0._src.path.fs_probe import *
from orbax.checkpoint.v0._src.path.fs_probe import _MISSING

sys.modules[__name__] = _mod
