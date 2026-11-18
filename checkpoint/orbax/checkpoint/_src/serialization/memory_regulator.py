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

"""Manages memory limits for checkpoint operations via PID control and profiling."""

# pylint: disable=g-bad-import-order,g-importing-member,unused-import,unused-wildcard-import,wildcard-import
import sys
from orbax.checkpoint.v0._src.serialization import memory_regulator as _mod
from orbax.checkpoint.v0._src.serialization.memory_regulator import *
from orbax.checkpoint.v0._src.serialization.memory_regulator import _BYTES_TO_GIB
from orbax.checkpoint.v0._src.serialization.memory_regulator import _profiler
from orbax.checkpoint.v0._src.serialization.memory_regulator import _resolve_profiler

sys.modules[__name__] = _mod
