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

"""Definitions and helpers for constructing OCDBT process spec strings.

These are used by Orbax OCDBT integration to name/identify per-process
subdirectories where OCDBT checkpoints are written.

When writing an OCDBT TensorStore checkpoint, per-process subdirectories are
used to isolate processes from each other. The names of these subdirectories
must conform to a specific format, defined in this module as OcdbtProcessSpec.
"""

# pylint: disable=g-bad-import-order,g-importing-member,unused-import,unused-wildcard-import,wildcard-import
import sys
from orbax.checkpoint.v0._src.serialization import ocdbt_process_spec as _mod
from orbax.checkpoint.v0._src.serialization.ocdbt_process_spec import *
from orbax.checkpoint.v0._src.serialization.ocdbt_process_spec import _PROCESS_ID_RE
from orbax.checkpoint.v0._src.serialization.ocdbt_process_spec import _PROCESS_PREFIX_RE
from orbax.checkpoint.v0._src.serialization.ocdbt_process_spec import _OCDBT_SPEC_PATTERN

sys.modules[__name__] = _mod
