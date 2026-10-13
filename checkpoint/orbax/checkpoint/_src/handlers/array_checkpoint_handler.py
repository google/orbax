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

"""ArrayCheckpointHandler for saving and restoring individual arrays/scalars."""

# pylint: disable=g-bad-import-order,g-importing-member,unused-import,unused-wildcard-import,wildcard-import
import sys
from orbax.checkpoint.v0._src.handlers import array_checkpoint_handler as _mod
from orbax.checkpoint.v0._src.handlers.array_checkpoint_handler import *
from orbax.checkpoint.v0._src.handlers.array_checkpoint_handler import _ELEMENT_KEY
from orbax.checkpoint.v0._src.handlers.array_checkpoint_handler import _USE_OCDBT_FOR_SAVE

sys.modules[__name__] = _mod
