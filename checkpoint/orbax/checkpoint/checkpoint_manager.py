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

"""A class providing functionalities for managing a series of checkpoints."""

# pylint: disable=g-bad-import-order,g-importing-member,unused-import,unused-wildcard-import,wildcard-import
import sys
from orbax.checkpoint.v0 import checkpoint_manager as _mod
from orbax.checkpoint.v0.checkpoint_manager import *
from orbax.checkpoint.v0.checkpoint_manager import _INIT_TIME
from orbax.checkpoint.v0.checkpoint_manager import _WAIT_FOR_PREV_SAVE_WARNING_THRESHOLD_SECS
from orbax.checkpoint.v0.checkpoint_manager import _metrics_file_exists
from orbax.checkpoint.v0.checkpoint_manager import _FinalizeThread
from orbax.checkpoint.v0.checkpoint_manager import _ShouldSaveFnPolicy
from orbax.checkpoint.v0.checkpoint_manager import _get_default_save_decision_policy
from orbax.checkpoint.v0.checkpoint_manager import _get_default_preservation_policy
from orbax.checkpoint.v0.checkpoint_manager import _get_args_for_key
from orbax.checkpoint.v0.checkpoint_manager import _create_root_directory
from orbax.checkpoint.v0.checkpoint_manager import _determine_default_item_mode_from_directory

sys.modules[__name__] = _mod
