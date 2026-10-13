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

"""Utilities for interacting with the atomicity module.

Mainly provides helper functions like `get_default_temporary_path_class` that
help interacting with the `atomicity` module, but can have larger dependencies
than we would want to introduce into the base `atomicity` module.
"""

# pylint: disable=g-bad-import-order,g-importing-member,unused-import,unused-wildcard-import,wildcard-import
import sys
from orbax.checkpoint.v0._src.path import atomicity_defaults as _mod
from orbax.checkpoint.v0._src.path.atomicity_defaults import *
from orbax.checkpoint.v0._src.path.atomicity_defaults import _ATOMICITY_MODE_TO_PATH_CLASS
from orbax.checkpoint.v0._src.path.atomicity_defaults import _resolve_temporary_path_class_from_options

sys.modules[__name__] = _mod
