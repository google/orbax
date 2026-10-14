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

"""Utils for orbax tests."""

# pylint: disable=g-bad-import-order,g-importing-member,unused-import,unused-wildcard-import,wildcard-import
import sys
from orbax.checkpoint.v0 import test_utils as _mod
from orbax.checkpoint.v0.test_utils import *
from orbax.checkpoint.v0.test_utils import _patch_unique_barrier_key
from orbax.checkpoint.v0.test_utils import _get_test_wrapper_function
from orbax.checkpoint.v0.test_utils import _get_simple_tensorstore_read_spec
from orbax.checkpoint.v0.test_utils import _replace_abstract_array_sharding_with_mesh

sys.modules[__name__] = _mod
