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

"""Higher-level utilities for working with tree structures."""

# pylint: disable=g-bad-import-order,g-importing-member,unused-import,unused-wildcard-import,wildcard-import
import sys
from orbax.checkpoint.v0._src.tree import structure_utils as _mod
from orbax.checkpoint.v0._src.tree.structure_utils import *
from orbax.checkpoint.v0._src.tree.structure_utils import _FlattenedNode
from orbax.checkpoint.v0._src.tree.structure_utils import _jax_internal_node_to_dict
from orbax.checkpoint.v0._src.tree.structure_utils import _wrap_callback_for_conversion
from orbax.checkpoint.v0._src.tree.structure_utils import _tree_trim
from orbax.checkpoint.v0._src.tree.structure_utils import _tree_trim_impl
from orbax.checkpoint.v0._src.tree.structure_utils import _check_for_common_keys
from orbax.checkpoint.v0._src.tree.structure_utils import _recursive_merge
from orbax.checkpoint.v0._src.tree.structure_utils import _ErrorType

sys.modules[__name__] = _mod
