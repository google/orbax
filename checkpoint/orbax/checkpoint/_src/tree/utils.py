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

"""Tree utilities."""

# pylint: disable=g-bad-import-order,g-importing-member,unused-import,unused-wildcard-import,wildcard-import
import sys
from orbax.checkpoint.v0._src.tree import utils as _mod
from orbax.checkpoint.v0._src.tree.utils import *
from orbax.checkpoint.v0._src.tree.utils import _internal_node_as_dict
from orbax.checkpoint.v0._src.tree.utils import _raise_unsupported_key_error
from orbax.checkpoint.v0._src.tree.utils import _extend_list
from orbax.checkpoint.v0._src.tree.utils import _pytree_key_as_tree_key

sys.modules[__name__] = _mod
