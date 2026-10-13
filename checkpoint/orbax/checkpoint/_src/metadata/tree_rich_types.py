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

"""Storage for PyTree checkpoint metadata with rich types."""

# pylint: disable=g-bad-import-order,g-importing-member,unused-import,unused-wildcard-import,wildcard-import
import sys
from orbax.checkpoint.v0._src.metadata import tree_rich_types as _mod
from orbax.checkpoint.v0._src.metadata.tree_rich_types import *
from orbax.checkpoint.v0._src.metadata.tree_rich_types import _new_namedtuple_type
from orbax.checkpoint.v0._src.metadata.tree_rich_types import _create_namedtuple
from orbax.checkpoint.v0._src.metadata.tree_rich_types import _module_and_class_name
from orbax.checkpoint.v0._src.metadata.tree_rich_types import _VALUE_METADATA_ENTRY_CLAZZ
from orbax.checkpoint.v0._src.metadata.tree_rich_types import _VALUE_METADATA_ENTRY_MODULE_AND_CLASS
from orbax.checkpoint.v0._src.metadata.tree_rich_types import _value_metadata_tree_for_json_dumps
from orbax.checkpoint.v0._src.metadata.tree_rich_types import _value_metadata_tree_for_json_loads

sys.modules[__name__] = _mod
