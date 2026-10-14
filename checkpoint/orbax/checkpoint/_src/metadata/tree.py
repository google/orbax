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

"""Main module for PyTree checkpoint metadata storage."""

# pylint: disable=g-bad-import-order,g-importing-member,unused-import,unused-wildcard-import,wildcard-import
import sys
from orbax.checkpoint.v0._src.metadata import tree as _mod
from orbax.checkpoint.v0._src.metadata.tree import *
from orbax.checkpoint.v0._src.metadata.tree import _KEY_NAME
from orbax.checkpoint.v0._src.metadata.tree import _KEY_TYPE
from orbax.checkpoint.v0._src.metadata.tree import _TREE_METADATA_KEY
from orbax.checkpoint.v0._src.metadata.tree import _KEY_METADATA_KEY
from orbax.checkpoint.v0._src.metadata.tree import _VALUE_METADATA_KEY
from orbax.checkpoint.v0._src.metadata.tree import _USE_OCDBT
from orbax.checkpoint.v0._src.metadata.tree import _USE_ZARR3
from orbax.checkpoint.v0._src.metadata.tree import _STORE_ARRAY_DATA_EQUAL_TO_FILL_VALUE
from orbax.checkpoint.v0._src.metadata.tree import _VALUE_METADATA_TREE
from orbax.checkpoint.v0._src.metadata.tree import _CUSTOM_METADATA
from orbax.checkpoint.v0._src.metadata.tree import _get_key_metadata_type
from orbax.checkpoint.v0._src.metadata.tree import _keypath_from_key_type
from orbax.checkpoint.v0._src.metadata.tree import _TreeMetadataImpl

sys.modules[__name__] = _mod
