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

"""Utils for writing in msgpack format. Derived from flax.serialization."""

# pylint: disable=g-bad-import-order,g-importing-member,unused-import,unused-wildcard-import,wildcard-import
import sys
from orbax.checkpoint.v0 import msgpack_utils as _mod
from orbax.checkpoint.v0.msgpack_utils import *
from orbax.checkpoint.v0.msgpack_utils import _ndarray_to_bytes
from orbax.checkpoint.v0.msgpack_utils import _dtype_from_name
from orbax.checkpoint.v0.msgpack_utils import _ndarray_from_bytes
from orbax.checkpoint.v0.msgpack_utils import _MsgpackExtType
from orbax.checkpoint.v0.msgpack_utils import _msgpack_ext_pack
from orbax.checkpoint.v0.msgpack_utils import _msgpack_ext_unpack
from orbax.checkpoint.v0.msgpack_utils import _np_convert_in_place
from orbax.checkpoint.v0.msgpack_utils import _tuple_to_dict
from orbax.checkpoint.v0.msgpack_utils import _dict_to_tuple
from orbax.checkpoint.v0.msgpack_utils import _chunk
from orbax.checkpoint.v0.msgpack_utils import _unchunk
from orbax.checkpoint.v0.msgpack_utils import _chunk_array_leaves_in_place
from orbax.checkpoint.v0.msgpack_utils import _unchunk_array_leaves_in_place

sys.modules[__name__] = _mod
