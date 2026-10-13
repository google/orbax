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

"""BasePyTreeCheckpointHandler class.

Implementation of `CheckpointHandler` interface dealing with JAX PyTrees. Much
of the underlying reading/writing logic for individual leaf types can be
customized, and is delegated to the `TypeHandler` class.
"""

# pylint: disable=g-bad-import-order,g-importing-member,unused-import,unused-wildcard-import,wildcard-import
import sys
from orbax.checkpoint.v0._src.handlers import base_pytree_checkpoint_handler as _mod
from orbax.checkpoint.v0._src.handlers.base_pytree_checkpoint_handler import *
from orbax.checkpoint.v0._src.handlers.base_pytree_checkpoint_handler import _update_array_restore_args
from orbax.checkpoint.v0._src.handlers.base_pytree_checkpoint_handler import _fill_missing_save_or_restore_args
from orbax.checkpoint.v0._src.handlers.base_pytree_checkpoint_handler import _format_bytes
from orbax.checkpoint.v0._src.handlers.base_pytree_checkpoint_handler import _is_prefix
from orbax.checkpoint.v0._src.handlers.base_pytree_checkpoint_handler import _get_partial_save_additions
from orbax.checkpoint.v0._src.handlers.base_pytree_checkpoint_handler import _filter_batch_requests

sys.modules[__name__] = _mod
