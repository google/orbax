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

"""PyTreeCheckpointHandler class.

Implementation of :py:class:`.CheckpointHandler` interface dealing with JAX
PyTrees. Much of the underlying reading/writing logic for individual leaf types
can be customized, and is delegated to the :py:class:`.TypeHandler` class.
"""

# pylint: disable=g-bad-import-order,g-importing-member,unused-import,unused-wildcard-import,wildcard-import
import sys
from orbax.checkpoint.v0._src.handlers import pytree_checkpoint_handler as _mod
from orbax.checkpoint.v0._src.handlers.pytree_checkpoint_handler import *
from orbax.checkpoint.v0._src.handlers.pytree_checkpoint_handler import _CHECKPOINT_FILE
from orbax.checkpoint.v0._src.handlers.pytree_checkpoint_handler import _maybe_set_default_restore_args
from orbax.checkpoint.v0._src.handlers.pytree_checkpoint_handler import _try_array_cast
from orbax.checkpoint.v0._src.handlers.pytree_checkpoint_handler import _maybe_shard_array
from orbax.checkpoint.v0._src.handlers.pytree_checkpoint_handler import _keystr
from orbax.checkpoint.v0._src.handlers.pytree_checkpoint_handler import _find_matching_input_args
from orbax.checkpoint.v0._src.handlers.pytree_checkpoint_handler import _has_use_fallback_transform
from orbax.checkpoint.v0._src.handlers.pytree_checkpoint_handler import _get_restore_parameters
from orbax.checkpoint.v0._src.handlers.pytree_checkpoint_handler import _multi_value_fns_with_args
from orbax.checkpoint.v0._src.handlers.pytree_checkpoint_handler import _transform_checkpoint
from orbax.checkpoint.v0._src.handlers.pytree_checkpoint_handler import _get_impl_save_args
from orbax.checkpoint.v0._src.handlers.pytree_checkpoint_handler import _concurrent_bytes

sys.modules[__name__] = _mod
