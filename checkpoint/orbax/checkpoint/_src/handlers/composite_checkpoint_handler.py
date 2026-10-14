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

"""Handler that combines other handlers.

Usage example::

  import orbax.checkpoint as ocp

  # A PyTree of jax.Arrays
  my_state = ...
  # A dictionary to be serialized as JSON
  json_dict = ...

  checkpointer = ocp.Checkpointer(ocp.CompositeCheckpointHandler())
  checkpointer.save(
      path,
      args=ocp.args.Composite(
          state=ocp.args.StandardSave(my_state),
          metadata=ocp.args.JsonSave(json_dict)
      )
  )

  restored = checkpointer.restore(
      path,
      args=ocp.args.Composite(
          state=ocp.args.StandardRestore(),
          metadata=ocp.args.JsonRestore()
      )
  )
  my_state = restored.state
  json_dict = restored.metadata
"""

# pylint: disable=g-bad-import-order,g-importing-member,unused-import,unused-wildcard-import,wildcard-import
import sys
from orbax.checkpoint.v0._src.handlers import composite_checkpoint_handler as _mod
from orbax.checkpoint.v0._src.handlers.composite_checkpoint_handler import *
from orbax.checkpoint.v0._src.handlers.composite_checkpoint_handler import _CONCURRENT_WORKERS
from orbax.checkpoint.v0._src.handlers.composite_checkpoint_handler import _DIRECTORY_CREATION_SIGNALS
from orbax.checkpoint.v0._src.handlers.composite_checkpoint_handler import _LegacyCheckpointHandlerWrapper
from orbax.checkpoint.v0._src.handlers.composite_checkpoint_handler import _WrapperArgs
from orbax.checkpoint.v0._src.handlers.composite_checkpoint_handler import _AsyncLegacyCheckpointHandlerWrapper
from orbax.checkpoint.v0._src.handlers.composite_checkpoint_handler import _AsyncWrapperArgs
from orbax.checkpoint.v0._src.handlers.composite_checkpoint_handler import _maybe_raise_reserved_item_error
from orbax.checkpoint.v0._src.handlers.composite_checkpoint_handler import _get_unique_registered_items_and_handlers
from orbax.checkpoint.v0._src.handlers.composite_checkpoint_handler import _add_to_handler_registry_from_global_registry

sys.modules[__name__] = _mod
