# Copyright 2025 The Orbax Authors.
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

"""Public API for CheckpointableHandlers."""

# pylint: disable=g-importing-member, g-multiple-import, unused-import, g-bad-import-order

import orbax.checkpoint.experimental.v1._src.handlers.global_registration
from orbax.checkpoint.experimental.v1._src.handlers.types import (
    CheckpointableHandler,
    StatefulCheckpointable,
)

from orbax.checkpoint.experimental.v1._src.handlers.pytree_handler import (
    PyTreeHandler,
)
from orbax.checkpoint.experimental.v1._src.handlers.proto_handler import (
    ProtoHandler,
)
from orbax.checkpoint.experimental.v1._src.handlers.json_handler import (
    JsonHandler,
)


from orbax.checkpoint.experimental.v1._src.handlers.registration import (
    CheckpointableHandlerRegistry,
    global_registry,
    local_registry,
    register_handler,
)
