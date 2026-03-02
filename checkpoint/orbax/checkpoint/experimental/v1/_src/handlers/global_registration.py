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

"""Defines all globally-registered handlers.

This must be imported to ensure necessary handlers are registered.

Registration order matters. The most recently registered valid handler for a
given checkpointable will be used.
"""

from typing import Sequence, Type

from orbax.checkpoint.experimental.v1._src.handlers import json_handler
from orbax.checkpoint.experimental.v1._src.handlers import proto_handler
from orbax.checkpoint.experimental.v1._src.handlers import pytree_handler
from orbax.checkpoint.experimental.v1._src.handlers import registration
from orbax.checkpoint.experimental.v1._src.handlers import stateful_checkpointable_handler
from orbax.checkpoint.experimental.v1._src.handlers import types as handler_types
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout


def _try_register_handler(
    handler_type: Type[handler_types.CheckpointableHandler],
    name: str | None = None,
    recognized_handler_typestrs: Sequence[str] | None = None,
):
  """Tries to register handler globally with name and recognized typestrs."""
  try:
    registration.global_registry().add(
        handler_type,
        name,
        recognized_handler_typestrs=recognized_handler_typestrs,
    )
  except registration.AlreadyExistsError:
    pass


_try_register_handler(
    proto_handler.ProtoHandler,
    recognized_handler_typestrs=[
        'orbax.checkpoint._src.handlers.proto_checkpoint_handler.ProtoCheckpointHandler',
    ],
)
_try_register_handler(
    json_handler.JsonHandler,
    recognized_handler_typestrs=[
        'orbax.checkpoint._src.handlers.json_checkpoint_handler.JsonCheckpointHandler',
    ],
)
_try_register_handler(
    stateful_checkpointable_handler.StatefulCheckpointableHandler
)
_try_register_handler(
    json_handler.MetricsHandler,
    checkpoint_layout.METRICS_CHECKPOINTABLE_KEY,
)
_try_register_handler(
    pytree_handler.PyTreeHandler,
    recognized_handler_typestrs=[
        'orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler',
        'orbax.checkpoint._src.handlers.standard_checkpoint_handler.StandardCheckpointHandler',
    ],
)
_try_register_handler(
    pytree_handler.PyTreeHandler, checkpoint_layout.PYTREE_CHECKPOINTABLE_KEY
)
