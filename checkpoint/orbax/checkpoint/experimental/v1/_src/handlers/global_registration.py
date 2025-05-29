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

"""Defines all globally-registered handlers.

This must be imported to ensure necessary handlers are registered.

Registration order matters. The most recently registered valid handler for a
given checkpointable will be used.
"""

from typing import Type

from orbax.checkpoint.experimental.v1._src.handlers import json_handler
from orbax.checkpoint.experimental.v1._src.handlers import proto_handler
from orbax.checkpoint.experimental.v1._src.handlers import pytree_handler
from orbax.checkpoint.experimental.v1._src.handlers import registration
from orbax.checkpoint.experimental.v1._src.handlers import stateful_checkpointable_handler
from orbax.checkpoint.experimental.v1._src.handlers import types as handler_types
from orbax.checkpoint.experimental.v1._src.path import format_utils


def _try_register_handler(
    handler_type: Type[handler_types.CheckpointableHandler],
    name: str | None = None,
):
  try:
    registration.global_registry().add(handler_type, name)
  except registration.AlreadyExistsError:
    pass


_try_register_handler(proto_handler.ProtoHandler)
_try_register_handler(json_handler.JsonHandler)
_try_register_handler(
    stateful_checkpointable_handler.StatefulCheckpointableHandler
)
_try_register_handler(
    json_handler.MetricsHandler,
    format_utils.METRICS_CHECKPOINTABLE_KEY,
)
_try_register_handler(pytree_handler.PyTreeHandler)
_try_register_handler(
    pytree_handler.PyTreeHandler, format_utils.PYTREE_CHECKPOINTABLE_KEY
)
