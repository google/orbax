# Copyright 2024 The Orbax Authors.
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

from orbax.checkpoint.experimental.v1._src.handlers import json_handler
from orbax.checkpoint.experimental.v1._src.handlers import proto_handler
from orbax.checkpoint.experimental.v1._src.handlers import pytree_handler
from orbax.checkpoint.experimental.v1._src.handlers import registration
from orbax.checkpoint.experimental.v1._src.path import format_utils


registration.global_registry().add(proto_handler.ProtoHandler)

registration.global_registry().add(json_handler.JsonHandler)

registration.global_registry().add(
    json_handler.MetricsHandler,
    format_utils.METRICS_CHECKPOINTABLE_KEY,
)

registration.global_registry().add(pytree_handler.PyTreeHandler)
