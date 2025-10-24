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

"""Public symbols for Pathways-related serialization."""

# pylint: disable=g-importing-member, unused-import, g-bad-import-order

from orbax.checkpoint._src.serialization.pathways_handler_registry import CheckpointingImpl
from orbax.checkpoint._src.serialization.pathways_handler_registry import register_pathways_handlers as register_type_handlers
