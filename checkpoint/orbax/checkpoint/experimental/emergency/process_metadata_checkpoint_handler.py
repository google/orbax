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

"""Backward-compatible re-exports for process metadata checkpoint handling."""

from orbax.checkpoint.experimental.emergency.multi_tier_checkpointing.process_metadata_checkpoint_handler import (
    ProcessMetadataCheckpointHandler,
)
from orbax.checkpoint.experimental.emergency.multi_tier_checkpointing.process_metadata_checkpoint_handler import (
    ProcessMetadataRestoreArgs,
)
from orbax.checkpoint.experimental.emergency.multi_tier_checkpointing.process_metadata_checkpoint_handler import (
    ProcessMetadataSaveArgs,
)

__all__ = [
    'ProcessMetadataCheckpointHandler',
    'ProcessMetadataRestoreArgs',
    'ProcessMetadataSaveArgs',
]
