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

"""Manages metadata of checkpoints at root and step level (not item level)."""

# pylint: disable=g-bad-import-order,g-importing-member,unused-import,unused-wildcard-import,wildcard-import
import sys
from orbax.checkpoint.v0._src.metadata import checkpoint as _mod
from orbax.checkpoint.v0._src.metadata.checkpoint import *
from orbax.checkpoint.v0._src.metadata.checkpoint import _STEP_METADATA_FILENAME
from orbax.checkpoint.v0._src.metadata.checkpoint import _ROOT_METADATA_FILENAME
from orbax.checkpoint.v0._src.metadata.checkpoint import _LEGACY_ROOT_METADATA_FILENAME
from orbax.checkpoint.v0._src.metadata.checkpoint import _sanitize_metadata_path
from orbax.checkpoint.v0._src.metadata.checkpoint import _MetadataStoreImpl
from orbax.checkpoint.v0._src.metadata.checkpoint import _BlockingMetadataStore
from orbax.checkpoint.v0._src.metadata.checkpoint import _NonBlockingMetadataStore
from orbax.checkpoint.v0._src.metadata.checkpoint import _METADATA_STORE_FOR_WRITES
from orbax.checkpoint.v0._src.metadata.checkpoint import _METADATA_STORE_FOR_READS
from orbax.checkpoint.v0._src.metadata.checkpoint import _METADATA_STORE_NON_BLOCKING_FOR_READS

sys.modules[__name__] = _mod
