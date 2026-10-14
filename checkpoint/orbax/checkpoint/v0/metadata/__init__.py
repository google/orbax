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

"""Defines exported symbols for package orbax.checkpoint.v0.metadata."""

# pylint: disable=g-importing-member, g-bad-import-order

from orbax.checkpoint.v0._src.metadata.checkpoint import RootMetadata
from orbax.checkpoint.v0._src.metadata.checkpoint import StepMetadata
from orbax.checkpoint.v0._src.metadata.checkpoint import MetadataStore
from orbax.checkpoint.v0._src.metadata.checkpoint import metadata_store
from orbax.checkpoint.v0._src.metadata.step_metadata_serialization import get_step_metadata

from orbax.checkpoint.v0._src.metadata.sharding import ShardingMetadata
from orbax.checkpoint.v0._src.metadata.sharding import NamedShardingMetadata
from orbax.checkpoint.v0._src.metadata.sharding import SingleDeviceShardingMetadata

from orbax.checkpoint.v0.metadata import value
from orbax.checkpoint.v0.metadata import tree

# Prefer to use metadata.value instead of the following symbols.
from orbax.checkpoint.v0._src.metadata.value import Metadata
from orbax.checkpoint.v0.metadata.value import ArrayMetadata
from orbax.checkpoint.v0.metadata.value import StringMetadata
from orbax.checkpoint.v0.metadata.value import ScalarMetadata
from orbax.checkpoint.v0.metadata.value import StorageMetadata
