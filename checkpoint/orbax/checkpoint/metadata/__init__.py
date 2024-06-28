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

"""Defines exported symbols for package orbax.checkpoint.metadata."""

# pylint: disable=g-importing-member, g-bad-import-order

from orbax.checkpoint.metadata.checkpoint import CheckpointMetadata
from orbax.checkpoint.metadata.checkpoint import CheckpointMetadataStore
from orbax.checkpoint.metadata.checkpoint import checkpoint_metadata_store

from orbax.checkpoint.metadata.sharding import ShardingMetadata
from orbax.checkpoint.metadata.sharding import NamedShardingMetadata
from orbax.checkpoint.metadata.sharding import SingleDeviceShardingMetadata
from orbax.checkpoint.metadata.sharding import GSPMDShardingMetadata
from orbax.checkpoint.metadata.sharding import PositionalShardingMetadata
from orbax.checkpoint.metadata.sharding import from_jax_sharding
from orbax.checkpoint.metadata.sharding import from_serialized_string
from orbax.checkpoint.metadata.sharding import get_sharding_or_none
from orbax.checkpoint.metadata.sharding import ShardingTypes

from orbax.checkpoint.metadata.value import Metadata
from orbax.checkpoint.metadata.value import ArrayMetadata
from orbax.checkpoint.metadata.value import StringMetadata
from orbax.checkpoint.metadata.value import ScalarMetadata
