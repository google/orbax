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

"""Internal IO utilities for metadata of a checkpoint at root level."""
from __future__ import annotations

import dataclasses
from typing import Any

from orbax.checkpoint._src.metadata import checkpoint
from orbax.checkpoint._src.metadata import metadata_serialization_utils as utils

SerializedMetadata = checkpoint.SerializedMetadata
RootMetadata = checkpoint.RootMetadata


@dataclasses.dataclass
class InternalRootMetadata:
  """Internal representation of `RootMetadata`.

  See documentation on `InternalCheckpointMetadata` for more design context.
  """

  internal_metadata: dict[str, Any] | None = dataclasses.field(
      default_factory=dict,
      metadata={'processor': utils.validate_and_process_internal_metadata},
  )
  custom_metadata: dict[str, Any] | None = dataclasses.field(
      default_factory=dict,
      metadata={'processor': utils.validate_and_process_custom_metadata},
  )

  def serialize(self) -> SerializedMetadata:
    """Serializes `self` to a JSON dictionary and performs validation."""
    return {
        field.name: field.metadata['processor'](getattr(self, field.name))
        for field in dataclasses.fields(self)
    }

  @classmethod
  def deserialize(
      cls, metadata_dict: SerializedMetadata
  ) -> InternalRootMetadata:
    """Deserializes `metadata_dict` to `InternalRootMetadata`."""
    assert isinstance(metadata_dict, dict)
    fields = dataclasses.fields(InternalRootMetadata)
    field_names = {field.name for field in fields}

    field_processor_args = {
        field_name: (metadata_dict.get(field_name, None),)
        for field_name in field_names
    }

    validated_metadata_dict = {
        field.name: field.metadata['processor'](
            *field_processor_args[field.name]
        )
        for field in fields
    }

    for k in metadata_dict:
      if k not in validated_metadata_dict:
        validated_metadata_dict['custom_metadata'][k] = (
            utils.process_unknown_key(k, metadata_dict)
        )
    return InternalRootMetadata(**validated_metadata_dict)

  def to_root_metadata(self) -> RootMetadata:
    return RootMetadata(
        internal_metadata=self.internal_metadata,
        custom_metadata=self.custom_metadata,
    )

  @classmethod
  def from_root_metadata(
      cls, root_metadata: RootMetadata
  ) -> InternalRootMetadata:
    return InternalRootMetadata(
        internal_metadata=root_metadata.internal_metadata,
        custom_metadata=root_metadata.custom_metadata,
    )


def serialize(metadata: RootMetadata) -> SerializedMetadata:
  """Serializes `metadata` to a dictionary."""
  return InternalRootMetadata.from_root_metadata(metadata).serialize()


def deserialize(metadata_dict: SerializedMetadata) -> RootMetadata:
  """Deserializes `metadata_dict` to `RootMetadata`."""
  return InternalRootMetadata.deserialize(metadata_dict).to_root_metadata()
