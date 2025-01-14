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

"""Internal IO utilities for metadata of a checkpoint at root level."""

from orbax.checkpoint._src.metadata import checkpoint
from orbax.checkpoint._src.metadata import metadata_serialization_utils as utils

SerializedMetadata = checkpoint.SerializedMetadata
RootMetadata = checkpoint.RootMetadata


# Mapping from field name to field validation and processing function.
_FIELD_PROCESSORS = {
    'custom': utils.validate_and_process_custom,
}


def serialize(metadata: RootMetadata) -> SerializedMetadata:
  """Serializes `metadata` to a dictionary."""
  return {
      field_name: process_field(metadata.__getattribute__(field_name))
      for field_name, process_field in _FIELD_PROCESSORS.items()
  }


def deserialize(metadata_dict: SerializedMetadata) -> RootMetadata:
  """Deserializes `metadata_dict` to `RootMetadata`."""
  validated_metadata_dict = {}

  for f in _FIELD_PROCESSORS:
    validated_metadata_dict[f] = _FIELD_PROCESSORS[f](
        metadata_dict.get(f, None)
    )

  for k in metadata_dict:
    if k not in validated_metadata_dict:
      validated_metadata_dict['custom'][k] = utils.process_unknown_key(
          k, metadata_dict
      )

  return RootMetadata(**validated_metadata_dict)
