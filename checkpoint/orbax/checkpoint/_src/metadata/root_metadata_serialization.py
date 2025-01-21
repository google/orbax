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

from absl import logging
from orbax.checkpoint._src.metadata import checkpoint
from orbax.checkpoint._src.metadata import metadata_serialization_utils as utils

SerializedMetadata = checkpoint.SerializedMetadata
RootMetadata = checkpoint.RootMetadata


def serialize(metadata: RootMetadata) -> SerializedMetadata:
  """Serializes `metadata` to a dictionary."""
  return {
      'custom': metadata.custom_metadata,
  }


def deserialize(metadata_dict: SerializedMetadata) -> RootMetadata:
  """Deserializes `metadata_dict` to `RootMetadata`."""
  validated_metadata_dict = {}

  if 'custom' in metadata_dict:
    utils.validate_type(metadata_dict['custom'], dict)
    for k in metadata_dict.get('custom', {}) or {}:
      utils.validate_type(k, str)
  validated_metadata_dict['custom'] = metadata_dict.get('custom', {})

  for k in metadata_dict:
    if k not in validated_metadata_dict:
      if 'custom' in metadata_dict and metadata_dict['custom']:
        raise ValueError(
            'Provided metadata contains unknown key %s, and the custom field '
            'is already defined.' % k
        )
      logging.warning(
          'Provided metadata contains unknown key %s. Adding it to custom.', k
      )
      validated_metadata_dict['custom'][k] = metadata_dict[k]

  # Rename to `custom_metadata`.
  validated_metadata_dict['custom_metadata'] = validated_metadata_dict.pop(
      'custom'
  )
  return RootMetadata(**validated_metadata_dict)
