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
  serialized_metadata = {}
  if metadata.format is not None:
    serialized_metadata['format'] = metadata.format
  if metadata.user_metadata:
    serialized_metadata['user_metadata'] = metadata.user_metadata
  return serialized_metadata


def deserialize(metadata_dict: SerializedMetadata) -> RootMetadata:
  """Deserializes `metadata_dict` to `RootMetadata`."""
  validated_metadata_dict = {}

  if 'format' in metadata_dict:
    utils.validate_field(metadata_dict, 'format', str)
  validated_metadata_dict['format'] = metadata_dict.get('format', None)

  if 'user_metadata' in metadata_dict:
    utils.validate_field(metadata_dict, 'user_metadata', dict)
    for k in metadata_dict.get('user_metadata', {}) or {}:
      utils.validate_dict_entry(metadata_dict, 'user_metadata', k, str)
  validated_metadata_dict['user_metadata'] = metadata_dict.get(
      'user_metadata', {}
  )

  for k in metadata_dict:
    if k not in validated_metadata_dict:
      if 'user_metadata' in metadata_dict and metadata_dict['user_metadata']:
        raise ValueError(
            'Provided metadata contains unknown key %s, and the `user_metadata`'
            ' field is already defined.' % k
        )
      logging.warning(
          'Provided metadata contains unknown key %s. Adding it to custom.', k
      )
      validated_metadata_dict['user_metadata'][k] = metadata_dict[k]

  return RootMetadata(**validated_metadata_dict)
