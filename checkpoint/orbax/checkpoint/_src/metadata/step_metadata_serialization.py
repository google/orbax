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

"""Internal IO utilities for metadata of a checkpoint at step level."""

from typing import Any

from orbax.checkpoint._src.logging import step_statistics
from orbax.checkpoint._src.metadata import checkpoint
from orbax.checkpoint._src.metadata import metadata_serialization_utils as utils

SerializedMetadata = checkpoint.SerializedMetadata
StepMetadata = checkpoint.StepMetadata
CompositeItemMetadata = checkpoint.CompositeItemMetadata
SingleItemMetadata = checkpoint.SingleItemMetadata
StepStatistics = step_statistics.SaveStepStatistics


# Mapping from field name to field validation and processing function.
_FIELD_PROCESSORS = {
    'item_handlers': utils.validate_and_process_item_handlers,
    'item_metadata': utils.validate_and_process_item_metadata,
    'metrics': utils.validate_and_process_metrics,
    'performance_metrics': utils.validate_and_process_performance_metrics,
    'init_timestamp_nsecs': utils.validate_and_process_init_timestamp_nsecs,
    'commit_timestamp_nsecs': utils.validate_and_process_commit_timestamp_nsecs,
    'custom': utils.validate_and_process_custom,
}


def serialize(metadata: StepMetadata) -> SerializedMetadata:
  """Serializes `metadata` to a dictionary."""

  serialization_processors = _FIELD_PROCESSORS.copy()
  # Per item metadata is already saved in the item subdirectories.
  del serialization_processors['item_metadata']

  return {
      field_name: process_field(metadata.__getattribute__(field_name))
      for field_name, process_field in serialization_processors.items()
  }

  # return {
  #     'item_handlers': _FIELD_PROCESSORS['item_handlers'](
  #         metadata.item_handlers
  #     ),
  #     # Per item metadata is already saved in the item subdirectories.
  #     'metrics': _FIELD_PROCESSORS['metrics'](metadata.metrics),
  #     'performance_metrics': _FIELD_PROCESSORS['performance_metrics'](
  #         metadata.performance_metrics
  #     ),
  #     'init_timestamp_nsecs': _FIELD_PROCESSORS['init_timestamp_nsecs'](
  #         metadata.init_timestamp_nsecs
  #     ),
  #     'commit_timestamp_nsecs': _FIELD_PROCESSORS['commit_timestamp_nsecs'](
  #         metadata.commit_timestamp_nsecs
  #     ),
  #     'custom': _FIELD_PROCESSORS['custom'](metadata.custom),
  # }


def serialize_for_update(**kwargs) -> SerializedMetadata:
  """Validates and serializes `kwargs` to a dictionary.

  To be used with MetadataStore.update().

  Args:
    **kwargs: The kwargs to be serialized.

  Returns:
    A dictionary of the serialized kwargs.
  """
  validated_kwargs = {}

  for k in kwargs:
    if k not in _FIELD_PROCESSORS:
      raise ValueError('Provided metadata contains unknown key %s.' % k)
    validated_kwargs[k] = _FIELD_PROCESSORS[k](kwargs[k])

  return validated_kwargs


def deserialize(
    metadata_dict: SerializedMetadata,
    item_metadata: CompositeItemMetadata | SingleItemMetadata | None = None,
    metrics: dict[str, Any] | None = None,
) -> StepMetadata:
  """Deserializes `metadata_dict` and other kwargs to `StepMetadata`."""
  validated_metadata_dict = {}

  validated_metadata_dict['item_handlers'] = _FIELD_PROCESSORS['item_handlers'](
      metadata_dict.get('item_handlers', None)
  )

  validated_metadata_dict['item_metadata'] = _FIELD_PROCESSORS['item_metadata'](
      item_metadata
  )

  validated_metadata_dict['metrics'] = _FIELD_PROCESSORS['metrics'](
      metadata_dict.get('metrics', None), metrics
  )

  validated_metadata_dict['performance_metrics'] = StepStatistics(
      **_FIELD_PROCESSORS['performance_metrics'](
          metadata_dict.get('performance_metrics', None)
      )
  )

  validated_metadata_dict['init_timestamp_nsecs'] = (
      _FIELD_PROCESSORS['init_timestamp_nsecs'](
          metadata_dict.get('init_timestamp_nsecs', None)
      )
  )

  validated_metadata_dict['commit_timestamp_nsecs'] = (
      _FIELD_PROCESSORS['commit_timestamp_nsecs'](
          metadata_dict.get('commit_timestamp_nsecs', None)
      )
  )

  validated_metadata_dict['custom'] = _FIELD_PROCESSORS['custom'](
      metadata_dict.get('custom', None)
  )

  for k in metadata_dict:
    if k not in validated_metadata_dict:
      validated_metadata_dict['custom'][k] = (
          utils.process_unknown_key(k, metadata_dict)
      )

  return StepMetadata(**validated_metadata_dict)
