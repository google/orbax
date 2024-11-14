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

import dataclasses
from typing import Any

from absl import logging
from orbax.checkpoint._src.logging import step_statistics
from orbax.checkpoint._src.metadata import checkpoint
from orbax.checkpoint._src.metadata import metadata_serialization_utils as utils

SerializedMetadata = checkpoint.SerializedMetadata
StepMetadata = checkpoint.StepMetadata
ItemMetadata = checkpoint.ItemMetadata
StepStatistics = step_statistics.SaveStepStatistics


def serialize(metadata: StepMetadata) -> SerializedMetadata:
  """Serializes `metadata` to a dictionary."""

  # Part of the StepMetadata api for user convenience, but not saved to disk.
  if metadata.item_metadata is not None:
    just_item_names = {k: None for k in metadata.item_metadata.keys()}
  else:
    just_item_names = None

  # Save only float performance metrics.
  performance_metrics = metadata.performance_metrics
  float_metrics = {
      metric: val
      for metric, val in dataclasses.asdict(performance_metrics).items()
      if isinstance(val, float)
  }

  return {
      'format': metadata.format,
      'item_handlers': metadata.item_handlers,
      'item_metadata': just_item_names,
      'metrics': metadata.metrics,
      'performance_metrics': float_metrics,
      'init_timestamp_nsecs': metadata.init_timestamp_nsecs,
      'commit_timestamp_nsecs': metadata.commit_timestamp_nsecs,
      'custom': metadata.custom,
  }


def deserialize(
    metadata_dict: SerializedMetadata,
    item_metadata: ItemMetadata | None = None,
    metrics: dict[str, Any] | None = None,
) -> StepMetadata:
  """Deserializes `metadata_dict` and other kwargs to `StepMetadata`."""
  validated_metadata_dict = {}

  utils.validate_field(metadata_dict, 'format', str)
  validated_metadata_dict['format'] = metadata_dict.get('format', None)

  utils.validate_field(metadata_dict, 'item_handlers', dict)
  for k in metadata_dict.get('item_handlers', {}) or {}:
    utils.validate_dict_entry(metadata_dict, 'item_handlers', k, str, str)
  validated_metadata_dict['item_handlers'] = metadata_dict.get(
      'item_handlers', {}
  )

  utils.validate_field(metadata_dict, 'item_metadata', dict)
  for k in metadata_dict.get('item_metadata', {}) or {}:
    utils.validate_dict_entry(metadata_dict, 'item_metadata', k, str)
  validated_metadata_dict['item_metadata'] = metadata_dict.get(
      'item_metadata', None
  )
  if item_metadata is not None:
    if validated_metadata_dict['item_metadata'] is None:
      validated_metadata_dict['item_metadata'] = {}
    utils.validate_field(item_metadata, 'item_metadata', dict)
    for k, v in item_metadata.items():
      utils.validate_dict_entry(metadata_dict, 'item_metadata', k, str)
      validated_metadata_dict['item_metadata'][k] = v

  utils.validate_field(metadata_dict, 'metrics', dict)
  for k in metadata_dict.get('metrics', {}) or {}:
    utils.validate_dict_entry(metadata_dict, 'metrics', k, str)
  validated_metadata_dict['metrics'] = metadata_dict.get('metrics', {})
  if metrics is not None:
    utils.validate_field(metadata_dict, 'metrics', dict)
    for k, v in metrics.items():
      utils.validate_dict_entry(metadata_dict, 'metrics', k, str)
      validated_metadata_dict['metrics'][k] = v

  utils.validate_field(metadata_dict, 'performance_metrics', dict)
  for k in metadata_dict.get('performance_metrics', {}) or {}:
    utils.validate_dict_entry(
        metadata_dict, 'performance_metrics', k, str, float
    )
  validated_metadata_dict['performance_metrics'] = StepStatistics(
      **metadata_dict.get('performance_metrics', {})
  )

  utils.validate_field(metadata_dict, 'init_timestamp_nsecs', int)
  validated_metadata_dict['init_timestamp_nsecs'] = (
      metadata_dict.get('init_timestamp_nsecs', None)
  )

  utils.validate_field(metadata_dict, 'commit_timestamp_nsecs', int)
  validated_metadata_dict['commit_timestamp_nsecs'] = (
      metadata_dict.get('commit_timestamp_nsecs', None)
  )

  utils.validate_field(metadata_dict, 'custom', dict)
  for k in metadata_dict.get('custom', {}) or {}:
    utils.validate_dict_entry(metadata_dict, 'custom', k, str)
  validated_metadata_dict['custom'] = metadata_dict.get('custom', {})

  for k in metadata_dict:
    if k not in validated_metadata_dict:
      logging.warning(
          'Provided metadata contains unknown key %s, ignoring.', k
      )

  return StepMetadata(**validated_metadata_dict)
