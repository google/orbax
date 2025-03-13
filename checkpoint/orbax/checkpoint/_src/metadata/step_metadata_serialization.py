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

from etils import epath
from orbax.checkpoint._src.logging import step_statistics
from orbax.checkpoint._src.metadata import checkpoint
from orbax.checkpoint._src.metadata import metadata_serialization_utils as utils

SerializedMetadata = checkpoint.SerializedMetadata
StepMetadata = checkpoint.StepMetadata
CompositeCheckpointHandlerTypeStrs = (
    checkpoint.CompositeCheckpointHandlerTypeStrs
)
CheckpointHandlerTypeStr = checkpoint.CheckpointHandlerTypeStr
CompositeItemMetadata = checkpoint.CompositeItemMetadata
SingleItemMetadata = checkpoint.SingleItemMetadata
StepStatistics = step_statistics.SaveStepStatistics


def get_step_metadata(path: epath.PathLike) -> StepMetadata:
  """Returns StepMetadata for a given checkpoint directory."""
  metadata_file_path = checkpoint.step_metadata_file_path(path)
  serialized_metadata = checkpoint.metadata_store(enable_write=False).read(
      metadata_file_path
  )
  if serialized_metadata is None:
    raise ValueError(f'Step metadata not found at {metadata_file_path}')
  return deserialize(serialized_metadata)


def serialize(metadata: StepMetadata) -> SerializedMetadata:
  """Serializes `metadata` to a dictionary."""

  # Per item metadata is already saved in the item subdirectories.
  del metadata.item_metadata

  # Save only float performance metrics.
  performance_metrics = metadata.performance_metrics
  float_metrics = {
      metric: val
      for metric, val in dataclasses.asdict(performance_metrics).items()
      if isinstance(val, float)
  }

  return {
      'item_handlers': metadata.item_handlers,
      'metrics': metadata.metrics,
      'performance_metrics': float_metrics,
      'init_timestamp_nsecs': metadata.init_timestamp_nsecs,
      'commit_timestamp_nsecs': metadata.commit_timestamp_nsecs,
      'custom_metadata': metadata.custom_metadata,
  }


# TODO(adamcogdell): Reduce code duplication with deserialize().
def serialize_for_update(**kwargs) -> SerializedMetadata:
  """Validates and serializes `kwargs` to a dictionary.

  To be used with MetadataStore.update().

  Args:
    **kwargs: The kwargs to be serialized.

  Returns:
    A dictionary of the serialized kwargs.
  """
  validated_kwargs = {}

  if 'item_handlers' in kwargs:
    utils.validate_type(kwargs['item_handlers'], [dict, str])
    item_handlers = kwargs.get('item_handlers')
    if isinstance(item_handlers, CompositeCheckpointHandlerTypeStrs):
      for k in kwargs.get('item_handlers'):
        utils.validate_type(k, str)
      validated_kwargs['item_handlers'] = item_handlers
    elif isinstance(item_handlers, CheckpointHandlerTypeStr):
      validated_kwargs['item_handlers'] = item_handlers

  if 'metrics' in kwargs:
    utils.validate_type(kwargs['metrics'], dict)
    for k in kwargs.get('metrics', {}) or {}:
      utils.validate_type(k, str)
    validated_kwargs['metrics'] = kwargs.get('metrics', {})

  if 'performance_metrics' in kwargs:
    utils.validate_type(kwargs['performance_metrics'], [dict, StepStatistics])
    performance_metrics = kwargs.get('performance_metrics', {})
    if isinstance(performance_metrics, StepStatistics):
      performance_metrics = dataclasses.asdict(performance_metrics)
    float_metrics = {
        metric: val
        for metric, val in performance_metrics.items()
        if isinstance(val, float)
    }
    validated_kwargs['performance_metrics'] = float_metrics

  if 'init_timestamp_nsecs' in kwargs:
    utils.validate_type(kwargs['init_timestamp_nsecs'], int)
    validated_kwargs['init_timestamp_nsecs'] = (
        kwargs.get('init_timestamp_nsecs', None)
    )

  if 'commit_timestamp_nsecs' in kwargs:
    utils.validate_type(kwargs['commit_timestamp_nsecs'], int)
    validated_kwargs['commit_timestamp_nsecs'] = (
        kwargs.get('commit_timestamp_nsecs', None)
    )

  if 'custom_metadata' in kwargs:
    if kwargs['custom_metadata'] is None:
      validated_kwargs['custom_metadata'] = {}
    else:
      utils.validate_type(kwargs['custom_metadata'], dict)
      for k in kwargs.get('custom_metadata', {}) or {}:
        utils.validate_type(k, str)
      validated_kwargs['custom_metadata'] = kwargs.get('custom_metadata', {})

  for k in kwargs:
    if k not in validated_kwargs:
      raise ValueError('Provided metadata contains unknown key %s.' % k)

  return validated_kwargs


def deserialize(
    metadata_dict: SerializedMetadata,
    item_metadata: CompositeItemMetadata | SingleItemMetadata | None = None,
    metrics: dict[str, Any] | None = None,
) -> StepMetadata:
  """Deserializes `metadata_dict` and other kwargs to `StepMetadata`."""
  validated_metadata_dict = {}

  validated_metadata_dict['item_handlers'] = (
      utils.validate_and_process_item_handlers(
          metadata_dict.get('item_handlers', None)
      )
  )

  validated_metadata_dict['item_metadata'] = (
      utils.validate_and_process_item_metadata(item_metadata)
  )

  validated_metadata_dict['metrics'] = utils.validate_and_process_metrics(
      metadata_dict.get('metrics', None), metrics
  )

  validated_metadata_dict['performance_metrics'] = StepStatistics(
      **utils.validate_and_process_performance_metrics(
          metadata_dict.get('performance_metrics', None)
      )
  )

  validated_metadata_dict['init_timestamp_nsecs'] = (
      utils.validate_and_process_init_timestamp_nsecs(
          metadata_dict.get('init_timestamp_nsecs', None)
      )
  )

  validated_metadata_dict['commit_timestamp_nsecs'] = (
      utils.validate_and_process_commit_timestamp_nsecs(
          metadata_dict.get('commit_timestamp_nsecs', None)
      )
  )

  validated_metadata_dict['custom_metadata'] = (
      utils.validate_and_process_custom_metadata(
          metadata_dict.get('custom_metadata', None)
      )
  )

  for k in metadata_dict:
    if k not in validated_metadata_dict:
      validated_metadata_dict['custom_metadata'][k] = utils.process_unknown_key(
          k, metadata_dict
      )

  return StepMetadata(**validated_metadata_dict)
