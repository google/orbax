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

"""Internal IO utilities for metadata of a checkpoint at step level."""

from __future__ import annotations

import dataclasses
from typing import Any

from etils import epath
from orbax.checkpoint._src.logging import step_statistics
from orbax.checkpoint._src.metadata import checkpoint
from orbax.checkpoint._src.metadata import metadata_serialization_utils as utils

SerializedMetadata = checkpoint.SerializedMetadata
CheckpointHandlerTypeStr = checkpoint.CheckpointHandlerTypeStr
StepMetadata = checkpoint.StepMetadata
CompositeItemMetadata = checkpoint.CompositeItemMetadata
SingleItemMetadata = checkpoint.SingleItemMetadata
StepStatistics = step_statistics.SaveStepStatistics


@dataclasses.dataclass
class InternalCheckpointMetadata:
  """An internal representation of checkpoint metadata.

  `CheckpointMetadata`, or `StepMetadata` (depending on V0 or V1), is a
  user-facing representation. It is a logical entity containing all the
  properties a user wishes to access. It is not tied to the actual
  representation of the metadata in the checkpoint, and is divorced from any
  internal implementation details.

  `InternalCheckpointMetadata` is a Python representation of the physical
  JSON file storing metadata properties. It excludes any properties that are
  handled separately, such as `item_metadata` or `descriptor`.

  The class provides `serialize` and `deserialize` methods allowing conversion
  between a JSON dict and this class, including validation.

  It also provides methods that convert between this class and the user-facing
  class. In particular, when converting to the user-facing class, it accepts
  additional metadata properties desired by the user that are stored separately
  from the JSON metadata file.
  """

  item_handlers: (
      dict[str, CheckpointHandlerTypeStr] | CheckpointHandlerTypeStr | None
  ) = dataclasses.field(
      default=None,
      metadata={'processor': utils.validate_and_process_item_handlers},
  )
  metrics: dict[str, Any] = dataclasses.field(
      default_factory=dict,
      metadata={'processor': utils.validate_and_process_metrics},
  )
  performance_metrics: StepStatistics = dataclasses.field(
      default_factory=StepStatistics,
      metadata={'processor': utils.validate_and_process_performance_metrics},
  )
  init_timestamp_nsecs: int | None = dataclasses.field(
      default=None,
      metadata={'processor': utils.validate_and_process_init_timestamp_nsecs},
  )
  commit_timestamp_nsecs: int | None = dataclasses.field(
      default=None,
      metadata={'processor': utils.validate_and_process_commit_timestamp_nsecs},
  )
  custom_metadata: dict[str, Any] = dataclasses.field(
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
  ) -> InternalCheckpointMetadata:
    """Deserializes `metadata_dict` to `InternalCheckpointMetadata`."""
    if not isinstance(metadata_dict, dict):
      raise ValueError(
          'Metadata dict must be a dictionary, but got'
          f' {type(metadata_dict).__name__}'
      )
    fields = dataclasses.fields(InternalCheckpointMetadata)
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

    validated_metadata_dict['performance_metrics'] = StepStatistics(
        **validated_metadata_dict['performance_metrics']
    )

    if not isinstance(validated_metadata_dict['custom_metadata'], dict):
      raise ValueError(
          'Custom metadata must be a dictionary, but got'
          f' {type(validated_metadata_dict["custom_metadata"]).__name__}'
      )
    for k in metadata_dict:
      if k not in validated_metadata_dict:
        validated_metadata_dict['custom_metadata'][k] = (
            utils.process_unknown_key(k, metadata_dict)
        )
    return InternalCheckpointMetadata(**validated_metadata_dict)

  def to_step_metadata(
      self,
      *,
      item_metadata: CompositeItemMetadata | SingleItemMetadata | None = None,
      additional_metrics: dict[str, Any] | None = None,
  ) -> StepMetadata:
    """Construct user-facing representation."""
    item_metadata = utils.validate_and_process_item_metadata(item_metadata)
    metrics = utils.validate_and_process_metrics(
        self.metrics, additional_metrics
    )
    return StepMetadata(
        item_handlers=self.item_handlers,
        item_metadata=item_metadata,
        metrics=metrics,
        performance_metrics=self.performance_metrics,
        init_timestamp_nsecs=self.init_timestamp_nsecs,
        commit_timestamp_nsecs=self.commit_timestamp_nsecs,
        custom_metadata=self.custom_metadata,
    )

  @classmethod
  def from_step_metadata(
      cls, step_metadata: StepMetadata
  ) -> InternalCheckpointMetadata:
    """Return internal representation, dropping fields handled separately."""
    return cls(
        item_handlers=step_metadata.item_handlers,
        metrics=step_metadata.metrics,
        performance_metrics=step_metadata.performance_metrics,
        init_timestamp_nsecs=step_metadata.init_timestamp_nsecs,
        commit_timestamp_nsecs=step_metadata.commit_timestamp_nsecs,
        custom_metadata=step_metadata.custom_metadata,
    )

  # TODO(b/407609701): Add `to_checkpoint_metadata` (like `to_step_metadata`)
  # for conversion to V1 metadata after dependencies are separated.

  @classmethod
  def create(
      cls,
      *,
      handler_typestrs: dict[str, str],
      init_timestamp_nsecs: int,
      commit_timestamp_nsecs: int,
      custom_metadata: dict[str, Any] | None = None,
  ) -> InternalCheckpointMetadata:
    """Return internal representation, dropping fields handled separately."""
    return cls(
        item_handlers=handler_typestrs,
        init_timestamp_nsecs=init_timestamp_nsecs,
        commit_timestamp_nsecs=commit_timestamp_nsecs,
        custom_metadata=custom_metadata,
    )


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
  metadata = InternalCheckpointMetadata.from_step_metadata(metadata)
  return metadata.serialize()


def serialize_for_update(**kwargs) -> SerializedMetadata:
  """Validates and serializes `kwargs` to a dictionary.

  To be used with MetadataStore.update().

  Args:
    **kwargs: The kwargs to be serialized.

  Returns:
    A dictionary of the serialized kwargs.
  """
  fields = dataclasses.fields(InternalCheckpointMetadata)
  field_names = {field.name for field in fields}

  for k in kwargs:
    if k not in field_names:
      raise ValueError('Provided metadata contains unknown key %s.' % k)

  validated_kwargs = {
      field.name: field.metadata['processor'](kwargs[field.name])
      for field in fields
      if field.name in kwargs
  }

  return validated_kwargs


def deserialize(
    metadata_dict: SerializedMetadata,
    item_metadata: CompositeItemMetadata | SingleItemMetadata | None = None,
    metrics: dict[str, Any] | None = None,
) -> StepMetadata:
  """Deserializes `metadata_dict` and other kwargs to `InternalCheckpointMetadata`."""
  return InternalCheckpointMetadata.deserialize(metadata_dict).to_step_metadata(
      item_metadata=item_metadata, additional_metrics=metrics
  )
