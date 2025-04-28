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

"""Metadata for `training.Checkpointer`."""

import dataclasses
from orbax.checkpoint._src.metadata import checkpoint_info
from orbax.checkpoint.experimental.v1._src.metadata import types as metadata_types
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types


CheckpointableMetadataT = metadata_types.CheckpointableMetadataT


CheckpointInfo = checkpoint_info.CheckpointInfo


class CheckpointMetadata(
    metadata_types.CheckpointMetadata[CheckpointableMetadataT]
):
  """Represents metadata for a single checkpoint (corresponding to a step).

  Like its parent, the class has a `metadata` attribute that is a generic type.

  See superclass documentation for more information, and for a list of base
  attributes. This class defines several additional attributes that are relevant
  to checkpoints in a sequence, but not necessarily to a singular checkpoint in
  isolation.

  Additional attributes:
    step: The step number of the checkpoint.
    metrics: User-provided metrics for the step (e.g. loss, accuracy, etc.)
  """

  def __init__(
      self,
      step: int,
      *,
      metadata: CheckpointableMetadataT,
      init_timestamp_nsecs: int | None = None,
      commit_timestamp_nsecs: int | None = None,
      custom_metadata: tree_types.JsonType | None = None,
      metrics: tree_types.JsonType | None = None,
  ):
    super().__init__(
        metadata=metadata,
        init_timestamp_nsecs=init_timestamp_nsecs,
        commit_timestamp_nsecs=commit_timestamp_nsecs,
        custom_metadata=custom_metadata,
    )
    self._step = step
    self._metrics = metrics

  @property
  def step(self) -> int:
    return self._step

  @property
  def metrics(self) -> tree_types.JsonType | None:
    return self._metrics


@dataclasses.dataclass(frozen=True, kw_only=True)
class RootMetadata:
  """Metadata of a sequence of checkpoint at root level (contains all steps).

  Attributes:
    custom_metadata: User-provided custom metadata. An arbitrary
      JSON-serializable dictionary the user can use to store additional
      information. The field is treated as opaque by Orbax.
  """

  custom_metadata: tree_types.JsonType | None = None
