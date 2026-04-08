# Copyright 2026 The Orbax Authors.
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

"""Metadata for :py:class:`~.v1.training.Checkpointer`."""

from __future__ import annotations

import dataclasses
import datetime
import pprint
import typing
from typing import TypeVar

from orbax.checkpoint.experimental.v1._src.metadata import types as metadata_types
from orbax.checkpoint.experimental.v1._src.path import types as path_types
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types

CheckpointableMetadataT = TypeVar('CheckpointableMetadataT')


@typing.final
class CheckpointMetadata(
    metadata_types.CheckpointMetadata[CheckpointableMetadataT],
):
  """Represents metadata for a single checkpoint (corresponding to a step).

  Like its parent, the class has a `metadata` attribute that is a generic type.
  The `.metadata` attribute contains checkpointable-specific metadata.
  If a PyTree was saved, it will contain :py:class:`~.v1.PyTreeMetadata`,
  otherwise if `Checkpointable`s were saved, it will be a dictionary mapping
  names to metadata.

  The Orbax checkpointing API provides two symmetric levels of interaction:

  1. **Higher level** (sequence-of-steps API): Accessed via
     :py:class:`~.v1.training.Checkpointer`.

  2. **Lower level** (individual path API): Accessed via free functions.

  `CheckpointMetadata` objects are returned by both API levels using the same
  core methods (:py:func:`~.v1.pytree_metadata` and
  :py:func:`~.v1.checkpointables_metadata`), reflecting this inherent symmetry.

  See superclass documentation for more information, and for a list of base
  attributes. This class defines several additional attributes that are relevant
  to checkpoints in a sequence, but not necessarily to a singular checkpoint in
  isolation.

  Example Usage::

    from orbax.checkpoint import v1 as ocp

    # Higher level (sequence-of-steps API)
    with ocp.training.Checkpointer('/path/to/my/checkpoints') as ckptr:
      ckpt_meta = ckptr.pytree_metadata(100)

    # Lower level (individual path API)
    ckpt_meta = ocp.pytree_metadata('/path/to/my/checkpoints/100')

    # Inspect checkpoint-level properties
    print(f'Init time (ns): {ckpt_meta.init_timestamp_nsecs}')
    print(f'Commit time (ns): {ckpt_meta.commit_timestamp_nsecs}')
    print(f'Custom metadata: {ckpt_meta.custom_metadata}')

    # The `.metadata` field contains checkpointable-specific metadata,
    # which will be `PyTreeMetadata` or dict[str, CheckpointableMetadata]
    # depending on what was saved.
    print(f'Checkpointable metadata: {ckpt_meta.metadata}')

  See also :py:class:`RootMetadata`.

  See the parent class, :py:class:`~.v1.CheckpointMetadata`, for base
  attributes.

  Additional Attributes:
    step: The step number of the checkpoint.
    metrics: An optional dictionary containing user-provided metrics saved
      alongside the checkpoint.
  """

  def __init__(
      self,
      step: int,
      path: path_types.Path,
      *,
      metadata: CheckpointableMetadataT,
      init_timestamp_nsecs: int | None = None,
      commit_timestamp_nsecs: int | None = None,
      custom_metadata: tree_types.JsonType | None = None,
      metrics: tree_types.JsonType | None = None,
  ):
    super().__init__(
        path=path,
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

  @property
  def time(self) -> datetime.datetime:
    return datetime.datetime.fromtimestamp(
        self.commit_timestamp_nsecs / 1e9, tz=datetime.timezone.utc
    )

  def _properties_strings(self) -> dict[str, str]:
    properties = super()._properties_strings()
    return {
        'step': str(self.step),
        **properties,
        'metrics': str(self.metrics),
    }


@typing.final
@dataclasses.dataclass(frozen=True, kw_only=True)
class RootMetadata:
  """Metadata of a sequence of checkpoint at root level (contains all steps).

  This class represents the top-level metadata for an entire checkpointing
  directory, distinct from step-specific metadata. It associates the physical
  storage location of the sequence with arbitrary, user-defined information
  that applies to all steps (e.g., experiment configuration).

  Example Usage:
    `RootMetadata` objects are returned by
    :py:meth:`~.v1.training.Checkpointer.root_metadata`.

    It can be used to inspect checkpoint-wide information, such as experiment
    configuration::

      import orbax.checkpoint.v1 as ocp
      ckptr = ocp.training.Checkpointer('/path/to/my/checkpoints')
      root_meta = ckptr.root_metadata()

      print(f'Directory: {root_meta.directory}')
      print(f'Custom metadata: {root_meta.custom_metadata}')

    See also :py:class:`CheckpointMetadata`.

  Attributes:
    directory: The directory of the checkpoint sequence.
    custom_metadata: User-provided custom metadata. An arbitrary
      JSON-serializable dictionary the user can use to store additional
      information. The field is treated as opaque by Orbax.
  """

  directory: path_types.Path
  custom_metadata: tree_types.JsonType | None = None

  def _properties_strings(self) -> dict[str, str]:
    return {
        'directory': str(self.directory),
        'custom_metadata': str(self.custom_metadata),
    }

  def __repr__(self):
    return f'RootMetadata({pprint.pformat(self._properties_strings())})'
