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

"""Defines policies for when a checkpoint is preserved."""

import typing
from typing import Any, Dict, Protocol, Sequence
from orbax.checkpoint._src.checkpoint_managers import preservation_policy as preservation_policy_lib
from orbax.checkpoint.experimental.v1._src.training.metadata import types


NestedDict = Dict[str, Any]
PyTree = Any

PreservationContext = preservation_policy_lib.PreservationContext
PreserveAll = preservation_policy_lib.PreserveAll
LatestN = preservation_policy_lib.LatestN
EveryNSeconds = preservation_policy_lib.EveryNSeconds
EveryNSteps = preservation_policy_lib.EveryNSteps
CustomSteps = preservation_policy_lib.CustomSteps
AnyPreservationPolicy = preservation_policy_lib.AnyPreservationPolicy
BestN = preservation_policy_lib.BestN


@typing.runtime_checkable
class PreservationPolicy(Protocol):
  """A policy that defines when checkpoints should be preserved."""

  def should_preserve(
      self,
      checkpoints: Sequence[types.CheckpointMetadata],
      *,
      context: PreservationContext,
  ) -> Sequence[bool]:
    """Indicates which checkpoints should be preserved.."""
    ...
