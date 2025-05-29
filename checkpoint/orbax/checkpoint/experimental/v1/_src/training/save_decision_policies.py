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

"""Defines policies for when a checkpoint is saved."""

import typing
from typing import Protocol, Sequence

from orbax.checkpoint._src.checkpoint_managers import save_decision_policy
from orbax.checkpoint.experimental.v1._src.training.metadata import types

ContinuousCheckpointingPolicy = (
    save_decision_policy.ContinuousCheckpointingPolicy
)
FixedIntervalPolicy = save_decision_policy.FixedIntervalPolicy
InitialSavePolicy = save_decision_policy.InitialSavePolicy
PreemptionCheckpointingPolicy = (
    save_decision_policy.PreemptionCheckpointingPolicy
)
SpecificStepsPolicy = save_decision_policy.SpecificStepsPolicy
AnySavePolicy = save_decision_policy.AnySavePolicy
DecisionContext = save_decision_policy.DecisionContext


@typing.runtime_checkable
class SaveDecisionPolicy(Protocol):
  """A policy that defines when to save a checkpoint."""

  def should_save(
      self,
      step: types.CheckpointMetadata,
      previous_steps: Sequence[types.CheckpointMetadata],
      *,
      context: DecisionContext
  ) -> bool:
    """Returns True if a checkpoint should be saved at the given step."""
    ...
