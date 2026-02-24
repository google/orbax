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

"""Defines policies for when a checkpoint is saved."""

import typing
from typing import Protocol, Sequence

from orbax.checkpoint._src.checkpoint_managers import save_decision_policy
from orbax.checkpoint.experimental.v1._src.training.metadata import types

# Checkpoint as often as possible, as long as a save is not ongoing.
ContinuousCheckpointingPolicy = (
    save_decision_policy.ContinuousCheckpointingPolicy
)

# Checkpoint at a fixed interval.
FixedIntervalPolicy = save_decision_policy.FixedIntervalPolicy

# Checkpoint as soon as possible if no checkpoints already exist.
InitialSavePolicy = save_decision_policy.InitialSavePolicy

# Save a checkpoint when a preemption is detected.
PreemptionCheckpointingPolicy = (
    save_decision_policy.PreemptionCheckpointingPolicy
)

# Checkpoint at specific steps.
SpecificStepsPolicy = save_decision_policy.SpecificStepsPolicy

# Evaluates all policies and saves if any of them returns True.
AnySavePolicy = save_decision_policy.AnySavePolicy

# A container for auxiliary information (e.g., metrics) used to inform decisions
DecisionContext = save_decision_policy.DecisionContext


@typing.runtime_checkable
class SaveDecisionPolicy(Protocol):
  """A policy that defines when to save a checkpoint.

  SaveDecisionPolicy is a protocol that defines the interface for making
  checkpoint save decisions. Implementations of this protocol should define the
  logic for when to save a checkpoint based on the given step and previous
  steps. Before implementing a new policy, users should check whether any of
  Orbax's existing policies (e.g., `FixedIntervalPolicy`,
  `ContinuousCheckpointingPolicy`, etc.) can be used.

  Examples:

  1. Configuring Checkpointer
  `SaveDecisionPolicy` instances can be passed to
  :class:`~orbax.checkpoint.experimental.v1.training.Checkpointer` to control
  save frequency. For example::

    from orbax.checkpoint.experimental.v1 import training
    policies = training.save_decision_policies

    # Save every 1000 steps, or when a preemption is detected.
    policy = policies.AnySavePolicy([
        policies.FixedIntervalPolicy(1000),
        policies.PreemptionCheckpointingPolicy(),
    ])
    checkpointer = training.Checkpointer(directory, save_decision_policy=policy)

  2. Implementing a custom policy
  To define custom saving rules, users may implement the SaveDecisionPolicy
  interface::

    class SaveEveryNSteps(SaveDecisionPolicy):
      def __init__(self, n: int):
        self.n = n

      def should_save(
          self,
          step: CheckpointMetadata,
          previous_steps: Sequence[CheckpointMetadata],
          *,
          context: DecisionContext
      ) -> bool:
        # step.step accesses the integer index of the current training step.
        return step.step % self.n == 0

  Methods:
    should_save(step, previous_steps, *, context):
      Evaluates the current state to return a boolean indicating whether a
      checkpoint should be saved.

      Args:
        step (CheckpointMetadata): Metadata for the current training step,
          containing the step index, timestamp, and metadata.
        previous_steps (Sequence[CheckpointMetadata]): A chronological list of
          metadata for all steps where a checkpoint was successfully saved.
        context (DecisionContext): A container for auxiliary information,
          such as validation loss or performance metrics, used to inform the
          save decision.
  """

  def should_save(
      self,
      step: types.CheckpointMetadata,
      previous_steps: Sequence[types.CheckpointMetadata],
      *,
      context: DecisionContext
  ) -> bool:
    """Returns True if a checkpoint should be saved at the given step."""
    ...
