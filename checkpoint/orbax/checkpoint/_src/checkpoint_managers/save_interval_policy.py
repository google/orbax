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

"""Defines policies for the interval at which checkpoints are saved."""

import dataclasses
import typing
from typing import Container, Protocol


@dataclasses.dataclass(kw_only=True)
class StepInfo:
  """Relevant information about a checkpoint step."""

  step: int
  is_saving_in_progress: bool
  reached_preemption: bool


@typing.runtime_checkable
class SaveIntervalPolicy(Protocol):
  """A policy that defines when to save a checkpoint.

  Implementations should return True from `should_save` when saving a checkpoint
  is desired at the given step.
  """

  def should_save(
      self, step: StepInfo, *, previous_steps: list[StepInfo]
  ) -> bool:
    ...


@dataclasses.dataclass
class FixedIntervalPolicy(SaveIntervalPolicy):
  """Checkpoint at a fixed interval."""

  interval: int

  def should_save(
      self, step: StepInfo, *, previous_steps: list[StepInfo]
  ) -> bool:
    del previous_steps
    return step.step % self.interval == 0


@dataclasses.dataclass
class SpecificStepsPolicy(SaveIntervalPolicy):

  steps: Container[int]

  def should_save(
      self, step: StepInfo, *, previous_steps: list[StepInfo]
  ) -> bool:
    del previous_steps
    return step.step in self.steps


class ContinuousCheckpointingPolicy(SaveIntervalPolicy):
  """Checkpoint as often as possible, as long as a save is not ongoing."""

  def should_save(
      self, step: StepInfo, *, previous_steps: list[StepInfo]
  ) -> bool:
    del previous_steps
    return not step.is_saving_in_progress


class PreemptionCheckpointingPolicy(SaveIntervalPolicy):
  """Save a checkpoint when a preemption is detected."""

  def should_save(
      self, step: StepInfo, *, previous_steps: list[StepInfo]
  ) -> bool:
    del previous_steps
    return step.reached_preemption


class InitialSavePolicy(SaveIntervalPolicy):
  """Checkpoint as soon as possible if no checkpoints already exist."""

  def should_save(
      self, step: StepInfo, *, previous_steps: list[StepInfo]
  ) -> bool:
    del step
    return not previous_steps


@dataclasses.dataclass
class AnySavePolicy(SaveIntervalPolicy):

  policies: list[SaveIntervalPolicy]

  def should_save(
      self, step: StepInfo, *, previous_steps: list[StepInfo]
  ) -> bool:
    return any(
        policy.should_save(step, previous_steps=previous_steps)
        for policy in self.policies
    )


@dataclasses.dataclass
class AllSavePolicy(SaveIntervalPolicy):

  policies: list[SaveIntervalPolicy]

  def should_save(
      self, step: StepInfo, *, previous_steps: list[StepInfo]
  ) -> bool:
    return any(
        policy.should_save(step, previous_steps=previous_steps)
        for policy in self.policies
    )
