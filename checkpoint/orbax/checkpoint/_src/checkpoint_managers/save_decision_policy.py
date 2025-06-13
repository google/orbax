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

import dataclasses
import datetime
import typing
from typing import Container, Protocol, Sequence
from orbax.checkpoint import options as options_lib
from orbax.checkpoint._src.checkpoint_managers import policy_checkpoint_info
from orbax.checkpoint._src.futures import signaling_client
from orbax.checkpoint._src.futures import synchronization
from orbax.checkpoint._src.multihost import multihost


PolicyCheckpointInfo = policy_checkpoint_info.PolicyCheckpointInfo


@dataclasses.dataclass(kw_only=True)
class DecisionContext:
  """Additional properties for making a save decision."""

  is_saving_in_progress: bool
  reached_preemption: bool
  multiprocessing_options: options_lib.MultiprocessingOptions


@typing.runtime_checkable
class SaveDecisionPolicy(Protocol):
  """A policy that defines when to save a checkpoint."""

  def should_save(
      self,
      step: PolicyCheckpointInfo,
      previous_steps: Sequence[PolicyCheckpointInfo],
      *,
      context: DecisionContext,
  ) -> bool:
    """Returns True if a checkpoint should be saved at the given step."""
    ...


@dataclasses.dataclass
class FixedIntervalPolicy(SaveDecisionPolicy):
  """Checkpoint at a fixed interval."""

  interval: int

  def should_save(
      self,
      step: PolicyCheckpointInfo,
      previous_steps: Sequence[PolicyCheckpointInfo],
      *,
      context: DecisionContext,
  ) -> bool:
    del previous_steps
    del context
    return step.step % self.interval == 0


@dataclasses.dataclass
class SpecificStepsPolicy(SaveDecisionPolicy):
  """Checkpoint at specific steps."""

  steps: Container[int]

  def should_save(
      self,
      step: PolicyCheckpointInfo,
      previous_steps: Sequence[PolicyCheckpointInfo],
      *,
      context: DecisionContext,
  ) -> bool:
    del previous_steps
    del context
    return step.step in self.steps


@dataclasses.dataclass(kw_only=True)
class ContinuousCheckpointingPolicy(SaveDecisionPolicy):
  """Checkpoint as often as possible, as long as a save is not ongoing."""

  minimum_interval_secs: int | None = None

  def should_save(
      self,
      step: PolicyCheckpointInfo,
      previous_steps: Sequence[PolicyCheckpointInfo],
      *,
      context: DecisionContext,
  ) -> bool:

    def _get_should_save_result() -> bool:
      if context.is_saving_in_progress:
        return False
      if not previous_steps or self.minimum_interval_secs is None:
        return True
      return step.time - previous_steps[-1].time >= datetime.timedelta(
          seconds=self.minimum_interval_secs
      )

    is_primary_host = multihost.is_primary_host(
        context.multiprocessing_options.primary_host
    )
    client = signaling_client.get_signaling_client()
    operation_id = synchronization.OperationIdGenerator.next_operation_id()
    result_barrier_key = f'{operation_id}_continuous_checkpointing_policy_should_save_{step.step}/'
    # Make time based and save in progress based decision only on primary host
    # and broadcast to all hosts.
    if is_primary_host:
      save_result = int(_get_should_save_result())
      client.key_value_set(
          result_barrier_key,
          str(save_result),
          allow_overwrite=True,
      )

    save_result = int(
        client.blocking_key_value_get(
            result_barrier_key,
            timeout_secs=600,
        )
    )
    return bool(save_result)


class PreemptionCheckpointingPolicy(SaveDecisionPolicy):
  """Save a checkpoint when a preemption is detected."""

  def should_save(
      self,
      step: PolicyCheckpointInfo,
      previous_steps: Sequence[PolicyCheckpointInfo],
      *,
      context: DecisionContext,
  ) -> bool:
    del step
    del previous_steps
    return context.reached_preemption


class InitialSavePolicy(SaveDecisionPolicy):
  """Checkpoint as soon as possible if no checkpoints already exist."""

  def should_save(
      self,
      step: PolicyCheckpointInfo,
      previous_steps: Sequence[PolicyCheckpointInfo],
      *,
      context: DecisionContext,
  ) -> bool:
    del step
    del context
    return not previous_steps


@dataclasses.dataclass
class AnySavePolicy(SaveDecisionPolicy):
  """Evaluates all policies and saves if any of them returns True.

  Each policy is evaluated in order, and if all are False, the final result is
  False. If at least one is True, the final result is True.
  """

  policies: Sequence[SaveDecisionPolicy]

  def should_save(
      self,
      step: PolicyCheckpointInfo,
      previous_steps: Sequence[PolicyCheckpointInfo],
      *,
      context: DecisionContext,
  ) -> bool:
    return any(
        policy.should_save(step, previous_steps=previous_steps, context=context)
        for policy in self.policies
    )
