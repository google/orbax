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

import dataclasses
import datetime
import typing
from typing import Container, Protocol, Sequence
from absl import logging
from orbax.checkpoint import options as options_lib
from orbax.checkpoint._src.checkpoint_managers import policy_checkpoint_info
from orbax.checkpoint._src.futures import signaling_client
from orbax.checkpoint._src.futures import synchronization
from orbax.checkpoint._src.multihost import multihost


PolicyCheckpointInfo = policy_checkpoint_info.PolicyCheckpointInfo


@dataclasses.dataclass(kw_only=True)
class DecisionContext:
  """Additional properties for making a save decision.

  This dataclass is populated by the checkpointer framework and passed into the
  `should_save` method of all `SaveDecisionPolicy` implementations. It provides
  essential external system context, allowing policies to make safe, state-aware
  decisions.

  Attributes:
    is_saving_in_progress: Indicates whether an asynchronous checkpoint
      save operation is currently running in the background. Policies (like
      `ContinuousCheckpointingPolicy`) use this to avoid triggering overlapping
      save operations.
    reached_preemption: Indicates whether a preemption signal has been
      received from the cluster manager, meaning the training job is about to
      be terminated.
    multiprocessing_options: Configuration details for distributed multihost
      training. This provides information such as primary host identification
      for synchronization barriers.
  """

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


def _log_save_decision(
    policy_name: str,
    step: PolicyCheckpointInfo,
    is_saving: bool,
) -> None:
  """Logs the save decision."""
  if is_saving:
    logging.vlog(
        1,
        f"{policy_name}: Saving checkpoint at step"
        f" {step.step}).",
    )
  else:
    logging.vlog(
        1,
        f"{policy_name}: Not saving checkpoint at step"
        f" {step.step}).",
    )


@dataclasses.dataclass
class FixedIntervalPolicy(SaveDecisionPolicy):
  """Checkpoint at a fixed interval.

  This policy evaluates to True whenever the current training step is an exact
  multiple of the configured `interval` (i.e., `step.step % interval == 0`).
  It makes its decision purely based on the current step index, strictly
  ignoring previous save history or external context.

  Attributes:
    interval: The frequency at which checkpoints should be saved. For example,
      an interval of 100 means a save is triggered at steps 100, 200, 300, etc.

  Methods:
    should_save(step, previous_steps, *, context):
      Evaluates whether the current step index is a multiple of the interval.

      Args:
        step (PolicyCheckpointInfo): Information about the current training
          step, primarily using the step index for modulo arithmetic.
        previous_steps (Sequence[PolicyCheckpointInfo]): Ignored by this policy.
        context (DecisionContext): Ignored by this policy.
  """

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
    result = step.step % self.interval == 0
    _log_save_decision(
        f"FixedIntervalPolicy (interval={self.interval})",
        step,
        result,
    )
    return result


@dataclasses.dataclass
class SpecificStepsPolicy(SaveDecisionPolicy):
  """Checkpoint at specific steps.

  This policy evaluates to True whenever the current training step index exists
  within the configured `steps` container (i.e., `step.step in steps`). It
  makes its decision purely based on the current step index, strictly ignoring
  previous save history or external context.

  Attributes:
    steps (Container[int]): A collection (such as a `set`, `list`, or `tuple`)
      of step indices where checkpoints should be saved.

  Methods:
    should_save(step, previous_steps, *, context):
      Evaluates whether the current step index exists in the steps container.

      Args:
        step (PolicyCheckpointInfo): Information about the current training
          step, primarily using the step index for membership testing.
        previous_steps (Sequence[PolicyCheckpointInfo]): Ignored by this policy.
        context (DecisionContext): Ignored by this policy.
  """

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
    result = step.step in self.steps
    _log_save_decision(
        f"SpecificStepsPolicy (steps={self.steps})",
        step,
        result,
    )
    return result


@dataclasses.dataclass(kw_only=True)
class ContinuousCheckpointingPolicy(SaveDecisionPolicy):
  """Checkpoint as often as possible, as long as a save is not ongoing.

  This policy evaluates to True as often as possible. It enforces two primary
  constraints to prevent blocking training or causing other regressions.

  1. It will never trigger a new save if a save is currently in progress
     (checked via the provided `DecisionContext`); this prevents blocking on an
     ongoing save, which would hurt accelerator utilization.
  2. It optionally respects a minimum time interval between saves if
     `minimum_interval_secs` is configured. This sets a floor on how
     frequently checkpoints are saved, which can be used to avoid excessive
     burden on the filesystem, or blocking too frequently (due to
     synchronous D2H).

  In a distributed training environment, to ensure perfect synchronization
  and avoid race conditions, the time and state-based save decision is
  computed exclusively on the primary host. The result is then broadcast
  to all other hosts via a blocking barrier.

  For usage examples, please refer to the parent class `SaveDecisionPolicy`.

  Attributes:
    minimum_interval_secs (int | None): The minimum time in seconds that must
      elapse between the timestamp of the previous checkpoint and the current
      step. If `None` (the default), back-to-back saves are permitted as soon
      as the ongoing save completes.

  Methods:
    should_save(step, previous_steps, *, context):
      Evaluates the current state and synchronizes across all hosts to return
      a boolean indicating whether a checkpoint should be saved.

      Args:
        step (PolicyCheckpointInfo): Information about the current training
          step, including the step index and timestamp.
        previous_steps (Sequence[PolicyCheckpointInfo]): A chronological list
          of metadata for all steps where a checkpoint was successfully saved.
        context (DecisionContext): A container for auxiliary information,
          such as the current saving state (`is_saving_in_progress`) and
          multiprocessing configuration used to inform the decision.
  """

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
    result_barrier_key = f"{operation_id}_continuous_checkpointing_policy_should_save_{step.step}/"
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
            timeout_secs=multihost.coordination_timeout(),
        )
    )
    _log_save_decision(
        "ContinuousCheckpointingPolicy"
        f" (minimum_interval_secs={self.minimum_interval_secs})",
        step,
        bool(save_result),
    )
    return bool(save_result)


class PreemptionCheckpointingPolicy(SaveDecisionPolicy):
  """Save a checkpoint when a preemption is detected.

  This policy evaluates to True strictly when the provided `DecisionContext`
  indicates that a preemption signal has been received (i.e.,
  `context.reached_preemption` is True). It can be useful for ensuring that
  training progress is safely stored before the job is killed by the cluster
  scheduler.

  Note that saving on preemption is not strictly necessary, however. For
  example, if continuous checkpointing is employed, and checkpoints are saved
  frequently enough, the cost of re-computing some amount of steps can be
  cheaper than the cost of waiting for a checkpoint to complete after
  preemption.

  Methods:
    should_save(step, previous_steps, *, context):
      Evaluates whether a preemption signal has been registered in the context.

      Args:
        step (PolicyCheckpointInfo): Ignored by this policy.
        previous_steps (Sequence[PolicyCheckpointInfo]): Ignored by this policy.
        context (DecisionContext): A container for auxiliary information. This
          policy specifically checks the `reached_preemption` boolean flag to
          make its decision.
  """

  def should_save(
      self,
      step: PolicyCheckpointInfo,
      previous_steps: Sequence[PolicyCheckpointInfo],
      *,
      context: DecisionContext,
  ) -> bool:
    _log_save_decision(
        "PreemptionCheckpointingPolicy",
        step,
        context.reached_preemption,
    )
    del step
    del previous_steps
    return context.reached_preemption


class InitialSavePolicy(SaveDecisionPolicy):
  """Save a checkpoint as soon as possible if no checkpoints already exist.

  This policy evaluates to True only if the `previous_steps` sequence is empty.
  It is highly useful for ensuring a baseline checkpoint is created immediately
  upon starting a fresh training run, while safely evaluating to False if the
  job is restarting from an existing checkpoint.

  Methods:
    should_save(step, previous_steps, *, context):
      Evaluates whether the `previous_steps` history is empty.

      Args:
        step (PolicyCheckpointInfo): Ignored by this policy.
        previous_steps (Sequence[PolicyCheckpointInfo]): A chronological list
          of metadata for previously saved steps. The policy checks if this is
          empty.
        context (DecisionContext): Ignored by this policy.
  """

  def should_save(
      self,
      step: PolicyCheckpointInfo,
      previous_steps: Sequence[PolicyCheckpointInfo],
      *,
      context: DecisionContext,
  ) -> bool:
    _log_save_decision(
        "InitialSavePolicy",
        step,
        not previous_steps,
    )
    del step
    del context
    return not previous_steps


@dataclasses.dataclass
class AnySavePolicy(SaveDecisionPolicy):
  """Evaluates all policies and saves if any of them returns True.

  This policy iterates through a provided sequence of child policies. It
  evaluates each one in order and returns True immediately if any child policy
  returns True. If all child policies return False, this policy returns False.
  It is highly useful for combining time-based, step-based, and event-based
  saving rules into a single, unified checkpointer configuration.

  Attributes:
    policies (Sequence[SaveDecisionPolicy]): An ordered collection of underlying
      policies to evaluate.

  Methods:
    should_save(step, previous_steps, *, context):
      Evaluates the sequence of configured policies.

      Args:
        step (PolicyCheckpointInfo): Passed down to each child policy.
        previous_steps (Sequence[PolicyCheckpointInfo]): Passed down to each
          child policy.
        context (DecisionContext): Passed down to each child policy.
  """

  policies: Sequence[SaveDecisionPolicy]

  def should_save(
      self,
      step: PolicyCheckpointInfo,
      previous_steps: Sequence[PolicyCheckpointInfo],
      *,
      context: DecisionContext,
  ) -> bool:
    logging.vlog(
        1,
        "AnySavePolicy: policies=%s, step=%d.",
        self.policies,
        step.step,
    )
    return any(
        policy.should_save(step, previous_steps=previous_steps, context=context)
        for policy in self.policies
    )
