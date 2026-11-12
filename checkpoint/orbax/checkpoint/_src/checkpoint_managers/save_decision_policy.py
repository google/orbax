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
import enum
import typing
from typing import Callable, Container, Protocol, Sequence
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
    is_saving_in_progress: Indicates whether an asynchronous checkpoint save
      operation is currently running in the background. Policies (like
      `ContinuousCheckpointingPolicy`) use this to avoid triggering overlapping
      save operations.
    in_progress_step: Step index of the asynchronous checkpoint save operation
      currently running in the background, or `None` if no save is in progress.
    reached_preemption: Indicates whether a preemption signal has been received
      from the cluster manager, meaning the training job is about to be
      terminated.
    multiprocessing_options: Configuration details for distributed multihost
      training. This provides information such as primary host identification
      for synchronization barriers.
    cancel_in_progress_save: Callback invoked by a policy during `save` when an
      in-progress background save should be cancelled to make way for the
      current step (or a no-op when `should_save` is queried without saving).
    save_durations_secs: Historical durations in seconds of completed checkpoint
      save operations, used to dynamically estimate deferral windows before
      target steps.
  """

  is_saving_in_progress: bool
  in_progress_step: int | None
  reached_preemption: bool
  multiprocessing_options: options_lib.MultiprocessingOptions
  cancel_in_progress_save: Callable[[], None]
  save_durations_secs: Sequence[float]


class _ContinuousSaveDecision(enum.IntEnum):
  SKIP = 0
  SAVE = 1
  CANCEL_AND_SAVE = 2


def _compute_percentile(values: Sequence[float], q: float) -> float:
  """Computes the q-th quantile (in (0.0, 1.0]) of values via linear interpolation."""
  sorted_values = sorted(values)
  if len(sorted_values) == 1:
    return sorted_values[0]
  pos = q * (len(sorted_values) - 1)
  lower_idx = int(pos)
  upper_idx = min(lower_idx + 1, len(sorted_values) - 1)
  weight = pos - lower_idx
  return (
      sorted_values[lower_idx] * (1.0 - weight)
      + sorted_values[upper_idx] * weight
  )


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

  This policy evaluates to True as often as possible. It enforces primary
  constraints to prevent blocking training or causing other regressions, while
  optionally aligning saves with a target step interval.

  1. It will never trigger a new opportunistic continuous save if a save is
     currently in progress (checked via the provided `DecisionContext`); this
     prevents blocking on an ongoing save, which would hurt accelerator
     utilization.
  2. When `target_interval_steps` is configured, exact multiples of
     `target_interval_steps` strictly evaluate to True (behaving like
     `FixedIntervalPolicy` at target steps and bypassing `minimum_interval_secs`
     and `is_saving_in_progress`), while opportunistic saves within
     `deferral_window_steps` before the next target step are deferred so the
     background writer stays free for the target step. If an opportunistic
     continuous save is still in progress when a target step arrives,
     `CheckpointManager.save` cancels the ongoing continuous save; if a previous
     target step is still in progress, `CheckpointManager.save` blocks until it
     completes.
  3. It optionally respects a minimum time interval between saves if
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
      step. If `None` (the default), back-to-back saves are permitted as soon as
      the ongoing save completes.
    target_interval_steps (int | None): Optional target step interval with which
      continuous checkpoints must strictly align. When set, steps where
      `step.step % target_interval_steps == 0` always evaluate to True, and
      opportunistic saves within `deferral_window_steps` steps before a target
      step are deferred.
    deferral_window_steps (int | None): Optional number of steps before each
      multiple of `target_interval_steps` during which new continuous saves are
      deferred to keep the background writer free for the upcoming target step.
      Only used when `target_interval_steps` is set. If `None` (the default),
      deferral is determined dynamically using `deferral_save_time_percentile`
      when historical save durations are available, or defaults to no deferral
      (`0`) otherwise.
    deferral_save_time_percentile (float | None): Optional percentile in `(0.0,
      1.0]` of historical checkpoint save durations
      (`context.save_durations_secs`) used to dynamically defer opportunistic
      saves before each multiple of `target_interval_steps` when
      `deferral_window_steps` is `None`. When historical save durations and step
      duration estimates are available, no opportunistic save is triggered if
      the estimated time to the next target step is within this percentile of
      historical save times. Defaults to `0.9` (the 90th percentile). Set to
      `None` (or set `deferral_window_steps=0`) to disable deferral.

  Methods:
    should_save(step, previous_steps, *, context):
      Evaluates the current state and synchronizes across all hosts to return
      a boolean indicating whether a checkpoint should be saved.

      Args:
        step (PolicyCheckpointInfo): Information about the current training
          step, including the step index and timestamp.
        previous_steps (Sequence[PolicyCheckpointInfo]): A chronological list of
          metadata for all steps where a checkpoint was successfully saved.
        context (DecisionContext): A container for auxiliary information, such
          as the current saving state (`is_saving_in_progress`) and
          multiprocessing configuration used to inform the decision.
  """

  minimum_interval_secs: int | None = None
  target_interval_steps: int | None = None
  deferral_window_steps: int | None = None
  deferral_save_time_percentile: float | None = 0.9
  _step_durations_secs: list[float] = dataclasses.field(
      default_factory=list, init=False, repr=False, compare=False
  )
  _last_observed_step_time: tuple[int, datetime.datetime] | None = (
      dataclasses.field(default=None, init=False, repr=False, compare=False)
  )

  def __post_init__(self):
    if (
        self.target_interval_steps is not None
        and self.target_interval_steps <= 0
    ):
      raise ValueError(
          "target_interval_steps must be positive, got"
          f" {self.target_interval_steps}."
      )
    if (
        self.deferral_window_steps is not None
        and self.deferral_window_steps < 0
    ):
      raise ValueError(
          "deferral_window_steps must be non-negative, got"
          f" {self.deferral_window_steps}."
      )
    if self.deferral_save_time_percentile is not None and not (
        0.0 < self.deferral_save_time_percentile <= 1.0
    ):
      raise ValueError(
          "deferral_save_time_percentile must be in (0.0, 1.0], got"
          f" {self.deferral_save_time_percentile}."
      )

  def _get_step_duration_secs(
      self,
      step: PolicyCheckpointInfo,
      previous_steps: Sequence[PolicyCheckpointInfo],
  ) -> float | None:
    """Returns the estimated duration in seconds of a single training step."""
    if self._step_durations_secs:
      return sum(self._step_durations_secs) / len(self._step_durations_secs)
    if previous_steps and step.step > previous_steps[-1].step:
      elapsed_secs = (step.time - previous_steps[-1].time).total_seconds()
      if elapsed_secs > 0:
        return elapsed_secs / (step.step - previous_steps[-1].step)
    return None

  def _should_defer_for_target_step(
      self,
      step: PolicyCheckpointInfo,
      previous_steps: Sequence[PolicyCheckpointInfo],
      context: DecisionContext,
  ) -> bool:
    """Returns True if an opportunistic save should be deferred for a target step."""
    if self.target_interval_steps is None:
      return False
    steps_until_target = self.target_interval_steps - (
        step.step % self.target_interval_steps
    )
    if self.deferral_window_steps is not None:
      return steps_until_target <= self.deferral_window_steps
    if (
        self.deferral_save_time_percentile is not None
        and context.save_durations_secs
    ):
      step_duration_secs = self._get_step_duration_secs(step, previous_steps)
      if step_duration_secs is not None:
        save_time_threshold_secs = _compute_percentile(
            context.save_durations_secs, self.deferral_save_time_percentile
        )
        return (
            steps_until_target * step_duration_secs <= save_time_threshold_secs
        )
    return False

  def should_save(
      self,
      step: PolicyCheckpointInfo,
      previous_steps: Sequence[PolicyCheckpointInfo],
      *,
      context: DecisionContext,
  ) -> bool:
    def _get_decision() -> _ContinuousSaveDecision:
      if (
          self.target_interval_steps is not None
          and step.step % self.target_interval_steps == 0
      ):
        if (
            context.in_progress_step is not None
            and context.in_progress_step % self.target_interval_steps != 0
        ):
          return _ContinuousSaveDecision.CANCEL_AND_SAVE
        return _ContinuousSaveDecision.SAVE
      if context.is_saving_in_progress:
        return _ContinuousSaveDecision.SKIP
      if not previous_steps:
        return _ContinuousSaveDecision.SAVE
      if self._should_defer_for_target_step(step, previous_steps, context):
        return _ContinuousSaveDecision.SKIP
      if self.minimum_interval_secs is None:
        return _ContinuousSaveDecision.SAVE
      if step.time - previous_steps[-1].time >= datetime.timedelta(
          seconds=self.minimum_interval_secs
      ):
        return _ContinuousSaveDecision.SAVE
      return _ContinuousSaveDecision.SKIP

    is_primary_host = multihost.is_primary_host(
        context.multiprocessing_options.primary_host
    )
    client = signaling_client.get_signaling_client()
    operation_id = synchronization.OperationIdGenerator.next_operation_id()
    result_barrier_key = f"{operation_id}_continuous_checkpointing_policy_should_save_{step.step}/"
    # Make time based and save in progress based decision only on primary host
    # and broadcast to all hosts.
    if is_primary_host:
      if (
          self._last_observed_step_time is not None
          and step.step > self._last_observed_step_time[0]
      ):
        elapsed_secs = (
            step.time - self._last_observed_step_time[1]
        ).total_seconds()
        if elapsed_secs > 0:
          self._step_durations_secs.append(
              elapsed_secs / (step.step - self._last_observed_step_time[0])
          )
      if (
          self._last_observed_step_time is None
          or step.step > self._last_observed_step_time[0]
      ):
        self._last_observed_step_time = (step.step, step.time)
      save_result = int(_get_decision())
      client.key_value_set(
          result_barrier_key,
          str(save_result),
          allow_overwrite=True,
      )

    decision = _ContinuousSaveDecision(
        int(
            client.blocking_key_value_get(
                result_barrier_key,
                timeout_secs=multihost.coordination_timeout(),
            )
        )
    )
    if decision == _ContinuousSaveDecision.CANCEL_AND_SAVE:
      context.cancel_in_progress_save()
    should_save_result = decision != _ContinuousSaveDecision.SKIP
    _log_save_decision(
        "ContinuousCheckpointingPolicy"
        f" (minimum_interval_secs={self.minimum_interval_secs},"
        f" target_interval_steps={self.target_interval_steps},"
        f" deferral_window_steps={self.deferral_window_steps})",
        step,
        should_save_result,
    )
    return should_save_result


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
