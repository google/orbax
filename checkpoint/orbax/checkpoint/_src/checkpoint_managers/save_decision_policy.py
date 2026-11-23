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

import collections
import dataclasses
import datetime
import enum
import statistics
import typing
from typing import Callable, Container, Protocol, Sequence
from absl import logging
from orbax.checkpoint import options as options_lib
from orbax.checkpoint._src.checkpoint_managers import policy_checkpoint_info
from orbax.checkpoint._src.futures import signaling_client
from orbax.checkpoint._src.futures import synchronization
from orbax.checkpoint._src.multihost import multihost


PolicyCheckpointInfo = policy_checkpoint_info.PolicyCheckpointInfo

# Number of most recent per-step duration observations used to estimate step
# time for dynamic deferral. A bounded window keeps slow warmup or post-restore
# steps from dominating the estimate for the rest of the run.
_STEP_DURATION_WINDOW_SIZE = 20

# Minimum number of completed saves required before dynamic deferral is
# applied. Dynamic deferral additionally waits until the step duration window
# is full, so that a single unusually slow or fast save or step at startup does
# not produce a bad deferral window.
_MIN_SAVE_DURATIONS_FOR_DYNAMIC_DEFERRAL = 3

# Number of most recent background save durations used for dynamic deferral.
_SAVE_DURATION_WINDOW_SIZE = 50

# Reasons recorded by `ContinuousCheckpointingPolicy` when it skips a save.
SKIP_REASON_SAVE_IN_PROGRESS = "save_in_progress"
SKIP_REASON_DEFERRED_FOR_TARGET_STEP = "deferred_for_target_step"
SKIP_REASON_MINIMUM_INTERVAL_NOT_MET = "minimum_interval_not_met"


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
  """

  is_saving_in_progress: bool
  in_progress_step: int | None
  reached_preemption: bool
  multiprocessing_options: options_lib.MultiprocessingOptions
  cancel_in_progress_save: Callable[[], None]


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

  Useful when absolute predictability is preferred. Too short an interval can
  block training on queued saves; too long loses more progress on interruption.

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

  Useful for most training workloads. Set `target_interval_steps` to an
  infrequent interval (e.g. for evals and rollbacks) that is much longer than
  the checkpoint write time, so recovery saves are rarely cancelled.

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
      saves are deferred dynamically if `deferral_save_time_percentile` is set,
      and not deferred otherwise.
    deferral_save_time_percentile (float | None): Optional percentile in `(0.0,
      1.0]` of recent background save durations used to dynamically defer
      opportunistic saves before each multiple of `target_interval_steps` when
      `deferral_window_steps` is `None`. No opportunistic save is triggered if
      the estimated time to the next target step is within this percentile of
      recent save times (e.g. `0.9` for the 90th percentile). Save durations
      are measured by the policy itself, from the step at which a save is
      first seen in progress (`context.in_progress_step`) until the first step
      at which it is no longer in progress, so they include cancelled saves and
      are accurate to within one step. Dynamic deferral only starts once at
      least 3 saves have been measured and enough steps have been observed to
      estimate step time. If `None` (the default), dynamic deferral is
      disabled.
    last_skip_reason (str | None): Why the most recent `should_save` call
      skipped (`save_in_progress`, `deferred_for_target_step` or
      `minimum_interval_not_met`), or `None` if it saved. Only set on the
      primary host, where the decision is made.

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
  deferral_save_time_percentile: float | None = None
  _step_durations_secs: collections.deque[float] = dataclasses.field(
      default_factory=lambda: collections.deque(
          maxlen=_STEP_DURATION_WINDOW_SIZE
      ),
      init=False,
      repr=False,
      compare=False,
  )
  _last_observed_step_time: tuple[int, datetime.datetime] | None = (
      dataclasses.field(default=None, init=False, repr=False, compare=False)
  )
  _save_durations_secs: collections.deque[float] = dataclasses.field(
      default_factory=lambda: collections.deque(
          maxlen=_SAVE_DURATION_WINDOW_SIZE
      ),
      init=False,
      repr=False,
      compare=False,
  )
  # Step and start time of the background save currently being timed.
  _timed_save: tuple[int, datetime.datetime] | None = dataclasses.field(
      default=None, init=False, repr=False, compare=False
  )
  last_skip_reason: str | None = dataclasses.field(
      default=None, init=False, repr=False, compare=False
  )

  def __post_init__(self):
    if (
        self.minimum_interval_secs is not None
        and self.minimum_interval_secs < 0
    ):
      raise ValueError(
          "minimum_interval_secs must be non-negative, got"
          f" {self.minimum_interval_secs}."
      )
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

  def _observe(
      self, step: PolicyCheckpointInfo, context: DecisionContext
  ) -> None:
    """Updates the step and save duration estimates with the current step."""
    last_observed = self._last_observed_step_time
    in_progress_step = (
        context.in_progress_step if context.is_saving_in_progress else None
    )
    # A timed save that is no longer in progress has completed or was
    # cancelled. A cancelled save would have taken at least this long, so it is
    # recorded too rather than biasing the estimate towards faster saves.
    if self._timed_save is not None and in_progress_step != self._timed_save[0]:
      elapsed_secs = (step.time - self._timed_save[1]).total_seconds()
      if elapsed_secs > 0:
        self._save_durations_secs.append(elapsed_secs)
      self._timed_save = None
    # A save is in progress from the step at which it was requested, which is
    # normally the last observed step.
    if (
        self._timed_save is None
        and in_progress_step is not None
        and last_observed is not None
        and last_observed[0] <= in_progress_step
    ):
      self._timed_save = (in_progress_step, last_observed[1])

    if last_observed is None or step.step > last_observed[0]:
      if last_observed is not None:
        elapsed_secs = (step.time - last_observed[1]).total_seconds()
        if elapsed_secs > 0:
          self._step_durations_secs.append(
              elapsed_secs / (step.step - last_observed[0])
          )
      self._last_observed_step_time = (step.step, step.time)

  def _get_step_duration_secs(self) -> float | None:
    """Returns the estimated duration in seconds of a single training step.

    Uses the median of the most recent `_STEP_DURATION_WINDOW_SIZE` observed
    per-step durations so that slow warmup or post-restore steps do not skew
    the estimate. Returns `None` until the window is full.
    """
    if len(self._step_durations_secs) < _STEP_DURATION_WINDOW_SIZE:
      return None
    return statistics.median(self._step_durations_secs)

  def _should_defer_for_target_step(self, step: PolicyCheckpointInfo) -> bool:
    """Returns True if an opportunistic save should be deferred for a target step."""
    if self.target_interval_steps is None:
      return False
    steps_until_target = self.target_interval_steps - (
        step.step % self.target_interval_steps
    )
    if self.deferral_window_steps is not None:
      return steps_until_target <= self.deferral_window_steps
    if (
        self.deferral_save_time_percentile is None
        or len(self._save_durations_secs)
        < _MIN_SAVE_DURATIONS_FOR_DYNAMIC_DEFERRAL
    ):
      return False
    step_duration_secs = self._get_step_duration_secs()
    if step_duration_secs is None:
      return False
    save_time_threshold_secs = _compute_percentile(
        self._save_durations_secs, self.deferral_save_time_percentile
    )
    return steps_until_target * step_duration_secs <= save_time_threshold_secs

  def should_save(
      self,
      step: PolicyCheckpointInfo,
      previous_steps: Sequence[PolicyCheckpointInfo],
      *,
      context: DecisionContext,
  ) -> bool:
    def _get_decision() -> tuple[_ContinuousSaveDecision, str | None]:
      """Returns the decision and, if it is to skip, the reason."""
      if (
          self.target_interval_steps is not None
          and step.step % self.target_interval_steps == 0
      ):
        if (
            context.in_progress_step is not None
            and context.in_progress_step % self.target_interval_steps != 0
        ):
          return _ContinuousSaveDecision.CANCEL_AND_SAVE, None
        return _ContinuousSaveDecision.SAVE, None
      if context.is_saving_in_progress:
        return _ContinuousSaveDecision.SKIP, SKIP_REASON_SAVE_IN_PROGRESS
      if not previous_steps:
        return _ContinuousSaveDecision.SAVE, None
      if self._should_defer_for_target_step(step):
        return (
            _ContinuousSaveDecision.SKIP,
            SKIP_REASON_DEFERRED_FOR_TARGET_STEP,
        )
      if self.minimum_interval_secs is None:
        return _ContinuousSaveDecision.SAVE, None
      if step.time - previous_steps[-1].time >= datetime.timedelta(
          seconds=self.minimum_interval_secs
      ):
        return _ContinuousSaveDecision.SAVE, None
      return _ContinuousSaveDecision.SKIP, SKIP_REASON_MINIMUM_INTERVAL_NOT_MET

    is_primary_host = multihost.is_primary_host(
        context.multiprocessing_options.primary_host
    )
    client = signaling_client.get_signaling_client()
    operation_id = synchronization.OperationIdGenerator.next_operation_id()
    result_barrier_key = f"{operation_id}_continuous_checkpointing_policy_should_save_{step.step}/"
    # Make time based and save in progress based decision only on primary host
    # and broadcast to all hosts.
    if is_primary_host:
      self._observe(step, context)
      primary_decision, self.last_skip_reason = _get_decision()
      save_result = int(primary_decision)
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
  preemption. Saving on preemption is often less resource-efficient, since the
  job must wait for a slow write before exiting. It is recommended when regular
  checkpoints are infrequent and a save fits comfortably within the preemption
  grace period. If the system cannot reliably finish a save between preemption
  notice and exit, saving on preemption is not a reliable checkpointing
  mechanism.

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
