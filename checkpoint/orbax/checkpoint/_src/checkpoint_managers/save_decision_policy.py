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
import math
import time
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


@typing.runtime_checkable
class SaveStartObserver(Protocol):
  """A policy that wants to be told when a save starts.

  `CheckpointManager.save` sends the save-start notification (calls
  `on_save_start`) on every host, at the same point of the save: after all
  hosts passed the `save_start` barrier, after the previous save finished, and
  after all checks that can reject the save (e.g. `StepAlreadyExistsError`),
  immediately before the write starts. So it is sent only for saves that
  really happen. Policies may use this to agree on state across hosts once per
  save instead of once per step.
  """

  def on_save_start(self, step: int, *, context: DecisionContext) -> None:
    ...


def notify_save_start(
    policy: SaveDecisionPolicy, step: int, *, context: DecisionContext
) -> None:
  """Sends the save-start notification to `policy` and any nested policies."""
  if isinstance(policy, AnySavePolicy):
    for p in policy.policies:
      notify_save_start(p, step, context=context)
  elif isinstance(policy, SaveStartObserver):
    policy.on_save_start(step, context=context)


_BOOTSTRAP_SAVE = -1
_BOOTSTRAP_WAIT = -2


@dataclasses.dataclass
class ScheduledContinuousCheckpointingPolicy(SaveDecisionPolicy):
  """EXPERIMENTAL. Continuous checkpointing with a per-save schedule.

  `ContinuousCheckpointingPolicy` runs a cross-host agreement on every step.
  This policy agrees once per save instead. During the save at step S, the
  primary host computes the step of the next save:

    next_step = S + ceil(target_interval_secs / step_time_estimate)

  and shares it with the other hosts. Between saves, `should_save` is a local
  comparison `step >= next_step`, with no cross-host messages.

  Differences from `ContinuousCheckpointingPolicy`:
    * If the previous write is still running at `next_step`, the save waits
      for it (normal `CheckpointManager.save` behavior) instead of skipping.
      To make this rare, `target_interval_secs` is stretched to
      `write_duration_margin * last_write_secs` when the previous write took
      longer than `minimum_interval_secs`.
    * `step_time_estimate` is a low quantile of recent per-step times, so
      errors make saves late rather than early. If the realized interval of
      the previous save was still shorter than `minimum_interval_secs`, the
      deficit is added to the next target.

  Until a schedule exists (fresh start or restart after restore), the policy
  runs a per-step agreement with the same save rule as
  `ContinuousCheckpointingPolicy`. As soon as the primary host has
  `min_samples_for_schedule` step time samples, it publishes a schedule
  anchored at the latest checkpoint, and all hosts switch to local decisions.
  After a restart, the interval is measured from the first step this process
  observes, not from the restored checkpoint's time, so downtime does not
  trigger an immediate save. A fresh start with no checkpoints is unchanged:
  the policy saves on the first step, like `InitialSavePolicy`.

  Every save re-anchors the schedule, including saves triggered by other
  policies (e.g. milestone or preemption saves): the next save is computed
  from the step and time of that save. A preemption save clears the schedule
  instead (see below).

  Step time is sampled in `should_save`. In an `AnySavePolicy`, put this
  policy first: an earlier policy returning True short-circuits it. The cost
  is one lost sample per save; correctness is not affected.

  Requires `CheckpointManager` to send the save-start notification
  (`notify_save_start`); it does.

  Preemption saves: when the save-start notification reports
  `reached_preemption`, the policy skips the next-step exchange (the job is
  about to exit). It records the save start time and clears the schedule, so
  if the job continues it re-bootstraps, as after a restart.

  Host 0 failure: the next-step exchange uses the JAX coordination service
  with `multihost.coordination_timeout()`, like Orbax's existing save
  barriers. The coordination service runs in process 0. If host 0 dies during
  the exchange, the other hosts' calls fail as they would at any of those
  barriers, and the job restarts from the latest checkpoint. The exposure is
  once per save, versus once per check for `ContinuousCheckpointingPolicy`.

  Attributes:
    minimum_interval_secs: Minimum wall time between consecutive saves. If 0,
      the next save is scheduled for when the previous write is expected to
      finish (see `write_duration_margin`).
    write_duration_margin: If > 0, the target interval is at least this multiple
      of the duration of the previous write. Set to 0 to disable.
    step_time_window: Number of recent per-step time samples to keep.
    step_time_quantile: Quantile of the samples used as step time estimate.
    min_samples_for_schedule: Step time samples needed before a schedule is
      published during bootstrap.
    num_bootstrap_decisions: Number of decisions made during bootstrap.
    num_local_decisions: Number of decisions made locally using schedule.
    schedule_history: History of scheduled steps as (current_step, next_step).
  """

  minimum_interval_secs: float
  write_duration_margin: float = 1.0
  step_time_window: int = 64
  step_time_quantile: float = 0.1
  min_samples_for_schedule: int = 4

  def __post_init__(self):
    if self.minimum_interval_secs < 0:
      raise ValueError("minimum_interval_secs must be >= 0.")
    if not 0.0 <= self.step_time_quantile <= 1.0:
      raise ValueError("step_time_quantile must be in [0, 1].")
    if self.min_samples_for_schedule < 1:
      raise ValueError("min_samples_for_schedule must be >= 1.")
    self._next_step: int | None = None
    self._samples: collections.deque[float] = collections.deque(
        maxlen=self.step_time_window
    )
    self._last_obs: tuple[int, float] | None = None
    # Time of the first step seen by `should_save` in this process.
    self._first_step_time: datetime.datetime | None = None
    self._samples_since_save: list[float] = []
    self._save_start_time: float | None = None
    self._save_in_flight = False
    self._last_write_secs: float | None = None
    self._last_target_secs: float | None = None
    # Stats, for tests and benchmarks.
    self.num_bootstrap_decisions = 0
    self.num_local_decisions = 0
    self.schedule_history: list[tuple[int, int | None]] = []

  @property
  def next_step(self) -> int | None:
    """Returns the next scheduled save step, or None if not yet scheduled."""
    return self._next_step

  def _observe(self, step: int, now: float, is_saving: bool) -> None:
    if self._save_in_flight and not is_saving:
      assert self._save_start_time is not None
      self._last_write_secs = now - self._save_start_time
      self._save_in_flight = False
    if self._last_obs is not None and step > self._last_obs[0]:
      sample = (now - self._last_obs[1]) / (step - self._last_obs[0])
      self._samples.append(sample)
      self._samples_since_save.append(sample)
    self._last_obs = (step, now)

  def _quantile(self, samples: Sequence[float]) -> float:
    s = sorted(samples)
    return s[min(len(s) - 1, int(self.step_time_quantile * len(s)))]

  def _step_time_estimate(self) -> float | None:
    """Low quantile of recent step times; smaller means later saves.

    Takes the minimum of the quantile over the whole window and over the
    samples since the last save, so a sudden speedup is picked up at the next
    save instead of after the window turns over.

    Returns:
      The estimate in seconds, or None without samples.
    """
    if not self._samples:
      return None
    est = self._quantile(self._samples)
    if self._samples_since_save:
      est = min(est, self._quantile(self._samples_since_save))
    return est

  def _compute_next_step(self, step: int, now: float) -> int | None:
    """Primary host only. Returns the step of the next save, or None."""
    est = self._step_time_estimate()
    if est is None or est <= 0:
      return None
    target = float(self.minimum_interval_secs)
    if self.write_duration_margin > 0 and self._last_write_secs is not None:
      target = max(target, self.write_duration_margin * self._last_write_secs)
    # Feedback: if the previous scheduled interval came out short, add the
    # deficit. Saves triggered early by other policies (preemption, forced)
    # are not "short"; they just re-anchor the schedule.
    is_scheduled_save = self._next_step is not None and step >= self._next_step
    if is_scheduled_save and self._save_start_time is not None:
      realized = now - self._save_start_time
      deficit = self.minimum_interval_secs - realized
      if deficit > 0:
        target += deficit
    self._last_target_secs = target
    return step + max(1, math.ceil(target / est))

  def on_save_start(self, step: int, *, context: DecisionContext) -> None:
    """Handles the save-start notification sent by `CheckpointManager.save`."""
    now = time.monotonic()
    if self._save_in_flight:
      # The previous write finished during this save's wait.
      assert self._save_start_time is not None
      self._last_write_secs = now - self._save_start_time
    next_step = None
    # On a preemption save the job is about to exit: skip the next-step
    # exchange and clear the schedule. Every host gets the same
    # `reached_preemption`, so they stay consistent. If the job continues, the
    # policy re-bootstraps, as after a restart.
    if not context.reached_preemption:
      is_primary_host = multihost.is_primary_host(
          context.multiprocessing_options.primary_host
      )
      if is_primary_host:
        next_step = self._compute_next_step(step, now)
      if multihost.process_count() > 1:
        client = signaling_client.get_signaling_client()
        operation_id = synchronization.OperationIdGenerator.next_operation_id()
        key = f"{operation_id}_scheduled_cc_next_step_{step}/"
        if is_primary_host:
          client.key_value_set(
              key,
              str(-1 if next_step is None else next_step),
              allow_overwrite=True,
          )
        else:
          value = int(
              client.blocking_key_value_get(
                  key, timeout_secs=multihost.coordination_timeout()
              )
          )
          next_step = None if value < 0 else value
    self._next_step = next_step
    self.schedule_history.append((step, next_step))
    self._save_start_time = now
    self._save_in_flight = True
    # Do not take a step time sample across the save.
    self._last_obs = None
    self._samples_since_save = []
    logging.info(
        "ScheduledContinuousCheckpointingPolicy: save at step %d, next save at"
        " step %s (target_secs=%s, last_write_secs=%s, step_time_est=%s).",
        step,
        next_step,
        self._last_target_secs,
        self._last_write_secs,
        self._step_time_estimate(),
    )

  def should_save(
      self,
      step: PolicyCheckpointInfo,
      previous_steps: Sequence[PolicyCheckpointInfo],
      *,
      context: DecisionContext,
  ) -> bool:
    self._observe(step.step, time.monotonic(), context.is_saving_in_progress)
    if self._first_step_time is None:
      self._first_step_time = step.time
    if self._next_step is None:
      return self._bootstrap_decision(step, previous_steps, context)
    result = self._local_decision(step)
    _log_save_decision(
        f"ScheduledContinuousCheckpointingPolicy (next_step={self._next_step})",
        step,
        result,
    )
    return result

  def _local_decision(self, step: PolicyCheckpointInfo) -> bool:
    assert self._next_step is not None
    self.num_local_decisions += 1
    return step.step >= self._next_step

  def _bootstrap_decision(
      self,
      step: PolicyCheckpointInfo,
      previous_steps: Sequence[PolicyCheckpointInfo],
      context: DecisionContext,
  ) -> bool:
    """Per-step agreement, used until a schedule exists.

    The primary host decides and broadcasts one value per step:
      * `_BOOTSTRAP_SAVE`: save now (same rule as
        `ContinuousCheckpointingPolicy`).
      * `_BOOTSTRAP_WAIT`: do not save, no schedule yet.
      * `n >= 0`: do not save; the next save is at step `n`. All hosts switch
        to local decisions from this step on.

    The schedule is published as soon as the primary host has
    `min_samples_for_schedule` step time samples, anchored at the later of the
    latest checkpoint in `previous_steps` and the first step observed by this
    process (so a restored checkpoint's age does not count). This keeps
    per-step agreement short after a fresh start or a restart.

    Args:
      step: current step.
      previous_steps: existing checkpoints.
      context: decision context.

    Returns:
      Whether to save at `step`.
    """
    self.num_bootstrap_decisions += 1
    is_primary_host = multihost.is_primary_host(
        context.multiprocessing_options.primary_host
    )
    value = _BOOTSTRAP_WAIT
    if is_primary_host:
      value = self._primary_bootstrap_value(step, previous_steps, context)
    if multihost.process_count() > 1:
      client = signaling_client.get_signaling_client()
      operation_id = synchronization.OperationIdGenerator.next_operation_id()
      key = f"{operation_id}_scheduled_cc_bootstrap_{step.step}/"
      if is_primary_host:
        client.key_value_set(key, str(value), allow_overwrite=True)
      else:
        value = int(
            client.blocking_key_value_get(
                key, timeout_secs=multihost.coordination_timeout()
            )
        )
    if value >= 0:
      self._next_step = value
      self.schedule_history.append((step.step, value))
      logging.info(
          "ScheduledContinuousCheckpointingPolicy: schedule set during"
          " bootstrap at step %d, next save at step %d.",
          step.step,
          value,
      )
      return self._local_decision(step)
    return value == _BOOTSTRAP_SAVE

  def _primary_bootstrap_value(
      self,
      step: PolicyCheckpointInfo,
      previous_steps: Sequence[PolicyCheckpointInfo],
      context: DecisionContext,
  ) -> int:
    """Primary host only. See `_bootstrap_decision`."""
    # Never save or publish a schedule while a write is in flight: a schedule
    # published now could fire before the write ends and force a wait.
    if context.is_saving_in_progress:
      return _BOOTSTRAP_WAIT
    if not previous_steps:
      return _BOOTSTRAP_SAVE
    # Measure from the later of the latest checkpoint and the first step this
    # process observed, so that downtime before a restart does not count.
    assert self._first_step_time is not None
    anchor = max(previous_steps[-1].time, self._first_step_time)
    elapsed = (step.time - anchor).total_seconds()
    if elapsed >= self.minimum_interval_secs:
      return _BOOTSTRAP_SAVE
    est = self._step_time_estimate()
    if (
        est is None
        or est <= 0
        or len(self._samples) < self.min_samples_for_schedule
    ):
      return _BOOTSTRAP_WAIT
    target = float(self.minimum_interval_secs)
    if self.write_duration_margin > 0 and self._last_write_secs is not None:
      target = max(target, self.write_duration_margin * self._last_write_secs)
    self._last_target_secs = target
    remaining = max(0.0, target - elapsed)
    return step.step + max(1, math.ceil(remaining / est))


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
