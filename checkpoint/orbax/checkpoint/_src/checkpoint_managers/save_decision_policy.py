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
from typing import Container, Literal, Protocol, Sequence
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


# Values of the per-save agreement exchange other than a planned step (>= 0).
_AGREED_SAVE = -1
_AGREED_WAIT = -2


@dataclasses.dataclass(kw_only=True)
class ContinuousCheckpointingPolicy(SaveDecisionPolicy):
  """Checkpoint as often as possible, as long as a save is not ongoing.

  The primary host decides with this rule, and all hosts follow it:

  1. When `target_interval_steps` is set, always save at every exact multiple
     of it (bypassing `minimum_interval_secs` and `is_saving_in_progress`). If
     an opportunistic continuous save is still in progress when a target step
     arrives, `CheckpointManager.save` cancels it; if a previous target save is
     still in progress, `CheckpointManager.save` blocks until it completes.
  2. Never save opportunistically while a save is in progress (checked via the
     `DecisionContext`); this prevents blocking on an ongoing save, which would
     hurt accelerator utilization.
  3. Save if no checkpoint exists yet.
  4. When `target_interval_steps` is set, do not save within
     `deferral_window_steps` steps before the next multiple, so the background
     writer stays free for the target step.
  5. Otherwise save once `minimum_interval_secs` have passed since the latest
     checkpoint (or immediately, if it is None).

  `agreement` selects when the hosts exchange the primary host's decision:

  * `'per_check'` (default): on every `should_save` call, through a key-value
    exchange.
  * `'per_save'` (EXPERIMENTAL): when a save starts, the primary host plans the
    step of the next save and shares it. Earlier steps decide locally, without
    messages. At the planned step the hosts exchange the primary host's
    decision (save, or a new planned step if the rule says not yet). Target
    steps always save, as with `'per_check'`. If the previous write is still
    running at a non-target planned step, the hosts skip the step and exchange
    again on each step until it ends. The plan is the earlier of the next
    target step and
    `max(minimum_interval_secs, write hint) / step_time_estimate` steps; a plan
    that falls inside the deferral window moves to the target step. The step
    time estimate is the median of recent step times (excluding the policy's
    own exchanges) and the write hint is `write_hint_fraction` times the
    shortest of the last `write_hint_window` writes. Saves are never early:
    the primary host checks the rule at the planned step. Until a plan exists
    (start, restart, preemption save) the hosts exchange on every step. After
    a restart, `minimum_interval_secs` counts from the first step this process
    observes. Requires the save-start notification from `CheckpointManager`
    (`notify_save_start`). In an `AnySavePolicy`, put this policy first so it
    observes every step.

  Attributes:
    minimum_interval_secs: The minimum time in seconds that must elapse between
      the timestamp of the previous checkpoint and the current step. If `None`
      (the default), back-to-back saves are permitted as soon as the ongoing
      save completes.
    target_interval_steps: Optional target step interval with which continuous
      checkpoints must strictly align. When set, steps where `step.step %
      target_interval_steps == 0` always evaluate to True, and opportunistic
      saves within `deferral_window_steps` steps before a target
      step are deferred.
    deferral_window_steps: Optional number of steps before each multiple of
      `target_interval_steps` during which new continuous saves are deferred to
      keep the background writer free for the upcoming target step. Only used
      when `target_interval_steps` is set. If `None`, defaults to
      `target_interval_steps // 2`.
    agreement: `'per_check'` or `'per_save'`; see above.
    write_hint_fraction: `'per_save'` only. Fraction of the shortest recent
      write duration used as the minimum planned interval. 0 disables it.
    write_hint_window: `'per_save'` only. Number of recent writes considered.
    step_time_window: `'per_save'` only. Number of step time samples kept.
    step_time_quantile: `'per_save'` only. Quantile of the samples used as the
      step time estimate. Larger values plan earlier.
    min_samples_for_schedule: `'per_save'` only. Samples needed before a plan
      is published without a save.
    num_bootstrap_decisions: `'per_save'` only (stats). Key-value exchanges in
      `should_save` while no plan existed after a start, restart or preemption
      save.
    num_planned_step_decisions: `'per_save'` only (stats). Key-value exchanges
      in `should_save` from a planned step until the save or a new plan.
    num_local_decisions: `'per_save'` only (stats). Decisions made locally,
      without messages.
    schedule_history: `'per_save'` only (stats). Plans as `(step, next_step)`.
  """

  minimum_interval_secs: float | None = None
  target_interval_steps: int | None = None
  deferral_window_steps: int | None = None
  agreement: Literal["per_check", "per_save"] = "per_check"
  write_hint_fraction: float = 0.8
  write_hint_window: int = 4
  step_time_window: int = 64
  step_time_quantile: float = 0.5
  min_samples_for_schedule: int = 4

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
    if self.agreement not in ("per_check", "per_save"):
      raise ValueError(
          "agreement must be 'per_check' or 'per_save', got"
          f" {self.agreement!r}."
      )
    if self.agreement == "per_save":
      if (
          self.minimum_interval_secs is not None
          and self.minimum_interval_secs < 0
      ):
        raise ValueError("minimum_interval_secs must be >= 0.")
      if not 0.0 <= self.write_hint_fraction <= 1.0:
        raise ValueError("write_hint_fraction must be in [0, 1].")
      if self.write_hint_window < 1:
        raise ValueError("write_hint_window must be >= 1.")
      if not 0.0 <= self.step_time_quantile <= 1.0:
        raise ValueError("step_time_quantile must be in [0, 1].")
      if self.min_samples_for_schedule < 1:
        raise ValueError("min_samples_for_schedule must be >= 1.")
    # State of the 'per_save' agreement.
    self._next_step: int | None = None
    # True from a planned step until the save or a new planned step.
    self._at_planned_step = False
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
    self._recent_write_secs: collections.deque[float] = collections.deque(
        maxlen=self.write_hint_window
    )
    self._last_target_secs: float | None = None
    # Stats of the 'per_save' agreement, for tests and benchmarks.
    self.num_bootstrap_decisions = 0
    self.num_planned_step_decisions = 0
    self.num_local_decisions = 0
    self.schedule_history: list[tuple[int, int | None]] = []

  # Shared rule.

  def _deferral_window(self) -> int:
    assert self.target_interval_steps is not None
    if self.deferral_window_steps is not None:
      return self.deferral_window_steps
    return self.target_interval_steps // 2

  def _save_rule(
      self,
      step: PolicyCheckpointInfo,
      previous_steps: Sequence[PolicyCheckpointInfo],
      *,
      is_saving_in_progress: bool,
      anchor_time: datetime.datetime | None = None,
  ) -> bool:
    """The primary host's save decision (see the class docstring).

    Args:
      step: current step.
      previous_steps: existing checkpoints.
      is_saving_in_progress: whether a save is in progress.
      anchor_time: start of the minimum interval; defaults to the time of the
        latest checkpoint.

    Returns:
      Whether to save at `step`.
    """
    if (
        self.target_interval_steps is not None
        and step.step % self.target_interval_steps == 0
    ):
      return True
    if is_saving_in_progress:
      return False
    if not previous_steps:
      return True
    if self.target_interval_steps is not None:
      steps_until_target = self.target_interval_steps - (
          step.step % self.target_interval_steps
      )
      if steps_until_target <= self._deferral_window():
        return False
    if self.minimum_interval_secs is None:
      return True
    if anchor_time is None:
      anchor_time = previous_steps[-1].time
    return step.time - anchor_time >= datetime.timedelta(
        seconds=self.minimum_interval_secs
    )

  def should_save(
      self,
      step: PolicyCheckpointInfo,
      previous_steps: Sequence[PolicyCheckpointInfo],
      *,
      context: DecisionContext,
  ) -> bool:
    if self.agreement == "per_save":
      return self._per_save_should_save(step, previous_steps, context)
    is_primary_host = multihost.is_primary_host(
        context.multiprocessing_options.primary_host
    )
    client = signaling_client.get_signaling_client()
    operation_id = synchronization.OperationIdGenerator.next_operation_id()
    result_barrier_key = f"{operation_id}_continuous_checkpointing_policy_should_save_{step.step}/"
    # Make time based and save in progress based decision only on primary host
    # and broadcast to all hosts.
    if is_primary_host:
      save_result = int(
          self._save_rule(
              step,
              previous_steps,
              is_saving_in_progress=context.is_saving_in_progress,
          )
      )
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
        f" (minimum_interval_secs={self.minimum_interval_secs},"
        f" target_interval_steps={self.target_interval_steps},"
        f" deferral_window_steps={self.deferral_window_steps})",
        step,
        bool(save_result),
    )
    return bool(save_result)

  # 'per_save' agreement.

  @property
  def next_step(self) -> int | None:
    """Returns the next planned save step (`'per_save'`), or None."""
    return self._next_step

  def _record_write_end(self, now: float) -> None:
    assert self._save_start_time is not None
    self._last_write_secs = now - self._save_start_time
    self._recent_write_secs.append(self._last_write_secs)
    self._save_in_flight = False

  def _observe(self, step: int, now: float, is_saving: bool) -> None:
    if self._save_in_flight and not is_saving:
      self._record_write_end(now)
    if self._last_obs is not None and step > self._last_obs[0]:
      sample = (now - self._last_obs[1]) / (step - self._last_obs[0])
      self._samples.append(sample)
      self._samples_since_save.append(sample)
    self._last_obs = (step, now)

  def _quantile(self, samples: Sequence[float]) -> float:
    s = sorted(samples)
    return s[min(len(s) - 1, int(self.step_time_quantile * len(s)))]

  def _step_time_estimate(self) -> float | None:
    """Quantile (default: median) of recent step times.

    Larger values make plans earlier: an early planned step costs one more
    exchange (the primary host publishes a new planned step), while a late
    one delays the save by the error. Takes the maximum of the quantile over
    the whole window and over the samples since the last save, so a sudden
    slowdown is picked up at the next save instead of after the window turns
    over.

    Returns:
      The estimate in seconds, or None without samples.
    """
    if not self._samples:
      return None
    est = self._quantile(self._samples)
    if self._samples_since_save:
      est = max(est, self._quantile(self._samples_since_save))
    return est

  def _write_hint_secs(self) -> float:
    """Expected write time used for planning.

    Returns:
      `write_hint_fraction` times the shortest of the last `write_hint_window`
      write durations, or 0 until that many writes have completed in this
      process (so a slow first write, e.g. startup, does not delay saves).
    """
    if (
        self.write_hint_fraction <= 0
        or len(self._recent_write_secs) < self.write_hint_window
    ):
      return 0.0
    return self.write_hint_fraction * min(self._recent_write_secs)

  def _align_to_target(self, step: int, planned: int) -> int:
    """Moves a plan at or past the next target, or inside its window, to it."""
    if self.target_interval_steps is None:
      return planned
    next_target = (step // self.target_interval_steps + 1) * (
        self.target_interval_steps
    )
    if next_target - planned <= self._deferral_window():
      return next_target
    return planned

  def _compute_next_step(self, step: int) -> int | None:
    """Primary host only. Returns the planned step of the next save, or None."""
    est = self._step_time_estimate()
    if est is None or est <= 0:
      return None
    target = max(
        float(self.minimum_interval_secs or 0), self._write_hint_secs()
    )
    self._last_target_secs = target
    return self._align_to_target(step, step + max(1, math.ceil(target / est)))

  def on_save_start(self, step: int, *, context: DecisionContext) -> None:
    """Handles the save-start notification sent by `CheckpointManager.save`."""
    if self.agreement != "per_save":
      return
    now = time.monotonic()
    if self._save_in_flight:
      # The previous write finished during this save's wait (only for saves
      # triggered by other policies; this policy never waits).
      self._record_write_end(now)
    self._at_planned_step = False
    next_step = None
    # On a preemption save the job is about to exit: skip the next-step
    # exchange and clear the plan. Every host gets the same
    # `reached_preemption`, so they stay consistent. If the job continues, the
    # policy re-bootstraps, as after a restart.
    if not context.reached_preemption:
      is_primary_host = multihost.is_primary_host(
          context.multiprocessing_options.primary_host
      )
      if is_primary_host:
        next_step = self._compute_next_step(step)
      if multihost.process_count() > 1:
        client = signaling_client.get_signaling_client()
        operation_id = synchronization.OperationIdGenerator.next_operation_id()
        key = f"{operation_id}_continuous_checkpointing_next_step_{step}/"
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
        "ContinuousCheckpointingPolicy (per_save): save at step %d, next save"
        " at step %s (target_secs=%s, last_write_secs=%s, step_time_est=%s).",
        step,
        next_step,
        self._last_target_secs,
        self._last_write_secs,
        self._step_time_estimate(),
    )

  def _per_save_should_save(
      self,
      step: PolicyCheckpointInfo,
      previous_steps: Sequence[PolicyCheckpointInfo],
      context: DecisionContext,
  ) -> bool:
    """`should_save` with the `'per_save'` agreement."""
    self._observe(step.step, time.monotonic(), context.is_saving_in_progress)
    if self._first_step_time is None:
      self._first_step_time = step.time
    if self._next_step is not None:
      if step.step < self._next_step:
        self.num_local_decisions += 1
        _log_save_decision(
            "ContinuousCheckpointingPolicy (per_save,"
            f" next_step={self._next_step})",
            step,
            False,
        )
        return False
      # Planned step reached. Every host gets here at the same step, because
      # `next_step` is shared. Do not decide locally: the previous write may
      # still be running. Clear the plan and let the primary host decide, on
      # this step and, while the write runs, on each following step.
      self._next_step = None
      self._at_planned_step = True
    return self._agreed_decision(step, previous_steps, context)

  def _agreed_decision(
      self,
      step: PolicyCheckpointInfo,
      previous_steps: Sequence[PolicyCheckpointInfo],
      context: DecisionContext,
  ) -> bool:
    """One key-value exchange of the primary host's decision.

    Used whenever no plan exists: after a start, a restart or a preemption
    save, and from a planned step until the save. The primary host broadcasts:
      * `_AGREED_SAVE`: save now.
      * `_AGREED_WAIT`: do not save, no plan yet.
      * `n >= 0`: do not save; the next save is planned at step `n`. All hosts
        decide locally until then.

    Args:
      step: current step.
      previous_steps: existing checkpoints.
      context: decision context.

    Returns:
      Whether to save at `step`.
    """
    if self._at_planned_step:
      self.num_planned_step_decisions += 1
    else:
      self.num_bootstrap_decisions += 1
    is_primary_host = multihost.is_primary_host(
        context.multiprocessing_options.primary_host
    )
    value = _AGREED_WAIT
    if is_primary_host:
      value = self._primary_value(step, previous_steps, context)
    if multihost.process_count() > 1:
      client = signaling_client.get_signaling_client()
      operation_id = synchronization.OperationIdGenerator.next_operation_id()
      key = f"{operation_id}_continuous_checkpointing_decision_{step.step}/"
      if is_primary_host:
        client.key_value_set(key, str(value), allow_overwrite=True)
      else:
        value = int(
            client.blocking_key_value_get(
                key, timeout_secs=multihost.coordination_timeout()
            )
        )
    # Exclude this exchange from the next step time sample: planned steps are
    # decided locally, so the estimate must not include exchange time.
    self._last_obs = (step.step, time.monotonic())
    if value >= 0:
      self._next_step = value
      self._at_planned_step = False
      self.schedule_history.append((step.step, value))
      logging.info(
          "ContinuousCheckpointingPolicy (per_save): planned without a save"
          " at step %d, next save at step %d.",
          step.step,
          value,
      )
      return step.step >= value
    result = value == _AGREED_SAVE
    _log_save_decision(
        "ContinuousCheckpointingPolicy (per_save, agreed)", step, result
    )
    return result

  def _primary_value(
      self,
      step: PolicyCheckpointInfo,
      previous_steps: Sequence[PolicyCheckpointInfo],
      context: DecisionContext,
  ) -> int:
    """Primary host only. See `_agreed_decision`."""
    # Measure the minimum interval from the later of the latest checkpoint and
    # the first step this process observed, so that downtime before a restart
    # does not count.
    anchor = None
    if previous_steps:
      assert self._first_step_time is not None
      anchor = max(previous_steps[-1].time, self._first_step_time)
    # Target steps always save, even while a write is in flight.
    if self._save_rule(
        step,
        previous_steps,
        is_saving_in_progress=context.is_saving_in_progress,
        anchor_time=anchor,
    ):
      return _AGREED_SAVE
    # While a write is in flight: skip the step and ask again on the next one.
    if context.is_saving_in_progress:
      return _AGREED_WAIT
    # Not yet: inside the deferral window, or the interval has not passed.
    est = self._step_time_estimate()
    if (
        est is None
        or est <= 0
        or len(self._samples) < self.min_samples_for_schedule
    ):
      return _AGREED_WAIT
    # No write is in flight here, so the write-time hint does not apply.
    remaining = 0.0
    if self.minimum_interval_secs is not None and anchor is not None:
      elapsed = (step.time - anchor).total_seconds()
      remaining = max(0.0, self.minimum_interval_secs - elapsed)
    self._last_target_secs = remaining
    planned = step.step + max(1, math.ceil(remaining / est))
    return self._align_to_target(step.step, planned)


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
