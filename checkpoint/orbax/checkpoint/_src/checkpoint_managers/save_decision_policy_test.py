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

import datetime
from typing import Sequence

from absl.testing import absltest
from absl.testing import parameterized
from orbax.checkpoint import options as options_lib
from orbax.checkpoint._src.checkpoint_managers import save_decision_policy as save_decision_policy_lib
from orbax.checkpoint._src.metadata import checkpoint_info

CheckpointInfo = checkpoint_info.CheckpointInfo
DecisionContext = save_decision_policy_lib.DecisionContext
ContinuousCheckpointingPolicy = (
    save_decision_policy_lib.ContinuousCheckpointingPolicy
)


def _make_step_info(
    step: int,
    base_time: datetime.datetime,
    offset_secs: float,
) -> CheckpointInfo:
  return CheckpointInfo(
      step=step,
      time=base_time + datetime.timedelta(seconds=offset_secs),
      metrics=None,
  )


def _make_context(
    *,
    is_saving_in_progress: bool = False,
    in_progress_step: int | None = None,
    reached_preemption: bool = False,
) -> DecisionContext:
  return DecisionContext(
      is_saving_in_progress=is_saving_in_progress,
      in_progress_step=in_progress_step,
      reached_preemption=reached_preemption,
      multiprocessing_options=options_lib.MultiprocessingOptions(),
      cancel_in_progress_save=lambda: None,
  )


def _set_save_durations(
    policy: ContinuousCheckpointingPolicy, save_durations_secs: Sequence[float]
) -> None:
  """Sets the save durations the policy has measured."""
  policy._save_durations_secs.clear()
  policy._save_durations_secs.extend(save_durations_secs)


def _fill_step_duration_window(
    policy: ContinuousCheckpointingPolicy, step_duration_secs: float
) -> None:
  """Fills the policy's step duration window so the warmup guard passes."""
  policy._step_durations_secs.extend(
      [step_duration_secs]
      * save_decision_policy_lib._STEP_DURATION_WINDOW_SIZE
  )


class SaveDecisionPolicyTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.base_time = datetime.datetime(
        2026, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="negative_minimum_interval",
          target_interval_steps=10,
          deferral_window_steps=None,
          minimum_interval_secs=-1,
      ),
      dict(
          testcase_name="zero_target_interval",
          target_interval_steps=0,
          deferral_window_steps=None,
      ),
      dict(
          testcase_name="negative_target_interval",
          target_interval_steps=-5,
          deferral_window_steps=None,
      ),
      dict(
          testcase_name="negative_deferral_window",
          target_interval_steps=10,
          deferral_window_steps=-1,
      ),
      dict(
          testcase_name="zero_deferral_percentile",
          target_interval_steps=10,
          deferral_window_steps=None,
          deferral_save_time_percentile=0.0,
      ),
      dict(
          testcase_name="greater_than_one_deferral_percentile",
          target_interval_steps=10,
          deferral_window_steps=None,
          deferral_save_time_percentile=1.1,
      ),
  )
  def test_continuous_policy_invalid_arguments_raises_value_error(
      self,
      target_interval_steps: int | None,
      deferral_window_steps: int | None,
      minimum_interval_secs: int | None = None,
      deferral_save_time_percentile: float | None = 0.9,
  ):
    with self.assertRaises(ValueError):
      ContinuousCheckpointingPolicy(
          minimum_interval_secs=minimum_interval_secs,
          target_interval_steps=target_interval_steps,
          deferral_window_steps=deferral_window_steps,
          deferral_save_time_percentile=deferral_save_time_percentile,
      )

  def test_continuous_policy_saving_in_progress_behavior(self):
    in_progress_ctx = _make_context(is_saving_in_progress=True)
    unconstrained_policy = ContinuousCheckpointingPolicy()
    self.assertFalse(
        unconstrained_policy.should_save(
            _make_step_info(0, self.base_time, 0),
            [],
            context=in_progress_ctx,
        )
    )
    prev = [_make_step_info(0, self.base_time, 0)]
    self.assertFalse(
        unconstrained_policy.should_save(
            _make_step_info(10, self.base_time, 100),
            prev,
            context=in_progress_ctx,
        )
    )

    constrained_policy = ContinuousCheckpointingPolicy(
        target_interval_steps=10,
        deferral_window_steps=3,
    )
    # Non-target steps never save while a save is in progress.
    self.assertFalse(
        constrained_policy.should_save(
            _make_step_info(1, self.base_time, 10),
            prev,
            context=in_progress_ctx,
        )
    )
    self.assertFalse(
        constrained_policy.should_save(
            _make_step_info(5, self.base_time, 50),
            prev,
            context=in_progress_ctx,
        )
    )
    # Exact multiples of target_interval_steps strictly evaluate to True even
    # when a save is in progress.
    self.assertTrue(
        constrained_policy.should_save(
            _make_step_info(0, self.base_time, 0),
            [],
            context=in_progress_ctx,
        )
    )
    self.assertTrue(
        constrained_policy.should_save(
            _make_step_info(10, self.base_time, 100),
            prev,
            context=in_progress_ctx,
        )
    )

  def test_continuous_policy_no_dynamic_deferral_by_default(self):
    # With target_interval_steps=10 and no deferral options set, no
    # opportunistic saves are deferred prior to target steps, even with save
    # history and step duration estimates available.
    policy = ContinuousCheckpointingPolicy(target_interval_steps=10)
    _fill_step_duration_window(policy, 10.0)
    _set_save_durations(policy, [10.0, 20.0, 30.0])
    ctx = _make_context()
    previous_steps: list[CheckpointInfo] = []
    saved_steps: list[int] = []

    for step in range(11):
      info = _make_step_info(step, self.base_time, step * 10.0)
      if policy.should_save(info, previous_steps, context=ctx):
        saved_steps.append(step)
        previous_steps.append(info)

    self.assertSequenceEqual(saved_steps, list(range(11)))

  def test_continuous_policy_dynamic_deferral_save_time_percentile(self):
    # Historical save durations: [10s, 20s, 30s] -> p90 = 28s.
    # Step timestamps advance by 10s/step, tracked automatically by the policy.
    # With target_interval_steps=10, steps 8 (20s to target <= 28s) and 9
    # (10s to target <= 28s) are deferred, while steps 1..7 (>= 30s to target)
    # and target step 10 are saved.
    policy = ContinuousCheckpointingPolicy(
        target_interval_steps=10, deferral_save_time_percentile=0.9
    )
    _fill_step_duration_window(policy, 10.0)
    _set_save_durations(policy, [10.0, 20.0, 30.0])
    ctx = _make_context()
    previous_steps: list[CheckpointInfo] = []
    saved_steps: list[int] = []

    for step in range(11):
      info = _make_step_info(step, self.base_time, step * 10.0)
      if policy.should_save(info, previous_steps, context=ctx):
        saved_steps.append(step)
        previous_steps.append(info)

    self.assertSequenceEqual(saved_steps, [0, 1, 2, 3, 4, 5, 6, 7, 10])

  @parameterized.named_parameters(
      dict(
          testcase_name="too_few_saves",
          save_durations_secs=[10.0, 20.0],
          fill_step_duration_window=True,
      ),
      dict(
          testcase_name="step_duration_window_not_full",
          save_durations_secs=[10.0, 20.0, 30.0],
          fill_step_duration_window=False,
      ),
  )
  def test_continuous_policy_dynamic_deferral_warmup(
      self,
      save_durations_secs: Sequence[float],
      fill_step_duration_window: bool,
  ):
    # Same setup as the dynamic deferral test, but without enough history yet,
    # so no opportunistic saves are deferred.
    policy = ContinuousCheckpointingPolicy(
        target_interval_steps=10, deferral_save_time_percentile=0.9
    )
    if fill_step_duration_window:
      _fill_step_duration_window(policy, 10.0)
    _set_save_durations(policy, save_durations_secs)
    ctx = _make_context()
    previous_steps: list[CheckpointInfo] = []
    saved_steps: list[int] = []

    for step in range(11):
      info = _make_step_info(step, self.base_time, step * 10.0)
      if policy.should_save(info, previous_steps, context=ctx):
        saved_steps.append(step)
        previous_steps.append(info)

    self.assertSequenceEqual(saved_steps, list(range(11)))

  def test_continuous_policy_step_duration_estimate_ignores_outlier(
      self,
  ):
    # p90 of 25s historical saves is 25s. Step 20 takes 300s (e.g. a stall),
    # and every other step takes 1s. With a mean over the window, the estimate
    # at step 38 would be (300 + 19 * 1) / 20 ~= 16s/step, so 2 steps to target
    # ~= 32s > 25s and step 38 would start a save that is almost certain to be
    # cancelled at step 40. The median estimates 1s/step, so step 38 (2s to
    # target <= 25s) is deferred.
    policy = ContinuousCheckpointingPolicy(
        target_interval_steps=40, deferral_save_time_percentile=0.9
    )
    _set_save_durations(policy, [25.0, 25.0, 25.0])
    busy_ctx = _make_context(is_saving_in_progress=True)
    idle_ctx = _make_context()
    prev = [_make_step_info(0, self.base_time, 0)]
    offset_secs = 0.0
    policy.should_save(prev[0], prev, context=busy_ctx)
    for step in range(1, 38):
      offset_secs += 300.0 if step == 20 else 1.0
      policy.should_save(
          _make_step_info(step, self.base_time, offset_secs),
          prev,
          context=busy_ctx,
      )
    self.assertEqual(policy._get_step_duration_secs(), 1.0)

    offset_secs += 1.0
    self.assertFalse(
        policy.should_save(
            _make_step_info(38, self.base_time, offset_secs),
            prev,
            context=idle_ctx,
        )
    )

  def test_continuous_policy_step_duration_window_is_bounded(self):
    policy = ContinuousCheckpointingPolicy(target_interval_steps=1000)
    busy_ctx = _make_context(is_saving_in_progress=True)
    prev = [_make_step_info(0, self.base_time, 0)]
    policy.should_save(prev[0], prev, context=busy_ctx)
    # 30 slow steps (10s each) followed by enough fast steps (1s each) to fill
    # the window evicts every slow observation.
    offset_secs = 0.0
    window_size = save_decision_policy_lib._STEP_DURATION_WINDOW_SIZE
    for step in range(1, 31 + window_size):
      offset_secs += 10.0 if step <= 30 else 1.0
      policy.should_save(
          _make_step_info(step, self.base_time, offset_secs),
          prev,
          context=busy_ctx,
      )
    self.assertLen(policy._step_durations_secs, window_size)
    self.assertEqual(policy._get_step_duration_secs(), 1.0)

  def test_continuous_policy_measures_save_durations(self):
    # Steps are 10s apart. Each save is timed from the step it was requested at
    # until the first step at which it is no longer in progress, including a
    # save that is replaced (cancelled) by a target step.
    policy = ContinuousCheckpointingPolicy(target_interval_steps=10)
    prev = [_make_step_info(0, self.base_time, 0)]
    observations = [
        # (step, in_progress_step); None means no save in progress.
        (0, None),  # Target step 0 saves.
        (1, 0),
        (3, None),  # Save of step 0 took 30s; step 3 saves.
        (4, 3),
        (10, 3),  # Target step 10 cancels the save of step 3.
        (11, 10),  # Save of step 3 ran for 80s before being replaced.
        (12, None),  # Save of step 10 took 20s.
    ]
    for step, in_progress_step in observations:
      policy.should_save(
          _make_step_info(step, self.base_time, step * 10.0),
          prev,
          context=_make_context(
              is_saving_in_progress=in_progress_step is not None,
              in_progress_step=in_progress_step,
          ),
      )
    self.assertSequenceEqual(policy._save_durations_secs, [30.0, 80.0, 20.0])

  def test_continuous_policy_ignores_in_progress_step_of_finished_save(self):
    # `in_progress_step` may still name the last save after it has finished;
    # only `is_saving_in_progress` says whether a save is running.
    policy = ContinuousCheckpointingPolicy(target_interval_steps=10)
    prev = [_make_step_info(0, self.base_time, 0)]
    for step in range(1, 5):
      policy.should_save(
          _make_step_info(step, self.base_time, step * 10.0),
          prev,
          context=_make_context(
              is_saving_in_progress=False, in_progress_step=0
          ),
      )
    self.assertEmpty(policy._save_durations_secs)

  @parameterized.named_parameters(
      dict(
          testcase_name="explicit_window",
          deferral_window_steps=3,
          deferral_save_time_percentile=None,
          save_durations_secs=(),
      ),
      dict(
          testcase_name="dynamic_p90",
          deferral_window_steps=None,
          deferral_save_time_percentile=0.9,
          save_durations_secs=(10.0, 20.0, 30.0),
      ),
  )
  def test_continuous_policy_records_skip_reason(
      self,
      deferral_window_steps: int | None,
      deferral_save_time_percentile: float | None,
      save_durations_secs: Sequence[float],
  ):
    # Step 8 is 2 steps (20s at 10s/step) before target step 10, inside both a
    # fixed 3-step window and the dynamic window (p90 of historical saves is
    # 28s). A reached preemption does not change the decision: saving on
    # preemption is opt-in via `PreemptionCheckpointingPolicy`.
    policy = ContinuousCheckpointingPolicy(
        minimum_interval_secs=60,
        target_interval_steps=10,
        deferral_window_steps=deferral_window_steps,
        deferral_save_time_percentile=deferral_save_time_percentile,
    )
    _fill_step_duration_window(policy, 10.0)
    _set_save_durations(policy, save_durations_secs)
    prev = [_make_step_info(0, self.base_time, 0)]

    with self.subTest("deferred_for_target_step"):
      self.assertFalse(
          policy.should_save(
              _make_step_info(8, self.base_time, 80),
              prev,
              context=_make_context(reached_preemption=True),
          )
      )
      self.assertEqual(policy.last_skip_reason, "deferred_for_target_step")

    with self.subTest("save_in_progress"):
      self.assertFalse(
          policy.should_save(
              _make_step_info(8, self.base_time, 80),
              prev,
              context=_make_context(is_saving_in_progress=True),
          )
      )
      self.assertEqual(policy.last_skip_reason, "save_in_progress")

    with self.subTest("minimum_interval_not_met"):
      # Step 2 is outside the deferral window, but only 20s have elapsed.
      self.assertFalse(
          policy.should_save(
              _make_step_info(2, self.base_time, 20),
              prev,
              context=_make_context(),
          )
      )
      self.assertEqual(policy.last_skip_reason, "minimum_interval_not_met")

    with self.subTest("saved"):
      self.assertTrue(
          policy.should_save(
              _make_step_info(10, self.base_time, 100),
              prev,
              context=_make_context(),
          )
      )
      self.assertIsNone(policy.last_skip_reason)

  @parameterized.named_parameters(
      dict(
          testcase_name="explicit_window_3",
          deferral_window_steps=3,
          expected_saved_steps=[0, 1, 2, 3, 4, 5, 6, 10],
      ),
      dict(
          testcase_name="zero_deferral_window",
          deferral_window_steps=0,
          expected_saved_steps=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
      ),
      dict(
          testcase_name="full_interval_deferral_window",
          deferral_window_steps=10,
          expected_saved_steps=[0, 10],
      ),
  )
  def test_continuous_policy_explicit_deferral_window(
      self,
      deferral_window_steps: int,
      expected_saved_steps: Sequence[int],
  ):
    policy = ContinuousCheckpointingPolicy(
        target_interval_steps=10,
        deferral_window_steps=deferral_window_steps,
    )
    idle_ctx = _make_context(is_saving_in_progress=False)
    previous_steps: list[CheckpointInfo] = []
    saved_steps: list[int] = []

    for step in range(11):
      info = _make_step_info(step, self.base_time, step)
      if policy.should_save(info, previous_steps, context=idle_ctx):
        saved_steps.append(step)
        previous_steps.append(info)

    self.assertSequenceEqual(saved_steps, expected_saved_steps)

  def test_continuous_policy_target_step_bypasses_minimum_interval_secs(self):
    policy = ContinuousCheckpointingPolicy(
        minimum_interval_secs=60,
        target_interval_steps=10,
        deferral_window_steps=3,
    )
    idle_ctx = _make_context(is_saving_in_progress=False)

    # Initial save at step 0 (t=0s).
    step_0 = _make_step_info(0, self.base_time, 0)
    self.assertTrue(policy.should_save(step_0, [], context=idle_ctx))
    prev = [step_0]

    # Step 4 at t=20s: outside deferral window (steps_until_target=6 > 3), but
    # minimum_interval_secs (60s) has not elapsed -> False.
    step_4_early = _make_step_info(4, self.base_time, 20)
    self.assertFalse(policy.should_save(step_4_early, prev, context=idle_ctx))

    # Step 4 at t=65s: outside deferral window and >= 60s elapsed -> True.
    step_4_ready = _make_step_info(4, self.base_time, 65)
    self.assertTrue(policy.should_save(step_4_ready, prev, context=idle_ctx))
    prev.append(step_4_ready)

    # Step 8 at t=130s: >= 60s elapsed since step 4 (t=65s), but inside the
    # deferral window (steps_until_target=2 <= 3) -> False.
    step_8 = _make_step_info(8, self.base_time, 130)
    self.assertFalse(policy.should_save(step_8, prev, context=idle_ctx))

    # Step 10 at t=80s: only 15s elapsed since step 4 (t=65s), which is less
    # than minimum_interval_secs=60s, but step 10 is an exact target step
    # multiple and no save is in progress -> True.
    step_10 = _make_step_info(10, self.base_time, 80)
    self.assertTrue(policy.should_save(step_10, prev, context=idle_ctx))

  def test_continuous_policy_target_step_invokes_cancel_callback_for_non_target_in_progress(
      self,
  ):
    policy = ContinuousCheckpointingPolicy(target_interval_steps=10)
    prev = [_make_step_info(0, self.base_time, 0)]

    # 1. Target step 10 with non-target step 7 in progress -> invokes callback.
    cancel_calls = 0

    def record_cancel():
      nonlocal cancel_calls
      cancel_calls += 1

    ctx_non_target = DecisionContext(
        is_saving_in_progress=True,
        in_progress_step=7,
        reached_preemption=False,
        multiprocessing_options=options_lib.MultiprocessingOptions(),
        cancel_in_progress_save=record_cancel,
    )
    self.assertTrue(
        policy.should_save(
            _make_step_info(10, self.base_time, 100),
            prev,
            context=ctx_non_target,
        )
    )
    self.assertEqual(cancel_calls, 1)

    # 2. Target step 10 with previous target step 0 in progress -> does NOT
    # invoke cancel callback.
    cancel_calls = 0
    ctx_target_in_progress = DecisionContext(
        is_saving_in_progress=True,
        in_progress_step=0,
        reached_preemption=False,
        multiprocessing_options=options_lib.MultiprocessingOptions(),
        cancel_in_progress_save=record_cancel,
    )
    self.assertTrue(
        policy.should_save(
            _make_step_info(10, self.base_time, 100),
            prev,
            context=ctx_target_in_progress,
        )
    )
    self.assertEqual(cancel_calls, 0)

    # 3. Target step 10 with no save in progress -> does NOT invoke callback.
    cancel_calls = 0
    ctx_idle = DecisionContext(
        is_saving_in_progress=False,
        in_progress_step=None,
        reached_preemption=False,
        multiprocessing_options=options_lib.MultiprocessingOptions(),
        cancel_in_progress_save=record_cancel,
    )
    self.assertTrue(
        policy.should_save(
            _make_step_info(10, self.base_time, 100),
            prev,
            context=ctx_idle,
        )
    )
    self.assertEqual(cancel_calls, 0)

    # 4. Wrapped inside AnySavePolicy -> still invokes cancel callback at target
    # step when non-target save is in progress.
    any_policy = save_decision_policy_lib.AnySavePolicy([
        save_decision_policy_lib.InitialSavePolicy(),
        policy,
        save_decision_policy_lib.PreemptionCheckpointingPolicy(),
    ])
    cancel_calls = 0
    self.assertTrue(
        any_policy.should_save(
            _make_step_info(10, self.base_time, 100),
            prev,
            context=ctx_non_target,
        )
    )
    self.assertEqual(cancel_calls, 1)


if __name__ == "__main__":
  absltest.main()
