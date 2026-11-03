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


def _make_context(*, is_saving_in_progress: bool = False) -> DecisionContext:
  return DecisionContext(
      is_saving_in_progress=is_saving_in_progress,
      reached_preemption=False,
      multiprocessing_options=options_lib.MultiprocessingOptions(),
  )


class SaveDecisionPolicyTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.base_time = datetime.datetime(
        2026, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc
    )

  @parameterized.named_parameters(
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
  )
  def test_continuous_policy_invalid_arguments_raises_value_error(
      self,
      target_interval_steps: int | None,
      deferral_window_steps: int | None,
  ):
    with self.assertRaises(ValueError):
      ContinuousCheckpointingPolicy(
          target_interval_steps=target_interval_steps,
          deferral_window_steps=deferral_window_steps,
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

  def test_continuous_policy_default_deferral_window(self):
    # With target_interval_steps=10 and deferral_window_steps=None, the default
    # deferral window is 10 // 2 = 5 steps (steps 5..9 are deferred).
    policy = ContinuousCheckpointingPolicy(target_interval_steps=10)
    idle_ctx = _make_context(is_saving_in_progress=False)
    prev = [_make_step_info(0, self.base_time, 0)]

    for step in range(1, 5):
      self.assertTrue(
          policy.should_save(
              _make_step_info(step, self.base_time, step),
              prev,
              context=idle_ctx,
          ),
          f"Expected step {step} outside deferral window to save.",
      )

    for step in range(5, 10):
      self.assertFalse(
          policy.should_save(
              _make_step_info(step, self.base_time, step),
              prev,
              context=idle_ctx,
          ),
          f"Expected step {step} inside deferral window to defer.",
      )

    self.assertTrue(
        policy.should_save(
            _make_step_info(10, self.base_time, 10),
            prev,
            context=idle_ctx,
        )
    )

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


if __name__ == "__main__":
  absltest.main()
