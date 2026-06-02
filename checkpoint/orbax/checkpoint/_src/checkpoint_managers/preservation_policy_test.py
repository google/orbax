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

"""Tests for preservation policy."""

import datetime
from typing import Sequence
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from orbax.checkpoint._src.checkpoint_managers import preservation_policy as preservation_policy_lib
from orbax.checkpoint._src.metadata import checkpoint_info


class PreservationPolicyTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.preservation_context = preservation_policy_lib.PreservationContext()

  def get_checkpoints(self, steps: Sequence[int] = (0, 1, 2, 3, 4)):
    checkpoints = []
    time_increment = datetime.timedelta(seconds=1)
    start_time = datetime.datetime.now()
    for i, step in enumerate(steps):
      current_time = start_time + i * time_increment
      checkpoints.append(
          checkpoint_info.CheckpointInfo(
              step=step,
              time=current_time,
              metrics=None
          )
      )
    return checkpoints

  def get_preserved_checkpoints(
      self,
      checkpoints: Sequence[checkpoint_info.CheckpointInfo],
      policy: preservation_policy_lib.PreservationPolicy,
  ):
    should_preserve_flags = policy.should_preserve(
        checkpoints,
        context=preservation_policy_lib.PreservationContext(),
    )
    return [
        checkpoint.step
        for checkpoint, should_preserve_flag in zip(
            checkpoints, should_preserve_flags
        )
        if should_preserve_flag
    ]

  @parameterized.parameters(
      dict(
          n=None,
          expected_preserved_steps=[0, 1, 2, 3, 4],
      ),
      dict(
          n=3,
          expected_preserved_steps=[2, 3, 4],
      ),
      dict(
          n=15,
          expected_preserved_steps=[0, 1, 2, 3, 4],
      ),
  )
  def test_latest_n_policy(self, n, expected_preserved_steps):
    policy = preservation_policy_lib.LatestN(n=n)

    self.assertSequenceEqual(
        expected_preserved_steps,
        self.get_preserved_checkpoints(
            self.get_checkpoints(), policy),
    )

  @parameterized.parameters(
      dict(
          interval_secs=1,
          expected_preserved_steps=[0, 1, 2, 3, 4],
      ),
      dict(
          interval_secs=3,
          expected_preserved_steps=[0, 3],
      ),
      dict(
          interval_secs=6,
          expected_preserved_steps=[0],
      ),
  )
  def test_every_n_seconds_policy(
      self, interval_secs, expected_preserved_steps
  ):
    policy = preservation_policy_lib.EveryNSeconds(interval_secs=interval_secs)

    self.assertSequenceEqual(
        expected_preserved_steps,
        self.get_preserved_checkpoints(self.get_checkpoints(), policy),
    )

  @parameterized.parameters(
      dict(
          interval_steps=1,
          steps=[0, 1, 2, 3, 4],
          expected_preserved_steps=[0, 1, 2, 3, 4],
      ),
      dict(
          interval_steps=3,
          steps=[0, 1, 2, 4],
          expected_preserved_steps=[0],
      ),
      dict(
          exact_interval=False,
          interval_steps=3,
          steps=[0, 1, 2, 4],
          expected_preserved_steps=[0, 4],
      ),
      dict(
          interval_steps=6,
          steps=[0, 1, 2, 3, 4],
          expected_preserved_steps=[0],
      ),
      dict(
          exact_interval=False,
          interval_steps=3,
          steps=[0, 1, 2, 4, 5, 8, 9, 13, 14, 25],
          expected_preserved_steps=[0, 4, 8, 13, 25],
      ),
      dict(
          interval_steps=1,
          steps=[0, 1, 2, 3, 4],
          expected_preserved_steps=[2, 3, 4],
          max_to_keep=3,
      ),
      dict(
          exact_interval=False,
          interval_steps=3,
          steps=[0, 1, 2, 4, 5, 8, 9, 13, 14, 25],
          expected_preserved_steps=[8, 13, 25],
          max_to_keep=3,
      ),
  )
  def test_every_n_steps_policy(
      self,
      interval_steps,
      steps,
      expected_preserved_steps,
      exact_interval=True,
      max_to_keep=None,
  ):
    policy = preservation_policy_lib.EveryNSteps(
        interval_steps=interval_steps,
        exact_interval=exact_interval,
        max_to_keep=max_to_keep,
    )

    self.assertEqual(
        expected_preserved_steps,
        self.get_preserved_checkpoints(self.get_checkpoints(steps), policy),
    )

  @parameterized.parameters(
      dict(
          interval_steps=1,
          steps=[0, 1, 2, 3, 4],
          expected_preserved_steps=[0, 1, 2, 3, 4],
      ),
      dict(
          interval_steps=3,
          # 2 and 4 are both 1 step away from the same nominal target step (3).
          steps=[0, 1, 2, 4],
          # 4 is kept because it's the more recent and last checkpoint.
          expected_preserved_steps=[0, 4],
      ),
      dict(
          interval_steps=3,
          # 2 and 4 are both 1 step away from the same nominal target step (3).
          steps=[0, 1, 2, 4, 5, 8, 9, 13, 14, 25],
          # 4 is kept because it's the more recent one.
          expected_preserved_steps=[0, 4, 5, 9, 13, 14, 25],
      ),
      dict(
          interval_steps=1,
          steps=[0, 1, 2, 3, 4],
          expected_preserved_steps=[2, 3, 4],
          max_to_keep=3,
      ),
      dict(
          interval_steps=3,
          steps=[0, 1, 2, 4, 5, 8, 9, 13, 14, 25],
          expected_preserved_steps=[13, 14, 25],
          max_to_keep=3,
      ),
  )
  def test_every_n_steps_closest_policy(
      self,
      interval_steps,
      steps,
      expected_preserved_steps,
      max_to_keep=None,
  ):
    policy = preservation_policy_lib.EveryNStepsClosest(
        interval_steps=interval_steps,
        max_to_keep=max_to_keep,
    )

    self.assertEqual(
        expected_preserved_steps,
        self.get_preserved_checkpoints(self.get_checkpoints(steps), policy),
    )

  def test_every_zero_steps_closest_policy_raises_error(self):
    policy = preservation_policy_lib.EveryNStepsClosest(interval_steps=0)
    with self.assertRaises(ValueError):
      self.get_preserved_checkpoints(self.get_checkpoints(), policy)

  def test_every_zero_steps_policy_raises_error(self):
    policy = preservation_policy_lib.EveryNSteps(interval_steps=0)
    with self.assertRaises(ValueError):
      self.get_preserved_checkpoints(self.get_checkpoints(), policy)

  @parameterized.parameters(
      dict(
          steps=[0, 2, 3],
          expected_preserved_steps=[0, 2, 3],
      ),
      dict(
          steps=[0, 1, 2, 3, 4],
          expected_preserved_steps=[0, 1, 2, 3, 4],
      ),
      dict(
          steps=[14, 13],
          expected_preserved_steps=[],
      ),
  )
  def test_custom_steps_policy(self, steps, expected_preserved_steps):
    policy = preservation_policy_lib.CustomSteps(steps=steps)

    self.assertEqual(
        expected_preserved_steps,
        self.get_preserved_checkpoints(self.get_checkpoints(), policy),
    )

  @parameterized.parameters(
      dict(
          n=None,
          loss=[5, None, 3, None, 7],
          expected_preserved_steps=[0, 1, 2, 3, 4],
          keep_checkpoints_without_metrics=True,
      ),
      dict(
          n=1,
          loss=[7, None, 4, None, 5],
          expected_preserved_steps=[1, 2, 3],
          keep_checkpoints_without_metrics=True,
      ),
      dict(
          n=4,
          loss=[5, None, 4, None, 7],
          expected_preserved_steps=[0, 1, 2, 3, 4],
          keep_checkpoints_without_metrics=True,
      ),
      dict(
          n=4,
          loss=[5, None, 4, None, 7],
          expected_preserved_steps=[0, 2, 4],
          keep_checkpoints_without_metrics=False,
      ),
  )
  def test_best_n_policy(
      self, n, loss, expected_preserved_steps, keep_checkpoints_without_metrics
  ):
    policy = preservation_policy_lib.BestN(
        get_metric_fn=lambda metrics: metrics['loss'],
        reverse=True,
        n=n,
        keep_checkpoints_without_metrics=keep_checkpoints_without_metrics,
    )
    checkpoints = self.get_checkpoints()
    for i, checkpoint in enumerate(checkpoints):
      if loss[i]:
        checkpoint.metrics = {'loss': loss[i]}

    self.assertEqual(
        expected_preserved_steps,
        self.get_preserved_checkpoints(
            checkpoints, policy
        ),
    )

  def test_joint_preservation_policy(self):
    policy = preservation_policy_lib.AnyPreservationPolicy(
        policies=[
            preservation_policy_lib.LatestN(n=3),  # 9, 10, 11
            preservation_policy_lib.EveryNSeconds(
                interval_secs=3
            ),  # 0, 3, 6, 9
            preservation_policy_lib.CustomSteps(steps=[0, 3]),  # 0, 3
            preservation_policy_lib.BestN(
                get_metric_fn=lambda metrics: metrics['loss'],
                reverse=True,
                n=2,
            ),  # 1, 2, 3, 4, 5, 7, 9, 11
            preservation_policy_lib.EveryNSteps(interval_steps=6),  # 0, 6
        ]
    )
    loss = [5, None, 4, None, 3, None, 11, None, 8, None, 12, None]
    steps = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11]
    checkpoints = self.get_checkpoints(steps)
    for i, checkpoint in enumerate(checkpoints):
      if loss[i]:
        checkpoint.metrics = {'loss': loss[i]}

    self.assertEqual(
        [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11],
        self.get_preserved_checkpoints(
            checkpoints, policy
        ),
    )

  @parameterized.parameters(
      dict(
          duration=datetime.timedelta(hours=24),
          expected_preserved_steps=[2, 3],
      ),
      dict(
          duration=datetime.timedelta(hours=28),
          expected_preserved_steps=[1, 2, 3],
      ),
      dict(
          duration=datetime.timedelta(hours=5),
          expected_preserved_steps=[],
      ),
      dict(
          duration=datetime.timedelta(hours=100),
          expected_preserved_steps=[0, 1, 2, 3],
      ),
  )
  def test_latest_duration_policy(self, duration, expected_preserved_steps):
    fixed_now = datetime.datetime(2026, 5, 29, 12, 0, 0)
    checkpoints = [
        checkpoint_info.CheckpointInfo(
            step=0,
            time=fixed_now - datetime.timedelta(hours=30),
            metrics=None,
        ),
        checkpoint_info.CheckpointInfo(
            step=1,
            time=fixed_now - datetime.timedelta(hours=25),
            metrics=None,
        ),
        checkpoint_info.CheckpointInfo(
            step=2,
            time=fixed_now - datetime.timedelta(hours=23),
            metrics=None,
        ),
        checkpoint_info.CheckpointInfo(
            step=3,
            time=fixed_now - datetime.timedelta(hours=10),
            metrics=None,
        ),
    ]
    policy = preservation_policy_lib.LatestDuration(duration=duration)
    with mock.patch.object(
        preservation_policy_lib.datetime, 'datetime'
    ) as mock_datetime:
      mock_datetime.now.return_value = fixed_now
      mock_datetime.timedelta = datetime.timedelta
      self.assertEqual(
          expected_preserved_steps,
          self.get_preserved_checkpoints(checkpoints, policy),
      )


if __name__ == '__main__':
  absltest.main()
