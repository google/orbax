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

"""Tests for preservation policy."""

import datetime
from typing import Sequence
from absl.testing import absltest
from absl.testing import parameterized
from orbax.checkpoint._src.checkpoint_managers import preservation_policy as preservation_policy_lib
from orbax.checkpoint._src.metadata import checkpoint_info


class PreservationPolicyTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.preservation_context = preservation_policy_lib.PreservationContext()

  def get_checkpoints(self, n: int):
    checkpoints = []
    time_increment = datetime.timedelta(seconds=1)
    start_time = datetime.datetime.now()
    for i in range(n):
      current_time = start_time + i * time_increment
      checkpoints.append(
          checkpoint_info.CheckpointInfo(
              step=i,
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
        self.get_preserved_checkpoints(self.get_checkpoints(5), policy),
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
        self.get_preserved_checkpoints(self.get_checkpoints(5), policy),
    )

  @parameterized.parameters(
      dict(
          interval_steps=1,
          expected_preserved_steps=[0, 1, 2, 3, 4],
      ),
      dict(
          interval_steps=3,
          expected_preserved_steps=[0, 3],
      ),
      dict(
          interval_steps=6,
          expected_preserved_steps=[0],
      ),
  )
  def test_every_n_steps_policy(self, interval_steps, expected_preserved_steps):
    policy = preservation_policy_lib.EveryNSteps(interval_steps=interval_steps)

    self.assertEqual(
        expected_preserved_steps,
        self.get_preserved_checkpoints(self.get_checkpoints(5), policy),
    )

  def test_every_zero_steps_policy_raises_error(self):
    policy = preservation_policy_lib.EveryNSteps(interval_steps=0)
    with self.assertRaises(ValueError):
      self.get_preserved_checkpoints(self.get_checkpoints(5), policy)

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
        self.get_preserved_checkpoints(self.get_checkpoints(5), policy),
    )

  @parameterized.parameters(
      dict(
          n=None,
          loss=[5, None, 3, None, 7],
          expected_preserved_steps=[0, 1, 2, 3, 4],
      ),
      dict(
          n=1,
          loss=[7, None, 4, None, 5],
          expected_preserved_steps=[1, 2, 3],
      ),
      dict(
          n=4,
          loss=[5, None, 4, None, 7],
          expected_preserved_steps=[0, 1, 2, 3, 4],
      ),
  )
  def test_best_n_policy(self, n, loss, expected_preserved_steps):
    policy = preservation_policy_lib.BestN(
        get_metric_fn=lambda metrics: metrics['loss'],
        reverse=True,
        n=n,
    )
    checkpoints = self.get_checkpoints(5)
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
    checkpoints = self.get_checkpoints(12)
    for i, checkpoint in enumerate(checkpoints):
      if loss[i]:
        checkpoint.metrics = {'loss': loss[i]}

    self.assertEqual(
        [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11],
        self.get_preserved_checkpoints(
            checkpoints, policy
        ),
    )


if __name__ == '__main__':
  absltest.main()
