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

import datetime
import time

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from orbax.checkpoint import args
from orbax.checkpoint import checkpoint_manager
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.checkpoint_managers import preservation_policy as preservation_policy_lib


CheckpointManager = checkpoint_manager.CheckpointManager
CheckpointManagerOptions = checkpoint_manager.CheckpointManagerOptions


@test_utils.barrier_compatible_test
class PreservationPolicyTest(parameterized.TestCase):
  """Structure allows test to run as subclasses, not base class."""

  def setUp(self):
    super().setUp()
    pytree = test_utils.setup_pytree()
    self.pytree = pytree
    tz = datetime.timezone.utc
    self.current_datetime = datetime.datetime.now(tz=tz)
    self.directory = epath.Path(
        self.create_tempdir(name='preservation_policy_test').full_path
    )

  def wait_if_async(self, manager):
    manager.wait_until_finished()  # no-op if no async checkpointers.

  def _save_step(self, manager, step, params, metrics=None, force=False):
    return manager.save(
        step,
        args=args.Composite(params=args.PyTreeSave(params)),
        metrics=metrics,
        force=force,
    )

  def test_latest_n_policy(self):
    """Tests keeping only the latest N checkpoints."""
    n_to_keep = 3
    options = CheckpointManagerOptions(
        preservation_policy=preservation_policy_lib.LatestN(n=n_to_keep)
    )
    with CheckpointManager(self.directory, options=options) as manager:
      num_steps = 5
      for step in range(num_steps):
        self._save_step(manager, step, self.pytree)
      self.wait_if_async(manager)
      # Expected steps: 2, 3, 4 (latest 3 of 0, 1, 2, 3, 4)
      self.assertCountEqual(
          range(num_steps - n_to_keep, num_steps), manager.all_steps()
      )

  def test_every_n_steps_policy(self):
    """Tests keeping checkpoints every N steps."""
    interval_steps = 3
    options = CheckpointManagerOptions(
        preservation_policy=preservation_policy_lib.EveryNSteps(
            interval_steps=interval_steps
        )
    )
    with CheckpointManager(self.directory, options=options) as manager:
      num_steps = 10
      for step in range(num_steps):  # steps 0..9
        self._save_step(manager, step, self.pytree)
      self.wait_if_async(manager)
      # Expected steps: 0, 3, 6, 9 (multiples of 3)
      self.assertCountEqual(
          [i for i in range(num_steps) if i % interval_steps == 0],
          manager.all_steps(),
      )

  def test_custom_steps_policy(self):
    """Tests keeping custom checkpoints."""
    custom_steps = [0, 3, 6, 9]
    options = CheckpointManagerOptions(
        preservation_policy=preservation_policy_lib.CustomSteps(
            steps=custom_steps
        )
    )
    with CheckpointManager(self.directory, options=options) as manager:
      num_steps = 10
      for step in range(num_steps):
        self._save_step(manager, step, self.pytree)
      self.wait_if_async(manager)
      self.assertCountEqual(
          custom_steps,
          manager.all_steps(),
      )

  def test_every_n_seconds_policy(self):
    """Tests keeping checkpoints roughly every N seconds."""
    interval_secs = 4
    options = CheckpointManagerOptions(
        preservation_policy=preservation_policy_lib.EveryNSeconds(
            interval_secs=interval_secs
        )
    )
    with CheckpointManager(self.directory, options=options) as manager:
      num_steps = 7
      for step in range(num_steps):
        self._save_step(manager, step, self.pytree)
        time.sleep(2)
      self.wait_if_async(manager)
      self.assertCountEqual([0, 2, 4, 6], manager.all_steps())

  def test_best_n_policy(self):
    """Tests keeping the best N checkpoints based on a metric."""
    n_to_keep = 2
    all_metrics = {'loss': [5, 2, 4, 3, 7]}

    options = CheckpointManagerOptions(
        preservation_policy=preservation_policy_lib.BestN(
            best_fn=lambda metrics: metrics['loss'],
            reverse=True,  # Lower loss is better (min mode)
            n=n_to_keep,
            keep_checkpoints_without_metrics=False,
        )
    )
    with CheckpointManager(self.directory, options=options) as manager:
      num_steps = 5
      for step in range(num_steps):
        metrics = {k: v[step] for k, v in all_metrics.items()}
        self._save_step(manager, step, metrics=metrics, params=self.pytree)
      self.wait_if_async(manager)
      # Losses: 5, 2, 4, 3, 7 for steps 0, 1, 2, 3, 4
      # Best 2 losses are 2 (step 1) and 3 (step 3)
      self.assertCountEqual([1, 3], manager.all_steps())

  def test_joint_policy(self):
    """Tests combining multiple policies."""
    n_to_keep = 2
    interval_steps = 4
    interval_secs = 12
    custom_steps = [0, 3]
    all_metrics = {'loss': [5, 2, 4, 3, 7, 10, 11, 9, 8, 6, 12, 1]}
    policies = [
        preservation_policy_lib.BestN(
            best_fn=lambda metrics: metrics['loss'],
            reverse=True,
            n=n_to_keep,
            keep_checkpoints_without_metrics=False,
        ),  # 1, 11
        preservation_policy_lib.EveryNSteps(
            interval_steps=interval_steps
        ),  # 0, 4, 8
        preservation_policy_lib.EveryNSeconds(
            interval_secs=interval_secs
        ),  # 0, 6
        preservation_policy_lib.CustomSteps(steps=custom_steps),
    ]
    policy = preservation_policy_lib.JointPreservationPolicy(policies)
    options = CheckpointManagerOptions(preservation_policy=policy)
    with CheckpointManager(self.directory, options=options) as manager:
      num_steps = 12
      for step in range(num_steps):
        metrics = {k: v[step] for k, v in all_metrics.items()}
        self._save_step(manager, step, metrics=metrics, params=self.pytree)
        time.sleep(2)
      self.wait_if_async(manager)
      print('abhisekar: all_steps: ', manager.all_steps())
      self.assertCountEqual([0, 1, 3, 4, 6, 8, 11], manager.all_steps())

if __name__ == '__main__':
  absltest.main()
