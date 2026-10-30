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

"""Tests for continuous checkpointing options and helper functions."""

import datetime
from unittest import mock

from absl.testing import absltest
import numpy as np
from orbax.checkpoint import args as args_lib
from orbax.checkpoint import checkpoint_manager
from orbax.checkpoint._src.checkpoint_managers import continuous_checkpointing
from orbax.checkpoint._src.checkpoint_managers import preservation_policy as preservation_policy_lib
from orbax.checkpoint._src.checkpoint_managers import save_decision_policy as save_decision_policy_lib
from orbax.checkpoint._src.metadata import checkpoint_info

_EPOCH = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)


def _preserved_steps(options, steps):
  checkpoints = [
      checkpoint_info.CheckpointInfo(step=s, time=_EPOCH, metrics=None)
      for s in steps
  ]
  preserve = options.preservation_policy.should_preserve(
      checkpoints, context=preservation_policy_lib.PreservationContext()
  )
  return [s for s, keep in zip(steps, preserve) if keep]


def _save_policies(options):
  policy = options.save_decision_policy
  assert isinstance(policy, save_decision_policy_lib.AnySavePolicy)
  return list(policy.policies)


class ContinuousCheckpointingOptionsTest(absltest.TestCase):

  def test_policy_order_and_values(self):
    options = continuous_checkpointing.continuous_checkpointing_options(
        milestone_interval_steps=500,
        minimum_interval_secs=150,
        milestone_steps=(4321,),
        max_to_keep=3,
    )

    self.assertTrue(options.enable_async_checkpointing)
    # Other options keep the `CheckpointManagerOptions` defaults.
    self.assertFalse(options.enable_background_delete)
    self.assertIsNone(options.max_to_keep)
    policies = _save_policies(options)
    self.assertEqual(
        [type(p) for p in policies],
        [
            save_decision_policy_lib.FixedIntervalPolicy,
            save_decision_policy_lib.SpecificStepsPolicy,
            save_decision_policy_lib.PreemptionCheckpointingPolicy,
            save_decision_policy_lib.InitialSavePolicy,
            save_decision_policy_lib.ContinuousCheckpointingPolicy,
        ],
    )
    fixed, specific, _, _, cc = policies
    assert isinstance(fixed, save_decision_policy_lib.FixedIntervalPolicy)
    assert isinstance(specific, save_decision_policy_lib.SpecificStepsPolicy)
    assert isinstance(
        cc, save_decision_policy_lib.ContinuousCheckpointingPolicy
    )
    self.assertEqual(fixed.interval, 500)
    self.assertEqual(specific.steps, (4321,))
    self.assertEqual(cc.minimum_interval_secs, 150)

  def test_no_milestone_steps_omits_specific_steps_policy(self):
    options = continuous_checkpointing.continuous_checkpointing_options(
        milestone_interval_steps=1000
    )
    policies = _save_policies(options)

    self.assertEqual(
        [type(p) for p in policies],
        [
            save_decision_policy_lib.FixedIntervalPolicy,
            save_decision_policy_lib.PreemptionCheckpointingPolicy,
            save_decision_policy_lib.InitialSavePolicy,
            save_decision_policy_lib.ContinuousCheckpointingPolicy,
        ],
    )
    cc = policies[-1]
    assert isinstance(
        cc, save_decision_policy_lib.ContinuousCheckpointingPolicy
    )
    self.assertIsNone(cc.minimum_interval_secs)

  def test_preserves_latest_and_milestone_checkpoints(self):
    options = continuous_checkpointing.continuous_checkpointing_options(
        milestone_interval_steps=500, max_to_keep=2
    )

    kept = _preserved_steps(
        options, [500, 510, 520, 1000, 1010, 1500, 1510, 1520]
    )

    # Latest 2 checkpoints, plus the latest 2 milestone checkpoints.
    self.assertEqual(kept, [1000, 1500, 1510, 1520])

  def test_max_to_keep_none_preserves_all(self):
    options = continuous_checkpointing.continuous_checkpointing_options(
        milestone_interval_steps=500
    )
    steps = [500, 510, 1000, 1010]

    self.assertEqual(_preserved_steps(options, steps), steps)

  def test_rejects_invalid_arguments(self):
    with self.assertRaises(ValueError):
      continuous_checkpointing.continuous_checkpointing_options(
          milestone_interval_steps=0
      )
    with self.assertRaises(ValueError):
      continuous_checkpointing.continuous_checkpointing_options(
          milestone_interval_steps=10, minimum_interval_secs=-1
      )
    with self.assertRaises(ValueError):
      continuous_checkpointing.continuous_checkpointing_options(
          milestone_interval_steps=10, max_to_keep=0
      )


class ContinuousCheckpointingManagerTest(absltest.TestCase):
  """Runs the options through a real `CheckpointManager`."""

  def _manager(self, **kwargs):
    manager = checkpoint_manager.CheckpointManager(
        self.create_tempdir().full_path,
        options=continuous_checkpointing.continuous_checkpointing_options(
            milestone_interval_steps=500, milestone_steps=(4321,), **kwargs
        ),
    )
    self.addCleanup(manager.close)
    return manager

  def _save(self, manager, step):
    self.assertTrue(
        manager.save(step, args=args_lib.StandardSave({'x': np.arange(4)}))
    )
    manager.wait_until_finished()

  def test_initial_save(self):
    manager = self._manager(minimum_interval_secs=3600)

    self.assertTrue(manager.should_save(7))

  def test_milestone_steps_save_within_minimum_interval(self):
    manager = self._manager(minimum_interval_secs=3600)
    self._save(manager, 500)

    self.assertFalse(manager.should_save(510))
    self.assertTrue(manager.should_save(1000))
    self.assertTrue(manager.should_save(4321))

  def test_saves_after_minimum_interval(self):
    manager = self._manager(minimum_interval_secs=0)
    self._save(manager, 500)

    self.assertTrue(manager.should_save(510))

  def test_saves_on_preemption(self):
    manager = self._manager(minimum_interval_secs=3600)
    self._save(manager, 500)

    with mock.patch.object(manager, 'reached_preemption', return_value=True):
      self.assertTrue(manager.should_save(510))

  def test_does_not_save_while_saving(self):
    manager = self._manager(minimum_interval_secs=0)
    self._save(manager, 500)

    with mock.patch.object(manager, 'is_saving_in_progress', return_value=True):
      self.assertFalse(manager.should_save(510))


if __name__ == '__main__':
  absltest.main()
