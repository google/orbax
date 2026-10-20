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

"""Tests for policy-controlled checkpoint overwrites."""

import pathlib
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import numpy as np
import orbax.checkpoint.experimental.v1 as ocp


class OverwriteTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(self.create_tempdir().full_path) / 'run'
    self.state: Any = {'weights': np.arange(4)}
    self.replacement: Any = {'weights': np.arange(4) + 10}

  @parameterized.product(
      asynchronous=(False, True), checkpointables=(False, True)
  )
  def test_skipped_overwrite_preserves_checkpoint(
      self, asynchronous, checkpointables
  ):
    with ocp.training.Checkpointer(
        self.directory,
        save_decision_policy=(
            ocp.training.save_decision_policies.FixedIntervalPolicy(10)
        ),
    ) as ckptr:
      self.assertTrue(ckptr.save_checkpointables(
          5,
          {'state': self.state, 'optimizer': {'momentum': np.ones(4)}},
          force=True,
          metrics={'loss': 0.5},
          custom_metadata={'label': 'original'},
      ))
      files = {
          path: path.read_bytes()
          for path in pathlib.Path(self.directory / '5').rglob('*')
          if path.is_file()
      }
      method = 'save_checkpointables' if checkpointables else 'save'
      if asynchronous:
        method += '_async'
      value = (
          {'state': self.replacement} if checkpointables else self.replacement
      )
      response = getattr(ckptr, method)(5, value, overwrite=True)
      if asynchronous:
        self.assertIsNone(response)
      else:
        self.assertIs(response, False)
      self.assertEqual([c.step for c in ckptr.checkpoints], [5])
      self.assertTrue(files)
      for path, contents in files.items():
        self.assertEqual(path.read_bytes(), contents)
      restored = ckptr.load_checkpointables(5)
      np.testing.assert_array_equal(
          restored['state']['weights'], self.state['weights']
      )
      np.testing.assert_array_equal(
          restored['optimizer']['momentum'], np.ones(4)
      )
      metadata = ckptr.checkpointables_metadata(5)
      self.assertEqual(metadata.metrics, {'loss': 0.5})
      self.assertEqual(metadata.custom_metadata, {'label': 'original'})
      ckptr.reload()
      self.assertEqual([c.step for c in ckptr.checkpoints], [5])

  @parameterized.parameters(None, 10)
  def test_eligible_overwrite(self, interval):
    policy = (
        None if interval is None
        else ocp.training.save_decision_policies.FixedIntervalPolicy(interval)
    )
    with ocp.training.Checkpointer(
        self.directory, save_decision_policy=policy
    ) as ckptr:
      self.assertTrue(ckptr.save(10, self.state))
      self.assertTrue(ckptr.save(10, self.replacement, overwrite=True))
      np.testing.assert_array_equal(
          ckptr.load(10)['weights'], self.replacement['weights']
      )

  @parameterized.parameters(False, True)
  def test_overwrite_older_checkpoint(self, force):
    with ocp.training.Checkpointer(self.directory) as ckptr:
      self.assertTrue(ckptr.save(5, self.state))
      self.assertTrue(ckptr.save(10, self.state))
      self.assertEqual(
          ckptr.save(5, self.replacement, overwrite=True, force=force), force
      )
      self.assertSameElements([c.step for c in ckptr.checkpoints], [5, 10])
      expected = self.replacement if force else self.state
      np.testing.assert_array_equal(
          ckptr.load(5)['weights'], expected['weights']
      )
      np.testing.assert_array_equal(
          ckptr.load(10)['weights'], self.state['weights']
      )

  @parameterized.parameters(False, True)
  def test_custom_policy_runs_once_before_deletion(self, accepts):
    policy = mock.Mock(
        spec=ocp.training.save_decision_policies.SaveDecisionPolicy
    )
    with ocp.training.Checkpointer(
        self.directory, save_decision_policy=policy
    ) as ckptr:
      self.assertTrue(ckptr.save(0, self.state, force=True))
      self.assertTrue(ckptr.save(5, self.state, force=True))

      def decide(step, previous_steps, *, context):
        self.assertEqual(step.step, 5)
        self.assertEqual([c.step for c in previous_steps], [0])
        self.assertFalse(context.is_saving_in_progress)
        self.assertEqual([c.step for c in ckptr.checkpoints], [0, 5])
        self.assertTrue((self.directory / '5').exists())
        return accepts

      policy.should_save.side_effect = decide
      self.assertEqual(ckptr.save(5, self.replacement, overwrite=True), accepts)
      policy.should_save.assert_called_once()
      expected = self.replacement if accepts else self.state
      np.testing.assert_array_equal(
          ckptr.load(5)['weights'], expected['weights']
      )

  def test_force_skips_policy(self):
    policy = mock.Mock(
        spec=ocp.training.save_decision_policies.SaveDecisionPolicy
    )
    with ocp.training.Checkpointer(
        self.directory, save_decision_policy=policy
    ) as ckptr:
      self.assertTrue(ckptr.save(5, self.state, force=True))
      self.assertTrue(
          ckptr.save(5, self.replacement, overwrite=True, force=True)
      )
      policy.should_save.assert_not_called()
      np.testing.assert_array_equal(
          ckptr.load(5)['weights'], self.replacement['weights']
      )

  def test_overwrite_false_preserves_checkpoint(self):
    with ocp.training.Checkpointer(self.directory) as ckptr:
      self.assertTrue(ckptr.save(5, self.state))
      with self.assertRaises(ocp.training.errors.StepAlreadyExistsError):
        ckptr.save(5, self.replacement)
      np.testing.assert_array_equal(
          ckptr.load(5)['weights'], self.state['weights']
      )

  def test_policy_error_preserves_checkpoint(self):
    policy = mock.Mock(
        spec=ocp.training.save_decision_policies.SaveDecisionPolicy
    )
    with ocp.training.Checkpointer(
        self.directory, save_decision_policy=policy
    ) as ckptr:
      self.assertTrue(ckptr.save(5, self.state, force=True))
      policy.should_save.side_effect = RuntimeError('policy error')
      with self.assertRaisesRegex(RuntimeError, 'policy error'):
        ckptr.save(5, self.replacement, overwrite=True)
      np.testing.assert_array_equal(
          ckptr.load(5)['weights'], self.state['weights']
      )

  @parameterized.parameters(5, 10)
  def test_overwrite_missing_checkpoint(self, step):
    with ocp.training.Checkpointer(
        self.directory,
        save_decision_policy=(
            ocp.training.save_decision_policies.FixedIntervalPolicy(10)
        ),
    ) as ckptr:
      self.assertEqual(
          ckptr.save(step, self.state, overwrite=True), step == 10
      )
      self.assertEqual((self.directory / str(step)).exists(), step == 10)

  def test_in_progress_policy_preserves_checkpoint(self):
    with ocp.training.Checkpointer(
        self.directory,
        save_decision_policy=(
            ocp.training.save_decision_policies.ContinuousCheckpointingPolicy()
        ),
    ) as ckptr:
      self.assertTrue(ckptr.save(5, self.state))
      with mock.patch.object(
          ckptr._manager, 'is_saving_in_progress', return_value=True
      ):
        self.assertIsNone(ckptr.save_async(5, self.replacement, overwrite=True))
      np.testing.assert_array_equal(
          ckptr.load(5)['weights'], self.state['weights']
      )


if __name__ == '__main__':
  absltest.main()
