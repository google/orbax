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

"""Base test for Checkpointer."""

# pylint: disable=missing-class-docstring,protected-access,missing-function-docstring

from typing import Sequence
from absl.testing import parameterized
from etils import epath
from orbax.checkpoint import test_utils
import orbax.checkpoint.experimental.v1 as ocp
from orbax.checkpoint.experimental.v1._src.path import step as path_step_lib
from orbax.checkpoint.experimental.v1._src.testing import array_utils as array_test_utils


Checkpointer = ocp.training.Checkpointer
save_decision_policies = ocp.training.save_decision_policies

RootMetadata = ocp.training.RootMetadata
CheckpointMetadata = ocp.training.CheckpointMetadata


class CheckpointerTestBase:

  class Test(parameterized.TestCase):
    """Test class."""

    def setUp(self):
      super().setUp()

      self.pytree, self.abstract_pytree = array_test_utils.create_numpy_pytree()

      self.directory = epath.Path(
          self.create_tempdir(name='checkpointing_test').full_path
      )

      test_utils.set_tensorstore_driver_for_test()
      test_utils.sync_global_processes('CheckpointerTest:setup_complete')

    def tearDown(self):
      test_utils.sync_global_processes('CheckpointerTest:tests_complete')
      super().tearDown()

    def test_properties(self):
      checkpointer = Checkpointer(self.directory)
      self.assertEqual(checkpointer.directory, self.directory)
      self.assertEmpty(checkpointer.checkpoints)
      self.assertIsNone(checkpointer.latest)

    def test_save_restore_pytree(self):
      checkpointer = Checkpointer(self.directory)
      checkpointer.save_pytree(0, self.pytree)

      with self.subTest('with_abstract_pytree'):
        loaded = checkpointer.load_pytree(0, self.abstract_pytree)
        test_utils.assert_tree_equal(self, self.pytree, loaded)

      with self.subTest('without_abstract_pytree'):
        loaded = checkpointer.load_pytree(0)
        test_utils.assert_tree_equal(self, self.pytree, loaded)

    @parameterized.parameters((True,), (False,))
    def test_load_latest_pytree(self, latest_arg_is_none):
      checkpointer = Checkpointer(self.directory)
      checkpointer.save_pytree(0, self.pytree, metrics={'loss': 0.5})
      new_pytree, _ = array_test_utils.create_numpy_pytree(add=1)
      checkpointer.save_pytree(1, new_pytree, metrics={'loss': 0.4})

      latest = checkpointer.latest
      self.assertIsInstance(latest, ocp.training.CheckpointMetadata)
      self.assertEqual(latest.step, 1)
      self.assertDictEqual(latest.metrics, {'loss': 0.4})

      if latest_arg_is_none:
        loaded = checkpointer.load_pytree(abstract_pytree=self.abstract_pytree)
      else:
        loaded = checkpointer.load_pytree(latest, self.abstract_pytree)

      test_utils.assert_tree_equal(self, new_pytree, loaded)

    def test_force_overwrites(self):
      plus_one_pytree, _ = array_test_utils.create_numpy_pytree(add=1)
      checkpointer = Checkpointer(self.directory)
      checkpointer.save_pytree(0, self.pytree)
      checkpointer.save_pytree(0, plus_one_pytree, force=True)
      test_utils.assert_tree_equal(
          self, plus_one_pytree, checkpointer.load_pytree(0)
      )

    def test_step_already_exists(self):
      plus_one_pytree, _ = array_test_utils.create_numpy_pytree(add=1)
      checkpointer = Checkpointer(self.directory)
      checkpointer.save_pytree(0, self.pytree)
      with self.assertRaises(ocp.training.errors.StepAlreadyExistsError):
        checkpointer.save_pytree(0, plus_one_pytree)

    def test_load_non_existent_step(self):
      checkpointer = Checkpointer(self.directory)
      checkpointer.save_pytree(0, self.pytree)
      with self.subTest('load_pytree'):
        with self.assertRaises(FileNotFoundError):
          checkpointer.load_pytree(1)
      with self.subTest('pytree_metadata'):
        with self.assertRaises(FileNotFoundError):
          checkpointer.pytree_metadata(1)

    @parameterized.parameters(
        (save_decision_policies.ContinuousCheckpointingPolicy(), range(10)),
        (save_decision_policies.FixedIntervalPolicy(1), range(10)),
        (save_decision_policies.FixedIntervalPolicy(2), range(0, 10, 2)),
        (
            save_decision_policies.AnySavePolicy([
                save_decision_policies.SpecificStepsPolicy((2, 3, 6)),
                save_decision_policies.InitialSavePolicy(),
            ]),
            [0, 2, 3, 6],
        ),
    )
    def test_steps(
        self,
        policy: save_decision_policies.SaveDecisionPolicy,
        expected_steps: Sequence[int],
    ):
      num_steps = 10
      checkpointer = Checkpointer(self.directory, save_decision_policy=policy)
      for step in range(num_steps):
        self.assertEqual(checkpointer.should_save(step), step in expected_steps)
        saved = checkpointer.save_pytree(step, self.pytree)
        self.assertEqual(saved, step in expected_steps)

      self.assertLen(checkpointer.checkpoints, len(expected_steps))
      self.assertSequenceEqual(
          [c.step for c in checkpointer.checkpoints], expected_steps
      )
      assert checkpointer.latest is not None
      self.assertEqual(checkpointer.latest.step, expected_steps[-1])

    def test_garbage_collection(self):
      self.skipTest('TODO(cpgaffney): Implement.')

    def test_reload(self):
      checkpointer = Checkpointer(self.directory)
      checkpointer.save_pytree(0, self.pytree)
      checkpointer.save_pytree(1, self.pytree)
      self.assertLen(checkpointer.checkpoints, 2)
      assert checkpointer.latest is not None
      self.assertEqual(checkpointer.latest.step, 1)

      (self.directory / '1').rmtree(missing_ok=True)

      # Properties still reflect cached state before reload.
      self.assertLen(checkpointer.checkpoints, 2)
      assert checkpointer.latest is not None
      self.assertEqual(checkpointer.latest.step, 1)

      checkpointer.reload()

      self.assertLen(checkpointer.checkpoints, 1)
      assert checkpointer.latest is not None
      self.assertEqual(checkpointer.latest.step, 0)

    def test_skips_when_ongoing_save(self):
      checkpointer = Checkpointer(self.directory)
      saved_0 = checkpointer.save_pytree_async(0, self.pytree)
      saved_1 = checkpointer.save_pytree(1, self.pytree)
      self.assertTrue(saved_0.result())
      self.assertFalse(saved_1)
      self.assertLen(checkpointer.checkpoints, 1)
      assert checkpointer.latest is not None
      self.assertEqual(checkpointer.latest.step, 0)

    def test_save_async(self):
      checkpointer = Checkpointer(self.directory)
      step_path = self.directory / '0'
      response = checkpointer.save_pytree_async(0, self.pytree)
      self.assertFalse(step_path.exists())  # Not finalized yet.
      # But a tmp dir should have been created.
      self.assertNotEmpty(list(self.directory.iterdir()))
      response.result()
      self.assertTrue(step_path.exists())
      loaded = checkpointer.load_pytree(0)
      test_utils.assert_tree_equal(self, self.pytree, loaded)

    def test_close(self):
      checkpointer = Checkpointer(self.directory)
      step_path = self.directory / '0'
      checkpointer.save_pytree_async(0, self.pytree)
      self.assertFalse(step_path.exists())  # Not finalized yet.
      # But a tmp dir should have been created.
      self.assertNotEmpty(list(self.directory.iterdir()))
      checkpointer.close()
      self.assertTrue(step_path.exists())

    def test_context_manager_close(self):
      step_path = self.directory / '0'
      with Checkpointer(self.directory) as checkpointer:
        checkpointer.save_pytree_async(0, self.pytree)
        self.assertFalse(step_path.exists())  # Not finalized yet.
        # But a tmp dir should have been created.
        self.assertNotEmpty(list(self.directory.iterdir()))
      self.assertTrue(step_path.exists())

    def test_step_name_format(self):
      checkpointer = Checkpointer(
          self.directory,
          step_name_format=path_step_lib.standard_name_format(
              step_prefix='foo'
          ),
      )
      checkpointer.save_pytree(0, self.pytree)
      self.assertTrue((self.directory / 'foo_0').exists())
      self.assertFalse((self.directory / '0').exists())

    def test_metadata(self):
      checkpointer = Checkpointer(
          self.directory, custom_metadata={'foo': 'bar'}
      )
      checkpointer.save_pytree(
          0, self.pytree, metrics={'loss': 0.5}, custom_metadata={'baz': 'qux'}
      )

      with self.subTest('root_metadata'):
        root_metadata = checkpointer.root_metadata()
        self.assertIsInstance(root_metadata, RootMetadata)
        self.assertDictEqual(root_metadata.custom_metadata, {'foo': 'bar'})

      with self.subTest('checkpoint_metadata'):
        checkpoint_metadata = checkpointer.pytree_metadata(0)
        self.assertIsInstance(checkpoint_metadata, CheckpointMetadata)
        self.assertDictEqual(
            checkpoint_metadata.custom_metadata, {'baz': 'qux'}
        )
        self.assertDictEqual(checkpoint_metadata.metrics, {'loss': 0.5})
        self.assertIsNotNone(checkpoint_metadata.init_timestamp_nsecs)
        self.assertIsNotNone(checkpoint_metadata.commit_timestamp_nsecs)
        self.assertIsInstance(
            checkpoint_metadata.metadata, dict
        )
