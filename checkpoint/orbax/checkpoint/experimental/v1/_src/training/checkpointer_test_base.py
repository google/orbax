# Copyright 2025 The Orbax Authors.
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

import datetime
import threading
from typing import Any, Sequence
from unittest import mock

from absl.testing import parameterized
from etils import epath
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.serialization import serialization
import orbax.checkpoint.experimental.v1 as ocp
from orbax.checkpoint.experimental.v1._src.path import step as path_step_lib
from orbax.checkpoint.experimental.v1._src.synchronization import multihost
from orbax.checkpoint.experimental.v1._src.testing import array_utils as array_test_utils
from orbax.checkpoint.experimental.v1._src.testing import handler_utils
from orbax.checkpoint.experimental.v1._src.testing import tree_utils as tree_test_utils
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types

Checkpointer = ocp.training.Checkpointer
save_decision_policies = ocp.training.save_decision_policies
preservation_policies = ocp.training.preservation_policies

RootMetadata = ocp.training.RootMetadata
CheckpointMetadata = ocp.training.CheckpointMetadata

Foo = handler_utils.Foo
Bar = handler_utils.Bar
Baz = handler_utils.Baz
AbstractFoo = handler_utils.AbstractFoo
AbstractBar = handler_utils.AbstractBar
AbstractBaz = handler_utils.AbstractBaz

ocp.handlers.register_handler(handler_utils.BazHandler)


class CheckpointerTestBase:

  class Test(parameterized.TestCase):
    """Test class."""

    def setUp(self):
      super().setUp()

      pytree, abstract_pytree = array_test_utils.create_sharded_pytree()
      numpy_pytree, abstract_numpy_pytree = (
          array_test_utils.create_numpy_pytree()
      )
      self.pytree = {
          'jax_array': pytree,
          'numpy_array': numpy_pytree,
      }
      self.abstract_pytree = {
          'jax_array': abstract_pytree,
          'numpy_array': abstract_numpy_pytree,
      }

      self.directory = epath.Path(
          self.create_tempdir(name='checkpointing_test').full_path
      )

      test_utils.set_tensorstore_driver_for_test()
      test_utils.sync_global_processes('CheckpointerTest:setup_complete')

    def tearDown(self):
      test_utils.sync_global_processes('CheckpointerTest:tests_complete')
      super().tearDown()

    def save_pytree(
        self,
        checkpointer: Checkpointer,
        step: int,
        pytree: tree_types.PyTreeOf[tree_types.LeafType],
        metrics: tree_types.JsonType | None = None,
        custom_metadata: tree_types.JsonType | None = None,
    ) -> bool:
      """Saves pytree with v0 CheckpointManager or v1 Checkpointer."""
      raise NotImplementedError()

    def save_checkpointables(
        self,
        checkpointer: Checkpointer,
        step: int,
        checkpointables: dict[str, Any],
        metrics: tree_types.JsonType | None = None,
        custom_metadata: tree_types.JsonType | None = None,
    ) -> bool:
      """Saves checkpointables with v0 CheckpointManager or v1 Checkpointer."""
      raise NotImplementedError()

    def test_properties(self):
      checkpointer = Checkpointer(self.directory)
      self.enter_context(checkpointer)
      self.assertEqual(checkpointer.directory, self.directory)
      self.assertEmpty(checkpointer.checkpoints)
      self.assertIsNone(checkpointer.latest)

    def test_save_restore_pytree(self):
      checkpointer = Checkpointer(self.directory)
      self.enter_context(checkpointer)
      self.save_pytree(checkpointer, 0, self.pytree)

      with self.subTest('with_abstract_pytree'):
        loaded = checkpointer.load_pytree(0, self.abstract_pytree)
        test_utils.assert_tree_equal(self, self.pytree, loaded)

      with self.subTest('without_abstract_pytree'):
        loaded = checkpointer.load_pytree(0)
        test_utils.assert_tree_equal(self, self.pytree, loaded)

    @parameterized.parameters((True,), (False,))
    def test_load_latest_pytree(self, latest_arg_is_none):
      checkpointer = Checkpointer(self.directory)
      self.enter_context(checkpointer)
      self.save_pytree(checkpointer, 0, self.pytree, metrics={'loss': 0.5})
      new_pytree = {
          'jax_array': self.pytree['jax_array'],
          'numpy_array': array_test_utils.create_numpy_pytree(add=1)[0],
      }
      self.save_pytree(checkpointer, 1, new_pytree, metrics={'loss': 0.4})

      latest = checkpointer.latest
      self.assertIsInstance(latest, ocp.training.CheckpointMetadata)
      self.assertEqual(latest.step, 1)
      self.assertDictEqual(latest.metrics, {'loss': 0.4})

      if latest_arg_is_none:
        loaded = checkpointer.load_pytree(abstract_pytree=self.abstract_pytree)
      else:
        loaded = checkpointer.load_pytree(latest, self.abstract_pytree)

      test_utils.assert_tree_equal(self, new_pytree, loaded)

    def test_overwrites(self):
      plus_one_pytree, _ = array_test_utils.create_numpy_pytree(add=1)
      checkpointer = Checkpointer(self.directory)
      self.enter_context(checkpointer)
      self.save_pytree(checkpointer, 0, self.pytree)
      checkpointer.save_pytree(0, plus_one_pytree, overwrite=True)
      test_utils.assert_tree_equal(
          self, plus_one_pytree, checkpointer.load_pytree(0)
      )

    def test_step_already_exists(self):
      plus_one_pytree, _ = array_test_utils.create_numpy_pytree(add=1)
      checkpointer = Checkpointer(self.directory)
      self.enter_context(checkpointer)
      self.save_pytree(checkpointer, 0, self.pytree)
      with self.assertRaises(ocp.training.errors.StepAlreadyExistsError):
        checkpointer.save_pytree(0, plus_one_pytree)

    def test_load_non_existent_step(self):
      checkpointer = Checkpointer(self.directory)
      self.enter_context(checkpointer)
      self.save_pytree(checkpointer, 0, self.pytree)
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
      self.enter_context(checkpointer)
      for step in range(num_steps):
        self.assertEqual(checkpointer.should_save(step), step in expected_steps)
        saved = self.save_pytree(checkpointer, step, self.pytree)
        self.assertEqual(saved, step in expected_steps)

      self.assertLen(checkpointer.checkpoints, len(expected_steps))
      self.assertSequenceEqual(
          [c.step for c in checkpointer.checkpoints], expected_steps
      )
      assert checkpointer.latest is not None
      self.assertEqual(checkpointer.latest.step, expected_steps[-1])

    def test_force_save_ignores_save_decision_policy(self):
      checkpointer = Checkpointer(
          self.directory,
          save_decision_policy=save_decision_policies.FixedIntervalPolicy(2),
      )
      self.enter_context(checkpointer)

      self.assertTrue(checkpointer.save_pytree(0, self.pytree))
      self.assertFalse(checkpointer.save_pytree(1, self.pytree))
      self.assertTrue(checkpointer.save_pytree(2, self.pytree))

      self.assertLen(checkpointer.checkpoints, 2)
      self.assertSequenceEqual(
          [c.step for c in checkpointer.checkpoints], [0, 2]
      )

      self.assertTrue(checkpointer.save_pytree(3, self.pytree, force=True))
      self.assertTrue(checkpointer.save_pytree(4, self.pytree, force=True))
      self.assertTrue(checkpointer.save_pytree(5, self.pytree, force=True))

      self.assertLen(checkpointer.checkpoints, 5)
      self.assertSequenceEqual(
          [c.step for c in checkpointer.checkpoints], [0, 2, 3, 4, 5]
      )

    def test_garbage_collection(self):
      self.skipTest('TODO(cpgaffney): Implement.')

    def test_reload(self):
      checkpointer = Checkpointer(self.directory)
      self.enter_context(checkpointer)
      self.save_pytree(checkpointer, 0, self.pytree)
      self.save_pytree(checkpointer, 1, self.pytree)
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
      self.enter_context(checkpointer)
      saved_0 = checkpointer.save_pytree_async(0, self.pytree)
      saved_1 = checkpointer.save_pytree(1, self.pytree)
      self.assertTrue(saved_0.result())
      self.assertFalse(saved_1)
      self.assertLen(checkpointer.checkpoints, 1)
      assert checkpointer.latest is not None
      self.assertEqual(checkpointer.latest.step, 0)

    def test_save_pytree_async(self):
      checkpointer = Checkpointer(self.directory)
      self.enter_context(checkpointer)

      start_serialize = threading.Event()
      original_serialize = serialization.async_serialize_from_host

      def mock_serialize(*args, **kwargs):
        start_serialize.wait()  # Wait for explicit signal before proceeding.
        return original_serialize(*args, **kwargs)

      # Serialization to disk does not start until receiving an explicit signal.
      self.enter_context(
          mock.patch.object(
              serialization, 'async_serialize_from_host', new=mock_serialize
          )
      )

      step = 0
      response = checkpointer.save_pytree_async(step, self.pytree)
      initial_d_files_mtimes = tree_test_utils.get_d_files_mtimes(
          self.directory / str(step)
      )
      self.assertFalse(
          tree_test_utils.is_pytree_checkpoint_complete(
              self.directory / str(step)
          )
      )
      start_serialize.set()

      response.result()
      final_d_files_mtimes = tree_test_utils.get_d_files_mtimes(
          self.directory / str(step)
      )
      self.assertNotEmpty(final_d_files_mtimes)
      self.assertNotEqual(initial_d_files_mtimes, final_d_files_mtimes)
      self.assertTrue(
          tree_test_utils.is_pytree_checkpoint_complete(
              self.directory / str(step)
          )
      )

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
      self.enter_context(checkpointer)
      self.save_pytree(checkpointer, 0, self.pytree)
      self.assertTrue((self.directory / 'foo_0').exists())
      self.assertFalse((self.directory / '0').exists())

    @parameterized.product(
        reinitialize_checkpointer=(True, False),
    )
    def test_root_metadata(self, reinitialize_checkpointer):
      checkpointer = Checkpointer(
          self.directory, custom_metadata={'foo': 'bar'}
      )
      self.enter_context(checkpointer)
      if reinitialize_checkpointer:
        # Does not overwrite custom metadata.
        Checkpointer(self.directory, custom_metadata={'baz': 2})
      root_metadata = checkpointer.root_metadata()
      self.assertIsInstance(root_metadata, RootMetadata)
      self.assertDictEqual(root_metadata.custom_metadata, {'foo': 'bar'})

    @parameterized.product(
        reinitialize_checkpointer=(True, False),
    )
    def test_pytree_metadata(self, reinitialize_checkpointer):
      checkpointer = Checkpointer(self.directory)
      self.save_pytree(
          checkpointer,
          0,
          self.pytree,
          metrics={'loss': 0.5},
          custom_metadata={'baz': 'qux'},
      )
      if reinitialize_checkpointer:
        checkpointer.close()
        checkpointer = Checkpointer(self.directory)
        self.enter_context(checkpointer)
      checkpoint_metadata = checkpointer.pytree_metadata(0)
      self.assertIsInstance(checkpoint_metadata, CheckpointMetadata)
      self.assertDictEqual(checkpoint_metadata.custom_metadata, {'baz': 'qux'})
      self.assertDictEqual(checkpoint_metadata.metrics, {'loss': 0.5})
      self.assertIsNotNone(checkpoint_metadata.init_timestamp_nsecs)
      self.assertIsNotNone(checkpoint_metadata.commit_timestamp_nsecs)
      self.assertIsInstance(checkpoint_metadata.metadata, dict)
      self.assertSameElements(
          checkpoint_metadata.metadata.keys(), ['jax_array', 'numpy_array']
      )

    @parameterized.product(
        reinitialize_checkpointer=(True, False),
    )
    def test_checkpointables_metadata(self, reinitialize_checkpointer):
      checkpointer = Checkpointer(self.directory)
      self.save_checkpointables(
          checkpointer,
          0,
          {'pytree': self.pytree, 'baz': Baz(123, 'hi')},
          metrics={'loss': 0.5},
          custom_metadata={'baz': 'qux'},
      )
      if reinitialize_checkpointer:
        checkpointer.close()
        checkpointer = Checkpointer(self.directory)
        self.enter_context(checkpointer)
      checkpoint_metadata = checkpointer.checkpointables_metadata(0)
      self.assertIsInstance(checkpoint_metadata, CheckpointMetadata)
      self.assertDictEqual(checkpoint_metadata.custom_metadata, {'baz': 'qux'})
      self.assertDictEqual(checkpoint_metadata.metrics, {'loss': 0.5})
      self.assertIsNotNone(checkpoint_metadata.init_timestamp_nsecs)
      self.assertIsNotNone(checkpoint_metadata.commit_timestamp_nsecs)
      self.assertIsInstance(checkpoint_metadata.metadata, dict)
      self.assertSameElements(
          checkpoint_metadata.metadata.keys(), ['pytree', 'baz']
      )
      self.assertSameElements(
          checkpoint_metadata.metadata['pytree'].keys(),
          ['jax_array', 'numpy_array'],
      )
      self.assertIsInstance(checkpoint_metadata.metadata['baz'], AbstractBaz)

    def test_custom_checkpointables(self):
      """Test custom checkpointables are saved and loaded.

      Subclasses must call it within custom `checkpointables_options` Context.
      """
      checkpointables = {
          'pytree': self.pytree,
          'foo': Foo(123, 'hi'),
          'bar': Bar(456, 'bye'),
      }
      checkpointer = Checkpointer(self.directory)
      self.enter_context(checkpointer)
      self.save_checkpointables(checkpointer, 0, checkpointables)

      with self.subTest('load'):
        loaded = checkpointer.load_checkpointables(0)
        self.assertSameElements(loaded.keys(), ['pytree', 'foo', 'bar'])
        test_utils.assert_tree_equal(
            self, checkpointables['pytree'], loaded['pytree']
        )
        self.assertEqual(checkpointables['foo'], loaded['foo'])
        self.assertEqual(checkpointables['bar'], loaded['bar'])
      with self.subTest('load_with_free_function'):
        loaded = ocp.load_checkpointables(self.directory / '0')
        self.assertSameElements(loaded.keys(), ['pytree', 'foo', 'bar'])
        test_utils.assert_tree_equal(
            self, checkpointables['pytree'], loaded['pytree']
        )
        self.assertEqual(checkpointables['foo'], loaded['foo'])
        self.assertEqual(checkpointables['bar'], loaded['bar'])
      with self.subTest('load_with_abstract_checkpointables'):
        abstract_checkpointables = {
            'pytree': self.abstract_pytree,
            'foo': AbstractFoo(),
            'bar': AbstractBar(),
        }
        loaded = checkpointer.load_checkpointables(0, abstract_checkpointables)
        self.assertSameElements(loaded.keys(), ['pytree', 'foo', 'bar'])
        test_utils.assert_tree_equal(self, self.pytree, loaded['pytree'])
        self.assertEqual(checkpointables['foo'], loaded['foo'])
        self.assertEqual(checkpointables['bar'], loaded['bar'])
      with self.subTest('load_with_abstract_checkpointables_none_values'):
        abstract_checkpointables = {
            'pytree': None,
            'foo': None,
            'bar': None,
        }
        loaded = checkpointer.load_checkpointables(0, abstract_checkpointables)
        self.assertSameElements(loaded.keys(), ['pytree', 'foo', 'bar'])
        test_utils.assert_tree_equal(
            self, checkpointables['pytree'], loaded['pytree']
        )
        self.assertEqual(checkpointables['foo'], loaded['foo'])
        self.assertEqual(checkpointables['bar'], loaded['bar'])
      with self.subTest('load_partial'):
        abstract_checkpointables = {
            'foo': None,
        }
        loaded = checkpointer.load_checkpointables(0, abstract_checkpointables)
        self.assertSameElements(loaded.keys(), ['foo'])
        self.assertEqual(checkpointables['foo'], loaded['foo'])

    def test_load_with_switched_abstract_checkpointables(self):
      """Test load with switched abstract checkpointables.

      Subclasses must call it within custom `checkpointables_options` Context.
      """
      checkpointables = {
          'pytree': self.pytree,
          'foo': Foo(123, 'hi'),
          'bar': Bar(456, 'bye'),
      }
      checkpointer = Checkpointer(self.directory)
      self.enter_context(checkpointer)
      self.save_checkpointables(checkpointer, 0, checkpointables)

      abstract_checkpointables = {
          'foo': AbstractBar(),
          'bar': AbstractFoo(),
      }
      loaded = checkpointer.load_checkpointables(0, abstract_checkpointables)
      self.assertSameElements(loaded.keys(), ['foo', 'bar'])
      self.assertEqual(Bar(123, 'hi'), loaded['foo'])
      self.assertEqual(Foo(456, 'bye'), loaded['bar'])

    def test_different_custom_checkpointables(self):
      """Test different custom checkpointables are saved and loaded.

      Subclasses must call it within custom `checkpointables_options` Context.
      """
      checkpointer = Checkpointer(self.directory)
      self.enter_context(checkpointer)
      self.save_checkpointables(checkpointer, 0, {'foo': Foo(123, 'hi')})
      checkpointer.save_checkpointables(1, {'bar': Bar(456, 'bye')})

      loaded = checkpointer.load_checkpointables(0)
      self.assertSameElements(loaded.keys(), ['foo'])
      self.assertEqual(Foo(123, 'hi'), loaded['foo'])

      loaded = checkpointer.load_checkpointables(1)
      self.assertSameElements(loaded.keys(), ['bar'])
      self.assertEqual(Bar(456, 'bye'), loaded['bar'])

    def test_custom_save_decision_policy(self):
      save_delta = datetime.timedelta(seconds=0.03)

      class ArbitrarySavePolicy(save_decision_policies.SaveDecisionPolicy):

        def should_save(
            self,
            step: CheckpointMetadata,
            previous_steps: Sequence[CheckpointMetadata],
            *,
            context: save_decision_policies.DecisionContext,
        ) -> bool:
          save_result = False
          is_primary_host = multihost.is_primary_host(
              context.multiprocessing_options.primary_host
          )
          if is_primary_host:
            if not previous_steps:
              save_result = True
            else:
              time_delta = step.time - previous_steps[-1].time
              save_result = step.step % 2 == 0 and time_delta >= save_delta
          save_result = bool(
              multihost.broadcast_one_to_all(
                  save_result, is_source=is_primary_host
              )
          )
          return save_result

      checkpointer = Checkpointer(
          self.directory, save_decision_policy=ArbitrarySavePolicy()
      )
      self.enter_context(checkpointer)
      for step in range(0, 30):
        self.save_pytree(checkpointer, step, self.pytree)

      self.assertNotEmpty(checkpointer.checkpoints)
      self.assertLess(len(checkpointer.checkpoints), 30)
      checkpointer_metadata = [
          checkpointer.pytree_metadata(metadata.step)
          for metadata in checkpointer.checkpoints
      ]
      for i in range(1, len(checkpointer.checkpoints)):
        time_delta = (
            checkpointer_metadata[i].time - checkpointer_metadata[i - 1].time
        )
        self.assertGreaterEqual(time_delta, save_delta)

    @parameterized.parameters(
        (None, range(10)),
        (ocp.training.preservation_policies.PreserveAll(), range(10)),
        (preservation_policies.LatestN(3), range(7, 10)),
        (preservation_policies.EveryNSteps(4), [0, 4, 8]),
        (preservation_policies.EveryNSeconds(40), range(0, 10, 2)),
        (
            preservation_policies.AnyPreservationPolicy([
                preservation_policies.LatestN(3),
                preservation_policies.CustomSteps([1, 3, 9]),
            ]),
            [1, 3, 7, 8, 9],
        ),
    )
    def test_preservation(self, policy, expected_steps):
      num_steps = 10
      checkpointer = Checkpointer(self.directory, preservation_policy=policy)
      for step in range(num_steps):

        class CustomDatetime(datetime.datetime):

          @classmethod
          def now(cls, tz=None):
            return datetime.datetime.fromtimestamp(
                step * 20, tz=tz  # pylint: disable=cell-var-from-loop
            )

        with mock.patch('datetime.datetime', new=CustomDatetime):
          # mock_dt.now.return_value = datetime.datetime.fromtimestamp(
          #     checkpoint_times[step]
          # )
          # mock_dt.fromtimestamp.side_effect = datetime.datetime.fromtimestamp
          # mock_dt.timestamp.return_value = checkpoint_times[step]
          checkpointer.save_pytree(step, self.pytree)

      self.assertLen(checkpointer.checkpoints, len(expected_steps))
      self.assertSequenceEqual(
          [c.step for c in checkpointer.checkpoints], expected_steps
      )
      self.assertSequenceEqual(
          [c.time for c in checkpointer.checkpoints],
          [
              datetime.datetime.fromtimestamp(
                  step * 20, tz=datetime.timezone.utc
              )
              for step in expected_steps
          ],
      )
      checkpointer.close()

    @parameterized.parameters(
        (
            preservation_policies.BestN(
                get_metric_fn=lambda m: m['accuracy'], n=2
            ),
            [0, 4],
        ),
        (
            preservation_policies.BestN(
                get_metric_fn=lambda m: m['loss'], reverse=True, n=2
            ),
            [0, 1],
        ),
        (
            preservation_policies.BestN(
                get_metric_fn=lambda m: m['accuracy'], n=None
            ),
            range(5),
        ),
        (
            preservation_policies.BestN(
                get_metric_fn=lambda m: m['accuracy'], n=0
            ),
            [],
        ),
    )
    def test_preservation_metrics(self, policy, expected_steps):
      num_steps = 5
      all_metrics = [
          {'loss': 0.3, 'accuracy': 0.8},
          {'loss': 0.2, 'accuracy': 0.6},
          {'loss': 0.8, 'accuracy': 0.2},
          {'loss': 0.9, 'accuracy': 0.3},
          {'loss': 0.4, 'accuracy': 0.9},
      ]
      checkpointer = Checkpointer(self.directory, preservation_policy=policy)
      for step in range(num_steps):
        checkpointer.save_pytree(step, self.pytree, metrics=all_metrics[step])

      self.assertLen(checkpointer.checkpoints, len(expected_steps))
      self.assertSequenceEqual(
          [c.step for c in checkpointer.checkpoints], expected_steps
      )
      self.assertSequenceEqual(
          [c.metrics for c in checkpointer.checkpoints],
          [all_metrics[step] for step in expected_steps],
      )
      checkpointer.close()
