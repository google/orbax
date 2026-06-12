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

"""Test utilities and test suite for emergency.LocalCheckpointManager."""

# pylint: disable=protected-access

from typing import Any

from absl import logging
from absl.testing import flagsaver
from absl.testing import parameterized
from etils import epath
import jax
from orbax.checkpoint import args as args_lib
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.handlers import pytree_checkpoint_handler
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.multihost import multislice
from orbax.checkpoint._src.testing import multiprocess_test
from orbax.checkpoint.experimental.emergency import checkpoint_manager
from orbax.checkpoint.experimental.emergency import mesh_consistency
from orbax.checkpoint.experimental.emergency import path as emergency_path_utils
from orbax.checkpoint.experimental.emergency import process_metadata_checkpoint_handler


PyTree = Any
PyTreeCheckpointHandler = pytree_checkpoint_handler.PyTreeCheckpointHandler
PyTreeSaveArgs = pytree_checkpoint_handler.PyTreeSaveArgs
PyTreeRestoreArgs = pytree_checkpoint_handler.PyTreeRestoreArgs

CheckpointManager = checkpoint_manager.CheckpointManager
LocalCheckpointManager = checkpoint_manager._LocalCheckpointManager  # pylint: disable=protected-access
CheckpointManagerOptions = checkpoint_manager.CheckpointManagerOptions
LocalCheckpointOptions = checkpoint_manager.LocalCheckpointOptions
PersistentCheckpointOptions = checkpoint_manager.PersistentCheckpointOptions
barrier_compatible_test = test_utils.barrier_compatible_test
assert_tree_equal = test_utils.assert_tree_equal
get_fake_global_mesh_for_slices = test_utils.get_fake_global_mesh_for_slices
_STATE_ITEM_NAME = 'state'
_DATASET_ITEM_NAME = 'dataset'


class LocalCheckpointManagerTestSuite:
  """LocalCheckpointManager test suite."""

  @barrier_compatible_test
  class LocalCheckpointManagerTest(parameterized.TestCase):
    """Test case for LocalCheckpointManager."""

    def make_global_mesh(self) -> jax.sharding.Mesh:
      """Creates a global mesh for testing purposes."""
      self.assertEqual(jax.device_count(), 8)
      self.assertEqual(jax.process_count(), 4)
      self.assertEqual(jax.local_device_count(), 2)

      # setup global mesh info for 2-slice tests
      slice_processes = [{0, 1}, {2, 3}]
      return get_fake_global_mesh_for_slices(slice_processes)

    def setUp(self):
      super().setUp()
      self.enter_context(
          flagsaver.flagsaver(
              experimental_orbax_use_distributed_process_id=True
          )
      )
      if not multihost.is_runtime_to_distributed_ids_initialized():
        multihost.initialize_runtime_to_distributed_ids()
      if not multihost.is_distributed_to_device_ids_initialized():
        multihost.initialize_distributed_to_device_ids()

      self.global_mesh = self.make_global_mesh()

      self._fn = lambda ty: issubclass(ty, jax.Array)

      pytree, mesh_tree, axes_tree = test_utils.setup_sharded_pytree()
      doubled_pytree = test_utils.apply_function(pytree, lambda x: x * 2)

      self.pytree = pytree
      self.doubled_pytree = doubled_pytree
      self.mesh_tree = mesh_tree
      self.axes_tree = axes_tree
      self.replica_id = multislice.process_replica_id(
          multihost.process_index(),
          self.global_mesh,
          replica_axis_index=0,
      )

      # make sure each process is working on different directories
      self.local_directory = epath.Path(
          self.create_tempdir(
              name=f'checkpointing_test_pid{multihost.process_index()}'
          ).full_path
      )
      logging.info(
          'self.directory=%s',
          self.local_directory,
      )
      test_utils.set_tensorstore_driver_for_test()

      test_utils.sync_global_processes(
          'LocalCheckpointManagerTest:setup_complete'
      )

    def tearDown(self):
      super().tearDown()
      test_utils.sync_global_processes(
          'LocalCheckpointManagerTest:teardown_complete'
      )

    def test_local_save_restore(self):
      """Test case."""
      options = CheckpointManagerOptions(
          local=LocalCheckpointOptions(
              save_interval_steps=1,
              max_to_keep=1,
              debug_use_full_global_mesh=True,
          ),
          persistent=PersistentCheckpointOptions(
              save_interval_steps=1, max_to_keep=1
          ),
      )
      manager = LocalCheckpointManager(
          directory=self.local_directory,
          global_mesh=self.global_mesh,
          replica_id=self.replica_id,
          options=options,
      )

      manager.save(
          0,
          args=args_lib.Composite(
              state=PyTreeSaveArgs(self.pytree),
              process_metadata=process_metadata_checkpoint_handler.ProcessMetadataSaveArgs(
                  global_mesh=self.global_mesh
              ),
          ),
      )

      manager.wait_until_finished()

      restored = manager.restore(
          0,
          args=args_lib.Composite(
              state=PyTreeRestoreArgs(),
              process_metadata=process_metadata_checkpoint_handler.ProcessMetadataRestoreArgs(),
          ),
      )
      test_utils.assert_tree_equal(self, self.pytree, restored.state)

    def test_local_auto_cleanup(self):
      """Test case."""
      options = CheckpointManagerOptions(
          local=LocalCheckpointOptions(
              save_interval_steps=1,
              max_to_keep=1,
              debug_use_full_global_mesh=True,
          ),
          persistent=PersistentCheckpointOptions(
              save_interval_steps=1, max_to_keep=1
          ),
      )
      manager = LocalCheckpointManager(
          directory=self.local_directory,
          global_mesh=self.global_mesh,
          replica_id=self.replica_id,
          options=options,
      )

      manager.save(
          0,
          args=args_lib.Composite(
              state=PyTreeSaveArgs(self.pytree),
              process_metadata=process_metadata_checkpoint_handler.ProcessMetadataSaveArgs(
                  global_mesh=self.global_mesh
              ),
          ),
      )
      manager.wait_until_finished()

      manager.save(
          1,
          args=args_lib.Composite(
              state=PyTreeSaveArgs(self.doubled_pytree),
              process_metadata=process_metadata_checkpoint_handler.ProcessMetadataSaveArgs(
                  global_mesh=self.global_mesh
              ),
          ),
      )
      manager.wait_until_finished()

      # step 0 should have been removed
      with self.assertRaisesRegex(ValueError, 'No step path found'):
        manager.restore(
            0,
            args=args_lib.Composite(
                state=PyTreeRestoreArgs(),
                process_metadata=process_metadata_checkpoint_handler.ProcessMetadataRestoreArgs(),
            ),
        )

      restored = manager.restore(
          1,
          args=args_lib.Composite(
              state=PyTreeRestoreArgs(),
              process_metadata=process_metadata_checkpoint_handler.ProcessMetadataRestoreArgs,
          ),
      )

      test_utils.assert_tree_equal(self, self.doubled_pytree, restored.state)

    def test_default_max_to_keep_is_1(self):
      """Test case."""
      # the default max_to_keep should be 1 because we want to ensure that by
      # default, up to 2 steps are in local storage at the same time.
      options = CheckpointManagerOptions(
          local=LocalCheckpointOptions(
              save_interval_steps=1,
              # max_to_keep unset to use default value
              debug_use_full_global_mesh=True,
          ),
      )
      manager = LocalCheckpointManager(
          directory=self.local_directory,
          global_mesh=self.global_mesh,
          replica_id=self.replica_id,
          options=options,
      )
      manager.save(
          0,
          args=args_lib.Composite(
              state=PyTreeSaveArgs(self.pytree),
              process_metadata=process_metadata_checkpoint_handler.ProcessMetadataSaveArgs(
                  global_mesh=self.global_mesh
              ),
          ),
      )
      manager.save(
          1,
          args=args_lib.Composite(
              state=PyTreeSaveArgs(self.doubled_pytree),
              process_metadata=process_metadata_checkpoint_handler.ProcessMetadataSaveArgs(
                  global_mesh=self.global_mesh
              ),
          ),
      )
      manager.wait_until_finished()

      self.assertEqual(manager.all_steps(), [1])
      self.assertFalse((self.local_directory / '0').exists())
      self.assertTrue((self.local_directory / '1' / 'state').exists())
      self.assertTrue(
          (self.local_directory / '1' / 'process_metadata').exists()
      )

    @parameterized.parameters(
        (False),
        (True,),
    )
    def test_local_startup_cleanup(self, enable_async_checkpointing: bool):
      """Test case."""
      options = CheckpointManagerOptions(
          local=LocalCheckpointOptions(
              save_interval_steps=1,
              max_to_keep=3,
              debug_use_full_global_mesh=True,
          ),
          enable_async_checkpointing=enable_async_checkpointing,
      )
      manager = LocalCheckpointManager(
          directory=self.local_directory,
          global_mesh=self.global_mesh,
          replica_id=self.replica_id,
          options=options,
      )

      manager.save(
          0,
          args=args_lib.Composite(
              state=PyTreeSaveArgs(self.pytree),
              process_metadata=process_metadata_checkpoint_handler.ProcessMetadataSaveArgs(
                  global_mesh=self.global_mesh
              ),
          ),
      )
      manager.save(
          1,
          args=args_lib.Composite(
              state=PyTreeSaveArgs(self.pytree),
              process_metadata=process_metadata_checkpoint_handler.ProcessMetadataSaveArgs(
                  global_mesh=self.global_mesh
              ),
          ),
      )
      manager.save(
          2,
          args=args_lib.Composite(
              state=PyTreeSaveArgs(self.doubled_pytree),
              process_metadata=process_metadata_checkpoint_handler.ProcessMetadataSaveArgs(
                  global_mesh=self.global_mesh
              ),
          ),
      )
      manager.wait_until_finished()

      # 3 steps should exist
      self.assertEqual(manager.all_steps(), [0, 1, 2])
      for i in range(3):
        self.assertTrue((self.local_directory / str(i)).exists())
        self.assertTrue(
            mesh_consistency.process_metadata_folder(
                self.local_directory / str(i) / 'process_metadata'
            ).exists()
        )

      # create a new manager with max_to_keep=1
      options = CheckpointManagerOptions(
          local=LocalCheckpointOptions(
              save_interval_steps=1,
              max_to_keep=1,
              debug_use_full_global_mesh=True,
          ),
          enable_async_checkpointing=enable_async_checkpointing,
      )
      manager2 = LocalCheckpointManager(
          directory=self.local_directory,
          global_mesh=self.global_mesh,
          replica_id=self.replica_id,
          options=options,
      )

      # manager2 should have gc'd step 0 and 1
      self.assertEqual(manager2.all_steps(), [2])
      self.assertFalse((self.local_directory / '0').exists())
      self.assertFalse((self.local_directory / '1').exists())
      self.assertTrue((self.local_directory / '2' / 'state').exists())
      self.assertTrue(
          (self.local_directory / '2' / 'process_metadata').exists()
      )

      # new step save should work
      manager2.save(
          3,
          args=args_lib.Composite(
              state=PyTreeSaveArgs(self.doubled_pytree),
              process_metadata=process_metadata_checkpoint_handler.ProcessMetadataSaveArgs(
                  global_mesh=self.global_mesh
              ),
          ),
      )
      manager2.wait_until_finished()
      self.assertEqual(manager2.all_steps(), [3])
      self.assertFalse((self.local_directory / '2').exists())
      self.assertTrue((self.local_directory / '3' / 'state').exists())
      self.assertTrue(
          (self.local_directory / '3' / 'process_metadata').exists()
      )

    @parameterized.parameters(
        (False),
        (True,),
    )
    def test_local_garbage_collection(self, enable_async_checkpointing: bool):
      """Test case."""
      options = CheckpointManagerOptions(
          local=LocalCheckpointOptions(
              save_interval_steps=1,
              max_to_keep=2,
              debug_use_full_global_mesh=True,
          ),
          persistent=PersistentCheckpointOptions(
              save_interval_steps=1, max_to_keep=1
          ),
          enable_async_checkpointing=enable_async_checkpointing,
      )
      manager = LocalCheckpointManager(
          directory=self.local_directory,
          global_mesh=self.global_mesh,
          replica_id=self.replica_id,
          options=options,
      )

      manager.save(
          0,
          args=args_lib.Composite(
              state=PyTreeSaveArgs(self.pytree),
              process_metadata=process_metadata_checkpoint_handler.ProcessMetadataSaveArgs(
                  global_mesh=self.global_mesh
              ),
          ),
      )
      manager.save(
          1,
          args=args_lib.Composite(
              state=PyTreeSaveArgs(self.pytree),
              process_metadata=process_metadata_checkpoint_handler.ProcessMetadataSaveArgs(
                  global_mesh=self.global_mesh
              ),
          ),
      )
      manager.save(
          2,
          args=args_lib.Composite(
              state=PyTreeSaveArgs(self.doubled_pytree),
              process_metadata=process_metadata_checkpoint_handler.ProcessMetadataSaveArgs(
                  global_mesh=self.global_mesh
              ),
          ),
      )
      manager.wait_until_finished()
      self.assertFalse((self.local_directory / '0').exists())
      self.assertTrue((self.local_directory / '1' / 'state').exists())
      self.assertTrue(
          (self.local_directory / '1' / 'process_metadata').exists()
      )

      self.assertTrue((self.local_directory / '2' / 'state').exists())
      self.assertTrue(
          (self.local_directory / '2' / 'process_metadata').exists()
      )

      if multihost.process_index() == 0:
        test_utils.empty_directory(self.local_directory)

      manager.reload()

      if multihost.process_index() == 0:
        self.assertFalse((self.local_directory / '1').exists())
        self.assertFalse((self.local_directory / '2').exists())
      else:
        self.assertTrue((self.local_directory / '1' / 'state').exists())
        self.assertTrue(
            (self.local_directory / '1' / 'process_metadata').exists()
        )
        self.assertTrue((self.local_directory / '2' / 'state').exists())
        self.assertTrue(
            (self.local_directory / '2' / 'process_metadata').exists()
        )

      # subsequent save should be successful and gc should work
      manager.save(
          3,
          args=args_lib.Composite(
              state=PyTreeSaveArgs(self.pytree),
              process_metadata=process_metadata_checkpoint_handler.ProcessMetadataSaveArgs(
                  global_mesh=self.global_mesh
              ),
          ),
      )
      manager.wait_until_finished()

      if multihost.process_index() == 0:
        self.assertFalse((self.local_directory / '2').exists())
        self.assertTrue((self.local_directory / '3' / 'state').exists())
        self.assertTrue(
            (self.local_directory / '3' / 'process_metadata').exists()
        )
      else:
        self.assertFalse((self.local_directory / '1').exists())
        self.assertTrue((self.local_directory / '2' / 'state').exists())
        self.assertTrue(
            (self.local_directory / '2' / 'process_metadata').exists()
        )
        self.assertTrue((self.local_directory / '3' / 'state').exists())
        self.assertTrue(
            (self.local_directory / '3' / 'process_metadata').exists()
        )

      manager.save(
          4,
          args=args_lib.Composite(
              state=PyTreeSaveArgs(self.doubled_pytree),
              process_metadata=process_metadata_checkpoint_handler.ProcessMetadataSaveArgs(
                  global_mesh=self.global_mesh
              ),
          ),
      )
      manager.save(
          5,
          args=args_lib.Composite(
              state=PyTreeSaveArgs(self.pytree),
              process_metadata=process_metadata_checkpoint_handler.ProcessMetadataSaveArgs(
                  global_mesh=self.global_mesh
              ),
          ),
      )
      manager.wait_until_finished()

      self.assertFalse((self.local_directory / '2').exists())
      self.assertFalse((self.local_directory / '3').exists())
      self.assertTrue((self.local_directory / '4' / 'state').exists())
      self.assertTrue(
          (self.local_directory / '4' / 'process_metadata').exists()
      )
      self.assertTrue((self.local_directory / '5' / 'state').exists())
      self.assertTrue(
          (self.local_directory / '5' / 'process_metadata').exists()
      )

    @parameterized.parameters(
        ([[], [], [], []], {0: {}, 1: {}}),
        ([[1], [1], [0], [0]], {0: {1}, 1: {0}}),
        ([[], [], [0], [0]], {0: {}, 1: {0}}),
        ([[], [], [0], []], {0: {}, 1: {}}),
        ([[], [], [0], [1]], {0: {}, 1: {}}),
        ([[], [0], [], [0]], {0: {}, 1: {}}),
        ([[0], [1], [0], [1]], {0: {}, 1: {0}}),
        ([[1, 2], [1, 2], [4], [4, 5]], {0: {1, 2}, 1: {4}}),
        ([[-1, 0], [-1, 0], [1, 2], [1, 2]], {0: {-1, 0}, 1: {1, 2}}),
        (
            [[-1, 0, 1], [-1, 0, 1], [0, 1, -1], [1, 0, -1]],
            {0: {-1, 0, 1}, 1: {-1, 0, 1}},
        ),
    )
    def test_common_steps_per_slice(self, process_steps, expectation):
      """Test case."""
      per_process_steps = {
          pid: steps for pid, steps in enumerate(process_steps)
      }
      result = emergency_path_utils._common_values_per_replica(  # pylint: disable=protected-access
          per_process_steps, global_mesh=self.global_mesh, replica_axis_index=0
      )
      self.assertSameElements(result, expectation)


class LocalCheckpointManagerTest(
    LocalCheckpointManagerTestSuite.LocalCheckpointManagerTest,
    multiprocess_test.MultiProcessTest,
):
  pass


if __name__ == '__main__':
  multiprocess_test.main()
