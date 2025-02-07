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

"""Test base classes for emergency.ReplicatorCheckpointManager."""

import json
from typing import Any
from unittest import mock
from absl import logging
from absl.testing import flagsaver
from absl.testing import parameterized
from etils import epath
import jax
import numpy as np
from orbax.checkpoint import args as args_lib
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.handlers import pytree_checkpoint_handler
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.multihost import multislice
from orbax.checkpoint._src.serialization import type_handlers
from orbax.checkpoint.experimental.emergency import mesh_consistency
from orbax.checkpoint.experimental.emergency import replicator_checkpoint_manager
from orbax.checkpoint.experimental.emergency import test_utils as emergency_test_utils
from orbax.checkpoint.path import atomicity
from orbax.checkpoint.path import step as step_lib
from .learning.brain.research.jax.tests.multiprocess import multiprocess_test


PyTree = Any
PyTreeCheckpointHandler = pytree_checkpoint_handler.PyTreeCheckpointHandler
PyTreeSaveArgs = pytree_checkpoint_handler.PyTreeSaveArgs
PyTreeRestoreArgs = pytree_checkpoint_handler.PyTreeRestoreArgs

ReplicatorCheckpointManager = (
    replicator_checkpoint_manager.ReplicatorCheckpointManager
)
ReplicatorCheckpointManagerOptions = (
    replicator_checkpoint_manager.ReplicatorCheckpointManagerOptions
)
barrier_compatible_test = test_utils.barrier_compatible_test
assert_tree_equal = test_utils.assert_tree_equal
get_fake_global_mesh_for_slices = test_utils.get_fake_global_mesh_for_slices
swap_slices_in_mesh = emergency_test_utils.swap_slices_in_mesh


@barrier_compatible_test
class ReplicatorCheckpointManagerTest(
    parameterized.TestCase, multiprocess_test.MultiProcessTest
):

  def make_global_mesh(self, replica_axis_index: int = 0) -> jax.sharding.Mesh:
    if replica_axis_index not in [0, 1]:
      raise ValueError(
          'replica_axis_index must be 0 or 1 for this test. Got:'
          f' {replica_axis_index}.'
      )
    self.assertEqual(jax.device_count(), 8)
    self.assertEqual(jax.process_count(), 4)
    self.assertEqual(jax.local_device_count(), 2)

    # setup global mesh info for 2-slice tests
    slice_processes = [{0, 1}, {2, 3}]
    mesh = test_utils.get_fake_global_mesh_for_slices(
        slice_processes, replica_axis_index
    )
    if replica_axis_index == 0:
      assert mesh.devices.shape == (2, 4), mesh.devices.shape
    if replica_axis_index == 1:
      assert mesh.devices.shape == (4, 2), mesh.devices.shape
    return mesh

  def setup_pytree(self, global_mesh: jax.sharding.Mesh):
    pytree = {
        'a': test_utils.create_sharded_array(
            np.arange(8), global_mesh, jax.sharding.PartitionSpec(None)
        ),
        'b': test_utils.create_sharded_array(
            np.arange(16),
            global_mesh,
            jax.sharding.PartitionSpec('data'),
        ),
        'scalar': test_utils.create_sharded_array(
            123, global_mesh, jax.sharding.PartitionSpec()
        ),
    }
    restore_args = jax.tree_util.tree_map(
        lambda x: type_handlers.ArrayRestoreArgs(sharding=x.sharding),
        pytree,
    )
    return pytree, restore_args

  def setUp(self):
    super().setUp()
    self.enter_context(
        flagsaver.flagsaver(experimental_orbax_use_distributed_process_id=True)
    )
    self.enter_context(
        mock.patch.object(
            step_lib, 'is_gcs_path', autospec=True, return_value=True
        )
    )
    if not multihost.is_runtime_to_distributed_ids_initialized():
      multihost.initialize_runtime_to_distributed_ids()
      multihost.initialize_distributed_to_device_ids()

    self.global_mesh = self.make_global_mesh()

    self._fn = lambda ty: issubclass(ty, jax.Array)

    pytree, restore_args = self.setup_pytree(self.global_mesh)
    doubled_pytree = test_utils.apply_function(pytree, lambda x: x * 2)

    self.pytree = pytree
    self.doubled_pytree = doubled_pytree
    self.restore_args = restore_args

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
        'ReplicatorCheckpointManagerTest:setup_complete'
    )

  def tearDown(self):
    super().tearDown()
    test_utils.sync_global_processes(
        'ReplicatorCheckpointManagerTest:teardown_complete'
    )

  @parameterized.parameters((0,), (1,))
  def test_save(self, replica_axis_index: int):
    global_mesh = self.make_global_mesh(replica_axis_index=replica_axis_index)
    pytree, _ = self.setup_pytree(global_mesh)
    options = ReplicatorCheckpointManagerOptions(
        save_interval_steps=3,
    )
    manager = ReplicatorCheckpointManager(
        self.local_directory,
        options=options,
        global_mesh=global_mesh,
    )

    for i in range(17):
      manager.save(i, args=PyTreeSaveArgs(pytree))
    manager.wait_until_finished()

    self.assertNotEmpty(list(self.local_directory.iterdir()))
    self.assertEqual(manager.all_steps(), [0, 3, 6, 9, 12, 15])

  @parameterized.parameters((0,), (1,))
  def test_save_restore(self, replica_axis_index: int):
    global_mesh = self.make_global_mesh(replica_axis_index=replica_axis_index)
    pytree, restore_args = self.setup_pytree(global_mesh)
    options = ReplicatorCheckpointManagerOptions(
        save_interval_steps=1,
    )
    manager = ReplicatorCheckpointManager(
        self.local_directory,
        options=options,
        global_mesh=global_mesh,
    )

    manager.save(0, args=PyTreeSaveArgs(pytree))

    manager.wait_until_finished()

    restored = manager.restore(
        0, args=PyTreeRestoreArgs(restore_args=restore_args)
    )

    test_utils.assert_tree_equal(self, pytree, restored)

  def test_no_cleanup(self):
    options = ReplicatorCheckpointManagerOptions(
        save_interval_steps=1,
    )
    manager = ReplicatorCheckpointManager(
        self.local_directory,
        options=options,
        global_mesh=self.global_mesh,
    )

    manager.save(0, args=PyTreeSaveArgs(self.pytree))
    manager.wait_until_finished()

    manager.save(1, args=PyTreeSaveArgs(self.doubled_pytree))
    manager.wait_until_finished()

    restored = manager.restore(
        0, args=PyTreeRestoreArgs(restore_args=self.restore_args)
    )
    test_utils.assert_tree_equal(self, self.pytree, restored)
    restored = manager.restore(
        1, args=PyTreeRestoreArgs(restore_args=self.restore_args)
    )
    test_utils.assert_tree_equal(self, self.doubled_pytree, restored)

  def test_startup_cleanup(self):
    options = ReplicatorCheckpointManagerOptions(
        save_interval_steps=1,
    )
    manager = ReplicatorCheckpointManager(
        self.local_directory,
        options=options,
        global_mesh=self.global_mesh,
    )

    manager.save(0, args=PyTreeSaveArgs(self.pytree))
    manager.save(1, args=PyTreeSaveArgs(self.pytree))
    manager.save(2, args=PyTreeSaveArgs(self.doubled_pytree))
    manager.wait_until_finished()

    # 3 steps should exist
    self.assertEqual(manager.all_steps(), [0, 1, 2])
    for step in [0, 1, 2]:
      self.assertTrue((self.local_directory / str(step)).exists())
      self.assertTrue(
          (
              self.local_directory / str(step) / atomicity.COMMIT_SUCCESS_FILE
          ).exists()
      )

    # Step looks like uncommitted now.
    (self.local_directory / '0' / atomicity.COMMIT_SUCCESS_FILE).unlink()

    manager2 = ReplicatorCheckpointManager(
        self.local_directory,
        options=options,
        global_mesh=self.global_mesh,
    )

    # manager2 should have gc'd step 0
    self.assertEqual(manager2.all_steps(), [1, 2])
    self.assertFalse((self.local_directory / '0').exists())
    self.assertTrue((self.local_directory / '1').exists())
    self.assertTrue((self.local_directory / '2').exists())

    # new step save should work
    manager2.save(3, args=PyTreeSaveArgs(self.doubled_pytree))
    manager2.wait_until_finished()
    self.assertEqual(manager2.all_steps(), [1, 2, 3])
    self.assertTrue((self.local_directory / '2').exists())
    self.assertTrue((self.local_directory / '3').exists())

  @parameterized.parameters(
      (0, 1, True),
      (0, 2, True),
      (1, 1, True),
      (1, 1, True),
      (1, 2, False),
      (3, 5, False),
      (2, 1, True),
      (3, 2, False),
      (6, 3, True),
  )
  def test_should_save(
      self,
      step: int,
      interval: int,
      expectation: bool,
  ):
    options = ReplicatorCheckpointManagerOptions(
        save_interval_steps=interval,
    )
    manager = ReplicatorCheckpointManager(
        self.local_directory,
        options=options,
        global_mesh=self.global_mesh,
    )

    for i in range(step):
      manager.save(i, args=PyTreeSaveArgs(self.pytree))
      manager.wait_until_finished()

    self.assertEqual(manager.should_save(step), expectation)

  @parameterized.parameters(
      (1, 4, [0, 1, 2, 3]),
      (2, 4, [0, 2]),
      (3, 4, [0, 3]),
      (3, 6, [0, 3]),
  )
  def test_steps(
      self,
      interval,
      total_steps,
      expectation,
  ):
    expected_latest_step = expectation[-1] if expectation else None
    options = ReplicatorCheckpointManagerOptions(
        save_interval_steps=interval,
    )
    manager = ReplicatorCheckpointManager(
        self.local_directory,
        options=options,
        global_mesh=self.global_mesh,
    )

    for i in range(total_steps):
      manager.save(i, args=PyTreeSaveArgs(self.pytree))
      manager.wait_until_finished()

    self.assertEqual(sorted(manager.all_steps()), expectation)
    self.assertEqual(manager.latest_step(), expected_latest_step)

    manager2 = ReplicatorCheckpointManager(
        self.local_directory,
        options=options,
        global_mesh=self.global_mesh,
    )

    self.assertEqual(sorted(manager2.all_steps()), expectation)
    self.assertEqual(manager2.latest_step(), expected_latest_step)

  @parameterized.parameters(
      (0,),
      (1,),
  )
  def test_slice_setup(self, replica_axis_index: int):
    """Test restoration after slice swap (via global mesh)."""
    global_mesh = self.make_global_mesh(replica_axis_index=replica_axis_index)
    pytree, restore_args = self.setup_pytree(global_mesh)
    pytree_double = test_utils.apply_function(pytree, lambda x: x * 2)
    options = ReplicatorCheckpointManagerOptions(
        save_interval_steps=1,
    )
    with ReplicatorCheckpointManager(
        self.local_directory,
        options=options,
        global_mesh=global_mesh,
    ) as manager:
      manager.save(0, args=PyTreeSaveArgs(pytree))
      manager.save(1, args=PyTreeSaveArgs(pytree))
      manager.save(2, args=PyTreeSaveArgs(pytree_double))
      test_utils.assert_tree_equal(
          self,
          pytree_double,
          manager.restore(
              None,
              args=PyTreeRestoreArgs(restore_args=restore_args),
          ),
      )

    self.assertEqual(jax.process_count(), 4)
    self.assertEqual(
        multislice.process_slice_id(
            0, global_mesh, replica_axis_index=replica_axis_index
        ),
        0,
    )
    self.assertEqual(
        multislice.process_slice_id(
            1, global_mesh, replica_axis_index=replica_axis_index
        ),
        0,
    )
    self.assertEqual(
        multislice.process_slice_id(
            2, global_mesh, replica_axis_index=replica_axis_index
        ),
        1,
    )
    self.assertEqual(
        multislice.process_slice_id(
            3, global_mesh, replica_axis_index=replica_axis_index
        ),
        1,
    )
    new_global_mesh = swap_slices_in_mesh(
        global_mesh, replica_axis_index=replica_axis_index
    )
    self.assertEqual(
        multislice.process_slice_id(
            0, new_global_mesh, replica_axis_index=replica_axis_index
        ),
        1,
    )
    self.assertEqual(
        multislice.process_slice_id(
            1, new_global_mesh, replica_axis_index=replica_axis_index
        ),
        1,
    )
    self.assertEqual(
        multislice.process_slice_id(
            2, new_global_mesh, replica_axis_index=replica_axis_index
        ),
        0,
    )
    self.assertEqual(
        multislice.process_slice_id(
            3, new_global_mesh, replica_axis_index=replica_axis_index
        ),
        0,
    )

    with ReplicatorCheckpointManager(
        self.local_directory,
        options=options,
        global_mesh=new_global_mesh,
    ) as manager:
      self.assertEqual(manager.latest_step(), 2)
      restored = manager.restore(
          None,
          args=PyTreeRestoreArgs(restore_args=restore_args),
      )
      test_utils.assert_tree_equal(self, pytree_double, restored)

  @parameterized.parameters(
      (0,),
      (1,),
  )
  def test_slice_swap(self, replica_axis_index: int):
    """Test step detection across slice swap (via mesh)."""
    global_mesh = self.make_global_mesh(replica_axis_index=replica_axis_index)
    pytree, restore_args = self.setup_pytree(global_mesh)
    options = ReplicatorCheckpointManagerOptions(
        save_interval_steps=1,
    )
    with ReplicatorCheckpointManager(
        self.local_directory,
        options=options,
        global_mesh=global_mesh,
    ) as manager:
      manager.save(0, args=PyTreeSaveArgs(pytree))
      manager.save(1, args=PyTreeSaveArgs(pytree))
      manager.save(2, args=PyTreeSaveArgs(pytree))
      manager.wait_until_finished()
      self.assertSameElements([0, 1, 2], manager.all_steps())
      self.assertEqual(2, manager.latest_step())
      test_utils.assert_tree_equal(
          self,
          pytree,
          manager.restore(
              manager.latest_step(),
              args=PyTreeRestoreArgs(restore_args=restore_args),
          ),
      )

    new_global_mesh = swap_slices_in_mesh(
        global_mesh, replica_axis_index=replica_axis_index
    )
    with ReplicatorCheckpointManager(
        self.local_directory,
        options=options,
        global_mesh=new_global_mesh,
    ) as manager:
      self.assertSameElements([0, 1, 2], manager.all_steps())
      self.assertEqual(2, manager.latest_step())
      test_utils.assert_tree_equal(
          self,
          pytree,
          manager.restore(
              None,
              args=PyTreeRestoreArgs(restore_args=restore_args),
          ),
      )

  def test_process_index_metadata(self):
    with ReplicatorCheckpointManager(
        self.local_directory,
        global_mesh=self.global_mesh,
    ) as mngr:
      metadata_folder = mesh_consistency.process_metadata_folder(
          self.local_directory, 0
      )
      self.assertFalse(metadata_folder.exists())

      mngr.save(0, args=PyTreeSaveArgs(self.pytree))
      mngr.wait_until_finished()

      metadata_path = (
          metadata_folder / mesh_consistency._GLOBAL_PROCESS_METADATA_FILE_NAME
      )
      self.assertTrue(metadata_path.exists())
      contents = json.loads(metadata_path.read_text())
      self.assertListEqual(
          multihost.distributed_to_device_ids(),
          contents,
      )

      metadata_path = (
          metadata_folder / mesh_consistency._MESH_METADATA_FILE_NAME
      )
      self.assertTrue(metadata_path.exists())
      device_ids = json.loads(metadata_path.read_text())
      self.assertListEqual(
          device_ids,
          [int(id) for id in self.global_mesh.device_ids.flatten()],
      )

      (metadata_folder / atomicity.COMMIT_SUCCESS_FILE).unlink()
      with self.assertRaisesRegex(
          ValueError, 'Process metadata folder was not finalized'
      ):
        mesh_consistency.read_process_metadata(self.local_directory, 0)

  def test_should_save_fn(self):
    expectation = [1, 5, 7, 8]

    def should_save(step: int, _) -> bool:
      return step in expectation

    options = ReplicatorCheckpointManagerOptions(
        should_save_fn=should_save,
    )
    with ReplicatorCheckpointManager(
        self.local_directory,
        options=options,
        global_mesh=self.global_mesh,
    ) as manager:
      for i in range(10):
        manager.save(i, args=PyTreeSaveArgs(self.pytree))
        manager.wait_until_finished()

      self.assertSameElements(manager.all_steps(), expectation)

  @parameterized.parameters((0,), (1,))
  def test_multiple_items(self, replica_axis_index: int):
    global_mesh = self.make_global_mesh(replica_axis_index=replica_axis_index)
    pytree, restore_args = self.setup_pytree(global_mesh)
    doubled_pytree = test_utils.apply_function(pytree, lambda x: x * 2)
    with ReplicatorCheckpointManager(
        self.local_directory,
        global_mesh=global_mesh,
    ) as manager:
      manager.save(
          0,
          args=args_lib.Composite(
              state=PyTreeSaveArgs(pytree),
              other_state=PyTreeSaveArgs(doubled_pytree),
          ),
      )

    restored = manager.restore(
        0,
        args=args_lib.Composite(
            state=PyTreeRestoreArgs(restore_args=restore_args),
            other_state=PyTreeRestoreArgs(restore_args=restore_args),
        ),
    )
    self.assertIn('state', restored)
    self.assertIn('other_state', restored)
    test_utils.assert_tree_equal(self, pytree, restored.state)
    test_utils.assert_tree_equal(self, doubled_pytree, restored.other_state)

  def test_invalid_args_save(self):
    with ReplicatorCheckpointManager(
        self.local_directory,
        global_mesh=self.global_mesh,
    ) as manager:
      with self.assertRaises(TypeError):
        manager.save(0)
      with self.assertRaises(ValueError):
        manager.save(0, args=args_lib.StandardSave(self.pytree))
      with self.assertRaises(ValueError):
        manager.save(
            0, args=args_lib.Composite(state=args_lib.StandardSave(self.pytree))
        )

  def test_invalid_args_restore(self):
    with ReplicatorCheckpointManager(
        self.local_directory,
        global_mesh=self.global_mesh,
    ) as manager:
      manager.save(0, args=args_lib.PyTreeSave(self.pytree))
      restored = manager.restore(
          0, args=args_lib.PyTreeRestore(restore_args=self.restore_args)
      )
      test_utils.assert_tree_equal(self, self.pytree, restored)

      with self.assertRaises(ValueError):
        manager.restore(0, args=args_lib.PyTreeRestore(self.pytree))
      with self.assertRaises(ValueError):
        manager.restore(0, args=args_lib.StandardRestore(self.pytree))
      with self.assertRaises(ValueError):
        manager.restore(
            0,
            args=args_lib.Composite(
                state=args_lib.StandardRestore(self.pytree)
            ),
        )
      with self.assertRaises(ValueError):
        manager.restore(0)
      with self.assertRaises(FileNotFoundError):
        manager.restore(1, args=args_lib.PyTreeRestore(self.pytree))
      with self.assertRaises(ValueError):
        manager.restore(1)

    with ReplicatorCheckpointManager(
        self.local_directory,
        global_mesh=self.global_mesh,
    ) as manager:
      manager.save(0, args=args_lib.PyTreeSave(self.pytree))

  @parameterized.parameters(
      ((8,), ('data',)),
      ((8,), ('data',)),
      ((1, 8), ('replica', 'data')),
      ((8, 1), ('data', 'replica')),
  )
  def test_save_restore_single_slice(self, mesh_shape, axis_names):
    """Test single-slice save/restore."""
    # 1-d mesh
    global_mesh = jax.sharding.Mesh(
        np.asarray(jax.devices()).reshape(mesh_shape), axis_names
    )
    pytree, restore_args = self.setup_pytree(global_mesh)
    options = ReplicatorCheckpointManagerOptions(
        save_interval_steps=1,
    )
    manager = ReplicatorCheckpointManager(
        self.local_directory,
        options=options,
        global_mesh=global_mesh,
    )

    manager.save(0, args=PyTreeSaveArgs(pytree))
    manager.wait_until_finished()

    restored = manager.restore(
        0, args=PyTreeRestoreArgs(restore_args=restore_args)
    )
    test_utils.assert_tree_equal(self, pytree, restored)


if __name__ == '__main__':
  multiprocess_test.main()
