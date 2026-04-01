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

"""Test base classes for emergency.ReplicatorCheckpointManager."""

import json
from typing import Any
from unittest import mock

from absl import logging
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
from etils import epath
import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint import args as args_lib
from orbax.checkpoint import checkpoint_manager
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.checkpoint_managers import preservation_policy as preservation_policy_lib
from orbax.checkpoint._src.handlers import pytree_checkpoint_handler
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.multihost import multislice
from orbax.checkpoint._src.path import gcs_utils
from orbax.checkpoint._src.serialization import type_handlers
from orbax.checkpoint._src.testing import multiprocess_test
from orbax.checkpoint.experimental.emergency import checkpoint_manager as emergency_checkpoint_manager
from orbax.checkpoint.experimental.emergency import mesh_consistency
from orbax.checkpoint.experimental.emergency import process_metadata_checkpoint_handler
from orbax.checkpoint.experimental.emergency import replicator_checkpoint_manager
from orbax.checkpoint.experimental.emergency.test_utils import dataset_iterator_checkpoint_handler
from orbax.checkpoint.experimental.emergency.test_utils import test_base as emergency_test_utils
from orbax.checkpoint.path import atomicity
from orbax.checkpoint.path import step as step_lib


PyTree = Any
rcm_lib = replicator_checkpoint_manager
PyTreeCheckpointHandler = pytree_checkpoint_handler.PyTreeCheckpointHandler
PyTreeSaveArgs = pytree_checkpoint_handler.PyTreeSaveArgs
PyTreeRestoreArgs = pytree_checkpoint_handler.PyTreeRestoreArgs

ReplicatorCheckpointManager = (
    replicator_checkpoint_manager.ReplicatorCheckpointManager
)
ReplicatorCheckpointManagerOptions = (
    replicator_checkpoint_manager.ReplicatorCheckpointManagerOptions
)
DatasetIteratorCheckpointSave = (
    dataset_iterator_checkpoint_handler.DatasetIteratorCheckpointSave
)
DatasetIteratorCheckpointRestore = (
    dataset_iterator_checkpoint_handler.DatasetIteratorCheckpointRestore
)
DatasetIteratorCheckpointHandler = (
    dataset_iterator_checkpoint_handler.DatasetIteratorCheckpointHandler
)
barrier_compatible_test = test_utils.barrier_compatible_test
assert_tree_equal = test_utils.assert_tree_equal
get_fake_global_mesh_for_slices = test_utils.get_fake_global_mesh_for_slices
swap_slices_in_mesh = emergency_test_utils.swap_slices_in_mesh
STATE = 'state'


def get_composite_save_args(
    pytree: PyTree,
    checkpoint_args: args_lib.CheckpointArgs = PyTreeSaveArgs,
) -> args_lib.Composite:
  return args_lib.Composite(state=checkpoint_args(pytree))


def get_composite_restore_args(
    pytree: PyTree | None = None,
    restore_args: PyTree | None = None,
    checkpoint_args: args_lib.CheckpointArgs = PyTreeRestoreArgs,
) -> args_lib.Composite:
  if pytree is not None:
    return args_lib.Composite(state=checkpoint_args(pytree))
  return args_lib.Composite(state=checkpoint_args(restore_args=restore_args))


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
            gcs_utils, 'is_gcs_path', autospec=True, return_value=True
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
    self.non_replicated_directory = (
        epath.Path(self._get_tempdir_path_test())
        / 'non_replicated_checkpointing_test'
    )
    if multihost.is_primary_host(primary_host=0):
      self.non_replicated_directory = epath.Path(
          self.create_tempdir(
              name='non_replicated_checkpointing_test'
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

  def assert_process_metadata_files_exist(
      self,
      step: int,
      global_mesh: jax.sharding.Mesh,
  ):
    metadata_folder = mesh_consistency.process_metadata_folder(
        self.local_directory / str(step) / 'process_metadata'
    )
    metadata_path = (
        metadata_folder / mesh_consistency._GLOBAL_PROCESS_METADATA_FILE_NAME
    )
    self.assertTrue(metadata_path.exists())
    contents = json.loads(metadata_path.read_text())
    self.assertListEqual(
        multihost.distributed_to_device_ids(),
        contents,
    )

    metadata_path = metadata_folder / mesh_consistency._MESH_METADATA_FILE_NAME
    self.assertTrue(metadata_path.exists())
    device_ids = json.loads(metadata_path.read_text())
    self.assertListEqual(
        device_ids,
        [int(id) for id in global_mesh.device_ids.flatten()],
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
      manager.save(
          i,
          args=get_composite_save_args(pytree),
      )
    manager.wait_until_finished()

    expected_steps = [0, 3, 6, 9, 12, 15]
    self.assertNotEmpty(list(self.local_directory.iterdir()))
    self.assertEqual(manager.all_steps(), expected_steps)
    for i in expected_steps:
      self.assert_process_metadata_files_exist(i, global_mesh)

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

    manager.save(
        0,
        args=get_composite_save_args(pytree),
    )
    manager.wait_until_finished()
    self.assert_process_metadata_files_exist(0, global_mesh)

    restored = manager.restore(
        0, args=get_composite_restore_args(restore_args=restore_args)
    )

    test_utils.assert_tree_equal(self, pytree, restored.state)

  def test_preservation_policy(self):
    global_mesh = self.make_global_mesh(replica_axis_index=0)
    pytree, _ = self.setup_pytree(global_mesh)
    options = ReplicatorCheckpointManagerOptions(
        save_interval_steps=1,
        preservation_policy=preservation_policy_lib.EveryNSteps(2),
    )
    manager = ReplicatorCheckpointManager(
        self.local_directory,
        options=options,
        global_mesh=global_mesh,
    )

    for i in range(3):
      manager.save(
          i,
          args=get_composite_save_args(pytree),
      )
      manager.wait_until_finished()
    self.assertEqual(manager.all_steps(), [0, 2])

  @parameterized.parameters((0,), (1,))
  def test_save_restore_dataset_iterator(self, replica_axis_index: int):
    global_mesh = self.make_global_mesh(replica_axis_index=replica_axis_index)
    pytree, restore_args = self.setup_pytree(global_mesh)
    options = ReplicatorCheckpointManagerOptions(
        save_interval_steps=1,
        preservation_policy=preservation_policy_lib.LatestN(1),
    )
    manager = ReplicatorCheckpointManager(
        self.local_directory,
        persistent_directory=self.non_replicated_directory,
        options=options,
        global_mesh=global_mesh,
    )

    dummy_dataset = [
        ('hello', 'hola'),
        ('world', 'mundo'),
        ('test', 'prueba'),
    ]
    dummy_iterator = DatasetIteratorCheckpointHandler.DummyIterator(
        dummy_dataset
    )

    manager.save(
        0,
        args=args_lib.Composite(
            state=PyTreeSaveArgs(pytree),
            dataset=DatasetIteratorCheckpointSave(dummy_iterator),
        ),
    )
    manager.wait_until_finished()
    self.assert_process_metadata_files_exist(0, global_mesh)

    restored = manager.restore(
        0,
        args=args_lib.Composite(
            state=PyTreeRestoreArgs(restore_args=restore_args),
            dataset=DatasetIteratorCheckpointRestore(dummy_iterator),
        ),
    )
    test_utils.assert_tree_equal(self, pytree, restored.state)
    test_utils.assert_tree_equal(self, dummy_iterator, restored.dataset)

    manager.save(
        1,
        args=args_lib.Composite(
            state=PyTreeSaveArgs(pytree),
            dataset=DatasetIteratorCheckpointSave(dummy_iterator),
        ),
    )
    manager.wait_until_finished()
    self.assertEqual(manager.all_steps(), [1])

  def test_no_cleanup(self):
    options = ReplicatorCheckpointManagerOptions(
        save_interval_steps=1,
    )
    manager = ReplicatorCheckpointManager(
        self.local_directory,
        options=options,
        global_mesh=self.global_mesh,
    )

    manager.save(
        0,
        args=get_composite_save_args(self.pytree),
    )
    manager.wait_until_finished()
    self.assert_process_metadata_files_exist(0, self.global_mesh)

    manager.save(
        1,
        args=get_composite_save_args(self.doubled_pytree),
    )
    manager.wait_until_finished()
    self.assert_process_metadata_files_exist(1, self.global_mesh)

    restored = manager.restore(
        0,
        args=get_composite_restore_args(restore_args=self.restore_args),
    )
    test_utils.assert_tree_equal(self, self.pytree, restored.state)
    restored = manager.restore(
        1,
        args=get_composite_restore_args(restore_args=self.restore_args),
    )
    test_utils.assert_tree_equal(self, self.doubled_pytree, restored.state)

  def test_startup_cleanup(self):
    options = ReplicatorCheckpointManagerOptions(
        save_interval_steps=1,
    )
    manager = ReplicatorCheckpointManager(
        self.local_directory,
        options=options,
        global_mesh=self.global_mesh,
    )

    manager.save(
        0,
        args=get_composite_save_args(self.pytree),
    )
    manager.save(
        1,
        args=get_composite_save_args(self.pytree),
    )
    manager.save(
        2,
        args=get_composite_save_args(self.doubled_pytree),
    )
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
      self.assert_process_metadata_files_exist(step, self.global_mesh)

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
    manager2.save(
        3,
        args=get_composite_save_args(self.doubled_pytree),
    )
    manager2.wait_until_finished()
    self.assertEqual(manager2.all_steps(), [1, 2, 3])
    self.assertTrue((self.local_directory / '2').exists())
    self.assertTrue((self.local_directory / '3').exists())
    self.assert_process_metadata_files_exist(3, self.global_mesh)

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
      manager.save(
          i,
          args=get_composite_save_args(self.pytree),
      )
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
      manager.save(
          i,
          args=get_composite_save_args(self.pytree),
      )
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
      manager.save(
          0,
          args=get_composite_save_args(pytree),
      )
      manager.save(
          1,
          args=get_composite_save_args(pytree),
      )
      manager.save(
          2,
          args=get_composite_save_args(self.doubled_pytree),
      )
      test_utils.assert_tree_equal(
          self,
          pytree_double,
          manager.restore(
              None, args=get_composite_restore_args(restore_args=restore_args)
          ).state,
      )

    self.assertEqual(jax.process_count(), 4)
    self.assertEqual(
        multislice.process_replica_id(
            0, global_mesh, replica_axis_index=replica_axis_index
        ),
        0,
    )
    self.assertEqual(
        multislice.process_replica_id(
            1, global_mesh, replica_axis_index=replica_axis_index
        ),
        0,
    )
    self.assertEqual(
        multislice.process_replica_id(
            2, global_mesh, replica_axis_index=replica_axis_index
        ),
        1,
    )
    self.assertEqual(
        multislice.process_replica_id(
            3, global_mesh, replica_axis_index=replica_axis_index
        ),
        1,
    )
    new_global_mesh = swap_slices_in_mesh(
        global_mesh, replica_axis_index=replica_axis_index
    )
    self.assertEqual(
        multislice.process_replica_id(
            0, new_global_mesh, replica_axis_index=replica_axis_index
        ),
        1,
    )
    self.assertEqual(
        multislice.process_replica_id(
            1, new_global_mesh, replica_axis_index=replica_axis_index
        ),
        1,
    )
    self.assertEqual(
        multislice.process_replica_id(
            2, new_global_mesh, replica_axis_index=replica_axis_index
        ),
        0,
    )
    self.assertEqual(
        multislice.process_replica_id(
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
          args=get_composite_restore_args(restore_args=restore_args),
      )
      test_utils.assert_tree_equal(self, pytree_double, restored.state)

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
      manager.save(
          0,
          args=get_composite_save_args(pytree),
      )
      manager.save(
          1,
          args=get_composite_save_args(pytree),
      )
      manager.save(
          2,
          args=get_composite_save_args(pytree),
      )
      manager.wait_until_finished()
      self.assertSameElements([0, 1, 2], manager.all_steps())
      self.assertEqual(2, manager.latest_step())
      test_utils.assert_tree_equal(
          self,
          pytree,
          manager.restore(
              manager.latest_step(),
              args=get_composite_restore_args(restore_args=restore_args),
          ).state,
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
              None, args=get_composite_restore_args(restore_args=restore_args)
          ).state,
      )

  def test_process_index_metadata(self):
    with ReplicatorCheckpointManager(
        self.local_directory,
        global_mesh=self.global_mesh,
    ) as mngr:
      metadata_folder = mesh_consistency.process_metadata_folder(
          self.local_directory / str(0) / 'process_metadata'
      )
      self.assertFalse(metadata_folder.exists())

      mngr.save(
          0,
          args=get_composite_save_args(self.pytree),
      )
      mngr.wait_until_finished()

      self.assert_process_metadata_files_exist(0, self.global_mesh)

      (metadata_folder / atomicity.COMMIT_SUCCESS_FILE).unlink()
      with self.assertRaisesRegex(
          ValueError, 'Process metadata folder was not finalized'
      ):
        mesh_consistency.read_process_metadata(metadata_folder)

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
        manager.save(
            i,
            args=get_composite_save_args(self.pytree),
        )
        manager.wait_until_finished()

      self.assertSameElements(manager.all_steps(), expectation)
      for step in expectation:
        self.assert_process_metadata_files_exist(step, self.global_mesh)

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
      self.assert_process_metadata_files_exist(0, global_mesh)

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
        manager.save(
            0,
            args=get_composite_save_args(self.pytree, args_lib.StandardRestore),
        )
      with self.assertRaises(ValueError):
        manager.save(
            0,
            args=get_composite_save_args(self.pytree, args_lib.StandardRestore),
        )

  def test_invalid_args_restore(self):
    with ReplicatorCheckpointManager(
        self.local_directory,
        global_mesh=self.global_mesh,
    ) as manager:
      manager.save(
          0,
          args=get_composite_save_args(self.pytree, args_lib.PyTreeSave),
      )
      restored = manager.restore(
          0,
          args=get_composite_restore_args(
              restore_args=self.restore_args,
              checkpoint_args=args_lib.PyTreeRestore,
          ),
      )
      test_utils.assert_tree_equal(self, self.pytree, restored.state)

      with self.assertRaises(ValueError):
        manager.restore(
            0,
            args=get_composite_restore_args(
                self.pytree, checkpoint_args=args_lib.StandardRestore
            ),
        )
      with self.assertRaises(TypeError):
        manager.restore(0)
      with self.assertRaises(FileNotFoundError):
        manager.restore(
            1,
            args=get_composite_restore_args(
                self.pytree, checkpoint_args=args_lib.PyTreeRestore
            ),
        )
      with self.assertRaises(TypeError):
        manager.restore(1)

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

    manager.save(
        0, args=get_composite_save_args(self.pytree, args_lib.PyTreeSave)
    )
    manager.wait_until_finished()

    restored = manager.restore(
        0, args=get_composite_restore_args(restore_args=restore_args)
    )
    test_utils.assert_tree_equal(self, pytree, restored.state)


class ReplicatorCheckpointManagerInternalTest(absltest.TestCase):

  def _make_manager(self):
    manager = rcm_lib.ReplicatorCheckpointManager.__new__(
        rcm_lib.ReplicatorCheckpointManager
    )
    device = jax.devices()[0]
    manager._global_mesh = jax.sharding.Mesh(np.array([device]), ('d',))
    manager._persistent_checkpoint_manager = None
    manager._process_metadata_handler = mock.Mock()
    manager._impl = mock.Mock()
    manager._colocated_controller = None
    manager._active_processes = None
    manager._local_handler_registry = None
    manager._get_distributed_to_device_ids = mock.Mock(return_value=[[0]])
    manager._global_mesh = jax.sharding.Mesh(np.array([device]), ('d',))
    manager._local_directory = epath.Path('/tmp/local')
    manager._impl.directory = manager._local_directory
    manager._step_name_format = step_lib.standard_name_format()
    manager._state_handler = mock.Mock()
    return manager

  def test_standard_save_writes_stable_process_metadata(self):
    manager = self._make_manager()
    manager._impl.directory = epath.Path('/tmp/local')
    manager._impl.save.return_value = True
    args = args_lib.Composite(state=args_lib.PyTreeSave({'x': 1}))

    saved = manager._standard_save(7, args, force=False)

    self.assertTrue(saved)
    manager._impl.save.assert_called_once()
    save_args = manager._impl.save.call_args.kwargs['args']
    self.assertIn('process_metadata', save_args.keys())
    manager._process_metadata_handler.save.assert_called_once_with(
        epath.Path('/tmp/local/process_metadata'),
        process_metadata_checkpoint_handler.ProcessMetadataSaveArgs(
            global_mesh=manager._global_mesh,
        ),
    )

  def test_standard_restore_prefers_stable_process_metadata(self):
    manager = self._make_manager()
    manager._impl.directory = epath.Path('/tmp/local')
    stable_metadata = ([[0]], [0])
    manager._process_metadata_handler.restore.return_value = stable_metadata
    restore_args = args_lib.Composite(
        state=args_lib.PyTreeRestore(
            restore_args={'x': mock.Mock(sharding=mock.Mock())}
        )
    )
    manager._get_mesh_consistent_args = mock.Mock(
        return_value=(restore_args, restore_args)
    )
    restored = args_lib.Composite(state={'x': 1})
    manager._impl.restore.return_value = restored
    manager._get_mesh_consistent_result = mock.Mock(return_value=restored)

    result = manager._standard_restore(3, restore_args)

    self.assertEqual(result, restored)
    manager._process_metadata_handler.restore.assert_called_once()
    manager._impl.restore.assert_called_once_with(3, args=restore_args)

  def test_standard_restore_falls_back_to_step_process_metadata(self):
    manager = self._make_manager()
    manager._impl.directory = epath.Path('/tmp/local')
    manager._process_metadata_handler.restore.side_effect = FileNotFoundError
    restore_args = args_lib.Composite(
        state=args_lib.PyTreeRestore(
            restore_args={'x': mock.Mock(sharding=mock.Mock())}
        )
    )
    manager._get_mesh_consistent_args = mock.Mock(
        return_value=(restore_args, restore_args)
    )
    manager._get_mesh_consistent_result = mock.Mock(return_value='restored')
    manager._impl.restore.side_effect = [
        args_lib.Composite(process_metadata=([[0]], [0])),
        args_lib.Composite(state={'x': 1}),
    ]

    result = manager._standard_restore(5, restore_args)

    self.assertEqual(result, 'restored')
    self.assertEqual(manager._impl.restore.call_count, 2)
    first_restore_args = manager._impl.restore.call_args_list[0].kwargs['args']
    self.assertEqual(tuple(first_restore_args.keys()), ('process_metadata',))

  def test_standard_restore_restores_requested_step_only(self):
    manager = self._make_manager()
    restore_args = args_lib.Composite(
        state=args_lib.PyTreeRestore(restore_args={'x': mock.Mock()})
    )
    manager._validate_and_standardize_args = mock.Mock(
        return_value=restore_args
    )
    manager._standard_restore_single_step = mock.Mock(
        return_value=args_lib.Composite(state={'x': 1})
    )

    result = manager._standard_restore(209, restore_args)

    self.assertEqual(result, args_lib.Composite(state={'x': 1}))
    self.assertEqual(
        manager._standard_restore_single_step.call_args.args[0], 209
    )

  def test_mesh_consistent_args_accepts_single_device_sharding(self):
    manager = self._make_manager()
    device = jax.devices()[0]
    arg = type_handlers.ArrayRestoreArgs(
        sharding=jax.sharding.SingleDeviceSharding(device),
        global_shape=(1,),
        dtype=jnp.float32,
    )

    _, consistent_args = manager._get_mesh_consistent_args(
        previous_distributed_to_device_ids=[[0]],
        previous_device_ids=[0],
        args=args_lib.Composite(
            state=args_lib.PyTreeRestore(
                restore_args={'x': arg},
            )
        ),
    )

    self.assertIs(
        consistent_args['state'].restore_args['x'].sharding,
        arg.sharding,
    )

  def test_validate_args_rejects_unsupported_items_in_colocated_mode(self):
    manager = self._make_manager()
    manager._colocated_controller = mock.Mock()
    args = args_lib.Composite(
        state=args_lib.PyTreeSave({'x': 1}),
        extra=args_lib.PyTreeSave({'y': 2}),
    )

    with self.assertRaisesRegex(
        ValueError, 'colocated mode only supports the following items'
    ):
      manager._validate_and_standardize_args(args)

  def test_local_checkpoint_handler_matches_standard_emergency_state_handler(
      self,
  ):
    multiprocessing_options = checkpoint_manager.MultiprocessingOptions(
        primary_host=None
    )

    mtc_state_handler, process_metadata_handler = (
        rcm_lib._local_checkpoint_handler(multiprocessing_options)
    )
    standard_state_handler = (
        emergency_checkpoint_manager._local_checkpoint_handler(
            multiprocessing_options
        )
    )

    self.assertEqual(
        mtc_state_handler._use_ocdbt, standard_state_handler._use_ocdbt
    )
    self.assertEqual(
        mtc_state_handler._use_zarr3, standard_state_handler._use_zarr3
    )
    self.assertEqual(
        mtc_state_handler._type_handler_registry.get(jax.Array).__class__,
        standard_state_handler._type_handler_registry.get(jax.Array).__class__,
    )
    mtc_array_handler = mtc_state_handler._type_handler_registry.get(jax.Array)
    standard_array_handler = standard_state_handler._type_handler_registry.get(
        jax.Array
    )
    self.assertEqual(
        mtc_array_handler._primary_host, standard_array_handler._primary_host
    )
    self.assertEqual(
        mtc_array_handler._replica_id, standard_array_handler._replica_id
    )
    self.assertEqual(
        mtc_array_handler._use_replica_parallel,
        standard_array_handler._use_replica_parallel,
    )
    self.assertIsInstance(
        process_metadata_handler,
        process_metadata_checkpoint_handler.ProcessMetadataCheckpointHandler,
    )


if __name__ == '__main__':
  multiprocess_test.main()
