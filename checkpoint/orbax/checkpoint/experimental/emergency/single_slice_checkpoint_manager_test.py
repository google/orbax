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

"""Tests for emergency.CheckpointManager in a single slice.

It can be problematic to test the single-slice version alongside the multislice
version, because the classes do different things on different "slices" of hosts,
which causes barrier names to get out of sync. This would not arise in a real
context, where the number slices/processes is fixed within a training run
(excluding things like preemption/restart).
"""

# pylint: disable=protected-access

from absl.testing import flagsaver
from absl.testing import parameterized
from etils import epath
import jax
import numpy as np
import optax
from orbax.checkpoint import args as args_lib
from orbax.checkpoint import test_utils
from orbax.checkpoint import utils
from orbax.checkpoint._src.handlers import pytree_checkpoint_handler
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.testing import multiprocess_test
from orbax.checkpoint.experimental.emergency import checkpoint_manager

PyTreeSaveArgs = pytree_checkpoint_handler.PyTreeSaveArgs

CheckpointManager = checkpoint_manager.CheckpointManager
CheckpointManagerOptions = checkpoint_manager.CheckpointManagerOptions
LocalCheckpointOptions = checkpoint_manager.LocalCheckpointOptions
PersistentCheckpointOptions = checkpoint_manager.PersistentCheckpointOptions
barrier_compatible_test = test_utils.barrier_compatible_test


@test_utils.barrier_compatible_test
class CheckpointManagerTest(
    parameterized.TestCase, multiprocess_test.MultiProcessTest
):

  def setUp(self):
    super().setUp()
    self.enter_context(
        flagsaver.flagsaver(experimental_orbax_use_distributed_process_id=True)
    )
    if not multihost.is_runtime_to_distributed_ids_initialized():
      multihost.initialize_runtime_to_distributed_ids()
      multihost.initialize_distributed_to_device_ids()

    self.local_directory = epath.Path(
        self.create_tempdir(
            name=f'local_checkpointing_test_pid{multihost.process_index()}'
        ).full_path
    )
    self.persistent_directory = epath.Path(
        self.create_tempdir(name='persistent_checkpointing_test').full_path
    )

    self.assertEqual(jax.device_count(), 8)
    self.assertEqual(jax.process_count(), 4)
    self.assertEqual(jax.local_device_count(), 2)

    test_utils.set_tensorstore_driver_for_test()
    test_utils.sync_global_processes('CheckpointManagerTest:setup_complete')

  def tearDown(self):
    super().tearDown()
    test_utils.sync_global_processes('CheckpointManagerTest:teardown_complete')

  def setup_pytree(self, global_mesh: jax.sharding.Mesh):
    """Create mesh and pytree."""
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
        'empty_dict': {},
        'empty_node': optax.EmptyState(),
    }
    return global_mesh, pytree

  @parameterized.parameters(
      ((8,), 0, ('data',)),
      ((8,), 1, ('data',)),
      ((1, 8), 0, ('replica', 'data')),
      ((8, 1), 1, ('data', 'replica')),
  )
  def test_save_restore_single_slice(
      self, mesh_shape, replica_axis_index, axis_names
  ):
    """Test single-slice save/restore."""
    # 1-d mesh
    global_mesh = jax.sharding.Mesh(
        np.asarray(jax.devices()).reshape(mesh_shape), axis_names
    )
    global_mesh, pytree = self.setup_pytree(global_mesh)
    abstract_state = jax.tree.map(utils.to_shape_dtype_struct, pytree)
    options = CheckpointManagerOptions(
        local=LocalCheckpointOptions(save_interval_steps=3, max_to_keep=2),
        persistent=PersistentCheckpointOptions(
            save_interval_steps=5, max_to_keep=3
        ),
        replica_axis_index=replica_axis_index,
    )
    manager = CheckpointManager(
        local_directory=self.local_directory,
        persistent_directory=self.persistent_directory,
        global_mesh=global_mesh,
        abstract_state=abstract_state,
        options=options,
    )
    self.assertEqual(1, manager._slice_count)
    self.assertTrue(manager.in_primary_slice)

    for i in range(17):
      manager.save(i, args=args_lib.Composite(state=PyTreeSaveArgs(pytree)))
    manager.wait_until_finished()

    self.assertEqual(manager.all_steps(), [5, 10, 15])
    test_utils.assert_tree_equal(self, pytree, manager.restore(None).state)


if __name__ == '__main__':
  multiprocess_test.main()
