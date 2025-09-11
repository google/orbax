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

import time

from absl import flags
from absl.testing import flagsaver
from absl.testing import parameterized
from etils import epath
import jax
import numpy as np
from orbax.checkpoint import args
from orbax.checkpoint import checkpoint_manager
from orbax.checkpoint import checkpoint_utils
from orbax.checkpoint import test_utils
from orbax.checkpoint import utils
from orbax.checkpoint._src.handlers import handler_registration
from orbax.checkpoint._src.handlers import pytree_checkpoint_handler
from orbax.checkpoint._src.metadata import array_metadata_store as array_metadata_store_lib
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.serialization import type_handlers
from orbax.checkpoint._src.testing import multiprocess_test


FLAGS = flags.FLAGS
PyTreeCheckpointHandler = pytree_checkpoint_handler.PyTreeCheckpointHandler
CheckpointManager = checkpoint_manager.CheckpointManager
CheckpointManagerOptions = checkpoint_manager.CheckpointManagerOptions


@test_utils.barrier_compatible_test
class CheckpointManagerSliceTest(
    parameterized.TestCase, multiprocess_test.MultiProcessTest
):
  """Structure allows test to run as subclasses, not base class."""

  def setUp(self):
    super().setUp()

    if not multihost.is_runtime_to_distributed_ids_initialized():
      multihost.initialize_runtime_to_distributed_ids()

    self.assertEqual(jax.device_count(), 8)
    self.assertEqual(jax.process_count(), 4)
    self.assertEqual(jax.local_device_count(), 2)

    self.directory = epath.Path(
        self.create_tempdir(name='checkpoint_manager_slice_test').full_path
    )
    test_utils.set_tensorstore_driver_for_test()

    test_utils.sync_global_processes(
        'CheckpointManagerSliceTest:setup_complete'
    )

  def tearDown(self):
    test_utils.sync_global_processes(
        'CheckpointManagerSliceTest:tests_complete'
    )
    super().tearDown()

  def wait_if_async(self, manager):
    manager.wait_until_finished()  # no-op if no async checkpointers.

  @parameterized.product(
      enable_async_checkpointing=(False, True),
      array_metadata_store=(None, array_metadata_store_lib.Store()),
  )
  def test_slice(
      self,
      enable_async_checkpointing: bool,
      array_metadata_store: array_metadata_store_lib.Store | None,
  ):
    """Test slice."""
    self.enter_context(
        flagsaver.flagsaver(experimental_orbax_use_distributed_process_id=True)
    )
    global_mesh = test_utils.get_fake_global_mesh_for_slices([{0, 1}, {2, 3}])

    mesh_axes = jax.sharding.PartitionSpec('data')
    arrays = [
        test_utils.create_sharded_array(arr, global_mesh, mesh_axes)
        for arr in [np.arange(8), np.arange(16)]
    ]
    assert len(global_mesh.devices[0]) == 4
    assert jax.process_count() == 4
    active_processes = {0, 1}
    primary_host = 0
    if multihost.process_index() in active_processes:
      single_slice_arrays = test_utils.select_single_replica(
          arrays, global_mesh
      )
      options = CheckpointManagerOptions(
          create=False,
          enable_async_checkpointing=enable_async_checkpointing,
          multiprocessing_options=checkpoint_manager.MultiprocessingOptions(
              primary_host=primary_host,
              active_processes=active_processes,
          ),
      )
      type_handler_registry = type_handlers.create_type_handler_registry(
          (
              jax.Array,
              type_handlers.ArrayHandler(
                  primary_host=None,
                  replica_id=None,
                  use_replica_parallel=False,
                  array_metadata_store=array_metadata_store,
              ),
          ),
      )
      handler = PyTreeCheckpointHandler(
          multiprocessing_options=options.multiprocessing_options,
          type_handler_registry=type_handler_registry,
      )
      registry = handler_registration.DefaultCheckpointHandlerRegistry()
      registry.add(None, args.PyTreeSave, handler)
      registry.add(None, args.PyTreeRestore, handler)
      with CheckpointManager(
          self.directory,
          options=options,
          handler_registry=registry,
      ) as manager:
        self.assertTrue(manager.save(0, args=args.PyTreeSave(arrays)))
        time.sleep(10)
        self.wait_if_async(manager)
        abstract_target = jax.tree.map(
            utils.to_shape_dtype_struct, single_slice_arrays
        )
        restore_args = checkpoint_utils.construct_restore_args(abstract_target)
        restored = manager.restore(
            0, args=args.PyTreeRestore(restore_args=restore_args)
        )
        test_utils.assert_tree_equal(self, single_slice_arrays, restored)


if __name__ == '__main__':
  multiprocess_test.main()
