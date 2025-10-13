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

from typing import Sequence

from absl import flags
from absl.testing import parameterized
from etils import epath
from flax.training import train_state
import jax
from orbax.checkpoint import args
from orbax.checkpoint import checkpoint_manager
from orbax.checkpoint import handlers
from orbax.checkpoint import test_utils
from orbax.checkpoint import utils
from orbax.checkpoint._src.arrays import types as arrays_types
from orbax.checkpoint._src.checkpointers import async_checkpointer
from orbax.checkpoint._src.checkpointers import checkpointer
from orbax.checkpoint._src.handlers import pytree_checkpoint_handler
from orbax.checkpoint._src.handlers import standard_checkpoint_handler
from orbax.checkpoint._src.metadata import checkpoint as metadata_lib
from orbax.checkpoint._src.metadata import sharding as sharding_metadata
from orbax.checkpoint._src.metadata import step_metadata_serialization
from orbax.checkpoint._src.metadata import value as value_metadata
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.path import step as step_lib
from orbax.checkpoint._src.serialization import type_handlers
from orbax.checkpoint._src.testing import multiprocess_test


FLAGS = flags.FLAGS
register_with_handler = args.register_with_handler
TrainState = train_state.TrainState
Mesh = jax.sharding.Mesh
PartitionSpec = jax.sharding.PartitionSpec
NamedSharding = jax.sharding.NamedSharding
NamedShardingMetadata = sharding_metadata.NamedShardingMetadata
PyTreeCheckpointHandler = pytree_checkpoint_handler.PyTreeCheckpointHandler
ArrayCheckpointHandler = handlers.ArrayCheckpointHandler
RestoreArgs = pytree_checkpoint_handler.RestoreArgs
ArrayRestoreArgs = pytree_checkpoint_handler.ArrayRestoreArgs
AsyncCheckpointer = async_checkpointer.AsyncCheckpointer
Checkpointer = checkpointer.Checkpointer
CheckpointManager = checkpoint_manager.CheckpointManager
CheckpointManagerOptions = checkpoint_manager.CheckpointManagerOptions
FileOptions = checkpoint_manager.FileOptions
JsonCheckpointHandler = handlers.JsonCheckpointHandler
ArrayMetadata = value_metadata.ArrayMetadata
StandardCheckpointHandler = (
    standard_checkpoint_handler.StandardCheckpointHandler
)

_DEFAULT_ITEM_NAME = checkpoint_manager.DEFAULT_ITEM_NAME
PLACEHOLDER = type_handlers.PLACEHOLDER


# Args registration not included.
class MyPyTreeCheckpointHandler(PyTreeCheckpointHandler):
  pass


def build_storage_metadata(
    chunk_shape: arrays_types.Shape, is_array_metadata_store_enabled: bool
) -> value_metadata.StorageMetadata:
  return value_metadata.StorageMetadata(
      chunk_shape=chunk_shape,
      write_shape=chunk_shape if is_array_metadata_store_enabled else None,
  )


@test_utils.barrier_compatible_test
class CheckpointManagerTest(
    parameterized.TestCase, multiprocess_test.MultiProcessTest
):
  """Structure allows test to run as subclasses, not base class."""

  def setUp(self):
    jax.distributed.initialize()
    test_utils.sync_global_processes('CheckpointManagerTest:jax_init')
    if not multihost.is_pathways_backend():
      multiprocess_test.MultiProcessTest.setUp(self)
      if not multihost.is_runtime_to_distributed_ids_initialized():
        multihost.initialize_runtime_to_distributed_ids()
      self.assertEqual(jax.process_count(), 4)
      self.assertEqual(jax.local_device_count(), 1)
    else:
      # Pathways tests, skip MultiProcessTest.setUp()
      parameterized.TestCase.setUp(self)
      self.assertEqual(jax.process_count(), 1)
      self.assertEqual(jax.local_device_count(), 8)

    self.assertEqual(jax.device_count(), 4)
    pytree, mesh_tree, axes_tree = test_utils.setup_sharded_pytree()
    doubled_pytree = test_utils.apply_function(pytree, lambda x: x * 2)

    self.empty_pytree = jax.tree.map(
        lambda x: object(), pytree, is_leaf=test_utils.is_leaf
    )
    self.pytree = pytree
    self.doubled_pytree = doubled_pytree
    self.mesh_tree = mesh_tree
    self.axes_tree = axes_tree
    self.pytree_restore_args = jax.tree.map(
        lambda mesh, axes: ArrayRestoreArgs(mesh=mesh, mesh_axes=axes),
        self.mesh_tree,
        self.axes_tree,
    )
    self.directory = epath.Path(
        self.create_tempdir(name='checkpoint_manager_test').full_path
    )
    self.secondary_directory = epath.Path(
        self.create_tempdir(name='checkpoint_manager_test_secondary').full_path
    )
    test_utils.set_tensorstore_driver_for_test()

    test_utils.sync_global_processes('CheckpointManagerTest:setup_complete')

  def tearDown(self):
    test_utils.sync_global_processes('CheckpointManagerTest:tests_complete')
    super().tearDown()

  def save_params(self, step, manager, params, metrics=None, force=False):
    return manager.save(
        step,
        args=args.Composite(params=args.PyTreeSave(params)),
        metrics=metrics,
        force=force,
    )

  def restore_params(self, step, manager, restore_args=None):
    restore_args = restore_args or self.pytree_restore_args
    return manager.restore(
        step,
        args=args.Composite(
            params=args.PyTreeRestore(restore_args=restore_args)
        ),
    ).params

  def wait_if_async(self, manager):
    manager.wait_until_finished()  # no-op if no async checkpointers.

  def assert_renamed_subdirs(
      self,
      directory: epath.Path,
      todelete_subdir: str,
      all_steps: Sequence[int],
      remaining_steps: Sequence[int],
  ):
    self.assertSameElements(remaining_steps, utils.checkpoint_steps(directory))
    deleted_steps = set(all_steps) - set(remaining_steps)
    for d in deleted_steps:
      self.assertTrue((directory / todelete_subdir / str(d)).exists())

  def assert_checkpoint_metadata(
      self,
      *,
      root: epath.Path,
      step_name_format: step_lib.NameFormat[step_lib.Metadata],
      step: int,
      assert_uncommitted: bool = True,
      assert_committed: bool = True,
      full_metadata: bool = True,
  ):
    metadata_store = metadata_lib.metadata_store(enable_write=False)
    if assert_uncommitted:
      tmp_paths = step_lib.all_temporary_paths(root)
      self.assertNotEmpty(tmp_paths)
      for p in tmp_paths:
        path = p.get()
        metadata_dict = metadata_store.read(
            file_path=metadata_lib.step_metadata_file_path(path)
        )
        step_metadata = step_metadata_serialization.deserialize(metadata_dict)
        self.assertIsNotNone(step_metadata)
        self.assertGreater(step_metadata.init_timestamp_nsecs, 0)
        self.assertIsNone(step_metadata.commit_timestamp_nsecs)
        if full_metadata:
          self.assertIsNotNone(step_metadata.metrics)
          self.assertIsNotNone(step_metadata.performance_metrics)
          self.assertIsNotNone(step_metadata.custom_metadata)
    if assert_committed:
      step_metadata: step_lib.Metadata = step_name_format.find_step(
          root, step=step
      )
      self.assertGreater(step_metadata.commit_timestamp_nsecs, 0)
      metadata_dict = metadata_store.read(
          file_path=metadata_lib.step_metadata_file_path(step_metadata.path)
      )
      checkpoint_step_metadata = step_metadata_serialization.deserialize(
          metadata_dict
      )
      self.assertIsNotNone(checkpoint_step_metadata)
      self.assertGreater(checkpoint_step_metadata.init_timestamp_nsecs, 0)
      self.assertGreater(checkpoint_step_metadata.commit_timestamp_nsecs, 0)
      if full_metadata:
        self.assertIsNotNone(checkpoint_step_metadata.metrics)
        self.assertIsNotNone(checkpoint_step_metadata.performance_metrics)
        self.assertIsNotNone(checkpoint_step_metadata.custom_metadata)

  def assert_directory_mode_equal(self, directory: epath.Path, mode: int):
    directory_mode = (
        directory.stat().mode - 0o040000
    )  # 04 is directory file type.
    if directory_mode != mode:
      self.fail(f'Directory mode {directory_mode:o} != {mode:o}')

  @parameterized.parameters((False, 8))
  def test_save_restore(self, enable_async, step_format_fixed_length):
    options = CheckpointManagerOptions(
        enable_async_checkpointing=enable_async,
        step_format_fixed_length=step_format_fixed_length,
        async_options=checkpoint_manager.AsyncOptions(timeout_secs=500),
    )
    with CheckpointManager(
        self.directory,
        item_names=('params',),
        options=options,
    ) as manager:
      self.assertTrue(self.save_params(0, manager, self.pytree))
      self.wait_if_async(manager)
      restored = self.restore_params(0, manager)
      test_utils.assert_tree_equal(self, self.pytree, restored)
      if enable_async:
        self.assertIsInstance(manager._checkpointer, AsyncCheckpointer)
        if isinstance(manager._checkpointer, AsyncCheckpointer):
          self.assertEqual(
              manager._checkpointer._async_manager._timeout_secs, 500
          )


if __name__ == '__main__':
  multiprocess_test.main()
