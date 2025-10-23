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

import ast
from concurrent import futures
import datetime
import os
import time
import typing
from typing import Optional, Sequence
from unittest import mock

from absl import flags
from absl import logging
from absl.testing import parameterized
from etils import epath
from flax import linen as nn
from flax.training import train_state
import jax
from jax.experimental import multihost_utils
from jax.experimental import pjit
import jax.numpy as jnp
import numpy as np
import optax
from orbax.checkpoint import args
from orbax.checkpoint import checkpoint_manager
from orbax.checkpoint import handlers
from orbax.checkpoint import test_utils
from orbax.checkpoint import utils
from orbax.checkpoint._src.arrays import types as arrays_types
from orbax.checkpoint._src.checkpoint_managers import preservation_policy as preservation_policy_lib
from orbax.checkpoint._src.checkpoint_managers import save_decision_policy as save_decision_policy_lib
from orbax.checkpoint._src.checkpointers import async_checkpointer
from orbax.checkpoint._src.checkpointers import checkpointer
from orbax.checkpoint._src.handlers import base_pytree_checkpoint_handler
from orbax.checkpoint._src.handlers import handler_registration
from orbax.checkpoint._src.handlers import pytree_checkpoint_handler
from orbax.checkpoint._src.handlers import standard_checkpoint_handler
from orbax.checkpoint._src.logging import composite_logger
from orbax.checkpoint._src.logging import standard_logger
from orbax.checkpoint._src.logging import step_statistics as step_stats
from orbax.checkpoint._src.metadata import array_metadata_store as array_metadata_store_lib
from orbax.checkpoint._src.metadata import checkpoint as metadata_lib
from orbax.checkpoint._src.metadata import root_metadata_serialization
from orbax.checkpoint._src.metadata import sharding as sharding_metadata
from orbax.checkpoint._src.metadata import step_metadata_serialization
from orbax.checkpoint._src.metadata import value as value_metadata
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.path import atomicity
from orbax.checkpoint._src.path import gcs_utils
from orbax.checkpoint._src.path import step as step_lib
from orbax.checkpoint._src.serialization import type_handler_registry
from orbax.checkpoint._src.serialization import type_handlers
from orbax.checkpoint._src.testing import multiprocess_test
from orbax.google.proto import descriptor_pb2


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
    if not multihost.is_pathways_backend():
      multiprocess_test.MultiProcessTest.setUp(self)
      if not multihost.is_runtime_to_distributed_ids_initialized():
        multihost.initialize_runtime_to_distributed_ids()
      self.assertEqual(jax.process_count(), 4)
      # self.assertEqual(jax.local_device_count(), 2)
    else:
      # Pathways tests, skip MultiProcessTest.setUp()
      parameterized.TestCase.setUp(self)
      self.assertEqual(jax.process_count(), 1)
      self.assertEqual(jax.local_device_count(), 8)

    # self.assertEqual(jax.device_count(), 8)
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

  @parameterized.parameters((False, 8), (False, None), (True, None))
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

  @parameterized.parameters((False,), (True,))
  def test_save_restore_with_concurrent_wait_until_finished(self, enable_async):
    """Tests no deadlocks when calling wait_until_finished concurrently."""
    num_save_waiter_threads = 10
    async_timeout_secs = 10
    options = CheckpointManagerOptions(
        enable_async_checkpointing=enable_async,
        async_options=checkpoint_manager.AsyncOptions(
            timeout_secs=async_timeout_secs
        ),
    )

    with (
        CheckpointManager(
            self.directory,
            item_names=('params',),
            options=options,
        ) as manager,
        futures.ThreadPoolExecutor(
            max_workers=num_save_waiter_threads,
            thread_name_prefix='save_waiter',
        ) as executor,
    ):
      self.assertTrue(self.save_params(0, manager, self.pytree))

      save_waiter_futures = []
      for _ in range(num_save_waiter_threads):
        # Call wait_until_finished concurrently from multiple threads.
        save_waiter_futures.append(executor.submit(manager.wait_until_finished))
      # Call wait_until_finished from the main thread to also make sure that
      # the main thread is not deadlocked.
      manager.wait_until_finished()  # wait_until_finished from Main thread.
      futures.wait(save_waiter_futures, timeout=5)  # Wait for all threads

      restored = self.restore_params(0, manager)
      test_utils.assert_tree_equal(self, self.pytree, restored)

  @parameterized.parameters((False, 8), (False, None), (True, None))
  def test_save_restore_with_descriptor(
      self, enable_async, step_format_fixed_length
  ):
    options = CheckpointManagerOptions(
        enable_async_checkpointing=enable_async,
        step_format_fixed_length=step_format_fixed_length,
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

      params_subdir = utils.get_save_directory(
          0,
          manager.directory,
          'params',
          step_format_fixed_length=step_format_fixed_length,
      )
      self.assertTrue((params_subdir / 'descriptor').exists())

      proto_handler = handlers.ProtoCheckpointHandler(
          filename=base_pytree_checkpoint_handler._DESCRIPTOR_FILENAME,
      )
      restored_descriptor = proto_handler.restore(
          params_subdir / base_pytree_checkpoint_handler._DESCRIPTOR_FOLDER,
          args=args.ProtoRestore(descriptor_pb2.Descriptor),
      )
      self.assertNotEmpty(restored_descriptor.uuid)
      uuid_filename = f'uuid-{restored_descriptor.uuid}'
      self.assertTrue((params_subdir / 'descriptor' / uuid_filename).exists())

  @parameterized.parameters(
      (None, 8), (None, None), ('checkpoint', 8), ('checkpoint', None)
  )
  def test_file_format(self, prefix, step_format_fixed_length):
    options = CheckpointManagerOptions(
        step_prefix=prefix, step_format_fixed_length=step_format_fixed_length
    )
    with CheckpointManager(
        self.directory,
        item_names=('params',),
        options=options,
    ) as manager:
      self.assertTrue(self.save_params(0, manager, self.pytree))
      self.wait_if_async(manager)
      step_str = '00000000' if step_format_fixed_length else '0'
      prefix_str = prefix + '_' if prefix else ''
      self.assertTrue(
          (self.directory / f'{prefix_str}{step_str}' / 'params').exists()
      )

  @parameterized.parameters((0o750,), (0o775,))
  def test_path_permissions(self, mode):
    old_umask = os.umask(0)
    try:
      with CheckpointManager(
          self.directory,
          item_names=('params',),
          options=CheckpointManagerOptions(
              file_options=FileOptions(path_permission_mode=mode)
          ),
      ) as manager:
        self.assertTrue(self.save_params(0, manager, self.pytree))
        self.wait_if_async(manager)
        step_directory = utils.get_save_directory(0, manager.directory)
        self.assert_directory_mode_equal(step_directory, mode)
        self.assert_directory_mode_equal(step_directory / 'params', mode)
    finally:
      os.umask(old_umask)

  def test_save_restore_no_kwargs(self):
    with CheckpointManager(self.directory, item_names=('params',)) as manager:
      expected = jax.tree.map(test_utils.replicate_sharded_array, self.pytree)
      expected = jax.tree.map(
          lambda x: np.asarray(x.addressable_data(0)), expected
      )
      self.assertTrue(self.save_params(0, manager, expected))
      self.wait_if_async(manager)
      restored = manager.restore(0)['params']
      test_utils.assert_tree_equal(self, expected, restored)

  def test_save_restore_invalid_item(self):
    with CheckpointManager(self.directory, item_names=('params',)) as manager:
      with self.assertRaisesRegex(
          ValueError, 'does not match with any registered handler'
      ):
        manager.save(
            0, args=args.Composite(invalid=args.PyTreeSave(self.pytree))
        )
      with self.assertRaises(FileNotFoundError):
        manager.restore(
            0,
            args=args.Composite(invalid=args.PyTreeRestore(self.pytree)),
        )

  def test_new_and_legacy_api_confusion(self):
    with CheckpointManager(self.directory, item_names=('params',)) as manager:
      self.assertTrue(
          manager.save(
              0,
              args=args.Composite(params=args.PyTreeSave(self.pytree)),
          )
      )
      self.wait_if_async(manager)
      with self.assertRaises(ValueError):
        manager.save(1, args.Composite(params=args.PyTreeSave(self.pytree)))
      with self.assertRaises(ValueError):
        manager.restore(
            0,
            args.Composite(
                params=args.PyTreeRestore(restore_args=self.pytree_restore_args)
            ),
        )

  def test_legacy_api_timeout(self):
    ckptr = AsyncCheckpointer(PyTreeCheckpointHandler(), timeout_secs=500)
    with CheckpointManager(self.directory, ckptr) as manager:
      self.assertIsInstance(manager._checkpointer, AsyncCheckpointer)
      self.assertEqual(
          typing.cast(
              AsyncCheckpointer, manager._checkpointer
          )._async_manager._timeout_secs,
          500,
      )

    with CheckpointManager(
        self.directory,
        {
            'params': AsyncCheckpointer(
                PyTreeCheckpointHandler(), timeout_secs=700
            ),
            'state': AsyncCheckpointer(
                PyTreeCheckpointHandler(), timeout_secs=800
            ),
        },
    ) as manager:
      self.assertIsInstance(manager._checkpointer, AsyncCheckpointer)
      self.assertEqual(
          typing.cast(
              AsyncCheckpointer, manager._checkpointer
          )._async_manager._timeout_secs,
          800,
      )

    with CheckpointManager(
        self.directory,
        {
            'params': AsyncCheckpointer(PyTreeCheckpointHandler()),
            'state': AsyncCheckpointer(PyTreeCheckpointHandler()),
        },
    ) as manager:
      self.assertIsInstance(manager._checkpointer, AsyncCheckpointer)
      self.assertEqual(
          typing.cast(
              AsyncCheckpointer, manager._checkpointer
          )._async_manager._timeout_secs,
          600,
      )

    with CheckpointManager(
        self.directory,
        {
            'params': AsyncCheckpointer(PyTreeCheckpointHandler()),
            'state': AsyncCheckpointer(
                PyTreeCheckpointHandler(), timeout_secs=600
            ),
        },
        options=CheckpointManagerOptions(
            async_options=checkpoint_manager.AsyncOptions(timeout_secs=500)
        ),
    ) as manager:
      self.assertIsInstance(manager._checkpointer, AsyncCheckpointer)
      self.assertEqual(
          typing.cast(
              AsyncCheckpointer, manager._checkpointer
          )._async_manager._timeout_secs,
          500,
      )

  def test_incorrect_single_usage(self):
    with CheckpointManager(self.directory) as manager:
      self.assertTrue(manager.save(0, args=args.PyTreeSave(self.pytree)))
      self.wait_if_async(manager)
      with self.assertRaises(ValueError):
        manager.save(
            1, args=args.Composite(params=args.PyTreeSave(self.pytree))
        )
      with self.assertRaises(ValueError):
        manager.restore(
            0,
            args=args.Composite(
                params=args.PyTreeRestore(restore_args=self.pytree_restore_args)
            ),
        )

  def test_incorrect_composite_usage(self):
    with CheckpointManager(self.directory, item_names=('params',)) as manager:
      self.assertTrue(
          manager.save(
              0,
              args=args.Composite(params=args.PyTreeSave(self.pytree)),
          )
      )
      self.wait_if_async(manager)
      with self.assertRaises(ValueError):
        manager.save(1, args=args.PyTreeSave(self.pytree))
      with self.assertRaises(ValueError):
        manager.restore(
            0,
            args=args.PyTreeRestore(restore_args=self.pytree_restore_args),
        )

  @parameterized.parameters((False, 8), (False, None), (True, None))
  def test_all_steps(self, enable_async, step_format_fixed_length):
    options = CheckpointManagerOptions(
        enable_async_checkpointing=enable_async,
        step_format_fixed_length=step_format_fixed_length,
    )
    with CheckpointManager(
        self.directory,
        item_names=('params',),
        options=options,
    ) as manager:
      num_steps = 5
      for step in range(num_steps):
        self.assertTrue(self.save_params(step, manager, self.pytree))
      self.wait_if_async(manager)

      test_utils.save_fake_tmp_dir(self.directory, num_steps, 'params')

      # Does not include step num_steps.
      self.assertSameElements(range(num_steps), manager.all_steps())
      self.assertEqual(manager.latest_step(), num_steps - 1)

  def test_all_steps_reload(self):
    initial_num_steps = 2
    total_num_steps = 4
    manager = CheckpointManager(
        self.directory,
        item_names=('params',),
    )
    for step in range(initial_num_steps):
      self.assertTrue(self.save_params(step, manager, self.pytree))
    self.wait_if_async(manager)
    self.assertSameElements(range(initial_num_steps), manager.all_steps())

    new_manager = CheckpointManager(
        self.directory,
        item_names=('params',),
    )
    self.assertSameElements(range(initial_num_steps), new_manager.all_steps())
    for step in range(2, 4):
      self.assertTrue(self.save_params(step, new_manager, self.pytree))
    self.wait_if_async(new_manager)
    self.assertSameElements(range(total_num_steps), new_manager.all_steps())

    self.assertSameElements(range(initial_num_steps), manager.all_steps())
    manager.reload()
    self.assertSameElements(range(total_num_steps), manager.all_steps())

    manager.close()
    new_manager.close()

  @parameterized.parameters((False, 1), (True, 2))
  def test_latest_step(self, enable_async, save_interval_steps):
    options = CheckpointManagerOptions(
        enable_async_checkpointing=enable_async,
        save_interval_steps=save_interval_steps,
    )
    with CheckpointManager(
        self.directory,
        item_names=('params',),
        options=options,
    ) as manager:
      num_steps = 6
      for step in range(num_steps):
        if step % save_interval_steps == 0:
          self.assertTrue(self.save_params(step, manager, self.pytree))
        else:
          self.assertFalse(self.save_params(step, manager, self.pytree))
      self.wait_if_async(manager)
      self.assertEqual(manager.latest_step(), num_steps - save_interval_steps)

      self.assertTrue(self.save_params(num_steps, manager, self.pytree))
      self.wait_if_async(manager)
      self.assertEqual(manager.latest_step(), num_steps)

  def test_latest_step_restore(self):
    """Test case."""
    options = CheckpointManagerOptions(save_interval_steps=2)
    with CheckpointManager(
        self.directory,
        item_names=('params',),
        options=options,
    ) as manager:
      num_steps = 6
      for step in range(num_steps):
        self.save_params(step, manager, {'step': step, **self.pytree})
      self.wait_if_async(manager)
      expected_step = 4
      self.assertEqual(manager.latest_step(), expected_step)
      restored = self.restore_params(
          None,
          manager,
          {'step': RestoreArgs(restore_type=int), **self.pytree_restore_args},
      )
      test_utils.assert_tree_equal(
          self, {'step': expected_step, **self.pytree}, restored
      )

  def test_no_overwrite_existing(self):
    """Test same step does not overwrite."""
    with CheckpointManager(self.directory, item_names=('params',)) as manager:
      self.assertTrue(self.save_params(0, manager, self.pytree))
      with self.assertRaises(ValueError):
        self.save_params(0, manager, self.doubled_pytree, force=True)
      self.wait_if_async(manager)
      restored = self.restore_params(0, manager)
      expected = self.pytree
      test_utils.assert_tree_equal(self, expected, restored)

  @parameterized.product(
      enable_async=[False, True],
      todelete_subdir=[None, 'ttl=1h'],
      enable_background_delete=[False, True],
  )
  def test_removes_old_saves(
      self, enable_async, todelete_subdir, enable_background_delete
  ):
    """Test old saves get removed."""
    options = CheckpointManagerOptions(
        enable_async_checkpointing=enable_async,
        max_to_keep=2,
        todelete_subdir=todelete_subdir,
        enable_background_delete=enable_background_delete,
    )
    with CheckpointManager(
        self.directory,
        item_names=('params',),
        options=options,
    ) as manager:

      for step in range(5):
        self.assertTrue(self.save_params(step, manager, self.pytree))
      self.wait_if_async(manager)
      self.assertSameElements([3, 4], manager.all_steps())
      manager.close()

    test_utils.sync_global_processes(f'test_removes_old_saves_{self.id()}')

    if todelete_subdir is not None:
      self.assert_renamed_subdirs(
          manager.directory,
          todelete_subdir,
          all_steps=range(5),
          remaining_steps=[3, 4],
      )

  @parameterized.parameters((None, Checkpointer), ('ttl=1h', AsyncCheckpointer))
  def test_max_to_keep_zero(self, todelete_subdir, ckptr):
    options = CheckpointManagerOptions(
        max_to_keep=0, todelete_subdir=todelete_subdir
    )
    with CheckpointManager(self.directory, options=options) as manager:
      for step in range(5):
        self.assertTrue(manager.save(step, args=args.PyTreeSave(self.pytree)))
        self.assertSameElements([], manager.all_steps())
      if ckptr is AsyncCheckpointer:
        manager.wait_until_finished()
        self.assertSameElements([], manager.all_steps())
      if todelete_subdir is not None:
        self.assert_renamed_subdirs(
            manager.directory,
            todelete_subdir,
            all_steps=range(5),
            remaining_steps=[],
        )

  @parameterized.parameters((None,), ('ttl=1h',))
  def test_max_to_keep_zero_other_conditions(self, todelete_subdir):
    tz = datetime.timezone.utc
    current_datetime = datetime.datetime.now(tz=tz)

    with mock.patch('datetime.datetime', autospec=True) as dt:
      options = CheckpointManagerOptions(
          max_to_keep=0,
          keep_period=3,
          keep_time_interval=datetime.timedelta(hours=5),
          todelete_subdir=todelete_subdir,
      )
      manager = CheckpointManager(
          self.directory,
          options=options,
      )
      for step in range(9):
        dt.now.return_value = current_datetime
        self.assertTrue(manager.save(step, args=args.PyTreeSave(self.pytree)))
        current_datetime += datetime.timedelta(hours=1)
      self.wait_if_async(manager)

    self.assertSameElements([0, 3, 5, 6], manager.all_steps())
    if todelete_subdir is not None:
      self.assert_renamed_subdirs(
          manager.directory,
          todelete_subdir,
          all_steps=range(9),
          remaining_steps=[0, 3, 5, 6],
      )

    manager.close()

  @parameterized.parameters(
      (False, None),
      (False, 'ttl=1h'),
      (True, None),
      (True, 'ttl=1h'),
  )
  def test_removes_old_saves_keep_period(self, enable_async, todelete_subdir):
    """Test old saves get removed."""
    options = CheckpointManagerOptions(
        enable_async_checkpointing=enable_async,
        max_to_keep=2,
        keep_period=4,
        save_interval_steps=2,
        todelete_subdir=todelete_subdir,
    )
    with CheckpointManager(
        self.directory,
        item_names=('params',),
        options=options,
    ) as manager:
      for step in range(12):
        if step % 2 == 0:
          self.assertTrue(self.save_params(step, manager, self.pytree))
      self.wait_if_async(manager)
      self.assertSameElements([0, 4, 8, 10], manager.all_steps())
      if todelete_subdir is not None:
        self.assert_renamed_subdirs(
            manager.directory,
            todelete_subdir,
            all_steps=[s for s in range(12) if s % 2 == 0],
            remaining_steps=[0, 4, 8, 10],
        )

  @parameterized.parameters((None,), ('ttl=1h',))
  def test_removes_old_saves_time_interval(self, todelete_subdir):
    tz = datetime.timezone.utc
    current_datetime = datetime.datetime.now(tz=tz)
    checkpoint_times = {}

    with mock.patch('datetime.datetime', autospec=True) as dt:
      options = CheckpointManagerOptions(
          max_to_keep=2,
          keep_time_interval=datetime.timedelta(hours=3),
          todelete_subdir=todelete_subdir,
      )
      manager = CheckpointManager(
          self.directory,
          item_names=('params',),
          options=options,
      )
      for step in range(10):
        dt.now.return_value = current_datetime
        self.assertTrue(self.save_params(step, manager, self.pytree))
        checkpoint_times[step] = current_datetime
        current_datetime += datetime.timedelta(hours=1)
      self.wait_if_async(manager)
      self.assertSameElements([0, 3, 6, 8, 9], manager.all_steps())
      if todelete_subdir is not None:
        self.assert_renamed_subdirs(
            manager.directory,
            todelete_subdir,
            all_steps=range(10),
            remaining_steps=[0, 3, 6, 8, 9],
        )

    # simulate restart
    new_manager = CheckpointManager(
        self.directory,
        item_names=('params',),
        options=options,
    )

    for checkpoint in new_manager._checkpoints:
      checkpoint.time = checkpoint_times[checkpoint.step]

    with mock.patch('datetime.datetime', autospec=True) as dt:
      for step in range(10, 20):
        dt.now.return_value = current_datetime
        self.assertTrue(self.save_params(step, new_manager, self.pytree))
        current_datetime += datetime.timedelta(hours=1)
      self.wait_if_async(new_manager)
      self.assertSameElements(
          [0, 3, 6, 9, 12, 15, 18, 19], new_manager.all_steps()
      )
      if todelete_subdir is not None:
        self.assert_renamed_subdirs(
            new_manager.directory,
            todelete_subdir,
            all_steps=range(20),
            remaining_steps=[0, 3, 6, 9, 12, 15, 18, 19],
        )

    manager.close()
    new_manager.close()

  @parameterized.parameters((None,), ('ttl=1h',))
  def test_removes_old_saves_time_interval_metrics(self, todelete_subdir):
    tz = datetime.timezone.utc
    current_datetime = datetime.datetime.now(tz=tz)
    steps = 10
    all_metrics = {'loss': list(range(steps))}

    with mock.patch('datetime.datetime', autospec=True) as dt:
      options = CheckpointManagerOptions(
          max_to_keep=3,
          best_mode='min',
          best_fn=lambda m: m['loss'],
          keep_time_interval=datetime.timedelta(hours=3),
          todelete_subdir=todelete_subdir,
      )
      manager = CheckpointManager(
          self.directory,
          item_names=('params',),
          options=options,
      )
      for step in range(steps):
        dt.now.return_value = current_datetime
        metrics = {k: v[step] for k, v in all_metrics.items()}
        self.assertTrue(
            manager.save(
                step,
                args=args.Composite(params=args.PyTreeSave(self.pytree)),
                metrics=metrics,
            )
        )
        current_datetime += datetime.timedelta(hours=1)
      self.wait_if_async(manager)

    # First three are kept because they are best, rest are kept because of time
    # interval.
    self.assertSameElements([0, 1, 2, 3, 6, 9], manager.all_steps())
    if todelete_subdir is not None:
      self.assert_renamed_subdirs(
          manager.directory,
          todelete_subdir,
          all_steps=range(steps),
          remaining_steps=[0, 1, 2, 3, 6, 9],
      )

    manager.close()

  def test_save_interval(self):
    """Test save interval > 1."""
    options = CheckpointManagerOptions(save_interval_steps=2)
    with CheckpointManager(
        self.directory,
        item_names=('params',),
        options=options,
    ) as manager:
      for step in range(6):
        saved = self.save_params(step, manager, self.pytree)
        if step % 2 == 0:
          self.assertTrue(saved)
        else:
          self.assertFalse(saved)
      self.wait_if_async(manager)
      self.assertSameElements([0, 2, 4], manager.all_steps())

  def test_save_on_steps(self):
    save_on_steps = frozenset({1, 3, 5})
    options = CheckpointManagerOptions(
        save_interval_steps=10000, save_on_steps=save_on_steps
    )
    with CheckpointManager(
        self.directory,
        item_names=('params',),
        options=options,
    ) as manager:
      for step in range(6):
        saved = self.save_params(step, manager, self.pytree)
        if step in {0} | save_on_steps:
          self.assertTrue(saved)
        else:
          self.assertFalse(saved)
      self.wait_if_async(manager)
      self.assertSameElements([0, 1, 3, 5], manager.all_steps())

  @parameterized.parameters((True,), (False,))
  def test_save_same_step(self, enable_async):
    """Test saving same step repeatedly."""
    options = CheckpointManagerOptions(enable_async_checkpointing=enable_async)
    with CheckpointManager(
        self.directory,
        item_names=('params',),
        options=options,
    ) as manager:
      # The below case tests an earlier bug where a dir is created, second save
      # is skipped, but leaves a dir present, third encounters error because tmp
      # dir still exists.
      step = 0
      self.assertTrue(self.save_params(step, manager, self.pytree, force=True))
      with self.assertRaises(checkpoint_manager.StepAlreadyExistsError):
        self.save_params(step, manager, self.pytree, force=True)
        self.save_params(step, manager, self.pytree, force=True)
      self.wait_if_async(manager)

      tmp_dir = (
          self.directory / str(step) / ('params' + '.orbax-checkpoint-tmp-*')
      )
      self.assertFalse(tmp_dir.exists())
      self.assertSameElements([0], manager.all_steps())

  def test_save_interval_force(self):
    """Test force option."""
    options = CheckpointManagerOptions(save_interval_steps=2)
    with CheckpointManager(
        self.directory,
        item_names=('params',),
        options=options,
    ) as manager:
      for step in range(6):
        saved = self.save_params(step, manager, self.pytree)
        if step % 2 == 0:
          self.assertTrue(saved)
        else:
          self.assertFalse(saved)
      self.wait_if_async(manager)
      self.assertTrue(self.save_params(5, manager, self.pytree, force=True))
      self.wait_if_async(manager)
      self.assertSameElements([0, 2, 4, 5], manager.all_steps())

  @parameterized.parameters(
      (False, None),
      (False, 'checkpoint'),
      (True, None),
  )
  def test_save_preempted(self, enable_async, prefix):
    """Simulate effects of preemption."""
    # Simulates the effects of preemption by creating a tmp directory and
    # ensuring it is cleaned up.
    tmp_dir = test_utils.save_fake_tmp_dir(
        self.directory, 0, 'params', subdirs=['subdir'], step_prefix=prefix
    )
    self.assertTrue(tmp_dir.exists())
    tmp_dir_items = list(tmp_dir.iterdir())
    self.assertLen(tmp_dir_items, 1)
    self.assertIn('subdir', tmp_dir_items[0].name)
    # Check for directory existence before initializing CheckpointManager, which
    # will clean up the above directories.
    test_utils.sync_global_processes('test_check_dirs')

    with CheckpointManager(
        self.directory,
        item_names=('params',),
        options=CheckpointManagerOptions(
            enable_async_checkpointing=enable_async,
            step_prefix=prefix,
            cleanup_tmp_directories=True,
        ),
    ) as manager:
      # Temp checkpoints cleaned up at creation.
      self.assertFalse(tmp_dir.exists())
      self.assertSameElements([], manager.all_steps())

      # Sync to check directories before a new tmp dir is created.
      test_utils.sync_global_processes('test_check_dirs_after_cleanup')

      tmp_dir = test_utils.save_fake_tmp_dir(
          self.directory, 0, 'params', subdirs=['subdir'], step_prefix=prefix
      )
      self.assertTrue(tmp_dir.exists())
      tmp_dir_items = list(tmp_dir.iterdir())
      self.assertLen(tmp_dir_items, 1)
      self.assertIn('subdir', tmp_dir_items[0].name)
      self.assertSameElements(
          [], manager.all_steps()
      )  # Only picks up finalized.

      # Do checks before tmp dirs are cleaned up by next save.
      test_utils.sync_global_processes('test_check_dirs_before_next_save')

      self.assertTrue(self.save_params(1, manager, self.pytree))
      self.wait_if_async(manager)

      self.assertSameElements([1], manager.all_steps())  # Step 0 not picked up.

  @parameterized.named_parameters(
      ('checkpointer_no_prefix', False, None, False),
      ('checkpointer_prefix', False, 'checkpoint', False),
      ('async_checkpointer_no_prefix', True, None, False),
      ('async_checkpointer_prefix', True, 'checkpoint', False),
      ('checkpointer_no_prefix_gcs', False, None, True),
      ('checkpointer_prefix_gcs', False, 'checkpoint', True),
      ('async_checkpointer_no_prefix_gcs', True, None, True),
      ('async_checkpointer_prefix_gcs', True, 'checkpoint', True),
  )
  def test_save_preempted_mock(self, enable_async, prefix, is_gcs):
    name_format = step_lib.standard_name_format(step_prefix=prefix)
    with mock.patch.object(
        gcs_utils, 'is_gcs_path', autospec=True, return_value=is_gcs
    ):
      with (
          mock.patch.object(atomicity.CommitFileTemporaryPath, 'finalize'),
          mock.patch.object(atomicity.AtomicRenameTemporaryPath, 'finalize'),
      ):
        manager = CheckpointManager(
            self.directory,
            item_names=('params',),
            options=CheckpointManagerOptions(
                enable_async_checkpointing=enable_async,
                step_name_format=name_format,
            ),
        )
        self.assertTrue(self.save_params(0, manager, self.pytree))
        self.wait_if_async(manager)

        # Manager thinks there are some steps unless we force a read. This would
        # not happen in real life, since a preemption would destroy the manager,
        # but it is a useful sanity check.
        self.assertNotEmpty(manager.all_steps())
        tmp_paths = step_lib.all_temporary_paths(manager.directory)
        self.assertLen(tmp_paths, 1)
        self.assert_checkpoint_metadata(
            root=self.directory,
            step_name_format=name_format,
            step=0,
            assert_committed=False,
            full_metadata=False,
        )
        manager.reload()
        self.assertEmpty(manager.all_steps())
        tmp_paths = step_lib.all_temporary_paths(manager.directory)
        self.assertLen(tmp_paths, 1)
        self.assert_checkpoint_metadata(
            root=self.directory,
            step_name_format=name_format,
            step=0,
            assert_committed=False,
            full_metadata=False,
        )

        test_utils.sync_global_processes(
            f'test_saved_first_checkpoint_{self.id()}'
        )

        # Simulate restart.
        manager = CheckpointManager(
            self.directory,
            item_names=('params',),
            options=CheckpointManagerOptions(
                enable_async_checkpointing=enable_async,
                step_name_format=name_format,
                multiprocessing_options=checkpoint_manager.MultiprocessingOptions(
                    barrier_sync_key_prefix='preempted'
                ),
            ),
        )
        self.assertEmpty(manager.all_steps())
        self.assertTrue(self.save_params(0, manager, self.pytree))
        self.wait_if_async(manager)
        self.assertLen(step_lib.all_temporary_paths(manager.directory), 1)

        if is_gcs:
          step_dir = utils.get_save_directory(
              0, manager.directory, step_name_format=name_format
          )
          self.assertTrue(step_dir.exists())
          self.assertFalse((step_dir / atomicity.COMMIT_SUCCESS_FILE).exists())
          self.assertTrue((step_dir / 'params').exists())
          self.assertFalse(
              (step_dir / 'params' / atomicity.COMMIT_SUCCESS_FILE).exists()
          )

        self.assertNotEmpty(manager.all_steps())
        manager.reload()
        # TODO(b/322223283): List step dir to debug why reload returns steps.
        for c in manager.directory.iterdir():
          logging.info('Root has: %s', c.name)
          if c.is_file():
            continue
          for gc in c.iterdir():
            logging.info('Root has: %s/%s', c.name, gc.name)
            if gc.is_file():
              continue
            for ggc in gc.iterdir():
              logging.info('Root has: %s/%s/%s', c.name, gc.name, ggc.name)
        # The following assertation fails flakily.
        self.assertEmpty(manager.all_steps())
      # utils.ensure_atomic_save mock.patch.object context closed.

      test_utils.sync_global_processes(f'test_ensure_atomic_save_{self.id()}')
      if multihost.process_index() == 0:
        # On non-GCS, since directories are created with a timestamp in the
        # name, it isn't really possible to get identical tmp checkpoints.
        # Thus, there is no need for them to be forcibly cleaned up, as with
        # GCS.
        self.assertLen(step_lib.all_temporary_paths(manager.directory), 1)

        tmp_dir = list(step_lib.all_temporary_paths(manager.directory))[0].get()
        test_utils.ensure_atomic_save(
            tmp_dir,
            utils.get_save_directory(
                0, manager.directory, step_name_format=name_format
            ),
            metadata_lib.metadata_store(enable_write=True, blocking_write=True),
        )
        self.assertEmpty(step_lib.all_temporary_paths(manager.directory))
        self.assert_checkpoint_metadata(
            root=manager.directory,
            step_name_format=name_format,
            step=0,
            assert_uncommitted=False,
            full_metadata=False,
        )

      manager.close()

  @parameterized.parameters((True,), (False,))
  @mock.patch.object(gcs_utils, 'is_gcs_path', autospec=True, return_value=True)
  def test_save_restore_gcs(self, enable_async, is_gcs_path):
    del is_gcs_path
    with CheckpointManager(
        self.directory,
        item_names=('params',),
        options=CheckpointManagerOptions(
            enable_async_checkpointing=enable_async
        ),
    ) as manager:
      self.assertTrue(self.save_params(0, manager, self.pytree))
      self.wait_if_async(manager)
      restored = self.restore_params(0, manager)
      test_utils.assert_tree_equal(self, self.pytree, restored)
      self.assertTrue(
          (
              utils.get_save_directory(0, self.directory, 'params')
              / atomicity.COMMIT_SUCCESS_FILE
          ).exists()
      )

  @parameterized.parameters((True,), (False,))
  @mock.patch.object(gcs_utils, 'is_gcs_path', autospec=True, return_value=True)
  def test_save_preempted_gcs(self, enable_async, is_gcs_path):
    """Simulate effects of preemption."""
    del is_gcs_path
    tmp_dir = test_utils.save_fake_tmp_dir(
        self.directory, 0, 'params', subdirs=['subdir']
    )
    self.assertTrue(tmp_dir.exists())
    self.assertTrue((tmp_dir / 'subdir').exists())

    # Check for directory existence before initializing CheckpointManager,
    # which will clean up the above directories.
    test_utils.sync_global_processes('test_check_dirs')

    with CheckpointManager(
        self.directory,
        item_names=('params',),
        options=CheckpointManagerOptions(
            cleanup_tmp_directories=True,
            enable_async_checkpointing=enable_async,
        ),
    ) as manager:
      # Temp checkpoints cleaned up at creation.
      self.assertFalse(tmp_dir.exists())
      self.assertFalse((tmp_dir / 'subdir').exists())
      self.assertSameElements([], manager.all_steps())

      # Sync to check directories before a new tmp dir is created.
      test_utils.sync_global_processes('test_check_dirs_after_cleanup')

      tmp_dir = test_utils.save_fake_tmp_dir(
          self.directory, 0, 'params', subdirs=['subdir']
      )
      self.assertTrue(tmp_dir.exists())
      self.assertTrue((tmp_dir / 'subdir').exists())
      self.assertSameElements(
          [], manager.all_steps()
      )  # Only picks up finalized.

      with self.assertRaisesRegex(ValueError, 'Found incomplete checkpoint'):
        manager.restore(0)

      self.assertTrue(self.save_params(1, manager, self.pytree))
      self.wait_if_async(manager)
      self.assertSameElements([1], manager.all_steps())  # Step 0 not picked up.
      self.assertTrue(
          (
              utils.get_save_directory(1, self.directory, 'params')
              / atomicity.COMMIT_SUCCESS_FILE
          ).exists()
      )

  @parameterized.parameters((False,), (True,))
  def test_save_default_item(self, enable_async):
    """Test managing single item."""
    with CheckpointManager(
        self.directory,
        options=CheckpointManagerOptions(
            enable_async_checkpointing=enable_async
        ),
    ) as manager:
      self.assertIsNone(manager._default_item.get())
      self.assertTrue(manager.save(0, args=args.PyTreeSave(self.pytree)))
      self.assertTrue(manager._default_item.get())
      self.wait_if_async(manager)
      restored = manager.restore(
          0, args=args.PyTreeRestore(restore_args=self.pytree_restore_args)
      )
      expected = self.pytree
      test_utils.assert_tree_equal(self, expected, restored)

  def test_multiple_items(self):
    """Test multiple different items."""
    with CheckpointManager(
        self.directory,
        item_names=('params', 'arr', 'metadata'),
    ) as manager:

      metadata = {
          'VERSION': 2,
          'optimizer': {
              'lr': 0.001,
              'type': 'adam',
          },
      }

      self.assertTrue(
          manager.save(
              0,
              args=args.Composite(**{
                  'params': args.PyTreeSave(self.pytree),
                  'arr': args.ArraySave(np.arange(16)),
                  'metadata': args.JsonSave(metadata),
              }),
          )
      )
      self.wait_if_async(manager)
      restored = manager.restore(
          0,
          args=args.Composite(**{
              'params': args.PyTreeRestore(
                  self.empty_pytree, self.pytree_restore_args
              ),
              'arr': args.ArrayRestore(),
              'metadata': args.JsonRestore(),
          }),
      )
      restored_params, restored_arr, restored_metadata = (
          restored.params,
          restored.arr,
          restored.metadata,
      )
      expected_params = self.pytree
      test_utils.assert_tree_equal(self, expected_params, restored_params)
      np.testing.assert_array_equal(restored_arr, np.arange(16))
      self.assertDictEqual(metadata, restored_metadata)

  @mock.patch.object(gcs_utils, 'is_gcs_path', autospec=True, return_value=True)
  def test_save_gcs_with_unfinalized_checkpoints(self, is_gcs_path):
    del is_gcs_path
    subdir = self.directory / '0'
    subdir.mkdir(parents=True, exist_ok=True)
    test_utils.sync_global_processes('test_make_unfinalized_checkpoint')
    with CheckpointManager(self.directory, item_names=('params',)) as manager:
      self.assertTrue(self.save_params(0, manager, self.pytree))
      self.wait_if_async(manager)
      self.assertSameElements([0], manager.all_steps())

      restored = self.restore_params(0, manager)
      test_utils.assert_tree_equal(self, self.pytree, restored)
      self.assertTrue(
          (
              utils.get_save_directory(0, self.directory, 'params')
              / atomicity.COMMIT_SUCCESS_FILE
          ).exists()
      )

  @parameterized.named_parameters(
      ('min_sync_delete', True, False, None),
      ('min_sync_rename', True, False, 'ttl=1h'),
      ('min_async_delete', True, True, None),
      ('min_async_rename', True, True, 'ttl=1h'),
      ('max_sync_delete', False, False, None),
      ('max_sync_rename', False, False, 'ttl=1h'),
      ('max_async_delete', False, True, None),
      ('max_async_rename', False, True, 'ttl=1h'),
  )
  def test_save_best(self, mode_min, enable_async, todelete_subdir):
    if mode_min:
      mode = 'min'
      metric_fn = lambda metrics: metrics['loss']
    else:
      mode = 'max'
      metric_fn = lambda metrics: metrics['accuracy']

    all_metrics = {
        'loss': [5, 2, 4, 3, 7] + [1, 10, 9, 7, 4],
        'accuracy': [30, 85, 70, 80, 60] + [100, 40, 45, 75, 75],
    }

    options = CheckpointManagerOptions(
        enable_async_checkpointing=enable_async,
        best_fn=metric_fn,
        best_mode=mode,
        max_to_keep=2,
        todelete_subdir=todelete_subdir,
    )

    with CheckpointManager(self.directory, options=options) as manager:
      for step in range(5):
        metrics = {k: v[step] for k, v in all_metrics.items()}
        self.assertTrue(
            manager.save(
                step, args=args.PyTreeSave(self.pytree), metrics=metrics
            )
        )
      self.wait_if_async(manager)

      # Simulate preemption - force new CheckpointManager to load
      # self._past_metrics from file.
      manager = CheckpointManager(self.directory, options=options)
      for step in range(5, 10):
        metrics = {k: v[step] for k, v in all_metrics.items()}
        self.assertTrue(
            manager.save(
                step, args=args.PyTreeSave(self.pytree), metrics=metrics
            )
        )
      self.wait_if_async(manager)

      self.assertSameElements([1, 5], manager.all_steps())
      if todelete_subdir is not None:
        self.assert_renamed_subdirs(
            manager.directory,
            todelete_subdir,
            all_steps=range(10),
            remaining_steps=[1, 5],
        )

  @parameterized.parameters((None,), ('ttl=1h',))
  def test_save_best_delete_no_metrics(self, todelete_subdir):
    options = CheckpointManagerOptions(
        best_fn=lambda metrics: metrics['loss'],
        best_mode='min',
        max_to_keep=2,
        todelete_subdir=todelete_subdir,
    )
    with CheckpointManager(self.directory, options=options) as manager:
      steps = 5
      for step in range(steps):
        metrics = None
        self.assertTrue(
            manager.save(
                step, args=args.PyTreeSave(self.pytree), metrics=metrics
            )
        )
      self.wait_if_async(manager)
      # Will keep 2 most recent, even without metrics.
      self.assertSameElements([0, 1, 2, 3, 4], manager.all_steps())
      self.assertIsNone(manager.best_step())  # No step has metrics set.
      self.assertEqual(manager.latest_step(), 4)
      if todelete_subdir is not None:
        self.assert_renamed_subdirs(
            manager.directory,
            todelete_subdir,
            all_steps=range(steps),
            remaining_steps=[0, 1, 2, 3, 4],
        )

  @parameterized.parameters((None,), ('ttl=1h',))
  def test_save_best_some_metrics(self, todelete_subdir):
    all_metrics = {
        'loss': [3, 6, 4, 1, 7, 7],
    }
    options = CheckpointManagerOptions(
        best_fn=lambda metrics: metrics['loss'],
        best_mode='min',
        max_to_keep=3,
        todelete_subdir=todelete_subdir,
    )
    with CheckpointManager(self.directory, options=options) as manager:
      steps = 6
      kept_steps = {
          0: [0],
          1: [0, 1],
          2: [0, 1, 2],
          3: [0, 1, 2, 3],
          4: [0, 1, 2, 3, 4],
          5: [0, 1, 2, 3, 4, 5],
      }
      for step in range(steps):
        if step % 2 == 0:
          metrics = {k: v[step] for k, v in all_metrics.items()}
        else:
          metrics = None
        self.assertTrue(
            manager.save(
                step, args=args.PyTreeSave(self.pytree), metrics=metrics
            )
        )
        self.assertSameElements(kept_steps[step], manager.all_steps())
      self.wait_if_async(manager)
      if todelete_subdir is not None:
        self.assert_renamed_subdirs(
            manager.directory,
            todelete_subdir,
            all_steps=range(steps),
            remaining_steps=kept_steps[steps - 1],
        )

  def test_flax_train_state(self):
    """Test using flax model."""

    class MLP(nn.Module):
      """A simple MLP model."""

      @nn.compact
      def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=8)(x)
        return x

    model = MLP()
    mesh = Mesh(np.asarray(jax.devices()), ('devices',))
    mesh_axes = PartitionSpec()

    @jax.jit
    def init_state():
      params = model.init(jax.random.PRNGKey(0), jnp.ones([8, 8]))
      tx = optax.adamw(learning_rate=0.001)
      state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
      return state

    init = pjit.pjit(init_state, in_shardings=None, out_shardings=mesh_axes)

    with Mesh(mesh.devices, mesh.axis_names):
      state = init()
      state_shape = jax.eval_shape(init)

    restore_args = jax.tree.map(
        lambda _: ArrayRestoreArgs(mesh=mesh, mesh_axes=mesh_axes), state_shape
    )

    with CheckpointManager(self.directory, item_names=('state',)) as manager:
      self.assertTrue(
          manager.save(
              0,
              args=args.Composite(state=args.PyTreeSave(state)),
          )
      )
      self.wait_if_async(manager)
      # Already fully replicated, don't need to provide args.
      restored = manager.restore(
          0,
          args=args.Composite(
              state=args.PyTreeRestore(state_shape, restore_args=restore_args)
          ),
      ).state
      test_utils.assert_tree_equal(self, state.params, restored.params)
      test_utils.assert_tree_equal(self, state.opt_state, restored.opt_state)

  def test_restore_independent(self):
    """Test restore from secondary location."""
    # Simulates pretrained checkpoint stored elsewhere.
    secondary_manager = CheckpointManager(
        self.secondary_directory,
        item_names=('params',),
    )
    self.assertTrue(self.save_params(0, secondary_manager, self.pytree))
    self.wait_if_async(secondary_manager)

    manager = CheckpointManager(self.directory, item_names=('params',))
    pytree_restore_args = jax.tree.map(
        lambda mesh, axes: ArrayRestoreArgs(mesh=mesh, mesh_axes=axes),
        self.mesh_tree,
        self.axes_tree,
    )

    with self.assertRaises(FileNotFoundError):
      manager.restore(
          0,
          args=args.Composite(
              params=args.PyTreeRestore(restore_args=pytree_restore_args)
          ),
      )

    restored = manager.restore(
        0,
        args=args.Composite(
            params=args.PyTreeRestore(restore_args=pytree_restore_args)
        ),
        directory=self.secondary_directory,
    ).params
    test_utils.assert_tree_equal(self, self.pytree, restored)

    secondary_manager.close()
    manager.close()

  @mock.patch.object(multihost_utils, 'reached_preemption_sync_point')
  def test_save_on_preemption(self, mock_reached_preemption_sync_point):
    if multihost.is_pathways_backend():
      self.skipTest('Not applicable to Pathways.')

    num_steps = 10
    options = CheckpointManagerOptions(save_interval_steps=4)
    with CheckpointManager(self.directory, options=options) as manager:

      preemption_step = 2
      mock_reached_preemption_sync_point.side_effect = (
          lambda step: step == preemption_step
      )

      for step in range(num_steps):
        manager.save(step, args=args.PyTreeSave(self.pytree))
      self.wait_if_async(manager)
      self.assertSameElements([0, preemption_step, 4, 8], manager.all_steps())

  def test_metadata(self):
    metadata = {
        'version': 1.1,
        'info': {
            'foo': 'bar',
            'baz': 5,
        },
    }
    with CheckpointManager(
        self.directory,
        metadata=metadata,
    ) as manager:
      self.assertTrue(manager._root_metadata_file_path().exists())
      self.assertDictEqual(manager.metadata().custom_metadata, metadata)

    new_metadata = metadata.copy()
    new_metadata.update({'version': 2.2})  # update doesn't return the dict.
    with CheckpointManager(
        self.directory,
        metadata=new_metadata,
    ) as manager:
      # Still equals original metadata.
      self.assertDictEqual(manager.metadata().custom_metadata, metadata)

  def test_empty_metadata(self):
    with CheckpointManager(
        self.directory,
    ) as manager:
      with self.assertRaisesRegex(
          ValueError, 'Metadata directory is not initialized'
      ):
        manager._root_metadata_file_path()
      self.assertDictEqual({}, manager.metadata().custom_metadata)

  def test_checkpoint_args_mismatch_item_handlers(self):
    with self.assertRaisesRegex(
        ValueError, 'does not match with any registered handler'
    ):
      with CheckpointManager(
          self.directory,
          item_handlers=handlers.PyTreeCheckpointHandler(),
      ) as manager:
        manager.save(0, args=args.StandardSave(self.pytree))

    with self.assertRaisesRegex(
        ValueError, 'does not match with any registered handler'
    ):
      with CheckpointManager(
          self.directory,
          item_handlers={'params': handlers.PyTreeCheckpointHandler()},
      ) as manager:
        manager.save(
            0,
            args=args.Composite(params=args.StandardSave(self.pytree)),
        )

  def test_item_names_with_single_item_handler(self):
    with self.assertRaises(ValueError):
      with CheckpointManager(
          self.directory,
          item_names=('params',),
          item_handlers=handlers.StandardCheckpointHandler(),
      ) as _:
        pass

  def test_default_item_metadata(self):
    with CheckpointManager(self.directory) as manager:
      self.assertIsNone(manager._default_item.get())
      state = {'step': 100}
      manager.save(100, args=args.StandardSave(state))
      self.assertTrue(manager._default_item.get())
      self.wait_if_async(manager)

      self.assertDictEqual(
          manager.metadata(100).item_metadata.tree,
          {
              'step': value_metadata.ScalarMetadata(
                  name='step',
                  directory=epath.Path(self.directory / _DEFAULT_ITEM_NAME),
                  dtype=jnp.int64,
              )
          },
      )

  def test_default_item_metadata_legacy(self):
    with CheckpointManager(self.directory) as manager:
      self.assertIsNone(manager._default_item.get())
      state = {'step': 100}
      manager.save(100, args=args.StandardSave(state))
      self.assertTrue(manager._default_item.get())
      self.wait_if_async(manager)

      self.assertDictEqual(
          manager.item_metadata(100).tree,
          {
              'step': value_metadata.ScalarMetadata(
                  name='step',
                  directory=epath.Path(self.directory / _DEFAULT_ITEM_NAME),
                  dtype=jnp.int64,
              )
          },
      )

  def test_default_item_metadata_with_new_checkpoint_manager(self):
    with CheckpointManager(self.directory) as manager:
      self.assertIsNone(manager._default_item.get())
      state = {'step': 100}
      manager.save(100, args=args.StandardSave(state))
      self.assertTrue(manager._default_item.get())
      manager.wait_until_finished()

      with CheckpointManager(
          self.directory,
          item_handlers=handlers.StandardCheckpointHandler(),
      ) as new_manager:
        self.assertTrue(new_manager._default_item.get())
        self.assertDictEqual(
            new_manager.metadata(100).item_metadata.tree,
            {
                'step': value_metadata.ScalarMetadata(
                    name='step',
                    directory=epath.Path(self.directory / _DEFAULT_ITEM_NAME),
                    dtype=jnp.int64,
                )
            },
        )

    with CheckpointManager(self.directory) as new_manager:
      self.assertIsNone(new_manager._default_item.get())
      self.assertIsNone(new_manager.metadata(100).item_metadata)

  def test_multiple_item_metadata(self):
    if multihost.is_pathways_backend():
      # TODO(b/408241116) Enable sharding metadata on Pathways.
      self.skipTest('Sharding metadata not present on Pathways.')

    manager = CheckpointManager(
        self.directory,
        item_names=('params', 'arr', 'metadata'),
    )
    self.assertFalse(manager._default_item.get())
    is_array_metadata_store_enabled = (
        array_metadata_store_lib.resolve_array_metadata_store(
            type_handler_registry.GLOBAL_TYPE_HANDLER_REGISTRY
        )
        is not None
    )
    metadata = {
        'VERSION': 2,
        'optimizer': {
            'lr': 0.001,
            'type': 'adam',
        },
    }
    self.assertTrue(
        manager.save(
            0,
            args=args.Composite(**{
                'params': args.PyTreeSave(self.pytree),
                'arr': args.ArraySave(np.arange(16)),
                'metadata': args.JsonSave(metadata),
            }),
        )
    )
    manager.wait_until_finished()
    local_shapes = jax.tree.map(
        test_utils.get_expected_chunk_shape, self.pytree
    )

    expected = {
        'arr': None,
        'metadata': None,
        'params': {
            'a': ArrayMetadata(
                name='a',
                directory=epath.Path(self.directory / '0' / 'params'),
                shape=(8,),
                sharding=NamedShardingMetadata(
                    shape=np.array([8]),
                    axis_names=['x'],
                    axis_types=(jax.sharding.AxisType.Auto,),
                    partition_spec=(None,),
                    device_mesh=sharding_metadata.DeviceMetadataMesh.from_jax_mesh(
                        self.pytree['a'].sharding.mesh
                    ),
                ),
                dtype=jnp.int32,
                storage=build_storage_metadata(
                    local_shapes['a'], is_array_metadata_store_enabled
                ),
            ),
            'b': ArrayMetadata(
                name='b',
                directory=epath.Path(self.directory / '0' / 'params'),
                shape=(16,),
                sharding=NamedShardingMetadata(
                    shape=np.array([8]),
                    axis_names=['x'],
                    axis_types=(jax.sharding.AxisType.Auto,),
                    partition_spec=('x',),
                    device_mesh=sharding_metadata.DeviceMetadataMesh.from_jax_mesh(
                        self.pytree['b'].sharding.mesh
                    ),
                ),
                dtype=jnp.int32,
                storage=build_storage_metadata(
                    local_shapes['b'], is_array_metadata_store_enabled
                ),
            ),
            'c': {
                'a': ArrayMetadata(
                    name='c.a',
                    directory=epath.Path(self.directory / '0' / 'params'),
                    shape=(2, 4),
                    sharding=NamedShardingMetadata(
                        shape=np.array([2, 4]),
                        axis_names=['x', 'y'],
                        axis_types=(
                            jax.sharding.AxisType.Auto,
                            jax.sharding.AxisType.Auto,
                        ),
                        partition_spec=('x', 'y'),
                        device_mesh=sharding_metadata.DeviceMetadataMesh.from_jax_mesh(
                            self.pytree['c']['a'].sharding.mesh
                        ),
                    ),
                    dtype=jnp.int32,
                    storage=build_storage_metadata(
                        local_shapes['c']['a'], is_array_metadata_store_enabled
                    ),
                ),
                'e': ArrayMetadata(
                    name='c.e',
                    directory=epath.Path(self.directory / '0' / 'params'),
                    shape=(4, 4),
                    sharding=NamedShardingMetadata(
                        shape=np.array([2, 4]),
                        axis_names=['x', 'y'],
                        axis_types=(
                            jax.sharding.AxisType.Auto,
                            jax.sharding.AxisType.Auto,
                        ),
                        partition_spec=('x', 'y'),
                        device_mesh=sharding_metadata.DeviceMetadataMesh.from_jax_mesh(
                            self.pytree['c']['e'].sharding.mesh
                        ),
                    ),
                    dtype=jnp.int32,
                    storage=build_storage_metadata(
                        local_shapes['c']['e'], is_array_metadata_store_enabled
                    ),
                ),
            },
        },
    }
    composite_metadata = manager.metadata(0).item_metadata
    for k in expected:
      self.assertIn(k, composite_metadata)
      if k != 'params':
        self.assertIsNone(composite_metadata[k])
      else:
        self.assertDictEqual(expected['params'], composite_metadata[k].tree)

    manager.close()

  def test_multiple_item_metadata_with_new_checkpoint_manager(self):
    if multihost.is_pathways_backend():
      # TODO(b/408241116) Enable sharding metadata on Pathways.
      self.skipTest('Sharding metadata not present on Pathways.')
    # Create a manager with items to be used by later managers.
    manager = CheckpointManager(
        self.directory,
        item_names=('params', 'arr', 'metadata'),
    )
    self.assertFalse(manager._default_item.get())
    metadata = {
        'VERSION': 2,
        'optimizer': {
            'lr': 0.001,
            'type': 'adam',
        },
    }
    self.assertTrue(
        manager.save(
            0,
            args=args.Composite(**{
                'params': args.PyTreeSave(self.pytree),
                'arr': args.ArraySave(np.arange(16)),
                'metadata': args.JsonSave(metadata),
            }),
        )
    )
    manager.wait_until_finished()
    self.assertSetEqual(
        set(manager.metadata(0).item_metadata.keys()),
        set(['params', 'arr', 'metadata']),
    )

    with self.subTest('no_names_no_handlers'):
      new_manager = CheckpointManager(self.directory)
      # User could provide named items or a single unnamed item when saving, so
      # the mode will be determined lazily.
      self.assertIsNone(new_manager._default_item.get())
      # Retrieve on-disk item metadata.
      self.assertSetEqual(
          set(new_manager.metadata(0).item_metadata.keys()),
          set(['params', 'arr', 'metadata']),
      )
      # `items` is now known to be dict[str, Any].
      self.assertFalse(new_manager._default_item.get())
      # No handlers means no metadata values.
      for v in new_manager.metadata(0).item_metadata.values():
        self.assertIsNone(v)
      new_manager.close()

    with self.subTest('names_but_no_handlers'):
      new_manager = CheckpointManager(
          self.directory,
          item_names=('params', 'arr', 'metadata'),
      )
      # `item_names` tells us that we are in named-item mode.
      self.assertFalse(new_manager._default_item.get())
      self.assertSetEqual(
          set(new_manager.metadata(0).item_metadata.keys()),
          set(['params', 'arr', 'metadata']),
      )
      for v in new_manager.metadata(0).item_metadata.values():
        self.assertIsNone(v)
      new_manager.close()

    with self.subTest('names_and_partial_handlers'):
      new_manager = CheckpointManager(
          self.directory,
          item_names=('params', 'arr', 'metadata'),
          item_handlers={
              'params': handlers.StandardCheckpointHandler(),
              'metadata': handlers.JsonCheckpointHandler(),
          },
      )
      self.assertFalse(new_manager._default_item.get())
      self.assertSetEqual(
          set(new_manager.metadata(0).item_metadata.keys()),
          set(['params', 'arr', 'metadata']),
      )
      item_metadata = new_manager.metadata(0).item_metadata
      self.assertIsNotNone(item_metadata.params)
      self.assertIsNone(item_metadata.arr)
      new_manager.close()

    with self.subTest('disjoint_names_and_handlers'):
      new_manager = CheckpointManager(
          self.directory,
          item_names=('arr',),
          item_handlers={
              'params': handlers.StandardCheckpointHandler(),
              'metadata': handlers.JsonCheckpointHandler(),
          },
      )
      self.assertFalse(new_manager._default_item.get())
      self.assertSetEqual(
          set(new_manager.metadata(0).item_metadata.keys()),
          set(['arr', 'params', 'metadata']),
      )
      new_manager.close()

    with self.subTest('handlers_but_no_names'):
      new_manager = CheckpointManager(
          self.directory,
          item_handlers={
              'params': handlers.StandardCheckpointHandler(),
              'metadata': handlers.JsonCheckpointHandler(),
              'arr': handlers.ArrayCheckpointHandler(),
          },
      )
      is_array_metadata_store_enabled = (
          array_metadata_store_lib.resolve_array_metadata_store(
              type_handler_registry.GLOBAL_TYPE_HANDLER_REGISTRY
          )
          is not None
      )
      self.assertFalse(new_manager._default_item.get())
      item_metadata = new_manager.metadata(0).item_metadata
      self.assertSameElements(
          ['arr', 'metadata', 'params'], item_metadata.keys()
      )
      local_shapes = jax.tree.map(
          test_utils.get_expected_chunk_shape, self.pytree
      )
      self.assertDictEqual(
          item_metadata['params'].tree,
          {
              'a': ArrayMetadata(
                  name='a',
                  directory=epath.Path(self.directory / '0' / 'params'),
                  shape=(8,),
                  sharding=NamedShardingMetadata(
                      shape=np.array([8]),
                      axis_names=['x'],
                      axis_types=(jax.sharding.AxisType.Auto,),
                      partition_spec=(None,),
                      device_mesh=sharding_metadata.DeviceMetadataMesh.from_jax_mesh(
                          self.pytree['a'].sharding.mesh
                      ),
                  ),
                  dtype=jnp.int32,
                  storage=build_storage_metadata(
                      local_shapes['a'], is_array_metadata_store_enabled
                  ),
              ),
              'b': ArrayMetadata(
                  name='b',
                  directory=epath.Path(self.directory / '0' / 'params'),
                  shape=(16,),
                  sharding=NamedShardingMetadata(
                      shape=np.array([8]),
                      axis_names=['x'],
                      axis_types=(jax.sharding.AxisType.Auto,),
                      partition_spec=('x',),
                      device_mesh=sharding_metadata.DeviceMetadataMesh.from_jax_mesh(
                          self.pytree['b'].sharding.mesh
                      ),
                  ),
                  dtype=jnp.int32,
                  storage=build_storage_metadata(
                      local_shapes['b'], is_array_metadata_store_enabled
                  ),
              ),
              'c': {
                  'a': ArrayMetadata(
                      name='c.a',
                      directory=epath.Path(self.directory / '0' / 'params'),
                      shape=(2, 4),
                      sharding=NamedShardingMetadata(
                          shape=np.array([2, 4]),
                          axis_names=['x', 'y'],
                          axis_types=(
                              jax.sharding.AxisType.Auto,
                              jax.sharding.AxisType.Auto,
                          ),
                          partition_spec=('x', 'y'),
                          device_mesh=sharding_metadata.DeviceMetadataMesh.from_jax_mesh(
                              self.pytree['c']['a'].sharding.mesh
                          ),
                      ),
                      dtype=jnp.int32,
                      storage=build_storage_metadata(
                          local_shapes['c']['a'],
                          is_array_metadata_store_enabled,
                      ),
                  ),
                  'e': ArrayMetadata(
                      name='c.e',
                      directory=epath.Path(self.directory / '0' / 'params'),
                      shape=(4, 4),
                      sharding=NamedShardingMetadata(
                          shape=np.array([2, 4]),
                          axis_names=['x', 'y'],
                          axis_types=(
                              jax.sharding.AxisType.Auto,
                              jax.sharding.AxisType.Auto,
                          ),
                          partition_spec=('x', 'y'),
                          device_mesh=sharding_metadata.DeviceMetadataMesh.from_jax_mesh(
                              self.pytree['c']['e'].sharding.mesh
                          ),
                      ),
                      dtype=jnp.int32,
                      storage=build_storage_metadata(
                          local_shapes['c']['e'],
                          is_array_metadata_store_enabled,
                      ),
                  ),
              },
          },
      )
      new_manager.close()

    manager.close()

  def test_directory_creation(self):
    directory = self.directory / 'mydir'
    self.assertFalse(directory.exists())
    # Finish check before object initialization creates the directory.
    test_utils.sync_global_processes(
        'CheckpointManagerTest:done_directory_check_0'
    )
    options = CheckpointManagerOptions(create=True)
    with CheckpointManager(directory, options=options) as _:
      self.assertTrue(directory.exists())
    test_utils.sync_global_processes(
        'CheckpointManagerTest:done_directory_check_1'
    )
    # Do it again to make sure we don't run into issues if the directory already
    # exists.
    with CheckpointManager(directory, options=options) as _:
      self.assertTrue(directory.exists())

  def test_delete(self):
    manager = CheckpointManager(self.directory)
    self.assertTrue(manager.save(0, args=args.JsonSave({'a': 1, 'b': 'hello'})))
    self.wait_if_async(manager)
    if multihost.process_index() == 0:
      self.assertSameElements([0], manager.all_steps())
    manager.delete(0)
    self.assertEmpty(manager.all_steps())
    self.assertEmpty(list(manager.directory.iterdir()))
    manager.close()

  def test_delete_with_todelete_subdir(self):
    todelete_subdir = 'ttl=1h'
    with CheckpointManager(
        self.directory,
        options=CheckpointManagerOptions(todelete_subdir=todelete_subdir),
    ) as manager:
      self.assertTrue(
          manager.save(0, args=args.JsonSave({'a': 1, 'b': 'hello'}))
      )
      self.assertTrue(
          manager.save(1, args=args.JsonSave({'a': 1, 'b': 'hello'}))
      )
      self.assertTrue(
          manager.save(2, args=args.JsonSave({'a': 1, 'b': 'hello'}))
      )
      self.wait_if_async(manager)
      if multihost.process_index() == 0:
        self.assertSameElements([0, 1, 2], manager.all_steps())
      manager.delete(0)
      manager.delete(1)
      self.assertSameElements([2], manager.all_steps())
      self.assert_renamed_subdirs(
          manager.directory,
          todelete_subdir,
          all_steps=[0, 1, 2],
          remaining_steps=[2],
      )

  def test_async_finalize(self):
    with CheckpointManager(self.directory) as manager:
      manager.save(0, args=args.JsonSave({'a': 1, 'b': 2}))
      self.wait_if_async(manager)
      time.sleep(5)  # allow time to finish save
      self.assertTrue(
          step_lib.is_path_finalized(
              step_lib.get_save_directory(0, manager.directory)
          )
      )
      self.assertFalse(
          step_lib.is_path_temporary(
              step_lib.get_save_directory(0, manager.directory)
          )
      )

  def test_async_is_save_in_progress(self):
    with CheckpointManager(
        self.directory,
        options=CheckpointManagerOptions(enable_async_checkpointing=True),
    ) as manager:
      manager.save(0, args=args.JsonSave({'a': 1, 'b': 2}))
      self.assertTrue(
          manager.is_saving_in_progress(),
          'Expected is_saving_in_progress() to be True when using async '
          'checkpointing.',
      )
      self.wait_if_async(manager)
      self.assertFalse(
          manager.is_saving_in_progress(),
          'Expected is_saving_in_progress() to be False after '
          'wait_until_finished().',
      )

  def test_sync_is_save_in_progress(self):
    with CheckpointManager(
        self.directory,
        options=CheckpointManagerOptions(enable_async_checkpointing=False),
    ) as manager:
      manager.save(0, args=args.JsonSave({'a': 1, 'b': 2}))
      self.assertFalse(
          manager.is_saving_in_progress(),
          'Expected is_saving_in_progress() to be False when using synchronous '
          'checkpointing.',
      )
      self.wait_if_async(manager)
      self.assertFalse(
          manager.is_saving_in_progress(),
          'Expected is_saving_in_progress() to be False after '
          'wait_until_finished().',
      )

  def test_should_save_with_older_step(self):
    step_name_format = step_lib.standard_name_format(step_prefix='step')
    (self.directory / step_name_format.build_name(10)).mkdir(
        parents=True, exist_ok=True
    )

    with CheckpointManager(
        self.directory,
        options=CheckpointManagerOptions(
            save_interval_steps=1,
            step_name_format=step_name_format,
        ),
    ) as manager:
      self.assertFalse(manager.should_save(step=5))

  def test_should_save_with_option_should_save_fn_empty_root_dir(self):
    def _should_save_fn(step: int, latest_step: Optional[int] = None) -> bool:
      del latest_step
      if step == 0:
        return False
      return step % 3 == 0 or step in [2]

    step_name_format = step_lib.standard_name_format(step_prefix='step')
    options = CheckpointManagerOptions(
        step_name_format=step_name_format,
        save_interval_steps=1,  # ignored due to to _should_save_fn
        save_on_steps=[0, 4, 5],  # ignored due to to _should_save_fn
        should_save_fn=_should_save_fn,
    )

    with CheckpointManager(self.directory, options=options) as manager:
      self.assertFalse(manager.should_save(step=0))
      self.assertFalse(manager.should_save(step=1))
      self.assertTrue(manager.should_save(step=2))
      self.assertTrue(manager.should_save(step=3))
      self.assertFalse(manager.should_save(step=4))
      self.assertFalse(manager.should_save(step=5))
      self.assertTrue(manager.should_save(step=6))

  def test_should_save_with_option_should_save_fn_non_empty_root_dir(self):
    def _should_save_fn(step: int, latest_step: Optional[int] = None) -> bool:
      del latest_step
      if step == 0:
        return False
      return step % 3 == 0 or step in [2]

    step_name_format = step_lib.standard_name_format(step_prefix='step')
    options = CheckpointManagerOptions(
        step_name_format=step_name_format,
        save_interval_steps=1,  # ignored due to to _should_save_fn
        save_on_steps=[0, 4, 5],  # ignored due to to _should_save_fn
        should_save_fn=_should_save_fn,
    )
    (self.directory / step_name_format.build_name(2)).mkdir(
        parents=True, exist_ok=True
    )

    with CheckpointManager(self.directory, options=options) as manager:
      self.assertFalse(manager.should_save(step=0))
      self.assertFalse(manager.should_save(step=1))
      self.assertFalse(manager.should_save(step=2))
      self.assertTrue(manager.should_save(step=3))
      self.assertFalse(manager.should_save(step=4))
      self.assertFalse(manager.should_save(step=5))
      self.assertTrue(manager.should_save(step=6))

  def test_should_save_without_option_should_save_fn_empty_root_dir(self):
    step_name_format = step_lib.standard_name_format(step_prefix='step')
    options = CheckpointManagerOptions(
        step_name_format=step_name_format,
        save_interval_steps=2,
        save_on_steps=[1],
        should_save_fn=None,
    )

    with CheckpointManager(self.directory, options=options) as manager:
      self.assertTrue(manager.should_save(step=0))
      self.assertTrue(manager.should_save(step=1))
      self.assertTrue(manager.should_save(step=2))
      self.assertTrue(manager.should_save(step=3))
      self.assertTrue(manager.should_save(step=4))
      self.assertTrue(manager.should_save(step=5))
      self.assertTrue(manager.should_save(step=6))

  def test_should_save_without_option_should_save_fn_non_empty_root_dir(self):
    step_name_format = step_lib.standard_name_format(step_prefix='step')
    options = CheckpointManagerOptions(
        step_name_format=step_name_format,
        save_interval_steps=2,
        save_on_steps=[1],
        should_save_fn=None,
    )
    (self.directory / step_name_format.build_name(0)).mkdir(
        parents=True, exist_ok=True
    )

    with CheckpointManager(self.directory, options=options) as manager:
      self.assertFalse(manager.should_save(step=0))
      self.assertTrue(manager.should_save(step=1))
      self.assertTrue(manager.should_save(step=2))
      self.assertFalse(manager.should_save(step=3))
      self.assertTrue(manager.should_save(step=4))
      self.assertFalse(manager.should_save(step=5))
      self.assertTrue(manager.should_save(step=6))

  def test_existing_dir_doesnt_err_when_read_only(self):
    with CheckpointManager(
        self.directory,
        options=CheckpointManagerOptions(
            save_interval_steps=0,
            create=False,
            read_only=True,
        ),
    ):
      self.assertTrue(self.directory.exists())

  def test_should_save_returns_false_when_read_only(self):
    with CheckpointManager(
        self.directory,
        options=CheckpointManagerOptions(
            save_interval_steps=0,
            create=False,
            read_only=True,
        ),
    ) as read_only_manager:
      self.assertFalse(read_only_manager.should_save(0))

  def test_save_returns_false_when_read_only(self):
    with CheckpointManager(
        self.directory,
        options=CheckpointManagerOptions(
            save_interval_steps=0,
            create=False,
            read_only=True,
        ),
    ) as read_only_manager:
      self.assertFalse(
          read_only_manager.save(0, args=args.JsonSave({'a': 1, 'b': 'hello'}))
      )

  def test_restore_works_normally_when_read_only(self):
    write_manager = CheckpointManager(
        self.directory,
        options=CheckpointManagerOptions(
            read_only=False,
        ),
    )
    self.assertTrue(
        write_manager.save(0, args=args.JsonSave({'a': 1, 'b': 'hello'}))
    )
    self.wait_if_async(write_manager)

    read_only_manager = CheckpointManager(
        self.directory,
        options=CheckpointManagerOptions(
            save_interval_steps=0,
            create=False,
            read_only=True,
        ),
    )
    self.assertEqual(
        {'a': 1, 'b': 'hello'},
        read_only_manager.restore(0, args=args.JsonRestore()),
    )

    write_manager.close()
    read_only_manager.close()

  def test_delete_skipped_when_read_only(self):
    write_manager = CheckpointManager(self.directory)
    self.assertTrue(
        write_manager.save(0, args=args.JsonSave({'a': 1, 'b': 'hello'}))
    )
    self.wait_if_async(write_manager)
    self.assertSameElements([0], write_manager.all_steps())

    read_only_manager = CheckpointManager(
        self.directory,
        options=CheckpointManagerOptions(
            save_interval_steps=0,
            create=False,
            read_only=True,
        ),
    )
    read_only_manager.delete(0)
    self.assertSameElements([0], read_only_manager.all_steps())

    write_manager.close()
    read_only_manager.close()

  @parameterized.parameters(
      ({'version': 1.1},),
      (None,),
  )
  def test_metadata_save_skipped_when_always_read_only(self, metadata):
    with CheckpointManager(
        self.directory,
        options=CheckpointManagerOptions(
            save_interval_steps=0,
            create=False,
            read_only=True,
        ),
        metadata=metadata,
    ) as read_only_manager:
      with self.assertRaisesRegex(
          ValueError, 'Metadata directory is not initialized'
      ):
        read_only_manager._root_metadata_file_path()
      self.assertEqual(
          read_only_manager.metadata().custom_metadata,
          metadata if metadata else {},
      )

    new_metadata = metadata.copy() if metadata else {}
    new_metadata.update({'version': 2.2})  # update doesn't return the dict.
    with CheckpointManager(
        self.directory,
        options=CheckpointManagerOptions(
            save_interval_steps=0,
            create=False,
            read_only=True,
        ),
        metadata=new_metadata,
    ) as read_only_manager:
      with self.assertRaisesRegex(
          ValueError, 'Metadata directory is not initialized'
      ):
        read_only_manager._root_metadata_file_path()
      # New metadata is returned as original metadata is not saved.
      self.assertEqual(
          read_only_manager.metadata().custom_metadata, new_metadata
      )

  @parameterized.parameters(
      ({'version': 1.1},),
      (None,),
  )
  def test_metadata_save_skipped_with_write_and_read_only(self, metadata):
    with CheckpointManager(
        self.directory,
        metadata=metadata,
    ) as write_manager:
      if metadata is None:
        with self.assertRaisesRegex(
            ValueError, 'Metadata directory is not initialized'
        ):
          write_manager._root_metadata_file_path()
        self.assertEqual(write_manager.metadata().custom_metadata, {})
      else:
        self.assertTrue(write_manager._root_metadata_file_path().exists())
        self.assertEqual(write_manager.metadata().custom_metadata, metadata)

    metadata2 = metadata.copy() if metadata else {}
    metadata2.update({'version': 2.2})  # update doesn't return the dict.
    with CheckpointManager(
        self.directory,
        options=CheckpointManagerOptions(
            save_interval_steps=0,
            create=False,
            read_only=True,
        ),
        metadata=metadata2,
    ) as read_only_manager:
      if metadata is None:
        with self.assertRaisesRegex(
            ValueError, 'Metadata directory is not initialized'
        ):
          read_only_manager._root_metadata_file_path()
        # Current metadata returned.
        self.assertEqual(
            read_only_manager.metadata().custom_metadata, metadata2
        )
      else:
        self.assertTrue(read_only_manager._root_metadata_file_path().exists())
        # Original metadata returned.
        self.assertEqual(read_only_manager.metadata().custom_metadata, metadata)

      metadata3 = metadata.copy() if metadata else {}
      metadata3.update({'version': 3.3})  # update doesn't return the dict.
      read_only_manager = CheckpointManager(
          self.directory,
          options=CheckpointManagerOptions(
              save_interval_steps=0,
              create=False,
              read_only=True,
          ),
          metadata=metadata3,
      )
      if metadata is None:
        with self.assertRaisesRegex(
            ValueError, 'Metadata directory is not initialized'
        ):
          read_only_manager._root_metadata_file_path()
        # Current metadata returned.
        self.assertEqual(
            read_only_manager.metadata().custom_metadata, metadata3
        )
      else:
        self.assertTrue(read_only_manager._root_metadata_file_path().exists())
        # Original metadata returned.
        self.assertEqual(read_only_manager.metadata().custom_metadata, metadata)

  def test_custom_metadata_with_metadata_store(self):
    custom_metadata = {'blah': 'hello'}
    with CheckpointManager(
        self.directory,
        metadata=custom_metadata,
    ) as write_manager:
      self.assertEqual(
          write_manager.metadata().custom_metadata, custom_metadata
      )
    metadata_dict = metadata_lib.metadata_store(enable_write=False).read(
        file_path=metadata_lib.root_metadata_file_path(
            self.directory / checkpoint_manager.METADATA_ITEM_NAME
        )
    )
    root_metadata = root_metadata_serialization.deserialize(metadata_dict)
    self.assertIsNotNone(root_metadata)

  def test_custom_metadata_with_legacy_metadata_file(self):
    with CheckpointManager(
        self.directory,
    ) as write_manager:
      self.assertEqual(write_manager.metadata().custom_metadata, {})

    custom_metadata = {'blah': 'hello'}
    legacy_metadata_checkpointer = checkpointer.Checkpointer(
        JsonCheckpointHandler()
    )
    legacy_metadata_checkpointer.save(
        self.directory / checkpoint_manager.METADATA_ITEM_NAME,
        args=args.JsonSave(custom_metadata),
    )

    with CheckpointManager(
        self.directory,
    ) as read_manager:
      self.assertEqual(read_manager.metadata().custom_metadata, custom_metadata)

  def _call_wait_until_finished_concurrently(
      self,
      num_threads: int,
      executor: futures.ThreadPoolExecutor,
      ckpt_manager: CheckpointManager,
  ):
    return [
        executor.submit(ckpt_manager.wait_until_finished)
        for _ in range(num_threads)
    ]

  @parameterized.parameters((None,), (2,))
  def test_async_error_subsequent_save(self, num_waiter_threads: Optional[int]):
    async_timeout_secs = 10
    with (
        futures.ThreadPoolExecutor(
            max_workers=num_waiter_threads,
            thread_name_prefix='save_waiter',
        ) as waiter_executor,
        futures.ThreadPoolExecutor(max_workers=1) as executor,
        CheckpointManager(
            self.directory,
            AsyncCheckpointer(
                test_utils.ErrorCheckpointHandler(
                    PyTreeCheckpointHandler(), executor=executor
                )
            ),
            options=CheckpointManagerOptions(
                max_to_keep=1,
                enable_async_checkpointing=True,
                async_options=checkpoint_manager.AsyncOptions(
                    timeout_secs=async_timeout_secs
                ),
            ),
        ) as manager,
    ):
      # Initiate save which will fail.
      self.assertTrue(manager.save(0, {'a': 1, 'b': 2}))

      if num_waiter_threads is not None:
        waiters = self._call_wait_until_finished_concurrently(
            num_waiter_threads, waiter_executor, manager
        )
        # All waiters should fail. None of them should clear the error.
        for waiter in waiters:
          with self.assertRaises(SystemError):
            waiter.result(timeout=5)

      # Subsequent save should fail. But also clear the error.
      with self.assertRaises(SystemError):
        manager.save(1, {'a': 1, 'b': 2})
      self.assertEmpty(manager.all_steps())
      self.assertLen(step_lib.all_temporary_paths(manager.directory), 1)

      # Error is cleared so it is not re-raised.
      manager.wait_until_finished()
      if num_waiter_threads is not None:
        waiters = self._call_wait_until_finished_concurrently(
            num_waiter_threads, waiter_executor, manager
        )
        # All waiters should succeed.
        for waiter in waiters:
          waiter.result(timeout=5)

  @parameterized.parameters((None,), (2,))
  def test_async_error_wait_until_finished(
      self, num_waiter_threads: Optional[int]
  ):
    async_timeout_secs = 10
    with (
        futures.ThreadPoolExecutor(
            max_workers=num_waiter_threads,
            thread_name_prefix='save_waiter',
        ) as waiter_executor,
        futures.ThreadPoolExecutor(max_workers=1) as executor,
        CheckpointManager(
            self.directory,
            AsyncCheckpointer(
                test_utils.ErrorCheckpointHandler(
                    PyTreeCheckpointHandler(), executor=executor
                )
            ),
            options=CheckpointManagerOptions(
                max_to_keep=1,
                enable_async_checkpointing=True,
                async_options=checkpoint_manager.AsyncOptions(
                    timeout_secs=async_timeout_secs
                ),
            ),
        ) as manager,
    ):
      # Initiate save which will fail.
      self.assertTrue(manager.save(0, {'a': 1, 'b': 2}))

      if num_waiter_threads is not None:
        waiters = self._call_wait_until_finished_concurrently(
            num_waiter_threads, waiter_executor, manager
        )
        # All waiters should fail. None of them should clear the error.
        for waiter in waiters:
          with self.assertRaises(SystemError):
            waiter.result(timeout=5)

      # Next wait_until_finished from main thread should fail. But also clear
      # the error.
      with self.assertRaises(SystemError):
        manager.wait_until_finished()
      self.assertEmpty(manager.all_steps())
      self.assertLen(step_lib.all_temporary_paths(manager.directory), 1)

      # Error is cleared so it is not re-raised.
      manager.wait_until_finished()
      if num_waiter_threads is not None:
        waiters = self._call_wait_until_finished_concurrently(
            num_waiter_threads, waiter_executor, manager
        )
        # All waiters should succeed.
        for waiter in waiters:
          waiter.result(timeout=5)

  @parameterized.parameters((None,), (2,))
  def test_async_error_multiple_steps(self, num_waiter_threads: Optional[int]):
    with CheckpointManager(
        self.directory,
        AsyncCheckpointer(PyTreeCheckpointHandler()),
        options=CheckpointManagerOptions(
            max_to_keep=2,
        ),
    ) as manager:
      self.assertTrue(manager.save(0, {'a': 1, 'b': 2}))
      self.assertTrue(manager.save(1, {'a': 1, 'b': 2}))
      self.assertTrue(manager.save(2, {'a': 1, 'b': 2}))
      manager.wait_until_finished()
      self.assertSameElements([1, 2], manager.all_steps())

    async_timeout_secs = 10
    with (
        futures.ThreadPoolExecutor(
            max_workers=num_waiter_threads,
            thread_name_prefix='save_waiter',
        ) as waiter_executor,
        futures.ThreadPoolExecutor(max_workers=1) as executor,
        CheckpointManager(
            self.directory,
            AsyncCheckpointer(
                test_utils.ErrorCheckpointHandler(
                    PyTreeCheckpointHandler(), executor=executor
                )
            ),
            options=CheckpointManagerOptions(
                max_to_keep=1,
                enable_async_checkpointing=True,
                async_options=checkpoint_manager.AsyncOptions(
                    timeout_secs=async_timeout_secs
                ),
            ),
        ) as manager,
    ):
      # Initiate save which will fail.
      self.assertTrue(manager.save(3, {'a': 1, 'b': 2}))

      if num_waiter_threads is not None:
        waiters = self._call_wait_until_finished_concurrently(
            num_waiter_threads, waiter_executor, manager
        )
        # All waiters should fail. None of them should clear the error.
        for waiter in waiters:
          with self.assertRaises(SystemError):
            waiter.result(timeout=5)

      # Next wait_until_finished from main thread should fail. But also clear
      # the error.
      with self.assertRaises(SystemError):
        manager.wait_until_finished()

      # Error is cleared so it is not re-raised.
      manager.wait_until_finished()
      if num_waiter_threads is not None:
        waiters = self._call_wait_until_finished_concurrently(
            num_waiter_threads, waiter_executor, manager
        )
        # All waiters should succeed.
        for waiter in waiters:
          waiter.result(timeout=5)

  def test_legacy_handler_default_item(self):
    if multihost.is_pathways_backend():
      self.skipTest('Not applicable to Pathways.')

    with CheckpointManager(
        self.directory,
        Checkpointer(MyPyTreeCheckpointHandler()),
    ) as manager:
      self.assertTrue(manager._default_item.get())
      manager.save(0, self.pytree)
      test_utils.assert_tree_equal(self, self.pytree, manager.restore(0))

  def test_legacy_handler_multiple_items(self):
    if multihost.is_pathways_backend():
      self.skipTest('Not applicable to Pathways.')

    with CheckpointManager(
        self.directory,
        {
            'state': Checkpointer(MyPyTreeCheckpointHandler()),
            'metadata': Checkpointer(JsonCheckpointHandler()),
        },
    ) as manager:
      manager.save(0, {'state': self.pytree, 'metadata': {'lang': 'en'}})
      restored = manager.restore(0)
      test_utils.assert_tree_equal(self, self.pytree, restored.state)
      test_utils.assert_tree_equal(self, {'lang': 'en'}, restored.metadata)

  @parameterized.parameters((Checkpointer,), (AsyncCheckpointer,))
  def test_save_restore_legacy_init(self, ckptr):
    if multihost.is_pathways_backend():
      self.skipTest('Not applicable to Pathways.')

    with CheckpointManager(
        self.directory,
        {'params': ckptr(PyTreeCheckpointHandler())},
    ) as manager:
      self.assertTrue(manager.save(0, {'params': self.pytree}))
      self.wait_if_async(manager)
      restored = manager.restore(
          0,
          restore_kwargs={
              'params': {
                  'restore_args': None,
              }
          },
      )['params']
      test_utils.assert_tree_equal(self, self.pytree, restored)

  @parameterized.parameters((Checkpointer,), (AsyncCheckpointer,))
  def test_save_restore_default_item_legacy_init(self, ckptr):
    with CheckpointManager(
        self.directory, ckptr(PyTreeCheckpointHandler())
    ) as manager:
      self.assertTrue(manager._default_item.get())
      self.assertTrue(manager.save(0, self.pytree))
      self.wait_if_async(manager)
      restored = manager.restore(
          0, restore_kwargs={'restore_args': self.pytree_restore_args}
      )
      expected = self.pytree
      test_utils.assert_tree_equal(self, expected, restored)

  @parameterized.parameters((Checkpointer,), (AsyncCheckpointer,))
  def test_multiple_items_legacy_init(self, ckptr):
    """Test multiple different items."""
    with CheckpointManager(
        self.directory,
        {
            'params': ckptr(PyTreeCheckpointHandler()),
            'arr': ckptr(ArrayCheckpointHandler()),
            'metadata': Checkpointer(
                JsonCheckpointHandler(filename='metadata')
            ),
        },
    ) as manager:
      metadata = {
          'VERSION': 2,
          'optimizer': {
              'lr': 0.001,
              'type': 'adam',
          },
      }

      self.assertTrue(
          manager.save(
              0,
              {
                  'params': self.pytree,
                  'arr': np.arange(16),
                  'metadata': metadata,
              },
          )
      )
      self.wait_if_async(manager)
      restored = manager.restore(
          0,
          {
              'params': self.empty_pytree,
              'arr': None,
              'metadata': None,
          },
          restore_kwargs={'params': {'restore_args': self.pytree_restore_args}},
      )
      restored_params, restored_arr, restored_metadata = (
          restored['params'],
          restored['arr'],
          restored['metadata'],
      )
      expected_params = self.pytree
      test_utils.assert_tree_equal(self, expected_params, restored_params)
      np.testing.assert_array_equal(restored_arr, np.arange(16))
      self.assertDictEqual(metadata, restored_metadata)

  @parameterized.product(
      array_metadata_store=(None, array_metadata_store_lib.Store())
  )
  def test_save_restore_local_fs(
      self, array_metadata_store: array_metadata_store_lib.Store | None
  ):
    """Test saving and restoring to local filesystem with various supported handlers."""

    # each process have different directory
    test_dir = self.directory / f'{multihost.process_index()}'
    test_dir.mkdir(parents=True)

    fn = lambda ty: issubclass(ty, jax.Array)
    with test_utils.register_type_handler(
        jax.Array,
        type_handlers.ArrayHandler(
            primary_host=None,
            replica_id=None,
            use_replica_parallel=False,
            array_metadata_store=array_metadata_store,
        ),
        fn,
    ):
      options = CheckpointManagerOptions(
          enable_async_checkpointing=True,
          multiprocessing_options=checkpoint_manager.MultiprocessingOptions(
              primary_host=None
          ),
      )
      with CheckpointManager(
          test_dir,
          options=options,
          item_handlers={
              'pytree': PyTreeCheckpointHandler(
                  multiprocessing_options=options.multiprocessing_options
              ),
              'json': JsonCheckpointHandler(
                  multiprocessing_options=options.multiprocessing_options
              ),
              'standard': StandardCheckpointHandler(
                  multiprocessing_options=options.multiprocessing_options
              ),
          },
      ) as manager:
        json_data = {'a': 1, 'b': 2}
        self.assertTrue(
            manager.save(
                0,
                args=args.Composite(
                    pytree=args.PyTreeSave(self.pytree),
                    json=args.JsonSave(json_data),
                    standard=args.StandardSave(self.doubled_pytree),
                ),
            )
        )
        self.wait_if_async(manager)
        restored = manager.restore(
            0,
            args=args.Composite(
                pytree=args.PyTreeRestore(),
                json=args.JsonRestore(),
                standard=args.StandardRestore(),
            ),
        )
        test_utils.assert_tree_equal(self, self.pytree, restored['pytree'])
        self.assertDictEqual(json_data, restored['json'])
        test_utils.assert_tree_equal(
            self, self.doubled_pytree, restored['standard']
        )

  def test_save_and_restore_composite_logger(self):
    test_logger = standard_logger.StandardLogger()
    with CheckpointManager(
        self.directory,
        AsyncCheckpointer(PyTreeCheckpointHandler()),
        options=CheckpointManagerOptions(
            max_to_keep=2,
        ),
        logger=composite_logger.CompositeLogger(test_logger),
    ) as manager:
      # Check that the step details are logged correctly.
      with self.assertLogs(level='INFO') as log_output:
        expected_step_statistics = step_stats.SaveStepStatistics()
        expected_step_statistics.step = 0
        expected_step_statistics.event_type = 'save'
        expected_step_statistics.reached_preemption = False
        expected_step_statistics.synchronous = False
        expected_step_statistics.directory = str(self.directory)
        self.assertTrue(manager.save(0, {'a': 1, 'b': 2}))
        dict_start_index = str(log_output[-1][-1]).find('{')
        step_statistics = ast.literal_eval(
            log_output[-1][-1][dict_start_index:]
        )
        self.assertEqual(expected_step_statistics.step, step_statistics['step'])
        self.assertEqual(
            expected_step_statistics.event_type, step_statistics['event_type']
        )
        self.assertEqual(
            expected_step_statistics.synchronous, step_statistics['synchronous']
        )
        self.assertEqual(
            expected_step_statistics.reached_preemption,
            step_statistics['reached_preemption'],
        )

        # Check that all the timestamps are set.
        self.assertIsNone(step_statistics['preemption_received_at'])
        self.assertIsNotNone(
            step_statistics['checkpoint_manager_blocking_start_time']
        )
        self.assertIsNotNone(
            step_statistics['checkpoint_manager_blocking_duration_secs']
        )
        self.assertIsNotNone(step_statistics['wait_for_prev_start_time'])
        self.assertIsNotNone(step_statistics['wait_for_prev_duration_secs'])
        self.assertIsNotNone(
            step_statistics['checkpointer_blocking_start_time']
        )
        self.assertIsNotNone(
            step_statistics['checkpointer_blocking_duration_secs']
        )

        self.assertIsNotNone(step_statistics['get_old_steps_start_time'])
        self.assertIsNotNone(step_statistics['get_old_steps_duration_secs'])
        self.assertEqual(
            step_statistics['directory'], expected_step_statistics.directory
        )
        self.wait_if_async(manager)
        manager.restore(0)

        expected_step_statistics = step_stats.RestoreStepStatistics()
        expected_step_statistics.step = 0
        expected_step_statistics.event_type = 'restore'
        expected_step_statistics.directory = str(self.directory)

        dict_start_index = str(log_output[-1][-1]).find('{')
        step_statistics = ast.literal_eval(
            log_output[-1][-1][dict_start_index:]
        )
        self.assertEqual(expected_step_statistics.step, step_statistics['step'])
        self.assertEqual(
            expected_step_statistics.event_type, step_statistics['event_type']
        )
        self.assertEqual(
            expected_step_statistics.directory,
            step_statistics['directory'],
        )
        self.assertIsNotNone(step_statistics['checkpointer_start_time'])
        self.assertIsNotNone(step_statistics['checkpointer_duration_secs'])
        self.assertIsNotNone(step_statistics['checkpoint_manager_start_time'])
        self.assertIsNotNone(
            step_statistics['checkpoint_manager_duration_secs']
        )

  def test_save_and_restore_standard_logger(self):
    with CheckpointManager(
        self.directory,
        AsyncCheckpointer(PyTreeCheckpointHandler()),
        options=CheckpointManagerOptions(
            max_to_keep=2,
        ),
    ) as manager:
      # Check that the step details are logged correctly.
      with self.assertLogs(level='INFO') as log_output:
        expected_step_statistics = step_stats.SaveStepStatistics()
        expected_step_statistics.step = 0
        expected_step_statistics.event_type = 'save'
        expected_step_statistics.reached_preemption = False
        expected_step_statistics.synchronous = False
        expected_step_statistics.directory = str(self.directory)
        self.assertTrue(manager.save(0, {'a': 1, 'b': 2}))
        dict_start_index = str(log_output[-1][-1]).find('{')
        step_statistics = ast.literal_eval(
            log_output[-1][-1][dict_start_index:]
        )
        self.assertEqual(expected_step_statistics.step, step_statistics['step'])
        self.assertEqual(
            expected_step_statistics.event_type, step_statistics['event_type']
        )
        self.assertEqual(
            expected_step_statistics.synchronous, step_statistics['synchronous']
        )
        self.assertEqual(
            expected_step_statistics.reached_preemption,
            step_statistics['reached_preemption'],
        )

        # Check that all the timestamps are set.
        self.assertIsNone(step_statistics['preemption_received_at'])
        self.assertIsNotNone(
            step_statistics['checkpoint_manager_blocking_start_time']
        )
        self.assertIsNotNone(
            step_statistics['checkpoint_manager_blocking_duration_secs']
        )
        self.assertIsNotNone(step_statistics['wait_for_prev_start_time'])
        self.assertIsNotNone(step_statistics['wait_for_prev_duration_secs'])
        self.assertIsNotNone(
            step_statistics['checkpointer_blocking_start_time']
        )
        self.assertIsNotNone(
            step_statistics['checkpointer_blocking_duration_secs']
        )

        self.assertIsNotNone(step_statistics['get_old_steps_start_time'])
        self.assertIsNotNone(step_statistics['get_old_steps_duration_secs'])
        self.assertEqual(
            step_statistics['directory'], expected_step_statistics.directory
        )
        self.wait_if_async(manager)
        manager.restore(0)

        expected_step_statistics = step_stats.RestoreStepStatistics()
        expected_step_statistics.step = 0
        expected_step_statistics.event_type = 'restore'
        expected_step_statistics.directory = str(self.directory)

        dict_start_index = str(log_output[-1][-1]).find('{')
        step_statistics = ast.literal_eval(
            log_output[-1][-1][dict_start_index:]
        )
        self.assertEqual(expected_step_statistics.step, step_statistics['step'])
        self.assertEqual(
            expected_step_statistics.event_type, step_statistics['event_type']
        )
        self.assertEqual(
            expected_step_statistics.directory,
            step_statistics['directory'],
        )
        self.assertIsNotNone(step_statistics['checkpointer_start_time'])
        self.assertIsNotNone(step_statistics['checkpointer_duration_secs'])
        self.assertIsNotNone(step_statistics['checkpoint_manager_start_time'])
        self.assertIsNotNone(
            step_statistics['checkpoint_manager_duration_secs']
        )

  def test_configure_atomicity(self):
    """Test case."""
    with CheckpointManager(
        self.directory,
        options=CheckpointManagerOptions(
            temporary_path_class=atomicity.CommitFileTemporaryPath
        ),
    ) as manager:
      manager.save(0, args=args.PyTreeSave(self.pytree))
      manager.wait_until_finished()
      restored = manager.restore(
          0, args=args.PyTreeRestore(self.pytree_restore_args)
      )
      test_utils.assert_tree_equal(self, self.pytree, restored)
      self.assertTrue(
          (self.directory / '0' / atomicity.COMMIT_SUCCESS_FILE).exists()
      )  # pylint: disable=protected-access

  def test_default_item_mode_with_handler_registry(self):
    # Test save args that mimics `args.StandardSave`. Required since
    # `args.StandardSave` and `args.StandardRestore` is already
    # registered in the global handler registry by default.
    class _TestSaveArgs(args.StandardSave):
      ...

    class _TestRestoreArgs(args.StandardRestore):
      ...

    step = 10

    handler_registry = handler_registration.DefaultCheckpointHandlerRegistry()
    handler = handlers.StandardCheckpointHandler()
    handler_registry.add(
        None,
        _TestSaveArgs,
        handler,
    )
    handler_registry.add(
        None,
        _TestRestoreArgs,
        handler,
    )

    state = {'step': step}
    placeholder_state = {'step': 0}
    with CheckpointManager(
        self.directory,
        handler_registry=handler_registry,
    ) as manager:
      self.assertIsNone(manager._default_item.get())
      manager.save(step, args=_TestSaveArgs(state))
      self.assertTrue(manager._default_item.get())
      manager.wait_until_finished()
      self.wait_if_async(manager)

      restored = manager.restore(
          step,
          args=_TestRestoreArgs(placeholder_state),
      )
      test_utils.assert_tree_equal(self, state, restored)

      # Restore without args.
      restored_with_none = manager.restore(
          step,
          args=None,
      )
      test_utils.assert_tree_equal(self, state, restored_with_none)

      # Restore metadata.
      self.assertDictEqual(
          manager.metadata(step).item_metadata.tree,
          {
              'step': value_metadata.ScalarMetadata(
                  name='step',
                  directory=epath.Path(self.directory / _DEFAULT_ITEM_NAME),
                  dtype=jnp.int64,
              )
          },
      )

    # Try restoring with a different manager.
    with CheckpointManager(
        self.directory,
        handler_registry=handler_registry,
    ) as manager:
      self.assertIsNone(manager._default_item.get())
      restored_different_manager = manager.restore(
          step,
          args=_TestRestoreArgs(placeholder_state),
      )
      self.assertTrue(manager._default_item.get())
      test_utils.assert_tree_equal(self, state, restored_different_manager)

  def test_multi_item_mode_with_handler_registry(self):
    step = 0
    state_to_save = {'small_state': 1}
    handler_registry = handler_registration.DefaultCheckpointHandlerRegistry()
    handler = handlers.PyTreeCheckpointHandler()
    handler_registry.add(
        'dataset',
        args.PyTreeSave,
        handler,
    )
    handler_registry.add(
        'dataset',
        args.PyTreeRestore,
        handler,
    )
    with CheckpointManager(
        self.directory,
        handler_registry=handler_registry,
    ) as manager:
      manager.save(
          step,
          args=args.Composite(
              state=args.StandardSave(state_to_save),
              dataset=args.PyTreeSave(self.pytree),
          ),
      )
      self.wait_if_async(manager)

      restored = manager.restore(
          step,
          args=args.Composite(
              state=args.StandardRestore(),
              dataset=args.PyTreeRestore(restore_args=self.pytree_restore_args),
          ),
      )
      test_utils.assert_tree_equal(self, state_to_save, restored.state)
      test_utils.assert_tree_equal(self, self.pytree, restored.dataset)

  @parameterized.parameters((None,), 'checkpoint')
  def test_default_init_default_item(self, step_prefix):
    pytree = {'a': 1, 'b': 2}
    options = CheckpointManagerOptions(step_prefix=step_prefix)
    with CheckpointManager(
        self.directory / 'default_item', options=options
    ) as manager:
      self.assertIsNone(manager._default_item.get())
      self.assertTrue(manager.save(0, args=args.StandardSave(pytree)))
      self.assertTrue(manager._default_item.get())
      self.assertDictEqual(pytree, manager.restore(0, args=None))
      self.assertDictEqual(
          pytree, manager.restore(0, args=args.StandardRestore())
      )
      with self.assertRaisesRegex(
          ValueError,
          r'Cannot provide `args` of type `Composite` when dealing with a '
          r'single, unnamed \(default\) checkpointable object.',
      ):
        manager.save(0, args=args.Composite(state=args.StandardSave(pytree)))

    with CheckpointManager(
        self.directory / 'default_item', options=options
    ) as manager:
      self.assertIsNone(manager._default_item.get())
      self.assertDictEqual(pytree, manager.restore(0))
      self.assertIsNotNone(manager.metadata(0).item_metadata)
      self.assertDictEqual(
          pytree, manager.restore(0, args=args.StandardRestore())
      )
      self.assertTrue(manager._default_item.get())

  @parameterized.parameters((None,), 'checkpoint')
  def test_default_init_multi_item(self, step_prefix):
    pytree = {'a': 1, 'b': 2}
    options = CheckpointManagerOptions(step_prefix=step_prefix)
    with CheckpointManager(
        self.directory / 'multi_item', options=options
    ) as manager:
      self.assertTrue(
          manager.save(
              0,
              args=args.Composite(
                  state=args.StandardSave(pytree),
                  embeddings=args.StandardSave(pytree),
              ),
          )
      )
      restored = manager.restore(0, args=None)
      self.assertDictEqual(pytree, restored.state)
      self.assertDictEqual(pytree, restored.embeddings)
      restored = manager.restore(0, args=args.Composite(state=None))
      self.assertDictEqual(pytree, restored.state)
      self.assertNotIn('embeddings', restored)

      with self.assertRaisesRegex(
          ValueError,
          'Must provide `args` of type `Composite` when dealing with multiple'
          ' checkpointable objects.',
      ):
        manager.save(1, args=args.StandardSave(pytree))

    with CheckpointManager(
        self.directory / 'multi_item', options=options
    ) as manager:
      restored = manager.restore(0)
      self.assertDictEqual(pytree, restored.state)
      self.assertDictEqual(pytree, restored.embeddings)
      item_metadata = manager.metadata(0).item_metadata
      self.assertIn('state', item_metadata)
      self.assertIn('embeddings', item_metadata)
      self.assertIsNotNone(item_metadata.state)
      self.assertIsNotNone(item_metadata.embeddings)

    with CheckpointManager(
        self.directory / 'multi_item', options=options
    ) as manager:
      with self.assertRaisesRegex(
          ValueError,
          'Provided `None` for `CheckpointArgs`, and the `CheckpointHandler`'
          ' for item "state" was not configured.',
      ):
        manager.restore(0, args=args.Composite(state=None))

    with CheckpointManager(
        self.directory / 'multi_item', options=options
    ) as manager:
      restored = manager.restore(
          0,
          args=args.Composite(
              state=args.StandardRestore(),
              embeddings=args.StandardRestore(),
          ),
      )
      self.assertDictEqual(pytree, restored.state)
      self.assertDictEqual(pytree, restored.embeddings)
      restored = manager.restore(0, args=args.Composite(state=None))
      self.assertDictEqual(pytree, restored.state)
      self.assertNotIn('embeddings', restored)

  def test_default_init_multi_item_dynamic_items(self):
    pytree = {'a': 1, 'b': 2}
    with CheckpointManager(self.directory / 'multi_item') as manager:
      self.assertTrue(
          manager.save(
              0,
              args=args.Composite(
                  state=args.StandardSave(pytree),
                  embeddings=args.StandardSave(pytree),
              ),
          )
      )
      restored = manager.restore(0, args=None)
      self.assertDictEqual(pytree, restored.state)
      self.assertDictEqual(pytree, restored.embeddings)

      self.assertTrue(
          manager.save(
              1,
              args=args.Composite(
                  state=args.StandardSave(pytree),
              ),
          )
      )
      restored = manager.restore(1, args=None)
      self.assertDictEqual(pytree, restored.state)
      self.assertNotIn('embeddings', restored)

      self.assertTrue(
          manager.save(
              2,
              args=args.Composite(
                  state=args.StandardSave(pytree),
                  embeddings=args.StandardSave(pytree),
                  extra=args.StandardSave(pytree),
              ),
          )
      )
      restored = manager.restore(2, args=None)
      self.assertDictEqual(pytree, restored.state)
      self.assertDictEqual(pytree, restored.embeddings)
      self.assertDictEqual(pytree, restored.extra)

  def test_save_root_metadata(self):
    with CheckpointManager(
        self.directory,
        options=CheckpointManagerOptions(
            save_root_metadata=True,
        ),
        metadata={'state': 123},
    ) as manager:
      self.assertEqual(manager.metadata().custom_metadata, {'state': 123})
    file_path = metadata_lib.root_metadata_file_path(
        self.directory / checkpoint_manager.METADATA_ITEM_NAME
    )
    self.assertTrue(file_path.exists())
    metadata_store = metadata_lib.metadata_store(
        enable_write=False, blocking_write=False
    )
    serialized_metadata = metadata_store.read(file_path)
    self.assertIsNotNone(serialized_metadata)
    self.assertEqual(
        root_metadata_serialization.deserialize(
            serialized_metadata
        ).custom_metadata,
        {'state': 123},
    )

  def test_save_root_metadata_disabled(self):
    metadata_dir = self.directory / checkpoint_manager.METADATA_ITEM_NAME
    with CheckpointManager(
        self.directory,
        options=CheckpointManagerOptions(
            save_root_metadata=False,
        ),
        metadata={'state': 123},
    ) as manager:
      self.assertEqual(manager.metadata().custom_metadata, {})
    self.assertFalse(metadata_dir.exists())

    with CheckpointManager(
        self.directory,
        options=CheckpointManagerOptions(
            save_root_metadata=True,
        ),
        metadata={'state': 123},
    ) as manager:
      self.assertEqual(manager.metadata().custom_metadata, {'state': 123})
    self.assertTrue(metadata_lib.root_metadata_file_path(metadata_dir).exists())

  def test_save_step_metadata(self):
    with CheckpointManager(
        self.directory,
        options=CheckpointManagerOptions(
            save_root_metadata=True,
        ),
    ) as manager:
      self.assertTrue(
          manager.save(
              0,
              args=args.StandardSave(self.pytree),
              custom_metadata={'a': 1, 'b': 2},
          )
      )
    serialized_metadata = metadata_lib.metadata_store(enable_write=False).read(
        metadata_lib.step_metadata_file_path(self.directory / '0')
    )
    self.assertIsNotNone(serialized_metadata)
    step_metadata = step_metadata_serialization.deserialize(serialized_metadata)

    self.assertDictEqual(
        step_metadata.item_handlers,
        {
            'default': handlers.StandardCheckpointHandler().typestr(),
        },
    )
    self.assertIsNone(step_metadata.item_metadata)
    self.assertEqual(step_metadata.metrics, {})
    self.assertEqual(
        step_metadata.performance_metrics,
        step_stats.SaveStepStatistics(),
    )
    self.assertGreater(step_metadata.init_timestamp_nsecs, 0)
    self.assertGreater(step_metadata.commit_timestamp_nsecs, 0)
    self.assertEqual(step_metadata.custom_metadata, {'a': 1, 'b': 2})

  @parameterized.named_parameters(
      ('checkpointer', False),
      ('async_checkpointer', True),
  )
  def test_metadata_save_preemption(self, enable_async):
    with mock.patch.object(checkpointer.Checkpointer, '_save_step_metadata'):
      manager = CheckpointManager(
          self.directory,
          item_names=('params',),
          options=CheckpointManagerOptions(
              enable_async_checkpointing=enable_async,
          ),
      )
      self.assertTrue(self.save_params(0, manager, self.pytree))
      self.wait_if_async(manager)

      serialized_metadata = manager._checkpointer._metadata_store.read(
          metadata_lib.step_metadata_file_path(self.directory)
      )
      self.assertIsNone(serialized_metadata)

  @parameterized.named_parameters(
      ('checkpointer', False),
      ('async_checkpointer', True),
  )
  def test_metadata_finalize_preemption(self, enable_async):
    with (
        mock.patch.object(atomicity.CommitFileTemporaryPath, 'finalize'),
        mock.patch.object(atomicity.AtomicRenameTemporaryPath, 'finalize'),
    ):
      manager = CheckpointManager(
          self.directory,
          item_names=('params',),
          options=CheckpointManagerOptions(
              enable_async_checkpointing=enable_async,
          ),
      )
      self.assertTrue(self.save_params(0, manager, self.pytree))
      self.wait_if_async(manager)

      step_dirs = list(self.directory.iterdir())
      self.assertLen(step_dirs, 1)
      step_dir = step_dirs[0]
      self.assertTrue(step_lib.is_path_temporary(step_dir))
      self.assertTrue(step_dir.exists())
      self.assertTrue(metadata_lib.step_metadata_file_path(step_dir).exists())
      self.assertFalse(
          metadata_lib.step_metadata_file_path(self.directory).exists()
      )
      serialized_metadata = manager._checkpointer._metadata_store.read(
          metadata_lib.step_metadata_file_path(step_dir)
      )
      self.assertIsNotNone(serialized_metadata)
      step_metadata = step_metadata_serialization.deserialize(
          serialized_metadata,
      )

      self.assertNotEmpty(step_metadata.item_handlers)
      self.assertIsNone(step_metadata.item_metadata)
      self.assertEmpty(step_metadata.metrics)
      self.assertEqual(
          step_metadata.performance_metrics, step_stats.SaveStepStatistics()
      )
      self.assertGreater(step_metadata.init_timestamp_nsecs, 0)
      self.assertIsNone(step_metadata.commit_timestamp_nsecs)
      self.assertEmpty(step_metadata.custom_metadata)

  @parameterized.named_parameters(
      ('checkpointer', False),
      ('async_checkpointer', True),
  )
  def test_item_metadata_access_with_no_handlers(self, enable_async):
    with CheckpointManager(
        self.directory,
        options=CheckpointManagerOptions(
            enable_async_checkpointing=enable_async,
        ),
    ) as manager:
      manager.save(0, args=args.Composite(state=args.StandardSave(self.pytree)))

    with CheckpointManager(
        self.directory,
    ) as manager:
      self.assertSameElements(manager.metadata(0).item_metadata, ['state'])

  @parameterized.named_parameters(
      ('checkpointer', False),
      ('async_checkpointer', True),
  )
  def test_restore_with_no_handlers(self, enable_async):
    pytree = {'a': 1, 'b': 2}
    with CheckpointManager(
        self.directory,
        options=CheckpointManagerOptions(
            enable_async_checkpointing=enable_async,
        ),
    ) as manager:
      manager.save(0, args=args.Composite(state=args.StandardSave(pytree)))

    with CheckpointManager(
        self.directory,
    ) as manager:
      self.assertDictEqual(pytree, manager.restore(0).state)

  @parameterized.named_parameters(
      ('checkpointer', False),
      ('async_checkpointer', True),
  )
  def test_item_metadata_access_no_handlers_and_default_item(
      self, enable_async
  ):
    with CheckpointManager(
        self.directory,
        options=CheckpointManagerOptions(
            enable_async_checkpointing=enable_async,
        ),
    ) as manager:
      manager.save(0, args=args.StandardSave(self.pytree))

    with self.subTest('restore'):
      with CheckpointManager(
          self.directory,
      ) as manager:
        manager.restore(0)
        self.assertIsNotNone(manager.metadata(0).item_metadata)

    with self.subTest('no_restore'):
      with CheckpointManager(
          self.directory,
      ) as manager:
        self.assertIsNone(manager.metadata(0).item_metadata)

  def test_root_metadata_save(self):
    with CheckpointManager(self.directory, metadata={'state': 123}) as manager:
      self.assertEqual(
          manager.metadata(),
          metadata_lib.RootMetadata(
              custom_metadata={'state': 123},
          ),
      )

  def test_read_only_manager_does_not_save_root_metadata(self):
    with CheckpointManager(
        self.directory,
        options=CheckpointManagerOptions(
            read_only=True,
        ),
        metadata={'state': 123},
    ) as manager:
      self.assertEqual(manager.metadata().custom_metadata, {'state': 123})

    with CheckpointManager(
        self.directory,
    ) as manager:
      self.assertEmpty(manager.metadata().custom_metadata)

  def test_root_metadata_does_not_overwrite(self):
    custom_metadata = {'state': 123}
    with CheckpointManager(self.directory, metadata=custom_metadata) as manager:
      self.assertEqual(manager.metadata().custom_metadata, custom_metadata)

    with CheckpointManager(
        self.directory, metadata={'new_state': 456}
    ) as manager:
      self.assertEqual(manager.metadata().custom_metadata, custom_metadata)

  @parameterized.named_parameters(
      ('checkpointer', False),
      ('async_checkpointer', True),
  )
  def test_step_metadata_save(self, enable_async):
    with CheckpointManager(
        self.directory,
        options=CheckpointManagerOptions(
            enable_async_checkpointing=enable_async,
            best_fn=lambda metrics: metrics['loss'],
        ),
        metadata={'state': 123},
    ) as manager:
      manager.save(
          0,
          args=args.Composite(
              state=args.StandardSave(self.pytree),
              dataset=args.StandardSave(self.pytree),
          ),
          metrics={'loss': 1.0},
      )
      self.wait_if_async(manager)
      step_metadata = manager.metadata(0)
      self.assertIsNotNone(step_metadata)
      self.assertDictContainsSubset(
          {
              'state': (
                  'orbax.checkpoint._src.handlers.standard_checkpoint_handler.StandardCheckpointHandler'
              ),
              'dataset': (
                  'orbax.checkpoint._src.handlers.standard_checkpoint_handler.StandardCheckpointHandler'
              ),
          },
          step_metadata.item_handlers,
      )
      self.assertSameElements(
          step_metadata.item_metadata, ['state', 'dataset', 'metrics']
      )
      self.assertIsNotNone(step_metadata.item_metadata['state'])
      self.assertIsNotNone(step_metadata.item_metadata['dataset'])
      self.assertDictEqual(step_metadata.metrics, {'loss': 1.0})
      self.assertIsInstance(
          step_metadata.performance_metrics, step_stats.SaveStepStatistics
      )
      self.assertGreater(step_metadata.init_timestamp_nsecs, 0)
      self.assertGreater(step_metadata.commit_timestamp_nsecs, 0)
      # Custom user metadata is currently only saved in the root metadata.
      # See b/390198468.
      self.assertEmpty(step_metadata.custom_metadata)

  @parameterized.parameters((True,), (False,))
  def test_save_decision_policy(self, with_initial_save):
    policies = [
        save_decision_policy_lib.FixedIntervalPolicy(3),
        save_decision_policy_lib.SpecificStepsPolicy(steps=[4]),
        save_decision_policy_lib.FixedIntervalPolicy(5),
    ]
    if with_initial_save:
      policies.append(save_decision_policy_lib.InitialSavePolicy())
    policy = save_decision_policy_lib.AnySavePolicy(policies)
    with CheckpointManager(
        self.directory,
        options=CheckpointManagerOptions(
            save_interval_steps=1,  # ignored
            save_decision_policy=policy,
        ),
    ) as manager:
      expected_steps = {3, 4, 5, 6, 9, 10}
      if with_initial_save:
        expected_steps.add(1)
      for step in range(1, 11):
        saved = manager.save(step, args=args.StandardSave(self.pytree))
        self.assertEqual(saved, step in expected_steps)

  def test_sync_continuous_checkpointing(self):
    with CheckpointManager(
        self.directory,
        options=CheckpointManagerOptions(
            save_interval_steps=5,  # ignored
            enable_async_checkpointing=False,
            save_decision_policy=save_decision_policy_lib.ContinuousCheckpointingPolicy(),
        ),
    ) as manager:
      for step in range(10):
        self.assertTrue(manager.save(step, args=args.StandardSave(self.pytree)))

  def test_async_continuous_checkpointing(self):
    original_on_commit_callback = async_checkpointer._on_commit_callback

    def mock_on_commit_callback(*a, **kw):
      time.sleep(3)
      return original_on_commit_callback(*a, **kw)

    with mock.patch.object(
        async_checkpointer, '_on_commit_callback', new=mock_on_commit_callback
    ):
      with CheckpointManager(
          self.directory,
          options=CheckpointManagerOptions(
              save_interval_steps=1,  # ignored
              save_decision_policy=save_decision_policy_lib.ContinuousCheckpointingPolicy(),
          ),
      ) as manager:
        self.assertTrue(manager.save(0, args=args.StandardSave(self.pytree)))
        self.assertFalse(manager.save(1, args=args.StandardSave(self.pytree)))
        self.assertFalse(manager.save(2, args=args.StandardSave(self.pytree)))
        self.wait_if_async(manager)
        self.assertTrue(manager.save(3, args=args.StandardSave(self.pytree)))
        self.assertFalse(manager.save(4, args=args.StandardSave(self.pytree)))
        self.wait_if_async(manager)
        self.assertTrue(manager.save(5, args=args.StandardSave(self.pytree)))
        self.assertFalse(manager.save(6, args=args.StandardSave(self.pytree)))

  def test_slow_save_logs_warning(self):
    original_on_commit_callback = async_checkpointer._on_commit_callback

    def mock_on_commit_callback(*a, **kw):
      time.sleep(1.1)
      return original_on_commit_callback(*a, **kw)

    with mock.patch.object(
        async_checkpointer, '_on_commit_callback', new=mock_on_commit_callback
    ):
      with CheckpointManager(
          self.directory,
          options=CheckpointManagerOptions(
              save_interval_steps=1,
              enable_async_checkpointing=True,
          ),
      ) as manager:
        self.assertTrue(manager.save(0, args=args.StandardSave(self.pytree)))
        # The second save should be blocked by the first one for > 1s,
        # thus triggering the warning.
        with self.assertLogs(level='WARNING') as log_context:
          self.assertTrue(manager.save(1, args=args.StandardSave(self.pytree)))
        self.assertLen(log_context.records, 1)
        self.assertRegex(
            log_context.records[0].getMessage(),
            'Waiting for previous save to complete took.*',
        )

  def test_preservation_joint_policy(self):
    """Tests combining multiple policies."""
    n_to_keep = 2
    interval_steps = 4
    custom_steps = [0, 3]
    all_metrics = {'loss': [5, 2, 4, 3, 7, 10, 11, 9, 8, 6, 12, 1]}
    policies = [
        preservation_policy_lib.BestN(
            get_metric_fn=lambda metrics: metrics['loss'],
            reverse=True,
            n=n_to_keep,
        ),  # 1, 11
        preservation_policy_lib.EveryNSteps(
            interval_steps=interval_steps
        ),  # 0, 4, 8
        preservation_policy_lib.CustomSteps(steps=custom_steps),
    ]
    policy = preservation_policy_lib.AnyPreservationPolicy(policies)
    options = CheckpointManagerOptions(preservation_policy=policy)
    with CheckpointManager(self.directory, options=options) as manager:
      num_steps = 12
      for step in range(num_steps):
        metrics = {k: v[step] for k, v in all_metrics.items()}
        self.save_params(step, manager, params=self.pytree, metrics=metrics)
      self.wait_if_async(manager)
      self.assertCountEqual([0, 1, 3, 4, 8, 11], manager.all_steps())

  def test_sync_continuous_checkpointing_with_minimum_interval_secs(self):
    with CheckpointManager(
        self.directory,
        options=CheckpointManagerOptions(
            save_interval_steps=5,  # ignored
            enable_async_checkpointing=False,
            save_decision_policy=save_decision_policy_lib.ContinuousCheckpointingPolicy(
                minimum_interval_secs=2
            ),
        ),
    ) as manager:
      self.assertTrue(manager.save(0, args=args.StandardSave(self.pytree)))
      self.assertFalse(manager.save(1, args=args.StandardSave(self.pytree)))
      time.sleep(2)
      self.assertTrue(manager.save(2, args=args.StandardSave(self.pytree)))

  @parameterized.parameters(
      (0,),
      (10,),
  )
  def test_initial_save(self, step):
    # By default, the first checkpoint always gets saved.
    with CheckpointManager(
        self.directory,
    ) as manager:
      self.assertTrue(manager.save(step, args=args.StandardSave(self.pytree)))

  def test_single_host_load_and_broadcast(self):

    if multihost.is_pathways_backend():
      self.skipTest('Not applicable to Pathways')

    options = CheckpointManagerOptions(
        save_interval_steps=1,
        single_host_load_and_broadcast=True,
        step_name_format=step_lib.standard_name_format(
            step_prefix='step',
            step_format_fixed_length=4,
            single_host_load_and_broadcast=True,
        ),
    )
    with CheckpointManager(self.directory, options=options) as manager:
      self.assertTrue(manager.save(0, args=args.StandardSave(self.pytree)))
      self.assertTrue(manager.save(1, args=args.StandardSave(self.pytree)))
      self.assertTrue(manager.save(2, args=args.StandardSave(self.pytree)))

    with self.subTest('reload'):
      with CheckpointManager(self.directory, options=options) as manager:
        manager.reload()
        self.assertEqual(manager.all_steps(), [0, 1, 2])

    with self.subTest('init_with_name_format'):
      with CheckpointManager(self.directory, options=options) as manager:
        self.assertEqual(manager.all_steps(), [0, 1, 2])

    with self.subTest('init_without_name_format'):
      with CheckpointManager(
          self.directory,
          options=CheckpointManagerOptions(
              save_interval_steps=1,
              step_prefix='step',
              step_format_fixed_length=4,
              single_host_load_and_broadcast=True,
          ),
      ) as manager:
        self.assertEqual(manager.all_steps(), [0, 1, 2])

    with self.subTest('broadcast_impl'):
      with mock.patch.object(
          multihost, 'broadcast_one_to_all'
      ) as mock_broadcast:
        mock_broadcast.side_effect = [3, np.array([0, 1, 2])]

        with CheckpointManager(self.directory, options=options):
          self.assertEqual(mock_broadcast.call_count, 2)

  def test_time_between_consecutive_saves_metric_sync(self):
    with (
        mock.patch('time.time') as mock_time,
        mock.patch(
            'jax.monitoring.record_event_duration_secs'
        ) as mock_record_event,
    ):
      mock_time.return_value = 1.0
      options = CheckpointManagerOptions(save_interval_steps=2)
      with CheckpointManager(self.directory, options=options) as manager:
        self.assertIsNone(manager._last_save_time)

        # save step 0
        mock_time.return_value = 2.0
        self.assertTrue(manager.save(0, args=args.PyTreeSave(self.pytree)))
        self.wait_if_async(manager)
        for call in mock_record_event.call_args_list:
          self.assertNotEqual(
              call[0][0],
              '/jax/orbax/checkpoint_manager/time_between_consecutive_saves_secs',
          )
        self.assertEqual(manager._last_save_time, 2.0)
        mock_record_event.reset_mock()

        # save step 1, should_save is False
        mock_time.return_value = 3.0
        self.assertFalse(manager.save(1, args=args.PyTreeSave(self.pytree)))
        self.wait_if_async(manager)
        for call in mock_record_event.call_args_list:
          self.assertNotEqual(
              call[0][0],
              '/jax/orbax/checkpoint_manager/time_between_consecutive_saves_secs',
          )
        self.assertEqual(manager._last_save_time, 2.0)  # not updated
        mock_record_event.reset_mock()

        # save step 2, should_save is True
        mock_time.return_value = 5.0
        self.assertTrue(manager.save(2, args=args.PyTreeSave(self.pytree)))
        self.wait_if_async(manager)
        mock_record_event.assert_any_call(
            '/jax/orbax/checkpoint_manager/time_between_consecutive_saves_secs',
            3.0,
        )
        self.assertEqual(manager._last_save_time, 5.0)

  def test_time_between_consecutive_saves_metric_async_can_be_negative(self):
    # Tests that time_between_consecutive_saves_secs can be negative in async
    # mode. If save(N) is called before _finalize(N-1) completes,
    # _last_save_time (set in _finalize) may be greater than
    # time_at_start_of_save_N (set in save), resulting in a negative value
    # for time_at_start_of_save_N - _last_save_time. This test simulates this
    # scenario by setting time=2.0 for save(0) and time=1.0 for save(2).
    with (
        mock.patch('time.time') as mock_time,
        mock.patch(
            'jax.monitoring.record_event_duration_secs'
        ) as mock_record_event,
    ):
      mock_time.return_value = 1.0
      options = CheckpointManagerOptions(
          save_interval_steps=1, enable_async_checkpointing=True
      )
      with CheckpointManager(self.directory, options=options) as manager:
        mock_time.return_value = 2.0
        self.assertTrue(manager.save(0, args=args.PyTreeSave(self.pytree)))
        self.wait_if_async(manager)
        self.assertEqual(manager._last_save_time, 2.0)
        mock_record_event.reset_mock()

        mock_time.return_value = 1.0
        self.assertTrue(manager.save(2, args=args.PyTreeSave(self.pytree)))
        self.wait_if_async(manager)
        mock_record_event.assert_any_call(
            '/jax/orbax/checkpoint_manager/time_between_consecutive_saves_secs',
            -1.0,
        )
        self.assertEqual(manager._last_save_time, 1.0)

  def test_partial_restore_with_placeholder(self):
    """Basic save and restore test."""
    directory = self.directory / 'partial_restore'

    with CheckpointManager(directory) as save_manager:
      save_manager.save(0, args=args.PyTreeSave(self.pytree))

    with self.subTest('success'):
      reference_item = self.empty_pytree.copy()
      reference_item['b'] = PLACEHOLDER
      reference_item['c']['e'] = PLACEHOLDER

      expected = self.pytree.copy()
      expected['b'] = PLACEHOLDER
      expected['c']['e'] = PLACEHOLDER

      with CheckpointManager(directory) as restore_manager:
        restored = restore_manager.restore(
            0,
            args=args.PyTreeRestore(
                reference_item,
                restore_args=self.pytree_restore_args,
            ),
        )
        test_utils.assert_tree_equal(self, expected, restored)

    with self.subTest('missing_leaf'):
      reference_item = self.empty_pytree.copy()
      reference_item['b'] = PLACEHOLDER
      reference_item['c']['e'] = PLACEHOLDER
      del reference_item['c']['a']

      with CheckpointManager(directory) as restore_manager:
        with self.assertRaisesRegex(
            ValueError, 'User-provided restore item and on-disk value'
        ):
          restore_manager.restore(
              0,
              args=args.PyTreeRestore(
                  reference_item,
                  restore_args=self.pytree_restore_args,
              ),
          )

    with self.subTest('non_leaf_placeholder'):
      reference_item = self.empty_pytree.copy()
      reference_item['c'] = PLACEHOLDER

      with CheckpointManager(directory) as restore_manager:
        with self.assertRaisesRegex(
            ValueError, 'User-provided restore item and on-disk value'
        ):
          restore_manager.restore(
              0,
              args=args.PyTreeRestore(
                  reference_item,
                  restore_args=self.pytree_restore_args,
              ),
          )

  def test_partial_restore_with_omission(self):
    """Basic save and restore test."""
    directory = self.directory / 'partial_restore'

    with CheckpointManager(directory) as save_manager:
      save_manager.save(0, args=args.PyTreeSave(self.pytree))

    with self.subTest('leaf_omission_success'):
      with CheckpointManager(directory) as restore_manager:
        reference_item = {
            'a': 0,
            # Omit 'b'
            'c': {
                'a': 0,
                # Omit 'e'
            },
        }
        expected = {
            'a': self.pytree['a'],
            'c': {
                'a': self.pytree['c']['a'],
            },
        }
        restored = restore_manager.restore(
            0,
            args=args.PyTreeRestore(
                reference_item,
                restore_args=self.pytree_restore_args,
                partial_restore=True,
            ),
        )
        test_utils.assert_tree_equal(self, expected, restored)

    with self.subTest('node_omission_success'):
      with CheckpointManager(directory) as restore_manager:
        reference_item = {
            'a': 0,
            'b': 0,
            # Omit 'c'
        }
        expected = {
            'a': self.pytree['a'],
            'b': self.pytree['b'],
        }
        restored = restore_manager.restore(
            0,
            args=args.PyTreeRestore(
                reference_item,
                restore_args=self.pytree_restore_args,
                partial_restore=True,
            ),
        )
        test_utils.assert_tree_equal(self, expected, restored)

    with self.subTest('extra_leaf'):
      with CheckpointManager(directory) as restore_manager:
        reference_item = {
            'a': 0,
            # Omit 'b'
            'c': {
                'a': 0,
                # Omit 'e'
            },
            'z': 0,
        }
        with self.assertRaisesRegex(
            ValueError,
            'Missing keys were found in the user-provided restore item.',
        ):
          restore_manager.restore(
              0,
              args=args.PyTreeRestore(
                  reference_item,
                  restore_args=self.pytree_restore_args,
                  partial_restore=True,
              ),
          )


if __name__ == '__main__':
  multiprocess_test.main()
