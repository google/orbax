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

"""Multiprocessing tests for P2P CheckpointManager."""

import asyncio
import os
import shutil
import threading
from typing import Any, Optional, Sequence
from unittest import mock

from absl import logging
from etils import epath
import grain.python as pygrain
import jax
from jax.experimental import multihost_utils
import numpy as np
from orbax.checkpoint import args as args_lib
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.futures import future
from orbax.checkpoint.experimental.emergency.p2p import args as p2p_args_lib
from orbax.checkpoint.experimental.emergency.p2p import checkpoint_manager as p2p_cm
from orbax.checkpoint.experimental.emergency.p2p import local
from orbax.checkpoint.experimental.emergency.p2p import options as options_lib
from orbax.checkpoint.experimental.emergency.p2p import persistent

from orbax.checkpoint._src.testing.oss import multiprocess_test

P = jax.sharding.PartitionSpec
Mesh = jax.sharding.Mesh
Composite = p2p_args_lib.Composite


class P2PCheckpointManagerMultiprocessTest(multiprocess_test.MultiProcessTest):

  def setUp(self):
    super().setUp()
    self.root_dir = self.create_tempdir('p2p_root')

  def initial_state(self, mesh):
    jax_processes = jax.process_count()
    self.assertEqual(jax_processes, 2)

    sharding = jax.sharding.NamedSharding(mesh, P('partition'))

    global_shape = (jax_processes * 2, 256)
    train_state = {
        'a': (
            np.arange(np.prod(global_shape), dtype=np.int32).reshape(
                global_shape
            )
        )
    }
    create_sharded_array = lambda x: jax.device_put(x, sharding)
    state = jax.tree_util.tree_map(create_sharded_array, train_state)

    abstract_state = jax.tree_util.tree_map(
        lambda x: jax.ShapeDtypeStruct(
            shape=x.shape, dtype=x.dtype, sharding=x.sharding
        ),
        state,
    )
    return state, abstract_state

  def test_save_restore_in_multiprocess(self):
    device_array = np.array(jax.devices()).reshape((2, jax.device_count() // 2))
    mesh = Mesh(device_array, axis_names=('replica', 'partition'))
    state, abstract_state = self.initial_state(mesh)
    process_id = jax.process_index()
    local_dir = self.root_dir.full_path + '/' + str(process_id)

    options = options_lib.CheckpointManagerOptions(
        local=options_lib.LocalCheckpointOptions(save_interval_steps=1),
        persistent=options_lib.PersistentCheckpointOptions(
            save_interval_steps=1
        ),
    )
    manager = p2p_cm.CheckpointManager(
        mesh,
        abstract_state,
        local_dir,
        options=options,
    )

    manager.save(1, args=Composite(state=args_lib.PyTreeSave(state)))
    manager.wait_until_finished()
    logging.info('Process %d finished saving step 1.', process_id)
    self.assertEqual(manager.latest_step(), 1)

    logging.info('All processes passed save_complete barrier.')

    restored = manager.restore(
        1, args=Composite(state=args_lib.PyTreeRestore(abstract_state))
    )
    self.assertIsNotNone(restored)
    self.assertIn('state', restored)
    test_utils.assert_tree_equal(self, restored.state, state)

    manager.close()

  def test_restore_via_p2p(self):
    device_array = np.array(jax.devices()).reshape((1, jax.device_count()))
    mesh = Mesh(device_array, axis_names=('partition', 'replica'))
    state, abstract_state = self.initial_state(mesh)
    process_id = jax.process_index()
    local_dir = self.root_dir.full_path + '/' + str(process_id)

    # Grain iterator
    ds = pygrain.MapDataset.source(list(range(100)))
    dl = pygrain.DataLoader(
        data_source=ds,
        sampler=pygrain.SequentialSampler(
            100 // jax.process_count(),
            pygrain.ShardOptions(jax.process_index(), jax.process_count()),
        ),
        operations=[pygrain.Batch(1)],
    )
    train_iter = iter(dl)
    for _ in range(5):
      next(train_iter)

    options = options_lib.CheckpointManagerOptions(
        local=options_lib.LocalCheckpointOptions(save_interval_steps=1),
        persistent=options_lib.PersistentCheckpointOptions(
            save_interval_steps=1
        ),
        replica_axis_index=1,
    )
    manager = p2p_cm.CheckpointManager(
        mesh,
        abstract_state,
        local_dir,
        options=options,
    )

    manager.save(
        1,
        args=Composite(
            state=args_lib.PyTreeSave(state),
            data_iter=pygrain.PyGrainCheckpointSave(train_iter),
        ),
    )
    manager.wait_until_finished()
    logging.info('Process %d finished saving step 1.', process_id)
    self.assertEqual(manager.latest_step(), 1)

    if process_id == 0:
      step_1_dir = os.path.join(local_dir, '1')
      logging.info('Process 0 deleting step 1 content: %s', step_1_dir)
      # Unable to call manager.delete() due to barrier
      shutil.rmtree(step_1_dir)
      logging.info('Process 0 deleted step 1 content.')
    else:
      manager._p2p.mark_registry_stale()

    # Reload manager to refresh state since checkpoint data is manipulated
    manager.reload()
    new_dl = pygrain.DataLoader(
        data_source=ds,
        sampler=pygrain.SequentialSampler(
            100 // jax.process_count(),
            pygrain.ShardOptions(jax.process_index(), jax.process_count()),
        ),
        operations=[pygrain.Batch(1)],
    )
    new_data_iter = iter(new_dl)

    restored = manager.restore(
        1,
        args=Composite(
            state=args_lib.PyTreeRestore(),
            data_iter=pygrain.PyGrainCheckpointRestore(new_data_iter),
        ),
    )
    self.assertIsNotNone(restored)
    self.assertIn('state', restored)
    test_utils.assert_tree_equal(self, restored.state, state)
    self.assertEqual(
        next(restored['data_iter']),
        [5 * jax.process_count() + jax.process_index()],
    )

    manager.close()

  def test_persistent_save_does_not_block_local_save(self):
    directory = (
        epath.Path(os.environ['TEST_TMPDIR'])
        / 'test_persistent_save_does_not_block_local_save'
    )
    local_dir = directory / 'local' / str(jax.process_index())
    persistent_dir = directory / 'persistent'
    local_dir.mkdir(parents=True, exist_ok=True)
    if jax.process_index() == 0:
      persistent_dir.mkdir(parents=True, exist_ok=True)
    multihost_utils.sync_global_devices('dirs_created')
    mesh = Mesh(
        np.array(jax.devices()).reshape(
            jax.process_count(), jax.local_device_count()
        ),
        axis_names=('replica', 'data'),
    )

    options = options_lib.CheckpointManagerOptions(
        local=options_lib.LocalCheckpointOptions(save_interval_steps=1),
        persistent=options_lib.PersistentCheckpointOptions(
            save_interval_steps=1
        ),
    )

    # Event to signal that persistent save has finished waiting
    _slow_save_event = threading.Event()

    class SlowSingleReplicaArrayHandler(
        persistent.type_handlers.SingleReplicaArrayHandler
    ):

      async def serialize(
          self,
          values: Sequence[jax.Array],
          infos: Sequence[Any],
          args: Optional[Sequence[Any]] = None,
      ) -> Sequence[future.Future]:
        async def slow_coro():
          logging.info('Persistent save sleeping...')
          await asyncio.sleep(2.0)
          logging.info('Persistent save woke up!')
          _slow_save_event.set()

        return [future.CommitFuture(slow_coro())]

    with mock.patch.object(
        persistent.type_handlers,
        'SingleReplicaArrayHandler',
        SlowSingleReplicaArrayHandler,
    ):
      persistent_manager = persistent.PersistentCheckpointManager(
          persistent_dir, mesh, replica_axis_index=0, options=options
      )
      local_manager = local.LocalCheckpointManager(
          local_dir, mesh, options=options
      )

      p_full = P(None, 'data')
      arr = jax.device_put(
          np.arange(jax.device_count(), dtype=np.int32).reshape(
              jax.process_count(), jax.local_device_count()
          ),
          jax.sharding.NamedSharding(mesh, p_full),
      )
      state = {'a': arr}
      save_args = Composite(state=args_lib.PyTreeSave(state))

      # Start persistent save (Async)
      logging.info('Starting persistent save')
      persistent_manager.save(1, args=save_args)

      # Start local save (Async)
      logging.info('Starting local save')
      local_manager.save(1, args=save_args)
      logging.info('Local save started (sync)')

      # Wait for local save to finish
      local_manager.wait_until_finished()
      logging.info('Local save finished waiting')

      # At this point, persistent save should still be sleeping or running,
      # so slow_save_event should NOT be set yet.
      self.assertFalse(
          _slow_save_event.is_set(),
          'Persistent save finished before Local save!',
      )

      # Now wait for persistent
      persistent_manager.wait_until_finished()
      self.assertTrue(_slow_save_event.is_set())

      persistent_manager.close()
      local_manager.close()

  def test_swap_local_checkpoint_data(self):
    device_array = np.array(jax.devices()).reshape(
        (jax.process_count(), jax.local_device_count())
    )
    mesh = Mesh(device_array, axis_names=('partition', 'replica'))
    state, abstract_state = self.initial_state(mesh)
    process_id = jax.process_index()
    local_dir = self.root_dir.full_path + '/' + str(process_id)

    options = options_lib.CheckpointManagerOptions(
        local=options_lib.LocalCheckpointOptions(save_interval_steps=1),
        persistent=options_lib.PersistentCheckpointOptions(
            save_interval_steps=1
        ),
    )
    manager = p2p_cm.CheckpointManager(
        mesh,
        abstract_state,
        local_dir,
        options=options,
    )

    manager.save(1, args=Composite(state=args_lib.PyTreeSave(state)))
    manager.wait_until_finished()
    logging.info('Process %d finished saving step 1.', process_id)
    self.assertEqual(manager.latest_step(), 1)

    multihost_utils.sync_global_devices('save_complete')

    if process_id == 0:
      dir_0 = os.path.join(self.root_dir.full_path, '0')
      dir_1 = os.path.join(self.root_dir.full_path, '1')
      dir_tmp = os.path.join(self.root_dir.full_path, 'tmp')

      logging.info('Swapping directories %s and %s', dir_0, dir_1)
      shutil.move(dir_0, dir_tmp)
      shutil.move(dir_1, dir_0)
      shutil.move(dir_tmp, dir_1)
      logging.info('Swapped directories.')

    multihost_utils.sync_global_devices('swap_complete')

    manager._p2p.mark_registry_stale()
    self.assertNotEmpty(manager.all_steps())
    self.assertEmpty(manager._local_manager.all_steps())
    restored = manager.restore(
        1, args=Composite(state=args_lib.PyTreeRestore(abstract_state))
    )
    self.assertIsNotNone(restored)
    self.assertIn('state', restored)
    test_utils.assert_tree_equal(self, restored.state, state)

    manager.close()

  def test_restore_downsize_replicas(self):
    device_array = np.array(jax.devices()).reshape((1, jax.device_count()))
    mesh = Mesh(device_array, axis_names=('partition', 'replica'))
    state, abstract_state = self.initial_state(mesh)
    process_id = jax.process_index()
    local_dir = self.root_dir.full_path + '/' + str(process_id)

    options = options_lib.CheckpointManagerOptions(
        local=options_lib.LocalCheckpointOptions(save_interval_steps=1),
        persistent=options_lib.PersistentCheckpointOptions(
            save_interval_steps=1
        ),
    )

    manager = p2p_cm.CheckpointManager(
        mesh,
        abstract_state,
        local_dir,
        options=options,
    )

    # 1. Save with 2 processes
    manager.save(1, args=Composite(state=args_lib.PyTreeSave(state)))
    manager.wait_until_finished()
    manager.close()

    multihost_utils.sync_global_devices('save_complete')

    # 2. Simulate Restore with 1 process (resizing scenario)
    # Different sharding for restore to verify argument propagation
    sharding_restore = jax.sharding.NamedSharding(
        mesh, P('partition', 'replica')
    )
    abstract_state_restore = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(
            shape=x.shape, dtype=x.dtype, sharding=sharding_restore
        ),
        state,
    )

    manager_restore = p2p_cm.CheckpointManager(
        mesh,
        abstract_state_restore,
        local_dir,
        options=options,
    )

    # Mock P2P layer to force local restore attempt
    with mock.patch.object(
        manager_restore._p2p, 'has_shard_for_step', return_value=True
    ):

      explicit_struct_args = Composite(
          state=args_lib.PyTreeRestore(
              item=abstract_state_restore, restore_args=abstract_state_restore
          )
      )
      restored_composite_explicit = manager_restore.restore(
          1, args=explicit_struct_args
      )
      self.assertIsNotNone(restored_composite_explicit)
      expected = jax.tree.map(
          lambda x: jax.device_put(x, sharding_restore), state
      )
      test_utils.assert_tree_equal(
          self, restored_composite_explicit.state, expected
      )

    manager_restore.close()


if __name__ == '__main__':
  multiprocess_test.main()
