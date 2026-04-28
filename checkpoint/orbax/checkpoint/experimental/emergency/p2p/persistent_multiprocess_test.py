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

import os
import shutil

from etils import epath
import grain.python as pygrain
import jax
from jax.experimental import multihost_utils
import numpy as np
from orbax.checkpoint import args as args_lib
from orbax.checkpoint import test_utils
from orbax.checkpoint.experimental.emergency.p2p import args as p2p_args
from orbax.checkpoint.experimental.emergency.p2p import options as options_lib
from orbax.checkpoint.experimental.emergency.p2p import persistent

from orbax.checkpoint._src.testing.oss import multiprocess_test

P = jax.sharding.PartitionSpec
Mesh = jax.sharding.Mesh


class PersistentMultiprocessTest(multiprocess_test.MultiProcessTest):

  def setUp(self):
    super().setUp()
    self.mesh = Mesh(
        np.array(jax.devices()).reshape(
            jax.process_count(), jax.local_device_count()
        ),
        axis_names=('replica', 'data'),
    )

  def _create_shared_tempdir(self, name: str) -> epath.Path:
    # helper function to create a temp dir for all processes
    # to simulate a persistent storage
    base_dir = epath.Path(os.environ['TEST_TMPDIR']) / name
    print(f'base_dir: {base_dir}')
    if jax.process_index() == 0:
      if base_dir.exists():
        shutil.rmtree(base_dir)
      base_dir.mkdir(parents=True, exist_ok=True)
      (base_dir / 'ckpt').mkdir(parents=True, exist_ok=True)
    multihost_utils.sync_global_devices(f'{name}_created')
    return base_dir / 'ckpt'

  def test_save_restore(self):
    directory = self._create_shared_tempdir('test_save_restore')

    options = options_lib.CheckpointManagerOptions(
        persistent=options_lib.PersistentCheckpointOptions(
            save_interval_steps=1
        )
    )
    manager = persistent.PersistentCheckpointManager(
        directory, self.mesh, replica_axis_index=0, options=options
    )

    p_full = P(None, 'data')
    p_1d_sharded = P('data')

    arr_a = jax.device_put(
        np.arange(jax.device_count(), dtype=np.int32).reshape(
            jax.process_count(), jax.local_device_count()
        ),
        jax.sharding.NamedSharding(self.mesh, p_full),
    )
    arr_b = jax.device_put(
        np.arange(jax.device_count(), dtype=np.int32),
        jax.sharding.NamedSharding(self.mesh, p_1d_sharded),
    )
    state = {'a': arr_a, 'b': arr_b}

    save_args = p2p_args.Composite(state=args_lib.PyTreeSave(state))
    manager.save(1, args=save_args)
    manager.wait_until_finished()
    multihost_utils.sync_global_devices('saved')

    if jax.process_index() == 0:
      self.assertFalse((directory / '1' / 'default').exists())
      self.assertTrue((directory / '1' / 'state').exists())

    def _to_abstract(x):
      if isinstance(x, jax.Array):
        return jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding)
      return x

    abstract_state = jax.tree.map(_to_abstract, state)
    restore_args = p2p_args.Composite(state=abstract_state)

    restored = manager.restore(1, args=restore_args)
    restored_state = restored.state

    test_utils.assert_tree_equal(self, state, restored_state)
    manager.close()

  def test_save_restore_with_grain_iterator(self):
    directory = self._create_shared_tempdir(
        'test_save_restore_with_grain_iterator'
    )

    options = options_lib.CheckpointManagerOptions(
        persistent=options_lib.PersistentCheckpointOptions(
            save_interval_steps=1
        )
    )
    manager = persistent.PersistentCheckpointManager(
        directory, self.mesh, replica_axis_index=0, options=options
    )

    ds = pygrain.MapDataset.source(list(range(10)))
    dl = pygrain.DataLoader(
        data_source=ds,
        sampler=pygrain.SequentialSampler(10, pygrain.ShardOptions(0, 1)),
        operations=[pygrain.Batch(1)],
    )
    data_iter = iter(dl)
    for _ in range(3):
      next(data_iter)

    p_full = P(None, 'data')
    arr_a = jax.device_put(
        np.arange(jax.device_count(), dtype=np.int32).reshape(
            jax.process_count(), jax.local_device_count()
        ),
        jax.sharding.NamedSharding(self.mesh, p_full),
    )
    state = {'a': arr_a}

    save_args = p2p_args.Composite(
        state=args_lib.PyTreeSave(state),
        data_iter=pygrain.PyGrainCheckpointSave(data_iter),
    )
    manager.save(1, args=save_args)
    manager.wait_until_finished()
    multihost_utils.sync_global_devices('saved')

    new_dl = pygrain.DataLoader(
        data_source=ds,
        sampler=pygrain.SequentialSampler(10, pygrain.ShardOptions(0, 1)),
        operations=[pygrain.Batch(1)],
    )
    new_data_iter = iter(new_dl)

    def _to_abstract(x):
      if isinstance(x, jax.Array):
        return jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding)
      return x

    # PersistentCheckpointManager expects the state with sharding information
    # in args.state.
    args = {'state': jax.tree.map(_to_abstract, state)}
    args['data_iter'] = pygrain.PyGrainCheckpointRestore(new_data_iter)

    restore_args = p2p_args.Composite(**args)
    restored = manager.restore(1, args=restore_args)

    self.assertIn('state', restored)
    self.assertIn('data_iter', restored)
    test_utils.assert_tree_equal(self, state, restored['state'])
    self.assertEqual(next(restored['data_iter']), 3)
    manager.close()


if __name__ == '__main__':
  multiprocess_test.main()
