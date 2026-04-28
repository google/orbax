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

from etils import epath
import grain.python as pygrain
import jax
import numpy as np
from orbax.checkpoint import args as args_lib
from orbax.checkpoint import test_utils
from orbax.checkpoint.experimental.emergency.p2p import args as p2p_args
from orbax.checkpoint.experimental.emergency.p2p import local
from orbax.checkpoint.experimental.emergency.p2p import options as options_lib

from orbax.checkpoint._src.testing.oss import multiprocess_test

P = jax.sharding.PartitionSpec
Mesh = jax.sharding.Mesh


class LocalMultiprocessTest(multiprocess_test.MultiProcessTest):

  def setUp(self):
    super().setUp()
    self.mesh = Mesh(np.array(jax.devices()), axis_names=('x',))

  def test_save_restore(self):
    directory = epath.Path(self.create_tempdir().full_path) / 'ckpt'
    options = options_lib.CheckpointManagerOptions(
        local=options_lib.LocalCheckpointOptions(save_interval_steps=1)
    )
    manager = local.LocalCheckpointManager(
        directory, self.mesh, options=options
    )

    sharding = jax.sharding.NamedSharding(self.mesh, P('x'))
    arr = jax.device_put(
        np.arange(jax.device_count(), dtype=np.int32), sharding
    )
    state = {
        'a': arr,
        'b': jax.device_put(
            np.arange(jax.device_count(), dtype=np.int32),
            sharding,
        ),
    }
    manager.save(1, args=p2p_args.Composite(state=args_lib.PyTreeSave(state)))
    manager.wait_until_finished()

    abstract_state = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding),
        state,
    )
    restored = manager.restore(
        1, args=p2p_args.Composite(state=args_lib.PyTreeRestore(abstract_state))
    )
    restored_state = restored.state

    test_utils.assert_tree_equal(self, state, restored_state)
    manager.close()

  def test_save_restore_with_grain_iterator(self):
    directory = epath.Path(self.create_tempdir().full_path) / 'ckpt'
    options = options_lib.CheckpointManagerOptions(
        local=options_lib.LocalCheckpointOptions(save_interval_steps=1)
    )
    manager = local.LocalCheckpointManager(
        directory, self.mesh, options=options
    )

    dl = pygrain.DataLoader(
        data_source=pygrain.RangeDataSource(0, 20, 1),
        sampler=pygrain.SequentialSampler(
            20 // jax.process_count(),
            pygrain.ShardOptions(jax.process_index(), jax.process_count()),
        ),
        operations=[pygrain.Batch(1)],
    )
    data_iter = iter(dl)
    for _ in range(3):
      next(data_iter)

    sharding = jax.sharding.NamedSharding(self.mesh, P('x'))
    arr = jax.device_put(np.arange(self.mesh.size, dtype=np.int32), sharding)
    state = {'a': arr}
    save_args = p2p_args.Composite(
        state=args_lib.PyTreeSave(state),
        data_iter=pygrain.PyGrainCheckpointSave(data_iter),
    )
    manager.save(1, args=save_args)
    manager.wait_until_finished()

    expected_content = """{
  "0": "{\\n    \\"version\\": 2,\\n    \\"last_seen_indices\\": {\\n        \\"0\\": 4\\n    },\\n    \\"last_worker_index\\": -1,\\n    \\"worker_count\\": 0,\\n    \\"sampler\\": \\"SequentialSampler(num_records=10, shard_options=ShardOptions(shard_index=0, shard_count=2, drop_remainder=False))\\",\\n    \\"data_source\\": \\"RangeDataSource(start=0, stop=20, step=1)\\"\\n}",
  "1": "{\\n    \\"version\\": 2,\\n    \\"last_seen_indices\\": {\\n        \\"0\\": 5\\n    },\\n    \\"last_worker_index\\": -1,\\n    \\"worker_count\\": 0,\\n    \\"sampler\\": \\"SequentialSampler(num_records=10, shard_options=ShardOptions(shard_index=1, shard_count=2, drop_remainder=False))\\",\\n    \\"data_source\\": \\"RangeDataSource(start=0, stop=20, step=1)\\"\\n}"
}"""

    data_iter_dir = directory / '1' / 'data_iter'
    for grain_ckpt_path in data_iter_dir.iterdir():
      self.assertEqual(grain_ckpt_path.read_text(), expected_content)

    new_data_iter = iter(dl)
    restore_args = p2p_args.Composite(
        state=args_lib.PyTreeRestore(),
        data_iter=pygrain.PyGrainCheckpointRestore(new_data_iter),
    )
    restored = manager.restore(1, args=restore_args)

    self.assertIn('state', restored)
    self.assertIn('data_iter', restored)
    test_utils.assert_tree_equal(self, state, restored['state'])
    self.assertEqual(
        next(restored['data_iter']),
        [3 * jax.process_count() + jax.process_index()],
    )
    manager.close()


if __name__ == '__main__':
  multiprocess_test.main()
