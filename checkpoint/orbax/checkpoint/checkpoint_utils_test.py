# Copyright 2022 The Orbax Authors.
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

"""Tests for checkpoint_utils."""

from absl.testing import absltest
from etils import epath
import jax
import numpy as np
from orbax.checkpoint import ArrayRestoreArgs
from orbax.checkpoint import checkpoint_utils
from orbax.checkpoint import Checkpointer
from orbax.checkpoint import CheckpointManager
from orbax.checkpoint import PyTreeCheckpointHandler
from orbax.checkpoint import RestoreArgs
from orbax.checkpoint import test_utils
from orbax.checkpoint import utils


class CheckpointUtilsTest(absltest.TestCase):

  def test_restore_args_from_target(self):
    devices = np.asarray(jax.devices())
    mesh = jax.sharding.Mesh(devices, ('x',))
    axes_tree = {
        'a': jax.sharding.PartitionSpec(
            'x',
        ),
        'x': None,
        'y': None,
    }
    pytree = {
        'a': test_utils.create_sharded_array(
            np.arange(16, dtype=np.int32) * 1,
            mesh,
            jax.sharding.PartitionSpec(
                'x',
            ),
        ),
        'x': np.ones(8, dtype=np.float64),
        'y': 1,
    }

    expected_restore_args = {
        'a': ArrayRestoreArgs(
            restore_type=jax.Array,
            mesh=mesh,
            mesh_axes=jax.sharding.PartitionSpec(
                'x',
            ),
            global_shape=(16,),
            dtype=np.int32,
        ),
        'x': RestoreArgs(restore_type=np.ndarray, dtype=np.float64),
        'y': RestoreArgs(restore_type=int),
    }
    restore_args = checkpoint_utils.restore_args_from_target(
        mesh, pytree, axes_tree
    )

    self.assertSameElements(expected_restore_args.keys(), restore_args.keys())

    def _check_restore_args(expected, actual):
      self.assertIsInstance(actual, RestoreArgs)
      self.assertEqual(expected.restore_type, actual.restore_type)
      self.assertEqual(expected.dtype, actual.dtype)

    def _check_array_restore_args(expected, actual):
      self.assertIsInstance(actual, ArrayRestoreArgs)
      self.assertEqual(expected.restore_type, jax.Array)
      self.assertEqual(expected.mesh, actual.mesh)
      self.assertEqual(expected.mesh_axes, actual.mesh_axes)
      self.assertEqual(expected.global_shape, actual.global_shape)
      self.assertEqual(expected.dtype, actual.dtype)

    with self.subTest(name='array_restore_args'):
      _check_array_restore_args(expected_restore_args['a'], restore_args['a'])
    with self.subTest(name='restore_args'):
      _check_restore_args(expected_restore_args['x'], restore_args['x'])
      _check_restore_args(expected_restore_args['y'], restore_args['y'])


class CheckpointIteratorTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(
        self.create_tempdir(name='checkpointing_test').full_path)
    self.manager = CheckpointManager(
        self.directory, {'params': Checkpointer(PyTreeCheckpointHandler())})
    self.items = {'params': {'a': 1, 'b': 2}}

  def check_saved_steps(self, expected_step):
    all_steps = list(
        checkpoint_utils.checkpoints_iterator(self.directory, timeout=0)
    )
    if expected_step is None:
      self.assertEmpty(all_steps)
    else:
      self.assertLen(all_steps, 1)
      self.assertEqual(all_steps[0], expected_step)

  def test_empty(self):
    self.check_saved_steps(None)

  def test_checkpoints(self):
    for i in range(2):
      self.manager.save(i, self.items)

    # Simulate incomplete checkpoint.
    test_utils.save_fake_tmp_dir(self.directory, 2, 'params')
    (self.directory / '3.foo').mkdir()
    self.assertSameElements([0, 1], utils.checkpoint_steps(self.directory))
    self.assertSameElements(
        [2],
        [
            utils.step_from_checkpoint_name(x)
            for x in utils.tmp_checkpoints(self.directory)
        ],
    )
    # Only returns latest and does not return incomplete checkpoints.
    self.check_saved_steps(1)

  def test_timeout_fn(self):
    timeout_fn_calls = [0]

    def timeout_fn():
      timeout_fn_calls[0] += 1
      return timeout_fn_calls[0] > 3

    results = list(
        checkpoint_utils.checkpoints_iterator(
            self.directory, timeout=0.1, timeout_fn=timeout_fn
        )
    )
    self.assertEqual([], results)
    self.assertEqual(4, timeout_fn_calls[0])


if __name__ == '__main__':
  absltest.main()
