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

"""Tests for checkpoint_utils."""

import functools
from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import jax
import numpy as np
import orbax.checkpoint as ocp
from orbax.checkpoint import checkpoint_utils
from orbax.checkpoint import test_utils
from orbax.checkpoint import utils
from orbax.checkpoint.metadata import value as value_metadata
from orbax.checkpoint.path import step as step_lib


class RestoreArgsTest(absltest.TestCase):

  def _check_restore_args(self, expected, actual):
    self.assertIsInstance(actual, ocp.RestoreArgs)
    self.assertEqual(expected.restore_type, actual.restore_type)
    if hasattr(expected, 'dtype'):
      self.assertEqual(expected.dtype, actual.dtype)

  def _check_array_restore_args(self, expected, actual):
    self.assertIsInstance(actual, ocp.ArrayRestoreArgs)
    self.assertEqual(expected.restore_type, jax.Array)
    self.assertEqual(expected.sharding.mesh, actual.sharding.mesh)
    self.assertEqual(expected.sharding.spec, actual.sharding.spec)
    self.assertEqual(expected.global_shape, actual.global_shape)
    self.assertEqual(expected.dtype, actual.dtype)

  def test_construct_restore_args(self):
    devices = np.asarray(jax.devices())
    mesh = jax.sharding.Mesh(devices, ('x',))
    sharding_tree = {
        'a': jax.sharding.NamedSharding(
            mesh,
            jax.sharding.PartitionSpec(
                'x',
            ),
        ),
        'x': None,
        'y': None,
        'z': None,
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
        'z': 'hi',
    }

    expected_restore_args = {
        'a': ocp.ArrayRestoreArgs(
            restore_type=jax.Array,
            sharding=jax.sharding.NamedSharding(
                mesh,
                jax.sharding.PartitionSpec(
                    'x',
                ),
            ),
            global_shape=(16,),
            dtype=np.int32,
        ),
        'x': ocp.RestoreArgs(restore_type=np.ndarray, dtype=np.float64),
        'y': ocp.RestoreArgs(restore_type=int),
        'z': ocp.RestoreArgs(restore_type=str),
    }
    restore_args = checkpoint_utils.construct_restore_args(
        pytree, sharding_tree
    )

    self.assertSameElements(expected_restore_args.keys(), restore_args.keys())

    with self.subTest(name='array_restore_args'):
      self._check_array_restore_args(
          expected_restore_args['a'], restore_args['a']
      )
    with self.subTest(name='restore_args'):
      self._check_restore_args(expected_restore_args['x'], restore_args['x'])
      self._check_restore_args(expected_restore_args['y'], restore_args['y'])
      self._check_restore_args(expected_restore_args['z'], restore_args['z'])

  def test_construct_restore_args_value_metadata(self):
    devices = np.asarray(jax.devices())
    mesh = jax.sharding.Mesh(devices, ('x',))
    pytree = {
        'a': value_metadata.ScalarMetadata(
            name='', directory=epath.Path(''), dtype=np.int32
        ),
        'b': value_metadata.ArrayMetadata(
            name='',
            directory=epath.Path(''),
            shape=(2, 4),
            sharding=None,
            dtype=np.float64,
        ),
        'c': value_metadata.ArrayMetadata(
            name='',
            directory=epath.Path(''),
            shape=(2, 4),
            sharding=jax.sharding.NamedSharding(
                mesh=mesh, spec=jax.sharding.PartitionSpec('x')
            ),
            dtype=np.float64,
        ),
        'd': value_metadata.StringMetadata(name='', directory=epath.Path('')),
    }

    expected_restore_args = {
        'a': ocp.RestoreArgs(restore_type=int, dtype=np.int32),
        'b': ocp.RestoreArgs(restore_type=np.ndarray, dtype=np.float64),
        'c': ocp.ArrayRestoreArgs(
            restore_type=jax.Array,
            sharding=jax.sharding.NamedSharding(
                mesh,
                jax.sharding.PartitionSpec(
                    'x',
                ),
            ),
            global_shape=(2, 4),
            dtype=np.float64,
        ),
        'd': ocp.RestoreArgs(restore_type=str),
    }
    restore_args = checkpoint_utils.construct_restore_args(pytree)

    self.assertSameElements(expected_restore_args.keys(), restore_args.keys())

    with self.subTest(name='array_restore_args'):
      self._check_array_restore_args(
          expected_restore_args['c'], restore_args['c']
      )
    with self.subTest(name='restore_args'):
      self._check_restore_args(expected_restore_args['a'], restore_args['a'])
      self._check_restore_args(expected_restore_args['b'], restore_args['b'])
      self._check_restore_args(expected_restore_args['d'], restore_args['d'])


class EvalUtilsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(
        self.create_tempdir(name='checkpointing_test').full_path
    )
    self.manager = ocp.CheckpointManager(
        self.directory, {'params': ocp.PyTreeCheckpointer()}
    )
    self.items = {'params': {'a': 1, 'b': 2}}

  def check_saved_steps(
      self, expected_step, step_prefix=None, step_format_fixed_length=None
  ):
    all_steps = list(
        checkpoint_utils.checkpoints_iterator(
            self.directory,
            timeout=0,
            step_prefix=step_prefix,
            step_format_fixed_length=step_format_fixed_length,
        )
    )
    if expected_step is None:
      self.assertEmpty(all_steps)
    else:
      self.assertLen(all_steps, 1)
      self.assertEqual(all_steps[0], expected_step)

  def test_empty(self):
    self.check_saved_steps(None)

  @parameterized.parameters(
      (None, None),
      (None, 8),
      ('checkpoint', None),
      ('checkpoint', 8),
  )
  def test_checkpoints_iterator(self, step_prefix, step_format_fixed_length):
    options = ocp.CheckpointManagerOptions(
        step_prefix=step_prefix,
        step_format_fixed_length=step_format_fixed_length,
    )
    with ocp.CheckpointManager(
        self.directory,
        {'params': ocp.PyTreeCheckpointer()},
        options=options,
    ) as manager:
      for i in range(2):
        manager.save(i, self.items)

    # Simulate incomplete checkpoint.
    test_utils.save_fake_tmp_dir(self.directory, 2, 'params')
    (self.directory / '3.foo').mkdir()
    self.assertSameElements(
        [0, 1],
        utils.checkpoint_steps(
            self.directory,
        ),
    )
    self.assertContainsSubset(
        {utils.any_checkpoint_step(self.directory)}, {0, 1}
    )
    self.assertSameElements(
        [2],
        [
            utils.step_from_checkpoint_name(x)
            for x in utils.tmp_checkpoints(self.directory)
        ],
    )
    # Only returns latest and does not return incomplete checkpoints.
    self.check_saved_steps(
        1,
        step_prefix=step_prefix,
        step_format_fixed_length=step_format_fixed_length,
    )

  def test_timeout_fn(self):
    timeout_fn_calls = [0]

    def timeout_fn():
      timeout_fn_calls[0] += 1
      return timeout_fn_calls[0] > 3

    results = list(
        checkpoint_utils.checkpoints_iterator(
            self.directory, timeout=5, timeout_fn=timeout_fn
        )
    )
    self.assertEqual([], results)
    self.assertEqual(5, timeout_fn_calls[0])

  def test_checkpoints_iterator_last_checkpoint(self):
    with ocp.CheckpointManager(self.directory) as manager:
      for i in range(2):
        manager.save(i, args=ocp.args.StandardSave(self.items))
        manager.wait_until_finished()
        self.check_saved_steps(i)
      manager.save(2, args=ocp.args.StandardSave(self.items))

    # Even though timeout_fn always returns true, we still have the
    # opportunity to evaluate on step 2.
    counter = 0
    for step in checkpoint_utils.checkpoints_iterator(
        self.directory, timeout=0, timeout_fn=lambda: True
    ):
      counter += 1
      self.assertEqual(step, 2)
    self.assertEqual(counter, 1)

  def test_locking(self):
    max_step = 5
    self.manager.save(0, self.items)
    previous_step = None
    for step in checkpoint_utils.checkpoints_iterator(
        self.directory, timeout=0
    ):
      if previous_step is not None:
        self.assertFalse(utils.is_locked(self.directory / str(previous_step)))
      self.assertTrue(utils.is_locked(self.directory / str(step)))
      previous_step = step

      if step + 1 < max_step:
        self.manager.save(step + 1, self.items)

  def test_unlock_existing(self):
    self.manager.save(0, self.items)
    self.manager.save(1, self.items)

    checkpoint_utils._lock_checkpoint(
        self.directory,
        step=0,
        step_name_format=step_lib.standard_name_format(
            step_prefix=None, step_format_fixed_length=None
        ),
    )
    self.assertTrue(utils.is_locked(self.directory / str(0)))
    self.assertFalse(utils.is_locked(self.directory / str(1)))

    for _ in checkpoint_utils.checkpoints_iterator(self.directory):
      break
    self.assertFalse(utils.is_locked(self.directory / str(0)))
    self.assertFalse(utils.is_locked(self.directory / str(1)))

  @parameterized.parameters(
      (None, None),
      (None, 8),
      ('checkpoint', None),
      ('checkpoint', 8),
  )
  def test_wait_for_new_checkpoint(self, step_prefix, step_format_fixed_length):
    options = ocp.CheckpointManagerOptions(
        step_prefix=step_prefix,
        step_format_fixed_length=step_format_fixed_length,
    )
    manager = ocp.CheckpointManager(
        self.directory,
        {'params': ocp.PyTreeCheckpointer()},
        options=options,
    )

    manager.save(0, self.items)
    manager.save(1, self.items)
    step_directory = functools.partial(
        utils.get_save_directory,
        directory=self.directory,
        step_prefix=step_prefix,
        step_format_fixed_length=step_format_fixed_length,
    )
    with checkpoint_utils.wait_for_new_checkpoint(
        self.directory,
        step_prefix=step_prefix,
        step_format_fixed_length=step_format_fixed_length,
    ) as step:
      self.assertEqual(step, 1)
      self.assertFalse(utils.is_locked(step_directory(0)))
      self.assertTrue(utils.is_locked(step_directory(1)))

    manager.save(2, self.items)
    manager.save(3, self.items)
    with checkpoint_utils.wait_for_new_checkpoint(
        self.directory,
        until_step=3,
        step_prefix=step_prefix,
        step_format_fixed_length=step_format_fixed_length,
    ) as step:
      self.assertEqual(step, 3)
      self.assertTrue(utils.is_locked(step_directory(3)))

    with checkpoint_utils.wait_for_new_checkpoint(
        self.directory,
        until_step=5,
        timeout=1,
        step_prefix=step_prefix,
        step_format_fixed_length=step_format_fixed_length,
    ) as step:
      self.assertEqual(step, -1)

    with checkpoint_utils.wait_for_new_checkpoint(
        self.directory,
        step_prefix=step_prefix,
        step_format_fixed_length=step_format_fixed_length,
    ) as step:
      self.assertEqual(step, 3)
      self.assertTrue(utils.is_locked(step_directory(3)))

    with checkpoint_utils.wait_for_new_checkpoint(
        self.directory,
        until_step=1,
        step_prefix=step_prefix,
        step_format_fixed_length=step_format_fixed_length,
    ) as step:
      self.assertEqual(step, 3)
      self.assertTrue(utils.is_locked(step_directory(3)))

  def test_wait_for_directory_creation(self):
    directory = self.directory / 'checkpoints'
    with checkpoint_utils.wait_for_new_checkpoint(directory, timeout=1) as step:
      self.assertEqual(step, -1)
    directory.mkdir()
    manager = ocp.CheckpointManager(
        directory, {'params': ocp.PyTreeCheckpointer()}
    )
    manager.save(0, self.items)
    with checkpoint_utils.wait_for_new_checkpoint(directory, timeout=1) as step:
      self.assertEqual(step, 0)


if __name__ == '__main__':
  absltest.main()
