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

"""Common tests for AbstractCheckpointManager subclasses."""
import glob

from absl.testing import parameterized
from flax import linen as nn
from flax.training.train_state import TrainState
import jax
from jax.experimental import multihost_utils
from jax.experimental import pjit
from jax.experimental.maps import Mesh
import jax.numpy as jnp
import numpy as np
import optax
from orbax.checkpoint import AsyncCheckpointManager
from orbax.checkpoint import CheckpointManagerOptions
from orbax.checkpoint import DatasetCheckpointHandler
from orbax.checkpoint import JsonCheckpointHandler
from orbax.checkpoint import PyTreeCheckpointHandler
from orbax.checkpoint import RestoreArgs
from orbax.checkpoint import test_utils
import tensorflow as tf

PyTree = type(jax.tree_structure(None))
jax.config.update('jax_parallel_functions_output_gda', True)


class CheckpointManagerTestBase:
  """Common tests for AbstractCheckpointManager subclasses."""

  class Test(parameterized.TestCase):
    """Structure allows test to run as subclasses, not base class."""

    def checkpoint_manager(self, *args, **kwargs):
      raise NotImplementedError

    def setUp(self):
      super().setUp()
      pytree, mesh_tree, axes_tree = test_utils.setup_gda_pytree()
      doubled_pytree = test_utils.apply_function(pytree, lambda x: x * 2)

      self.empty_pytree = jax.tree_multimap(
          lambda x: object(), pytree, is_leaf=test_utils.is_leaf)
      self.pytree = pytree
      self.doubled_pytree = doubled_pytree
      self.mesh_tree = mesh_tree
      self.axes_tree = axes_tree
      self.pytree_restore_args = jax.tree_map(
          lambda mesh, axes: RestoreArgs(mesh=mesh, mesh_axes=axes),
          self.mesh_tree, self.axes_tree)
      self.dataset = tf.data.Dataset.range(64)
      self.directory = self.create_tempdir(name='checkpointing_test').full_path
      self.secondary_directory = self.create_tempdir(
          name='checkpointing_test_secondary').full_path

      multihost_utils.sync_global_devices(
          'CheckpointManagerTest:setup_complete')

    def tearDown(self):
      multihost_utils.sync_global_devices(
          'CheckpointManagerTest:tests_complete')
      super().tearDown()

    def restore_params(self, step, ckpt_mngr):
      return ckpt_mngr.restore(
          step,
          restore_kwargs={
              'params': {
                  'restore_args': self.pytree_restore_args,
              }
          })['params']

    def wait_if_async(self, mngr):
      if isinstance(mngr, AsyncCheckpointManager):
        mngr.wait_until_finished()

    def test_save_restore(self):
      """Basic save and restore test."""
      mngr = self.checkpoint_manager(self.directory,
                                     {'params': PyTreeCheckpointHandler()})
      self.assertTrue(mngr.save(0, {'params': self.pytree}))
      self.wait_if_async(mngr)
      restored = self.restore_params(0, mngr)
      test_utils.assert_tree_equal(self, self.pytree, restored)

    def test_save_restore_no_kwargs(self):
      """Restore with no GDA args."""
      mngr = self.checkpoint_manager(self.directory,
                                     {'params': PyTreeCheckpointHandler()})
      self.assertTrue(mngr.save(0, {'params': self.pytree}))
      self.wait_if_async(mngr)
      restored = mngr.restore(0)['params']
      expected = jax.tree_map(test_utils.replicate_gda, self.pytree)
      expected = jax.tree_map(lambda x: np.asarray(x.local_data(0)), expected)
      test_utils.assert_tree_equal(self, expected, restored)

    def test_save_restore_invalid_item(self):
      """Restore with invalid item."""
      mngr = self.checkpoint_manager(self.directory,
                                     {'params': PyTreeCheckpointHandler()})
      with self.assertRaises(ValueError):
        mngr.save(0, {'invalid': self.pytree})
      with self.assertRaises(ValueError):
        mngr.restore(0, items={'invalid': None})

    def test_save_structure(self):
      ckptr = PyTreeCheckpointHandler()
      mngr = self.checkpoint_manager(self.directory, {'params': ckptr})
      self.assertTrue(mngr.save(0, {'params': self.pytree}))
      self.wait_if_async(mngr)
      structure = mngr.structure()['params']
      expected = ckptr.structure(
          tf.io.gfile.join(self.directory, '0', 'params'))
      test_utils.assert_tree_equal(self, expected, structure)

    def test_all_steps(self):
      """Test correct steps are saved."""
      mngr = self.checkpoint_manager(self.directory,
                                     {'params': PyTreeCheckpointHandler()})
      for step in range(5):
        self.assertTrue(mngr.save(step, {'params': self.pytree}))
      self.wait_if_async(mngr)

      test_utils.save_fake_tmp_dir(self.directory, 5, 'params')
      multihost_utils.sync_global_devices('save_incomplete')

      # Does not include step 5
      self.assertSameElements(range(5), mngr.all_steps())

    def test_latest_step(self):
      """Test latest_step."""
      mngr = self.checkpoint_manager(self.directory,
                                     {'params': PyTreeCheckpointHandler()})
      for step in range(5):
        self.assertTrue(mngr.save(step, {'params': self.pytree}))
      self.wait_if_async(mngr)
      self.assertEqual(mngr.latest_step(), 4)

      self.assertTrue(mngr.save(5, {'params': self.pytree}))
      self.wait_if_async(mngr)
      self.assertEqual(mngr.latest_step(), 5)

    def test_ordered_save(self):
      """Test save order enforced."""
      mngr = self.checkpoint_manager(self.directory,
                                     {'params': PyTreeCheckpointHandler()})
      self.assertTrue(mngr.save(0, {'params': self.pytree}))
      self.assertTrue(mngr.save(1, {'params': self.pytree}))
      self.wait_if_async(mngr)

      self.assertFalse(mngr.save(0, {'params': self.doubled_pytree}))
      self.wait_if_async(mngr)
      restored = self.restore_params(0, mngr)
      expected = self.pytree
      # Despite calling save with doubled_pytree, restored is unmodified.
      test_utils.assert_tree_equal(self, expected, restored)

      self.assertFalse(mngr.save(5, {'params': self.doubled_pytree}))
      self.wait_if_async(mngr)
      with self.assertRaises(FileNotFoundError):
        mngr.restore(5)

    def test_no_overwrite_existing(self):
      """Test same step does not overwrite."""
      mngr = self.checkpoint_manager(self.directory,
                                     {'params': PyTreeCheckpointHandler()})
      self.assertTrue(mngr.save(0, {'params': self.pytree}))
      self.wait_if_async(mngr)
      self.assertFalse(
          mngr.save(0, {'params': self.doubled_pytree}, force=True))
      self.wait_if_async(mngr)
      restored = self.restore_params(0, mngr)
      expected = self.pytree
      test_utils.assert_tree_equal(self, expected, restored)

    def test_removes_old_saves(self):
      """Test old saves get removed."""
      options = CheckpointManagerOptions(max_to_keep=2)
      mngr = self.checkpoint_manager(
          self.directory, {'params': PyTreeCheckpointHandler()},
          options=options)
      for step in range(5):
        self.assertTrue(mngr.save(step, {'params': self.pytree}))
      self.wait_if_async(mngr)
      self.assertSameElements([3, 4], mngr.all_steps())

    def test_save_interval(self):
      """Test save interval > 1."""
      options = CheckpointManagerOptions(save_interval_steps=2)
      mngr = self.checkpoint_manager(
          self.directory, {'params': PyTreeCheckpointHandler()},
          options=options)
      for step in range(6):
        saved = mngr.save(step, {'params': self.pytree})
        if step % 2 == 0:
          self.assertTrue(saved)
        else:
          self.assertFalse(saved)
      self.wait_if_async(mngr)
      self.assertSameElements([0, 2, 4], mngr.all_steps())

    def test_save_same_step(self):
      """Test saving same step repeatedly."""
      options = CheckpointManagerOptions()
      mngr = self.checkpoint_manager(
          self.directory, {'params': PyTreeCheckpointHandler()},
          options=options)
      # checks an earlier bug where a dir is created, second save is skipped,
      # but leaves a dir present, third encounters error because tmp dir still
      # exists.
      step = 0
      self.assertTrue(mngr.save(step, {'params': self.pytree}, force=True))
      self.assertFalse(mngr.save(step, {'params': self.pytree}, force=True))
      self.assertFalse(mngr.save(step, {'params': self.pytree}, force=True))
      self.wait_if_async(mngr)

      tmp_dir = tf.io.gfile.join(self.directory, str(step),
                                 'params') + '.orbax-checkpoint-tmp-*'
      self.assertEmpty(glob.glob(tmp_dir))
      self.assertSameElements([0], mngr.all_steps())

    def test_save_interval_force(self):
      """Test force option."""
      options = CheckpointManagerOptions(save_interval_steps=2)
      mngr = self.checkpoint_manager(
          self.directory, {'params': PyTreeCheckpointHandler()},
          options=options)
      for step in range(6):
        saved = mngr.save(step, {'params': self.pytree})
        if step % 2 == 0:
          self.assertTrue(saved)
        else:
          self.assertFalse(saved)
      self.wait_if_async(mngr)
      self.assertTrue(mngr.save(5, {'params': self.pytree}, force=True))
      self.wait_if_async(mngr)
      self.assertSameElements([0, 2, 4, 5], mngr.all_steps())

    def test_save_preempted(self):
      """Simulate effects of preemption."""
      # Simulates the effects of preemption by creating a tmp directory and
      # ensuring it is cleaned up.
      step = 0
      tmp_dir = test_utils.save_fake_tmp_dir(
          self.directory, step, 'params', subdirs=['subdir'])

      self.assertTrue(tf.io.gfile.exists(tmp_dir))
      self.assertTrue(tf.io.gfile.exists(tf.io.gfile.join(tmp_dir, 'subdir')))

      # Complete check before allowing save, which will remove these directories
      multihost_utils.sync_global_devices('test_check_dirs')

      step = 1
      mngr = self.checkpoint_manager(self.directory,
                                     {'params': PyTreeCheckpointHandler()})
      self.assertTrue(mngr.save(step, {'params': self.pytree}))
      self.wait_if_async(mngr)

      self.assertFalse(tf.io.gfile.exists(tmp_dir))
      self.assertFalse(tf.io.gfile.exists(tf.io.gfile.join(tmp_dir, 'subdir')))
      # step 0 not picked up.
      self.assertSameElements([1], mngr.all_steps())

    def test_save_single_item(self):
      """Test managing single item."""
      mngr = self.checkpoint_manager(self.directory, PyTreeCheckpointHandler())
      self.assertTrue(mngr.save(0, self.pytree))
      self.wait_if_async(mngr)
      restored = mngr.restore(
          0, restore_kwargs={'restore_args': self.pytree_restore_args})
      expected = self.pytree
      test_utils.assert_tree_equal(self, expected, restored)

    def test_multiple_items(self):
      """Test multiple different items."""
      mngr = self.checkpoint_manager(
          self.directory, {
              'params': PyTreeCheckpointHandler(),
              'dataset': DatasetCheckpointHandler(),
              'metadata': JsonCheckpointHandler(filename='metadata'),
          })

      metadata = {
          'VERSION': 2,
          'optimizer': {
              'lr': 0.001,
              'type': 'adam',
          }
      }

      iterator = iter(self.dataset)
      # change iterator state
      for _ in range(10):
        next(iterator)
      self.assertTrue(
          mngr.save(0, {
              'params': self.pytree,
              'dataset': iterator,
              'metadata': metadata,
          }))
      self.wait_if_async(mngr)
      restored = mngr.restore(
          0, {
              'params': self.empty_pytree,
              'dataset': iter(self.dataset),
              'metadata': None,
          },
          restore_kwargs={'params': {
              'restore_args': self.pytree_restore_args
          }})
      restored_params, restored_dataset, restored_metadata = restored[
          'params'], restored['dataset'], restored['metadata']
      expected_params = self.pytree
      test_utils.assert_tree_equal(self, expected_params, restored_params)
      self.assertEqual(10, next(restored_dataset).numpy())
      self.assertDictEqual(metadata, restored_metadata)

    @parameterized.named_parameters(('min', True), ('max', False))
    def test_save_best(self, mode_min):
      """Test best metrics saving."""
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
          best_fn=metric_fn, best_mode=mode, max_to_keep=2)

      mngr = self.checkpoint_manager(
          self.directory, PyTreeCheckpointHandler(), options=options)
      for step in range(5):
        metrics = {k: v[step] for k, v in all_metrics.items()}
        self.assertTrue(mngr.save(step, self.pytree, metrics=metrics))
      self.wait_if_async(mngr)

      # simulate preemption - force new CheckpointManager to load
      # self._past_metrics from file.
      mngr = self.checkpoint_manager(
          self.directory, PyTreeCheckpointHandler(), options=options)
      for step in range(5, 10):
        metrics = {k: v[step] for k, v in all_metrics.items()}
        self.assertTrue(mngr.save(step, self.pytree, metrics=metrics))
      self.wait_if_async(mngr)

      self.assertSameElements([1, 5], mngr.all_steps())

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
      mesh_axes = pjit.PartitionSpec()

      @jax.jit
      def init_state():
        params = model.init(jax.random.PRNGKey(0), jnp.ones([8, 8]))
        tx = optax.adamw(learning_rate=0.001)
        state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
        return state

      init = pjit.pjit(
          init_state, in_axis_resources=None, out_axis_resources=mesh_axes)

      with Mesh(mesh.devices, mesh.axis_names):
        state = init()
        state_shape = jax.eval_shape(init)

      restore_args = jax.tree_map(
          lambda _: RestoreArgs(mesh=mesh, mesh_axes=mesh_axes), state_shape)

      mngr = self.checkpoint_manager(self.directory, {
          'state': PyTreeCheckpointHandler(),
      })
      self.assertTrue(mngr.save(0, {
          'state': state,
      }))
      self.wait_if_async(mngr)
      # already fully replicated, don't need to provide args
      restored = mngr.restore(
          0,
          items={'state': state_shape},
          restore_kwargs={'state': {
              'restore_args': restore_args
          }})['state']
      test_utils.assert_tree_equal(self, state.params, restored.params)
      test_utils.assert_tree_equal(self, state.opt_state, restored.opt_state)

    def test_restore_independent(self):
      """Test restore from secondary location."""
      # simulates pretrained checkpoint stored elsewhere
      secondary_mngr = self.checkpoint_manager(
          self.secondary_directory, {'params': PyTreeCheckpointHandler()})
      self.assertTrue(secondary_mngr.save(0, {'params': self.pytree}))
      self.wait_if_async(secondary_mngr)

      mngr = self.checkpoint_manager(self.directory,
                                     {'params': PyTreeCheckpointHandler()})
      pytree_restore_args = jax.tree_map(
          lambda mesh, axes: RestoreArgs(mesh=mesh, mesh_axes=axes),
          self.mesh_tree, self.axes_tree)
      restore_kwargs = {'params': {'restore_args': pytree_restore_args,}}

      with self.assertRaises(FileNotFoundError):
        mngr.restore(0, restore_kwargs=restore_kwargs)

      restored = mngr.restore(
          0, restore_kwargs=restore_kwargs,
          directory=self.secondary_directory)['params']
      test_utils.assert_tree_equal(self, self.pytree, restored)
