# Copyright 2023 The Orbax Authors.
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

"""Test for standard_checkpoint_handler.py."""

import functools
from typing import Any

from absl.testing import parameterized
from etils import epath
import flax
import flax.training.train_state
import jax
from jax import numpy as jnp
import numpy as np
import optax
from orbax.checkpoint import standard_checkpoint_handler
from orbax.checkpoint import test_utils
from orbax.checkpoint import type_handlers
from orbax.checkpoint import utils


PyTree = Any
SaveArgs = type_handlers.SaveArgs
StandardSaveArgs = standard_checkpoint_handler.StandardSaveArgs
StandardRestoreArgs = standard_checkpoint_handler.StandardRestoreArgs


class StandardCheckpointHandler(
    standard_checkpoint_handler.StandardCheckpointHandler
):

  def save(self, directory, *args, **kwargs):
    super().save(directory, *args, **kwargs)
    if jax.process_index() == 0:
      self.finalize(directory)
    utils.sync_global_devices('StandardCheckpointHandler:finalize')


# Not in common util because we need to eliminate OSS dependency on flax.
def init_flax_model(model):
  params = model.init(jax.random.PRNGKey(0), jnp.ones([8, 8]))
  tx = optax.adamw(learning_rate=0.001)
  state = flax.training.train_state.TrainState.create(
      apply_fn=model.apply, params=params, tx=tx
  )
  return jax.tree_util.tree_map(np.asarray, state)


class StandardCheckpointHandlerTestBase:
  """Base test cases for StandardCheckpointHandler."""

  class Test(parameterized.TestCase):
    """Test class."""

    save_args_cls = StandardSaveArgs
    restore_args_cls = StandardRestoreArgs

    def setUp(self):
      super().setUp()

      self.numpy_pytree = test_utils.setup_pytree()
      pytree, _, _ = test_utils.setup_sharded_pytree(self.numpy_pytree)
      zeros_pytree = jax.tree_util.tree_map(
          np.zeros_like,
          self.numpy_pytree,
          is_leaf=test_utils.is_leaf,
      )
      zeros_pytree, _, _ = test_utils.setup_sharded_pytree(zeros_pytree)
      self.zeros_pytree = zeros_pytree

      self.numpy_pytree.update({'x': 4.5, 'y': 3})
      self.pytree = pytree
      self.mixed_pytree = {
          'sharded': self.pytree,
          'numpy': self.numpy_pytree,
      }

      self.directory = epath.Path(
          self.create_tempdir(name='checkpointing_test').full_path
      )
      test_utils.set_tensorstore_driver_for_test()

      utils.sync_global_devices('StandardCheckpointHandler:setup_complete')

    def tearDown(self):
      utils.sync_global_devices('StandardCheckpointHandler:tests_complete')
      self.handler.close()
      super().tearDown()

    @property
    def handler(self) -> StandardCheckpointHandler:
      return StandardCheckpointHandler()

    def test_basic(self):
      self.handler.save(self.directory, args=self.save_args_cls(self.pytree))
      restored = self.handler.restore(
          self.directory, args=self.restore_args_cls(self.zeros_pytree)
      )
      self.assertTrue(
          (self.directory / type_handlers._OCDBT_MANIFEST_FILE).exists()  # pylint: disable=protected-access
      )
      test_utils.assert_tree_equal(self, self.pytree, restored)

    def test_basic_no_item_arg(self):
      self.handler.save(self.directory, args=self.save_args_cls(self.pytree))
      restored = self.handler.restore(self.directory)
      self.assertTrue(
          (self.directory / type_handlers._OCDBT_MANIFEST_FILE).exists()  # pylint: disable=protected-access
      )
      test_utils.assert_tree_equal(self, self.pytree, restored)

    def test_shape_dtype_struct(self):
      """Test case."""
      self.handler.save(
          self.directory, args=self.save_args_cls(self.mixed_pytree)
      )
      restored = self.handler.restore(
          self.directory,
          args=self.restore_args_cls(
              jax.tree_util.tree_map(
                  utils.to_shape_dtype_struct, self.mixed_pytree
              )
          ),
      )
      test_utils.assert_tree_equal(self, self.mixed_pytree, restored)

    def test_save_aggregate(self):
      def _save_args(arr):
        return SaveArgs(aggregate=(np.asarray(arr).ndim < 2))

      save_args = jax.tree_util.tree_map(_save_args, self.numpy_pytree)
      with self.assertRaisesRegex(ValueError, 'Unsupported option `aggregate`'):
        self.handler.save(
            self.directory,
            args=self.save_args_cls(self.numpy_pytree, save_args=save_args),
        )

    def test_save_unsupported_type(self):
      pytree = {'str_key': 'str_value', **self.pytree}
      with self.assertRaisesRegex(ValueError, 'Unsupported type'):
        self.handler.save(self.directory, args=self.save_args_cls(pytree))

    def test_restore_unsupported_type(self):
      pytree = {'str_key': 'str_value', **self.pytree}
      self.handler.save(self.directory, args=self.save_args_cls(self.pytree))
      with self.assertRaisesRegex(ValueError, 'Unsupported type'):
        self.handler.restore(self.directory, args=self.restore_args_cls(pytree))

    def test_cast(self):
      """Test case."""
      save_args = jax.tree_util.tree_map(
          lambda _: SaveArgs(dtype=jnp.int16), self.pytree
      )
      self.handler.save(
          self.directory,
          args=self.save_args_cls(self.pytree, save_args=save_args),
      )
      metadata = self.handler.metadata(self.directory)
      jax.tree_util.tree_map(
          lambda m: self.assertEqual(m.dtype, jnp.int16), metadata
      )

      def check_dtype(x, dtype):
        if utils.is_scalar(x):
          self.assertIsInstance(x, int)
        else:
          self.assertEqual(x.dtype, dtype)

      restored = self.handler.restore(
          self.directory,
          args=self.restore_args_cls(
              jax.tree_util.tree_map(
                  functools.partial(
                      utils.to_shape_dtype_struct,
                      dtype=jnp.bfloat16,
                      scalar_dtype=int,
                  ),
                  self.pytree,
              )
          ),
      )
      jax.tree_util.tree_map(lambda x: check_dtype(x, jnp.bfloat16), restored)

    def test_flax_model(self):
      """Test case."""

      @flax.struct.dataclass
      class Params(flax.struct.PyTreeNode):
        params: Any
        opt_state: Any

      def make_params():
        return Params(
            params=self.numpy_pytree,
            opt_state=(optax.EmptyState(), optax.EmptyState()),
        )

      params = make_params()
      mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ('devices',))
      mesh_axes = jax.sharding.PartitionSpec()
      params = jax.tree_util.tree_map(
          lambda arr: test_utils.create_sharded_array(arr, mesh, mesh_axes),
          params,
      )
      target = jax.tree_util.tree_map(utils.to_shape_dtype_struct, params)

      self.handler.save(self.directory, args=self.save_args_cls(params))
      restored = self.handler.restore(
          self.directory, args=self.restore_args_cls(target)
      )
      test_utils.assert_tree_equal(self, params, restored)

    def test_empty_pytrees(self):
      """Test case."""
      with self.assertRaises(ValueError):
        self.handler.save(self.directory, args=self.save_args_cls({}))

      item = {'a': {}, 'b': 3}
      self.handler.save(self.directory, args=self.save_args_cls(item))
      restored = self.handler.restore(
          self.directory, args=self.restore_args_cls(item)
      )
      self.assertDictEqual(restored, item)

      item = {'c': None, 'd': 2}
      self.handler.save(self.directory, args=self.save_args_cls(item))
      restored = self.handler.restore(
          self.directory, args=self.restore_args_cls(item)
      )
      self.assertDictEqual(restored, item)

    def test_masked_shape_dtype_struct(self):
      """Test case."""

      def _should_mask(keypath):
        return keypath[0].key == 'a' or (
            keypath[0].key == 'c' and keypath[1].key == 'e'
        )

      def _mask(keypath, x):
        return optax.MaskedNode() if _should_mask(keypath) else x

      def _none(keypath, x):
        return None if _should_mask(keypath) else x

      masked_tree = jax.tree_util.tree_map_with_path(_mask, self.pytree)
      expected = jax.tree_util.tree_map_with_path(_none, self.pytree)

      self.handler.save(self.directory, args=self.save_args_cls(masked_tree))
      self.assertTrue(
          (self.directory / type_handlers._OCDBT_MANIFEST_FILE).exists()  # pylint: disable=protected-access
      )

      # Restore it with item which was given before applying masking.
      restored = self.handler.restore(
          self.directory,
          args=self.restore_args_cls(
              jax.tree_util.tree_map(utils.to_shape_dtype_struct, self.pytree)
          ),
      )
      test_utils.assert_tree_equal(self, expected, restored)

      # Restore it with item after applying masking to it.
      restored = self.handler.restore(
          self.directory,
          args=self.restore_args_cls(
              jax.tree_util.tree_map(utils.to_shape_dtype_struct, masked_tree)
          ),
      )
      test_utils.assert_tree_equal(self, expected, restored)

      # Restore it without any item.
      restored = self.handler.restore(self.directory)
      test_utils.assert_tree_equal(self, expected, restored)
