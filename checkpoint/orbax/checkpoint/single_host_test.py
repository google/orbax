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

"""To test Orbax in single-host setup."""

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from orbax.checkpoint import checkpoint_manager
from orbax.checkpoint import test_utils
from orbax.checkpoint import type_handlers
from orbax.checkpoint._src.handlers import json_checkpoint_handler
from orbax.checkpoint._src.handlers import pytree_checkpoint_handler
import tensorstore as ts


PyTreeCheckpointHandler = test_utils.PyTreeCheckpointHandler


class SingleHostTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.ckpt_dir = epath.Path(self.create_tempdir('ckpt').full_path)

  def test_save_and_restore_a_single_device_sharded_jax_array(self):
    handler = PyTreeCheckpointHandler()
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (10,))
    assert isinstance(x.sharding, jax.sharding.SingleDeviceSharding)
    handler.save(
        self.ckpt_dir,
        args=pytree_checkpoint_handler.PyTreeSaveArgs({'array_x': x}),
    )

    restored_tree = handler.restore(self.ckpt_dir)
    np.testing.assert_array_equal(x, restored_tree['array_x'])

    self.assertIsInstance(restored_tree['array_x'], jax.Array)
    self.assertEqual(x.sharding, restored_tree['array_x'].sharding)

  @parameterized.parameters([False, True])
  def test_save_and_restore_jax_array(self, use_zarr3):
    handler = PyTreeCheckpointHandler(use_zarr3=use_zarr3)
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (10,))
    handler.save(self.ckpt_dir, {'x': x})
    restored_tree = handler.restore(self.ckpt_dir)

    np.testing.assert_array_equal(x, restored_tree['x'])
    assert isinstance(restored_tree['x'], jax.Array)

  def test_save_and_restore_zarrv3_jax_array_default_chunk_size(self):
    handler = PyTreeCheckpointHandler(use_zarr3=True)
    key = jax.random.PRNGKey(0)
    array_shape = (10, 7)
    x = jax.random.normal(key, array_shape)
    pytree = {'x': x}
    handler.save(self.ckpt_dir, pytree)

    # validate the stored array is in the chunk_layout specified
    tsstore = ts.open({
        'driver': 'zarr3',
        'kvstore': {
            'driver': 'ocdbt',
            'base': f'file://{self.ckpt_dir}',
            'path': 'x',
        },
    }).result()

    np.testing.assert_array_equal(
        tsstore.chunk_layout.read_chunk.shape, array_shape
    )
    np.testing.assert_array_equal(
        tsstore.chunk_layout.write_chunk.shape, array_shape
    )

    # validate the restored_tree is identical as the written one
    restore_handler = PyTreeCheckpointHandler(use_zarr3=True)
    restored_tree = restore_handler.restore(self.ckpt_dir)
    np.testing.assert_array_equal(x, restored_tree['x'])
    self.assertIsInstance(restored_tree['x'], jax.Array)

  def test_save_and_restore_zarrv3_with_metadata(self):
    handler = PyTreeCheckpointHandler(use_zarr3=True)
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (10,))
    pytree = {'x': x}
    handler.save(self.ckpt_dir, pytree)

    # even use_zarr3 is set to False, checkpoint can still be restored
    restore_handler = PyTreeCheckpointHandler(use_zarr3=False)
    restored_tree = restore_handler.restore(self.ckpt_dir)

    # validate the restored_tree is identical as the written one
    np.testing.assert_array_equal(x, restored_tree['x'])
    self.assertIsInstance(restored_tree['x'], jax.Array)

  # TODO: b/354139177 - Add a test for invalid chunk_byte_size values.
  def test_chunk_byte_size(self):
    handler = PyTreeCheckpointHandler(use_zarr3=True)
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (10, 100, 200), dtype=jnp.dtype('float32'))
    pytree = {'x': x}

    save_args = jax.tree.map(
        lambda x: type_handlers.SaveArgs(
            chunk_byte_size=jnp.dtype('float32').itemsize * 5**3,
        ),
        pytree,
    )
    handler.save(self.ckpt_dir, pytree, save_args=save_args)

    # validate the stored array is in the chunk_layout specified
    tsstore = ts.open({
        'driver': 'zarr3',
        'kvstore': {
            'driver': 'ocdbt',
            'base': f'file://{self.ckpt_dir}',
            'path': 'x',
        },
    }).result()

    np.testing.assert_array_equal(
        tsstore.chunk_layout.read_chunk.shape, (5, 5, 5)
    )
    np.testing.assert_array_equal(
        tsstore.chunk_layout.write_chunk.shape, (5, 5, 5)
    )

    # validate the restored_tree is identical as the written one
    restore_handler = PyTreeCheckpointHandler(use_zarr3=True)
    restored_tree = restore_handler.restore(self.ckpt_dir)
    np.testing.assert_array_equal(x, restored_tree['x'])
    assert isinstance(restored_tree['x'], jax.Array)

  def test_validate_checkpoint_manager_and_handler_primary_host(self):
    options = checkpoint_manager.CheckpointManagerOptions(
        enable_async_checkpointing=True,
        save_interval_steps=1,
        max_to_keep=1,
    )

    # default is fine
    checkpoint_manager.CheckpointManager(
        options=options,
        directory=self.ckpt_dir,
    )

    with self.assertRaises(ValueError):
      options.multiprocessing_options = (
          checkpoint_manager.MultiprocessingOptions(primary_host=None)
      )
      checkpoint_manager.CheckpointManager(
          options=options, directory=self.ckpt_dir
      )

    # single handler with non-zero primary_host
    with self.assertRaises(ValueError):
      options.multiprocessing_options = (
          checkpoint_manager.MultiprocessingOptions(primary_host=0)
      )
      checkpoint_manager.CheckpointManager(
          options=options,
          directory=self.ckpt_dir,
          item_handlers=pytree_checkpoint_handler.PyTreeCheckpointHandler(
              multiprocessing_options=checkpoint_manager.MultiprocessingOptions(
                  primary_host=None
              )
          ),
      )

    # multple handlers with non-zero primary_host
    with self.assertRaises(ValueError):
      options.multiprocessing_options = (
          checkpoint_manager.MultiprocessingOptions(primary_host=0)
      )
      checkpoint_manager.CheckpointManager(
          options=options,
          directory=self.ckpt_dir,
          item_names={'x', 'y'},
          item_handlers={
              'x': pytree_checkpoint_handler.PyTreeCheckpointHandler(),
              'y': json_checkpoint_handler.JsonCheckpointHandler(
                  multiprocessing_options=checkpoint_manager.MultiprocessingOptions(
                      primary_host=None
                  )
              ),
          },
      )

    # manager is set to non-zero
    with self.assertRaises(ValueError):
      options.multiprocessing_options = (
          checkpoint_manager.MultiprocessingOptions(primary_host=None)
      )
      checkpoint_manager.CheckpointManager(
          options=options,
          directory=self.ckpt_dir,
          item_handlers=pytree_checkpoint_handler.PyTreeCheckpointHandler(),
      )

  @parameterized.product(
      dtype=[
          jnp.bfloat16,
          ml_dtypes.bfloat16,
          np.dtype('float32'),
          np.dtype('float64'),
          ml_dtypes.float8_e5m2fnuz,
          ml_dtypes.float8_e5m2,
          ml_dtypes.float8_e4m3fnuz,
          ml_dtypes.float8_e4m3fn,
          ml_dtypes.float8_e4m3b11fnuz,
          ml_dtypes.int4,
          # ml_dtypes.uint4, # TODO(b/295577703)
          jnp.uint8,
          jnp.uint16,
          jnp.uint32,
          jnp.uint64,
          jnp.int8,
          jnp.int16,
          jnp.int32,
          jnp.int64,
      ],
      use_ocdbt=[True, False],
      use_zarr3=[True, False],
  )
  def test_save_and_restore_different_dtypes(self, dtype, use_ocdbt, use_zarr3):
    handler = PyTreeCheckpointHandler(use_ocdbt=use_ocdbt, use_zarr3=use_zarr3)

    rand_arr = np.random.normal(scale=3, size=10)  # use int4 max
    x = jax.numpy.asarray(rand_arr, dtype=dtype)
    pytree = {'x': x}
    handler.save(self.ckpt_dir, pytree)

    restore_handler = PyTreeCheckpointHandler()
    restored_tree = restore_handler.restore(self.ckpt_dir)

    # validate the restored_tree is identical as the written one
    np.testing.assert_array_equal(x, restored_tree['x'])
    self.assertIsInstance(restored_tree['x'], jax.Array)


if __name__ == '__main__':
  absltest.main()
