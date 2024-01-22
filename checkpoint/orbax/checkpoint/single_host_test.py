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
import numpy as np
from orbax.checkpoint import pytree_checkpoint_handler
from orbax.checkpoint import type_handlers
from orbax.checkpoint import utils
import tensorstore as ts


class PyTreeCheckpointHandler(
    pytree_checkpoint_handler.PyTreeCheckpointHandler
):

  def save(self, directory, *args, **kwargs):
    super().save(directory, *args, **kwargs)
    if jax.process_index() == 0:
      self.finalize(directory)
    utils.sync_global_devices('PyTreeCheckpointHandler:finalize')


class SingleHostTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.ckpt_dir = epath.Path(self.create_tempdir('ckpt').full_path)

  @parameterized.parameters([False, True])
  def test_save_and_restore_a_single_device_sharded_jax_array(
      self, write_tree_metadata
  ):
    handler = PyTreeCheckpointHandler(write_tree_metadata=write_tree_metadata)
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (10,))
    assert isinstance(x.sharding, jax.sharding.SingleDeviceSharding)
    handler.save(
        self.ckpt_dir,
        args=pytree_checkpoint_handler.PyTreeSaveArgs({'array_x': x}),
    )

    restored_tree = handler.restore(self.ckpt_dir)
    np.testing.assert_array_equal(x, restored_tree['array_x'])

    if write_tree_metadata:
      self.assertIsInstance(restored_tree['array_x'], jax.Array)
      self.assertEqual(x.sharding, restored_tree['array_x'].sharding)
    else:
      # previously, Orbax just return numpyarray even the saved
      # the array is a SigmleDeviceSharded Jax Array.
      self.assertIsInstance(restored_tree['array_x'], np.ndarray)

  @parameterized.parameters([False, True])
  def test_save_and_restore_jax_array(self, use_zarr3):
    handler = PyTreeCheckpointHandler(use_zarr3=use_zarr3)
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (10,))
    handler.save(self.ckpt_dir, {'x': x})
    restored_tree = handler.restore(self.ckpt_dir)

    np.testing.assert_array_equal(x, restored_tree['x'])
    assert isinstance(restored_tree['x'], jax.Array)

  def test_save_and_restore_zarrv3_jax_array_custom_chunk_size(self):
    handler = PyTreeCheckpointHandler(use_zarr3=True)
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (10,))
    pytree = {'x': x}
    write_chunk_shape = (2,)
    read_chunk_shape = (1,)

    save_args = jax.tree_map(
        lambda x: type_handlers.SaveArgs(
            write_chunk_shape=write_chunk_shape,
            read_chunk_shape=read_chunk_shape,
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
        tsstore.chunk_layout.read_chunk.shape, read_chunk_shape
    )
    np.testing.assert_array_equal(
        tsstore.chunk_layout.write_chunk.shape, write_chunk_shape
    )

    # validate the restored_tree is identical as the written one
    restore_handler = PyTreeCheckpointHandler(use_zarr3=True)
    restored_tree = restore_handler.restore(self.ckpt_dir)
    np.testing.assert_array_equal(x, restored_tree['x'])
    self.assertIsInstance(restored_tree['x'], jax.Array)

  def test_save_and_restore_zarrv3_with_metadata(self):
    handler = PyTreeCheckpointHandler(use_zarr3=True, write_tree_metadata=True)
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

  @parameterized.parameters([
      ((3,), None),
      (None, (3,)),
      ((5,), (2,)),
  ])
  def test_save_zarrv3_jax_array_with_invalid_write_or_read_chunk_sizes(
      self, write_chunk_shape, read_chunk_shape
  ):
    handler = PyTreeCheckpointHandler(use_zarr3=True)
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (10,))
    pytree = {'x': x}

    save_args = jax.tree_map(
        lambda x: type_handlers.SaveArgs(
            write_chunk_shape=write_chunk_shape,
            read_chunk_shape=read_chunk_shape,
        ),
        pytree,
    )
    with self.assertRaises(ValueError):
      handler.save(self.ckpt_dir, pytree, save_args=save_args)

  def test_choose_chunk_shape_equal_global_shape(self):
    shape = (10, 100, 200)
    dtype = np.dtype('float32')

    # allow only 1 element
    chosen_shape = type_handlers._choose_chunk_shape(
        global_shape=shape,
        write_shape=shape,
        dtype=dtype,
        target_byte_size=dtype.itemsize,
    )
    np.testing.assert_array_equal(chosen_shape, (1, 1, 1))

    # allow 3 elements
    chosen_shape = type_handlers._choose_chunk_shape(
        global_shape=shape,
        write_shape=shape,
        dtype=dtype,
        target_byte_size=5**3 * dtype.itemsize,
    )
    np.testing.assert_array_equal(chosen_shape, (5, 5, 5))

    # allow 4 elements
    chosen_shape = type_handlers._choose_chunk_shape(
        global_shape=shape,
        write_shape=shape,
        dtype=dtype,
        target_byte_size=5**4 * dtype.itemsize,
    )
    np.testing.assert_array_equal(chosen_shape, (5, 10, 10))

    # not divisble target_byte_size should still result a correct shape
    chosen_shape = type_handlers._choose_chunk_shape(
        global_shape=shape,
        write_shape=shape,
        dtype=dtype,
        target_byte_size=5**4 * dtype.itemsize + 3,
    )
    np.testing.assert_array_equal(chosen_shape, (5, 10, 10))

  def test_choose_chunk_shape_for_sharded_array(self):
    local_shape = (10, 100, 200)
    dtype = np.dtype('float32')

    # allow to split on at the sharded axis
    chosen_shape = type_handlers._choose_chunk_shape(
        global_shape=(10, 500, 200),
        write_shape=local_shape,
        dtype=dtype,
        target_byte_size=10 * 5 * 200 * dtype.itemsize,
    )
    np.testing.assert_array_equal(chosen_shape, (10, 5, 200))

    # forced to split on unsharded axis when the target_byte_size is small
    chosen_shape = type_handlers._choose_chunk_shape(
        global_shape=(10, 500, 200),
        write_shape=local_shape,
        dtype=dtype,
        target_byte_size=10 * 1 * 100 * dtype.itemsize,
    )
    np.testing.assert_array_equal(chosen_shape, (10, 1, 100))

  def test_chunk_byte_size(self):
    handler = PyTreeCheckpointHandler(use_zarr3=True)
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (10, 100, 200), dtype=jnp.dtype('float32'))
    pytree = {'x': x}

    save_args = jax.tree_map(
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


if __name__ == '__main__':
  absltest.main()
