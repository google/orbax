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

import asyncio
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import jax
from jax.experimental import mesh_utils
import numpy as np
from orbax.checkpoint._src.metadata import array_metadata_store as array_metadata_store_lib
from orbax.checkpoint._src.serialization import cloud_pathways_array_handler
from orbax.checkpoint._src.serialization import cloud_pathways_helper
from orbax.checkpoint._src.serialization import jax_array_restore_args
from orbax.checkpoint._src.serialization import types


class CloudPathwaysArrayHandlerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(self.create_tempdir().full_path)
    self.devices = jax.devices()
    self.mesh = jax.sharding.Mesh(
        mesh_utils.create_device_mesh(
            (len(self.devices),), devices=self.devices
        ),
        ("x",),
    )

  def assert_array_equal(self, array, expected):
    if hasattr(expected, "dtype"):
      self.assertEqual(expected.dtype, array.dtype)
    self.assertIsInstance(array, type(expected))

    if isinstance(expected, jax.Array):
      if jax.dtypes.issubdtype(expected.dtype, jax.dtypes.prng_key):
        self.assertTrue(jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key))
        np.testing.assert_array_equal(
            jax.random.key_data(array),
            jax.random.key_data(expected),
        )
        self.assertEqual(
            jax.random.key_impl(array),
            jax.random.key_impl(expected),
        )
      else:
        self.assertLen(
            array.addressable_shards, len(expected.addressable_shards)
        )
        for expected_shard, array_shard in zip(
            expected.addressable_shards, array.addressable_shards
        ):
          np.testing.assert_array_equal(array_shard.data, expected_shard.data)

    elif isinstance(expected, (np.ndarray, jax.numpy.ndarray)):
      np.testing.assert_array_equal(array, expected)
    else:
      self.assertEqual(array, expected)

  @mock.patch.object(cloud_pathways_helper, "read_arrays")
  def test_deserialize_typed_prng_key(self, mock_read_arrays):
    sharding = jax.sharding.NamedSharding(
        self.mesh, jax.sharding.PartitionSpec("x")
    )
    key = jax.random.key(0)
    key_data_array = jax.random.key_data(key)

    fut = mock.MagicMock()
    fut.result.return_value = None
    mock_read_arrays.return_value = ([key_data_array], fut)

    store = array_metadata_store_lib.Store()
    handler = cloud_pathways_array_handler.CloudPathwaysArrayHandler(
        array_metadata_store=store
    )

    info = types.ParamInfo(
        name="random_key",
        path=self.directory / "random_key",
        parent_dir=self.directory,
    )
    restore_arg = jax_array_restore_args.ArrayRestoreArgs(
        dtype=key.dtype, global_shape=key.shape, sharding=sharding
    )

    async def run_deserialize():
      return await handler.deserialize([info], [restore_arg])

    [restored] = asyncio.run(run_deserialize())

    # Assert read_arrays received physical uint32 dtype and physical
    # shape/sharding
    mock_read_arrays.assert_called_once()
    call_args = mock_read_arrays.call_args[0]
    read_dtypes = call_args[2]
    read_shapes = call_args[3]
    read_shardings = call_args[4]

    self.assertEqual(read_dtypes[0], np.dtype("uint32"))
    self.assertEqual(read_shapes[0], key_data_array.shape)
    self.assertEqual(
        read_shardings[0].spec, jax.sharding.PartitionSpec("x", None)
    )

    # Assert returned array is re-wrapped back into key<fry>
    self.assertTrue(jax.dtypes.issubdtype(restored.dtype, jax.dtypes.prng_key))
    self.assertEqual(restored.dtype, key.dtype)
    np.testing.assert_array_equal(
        jax.random.key_data(restored), jax.random.key_data(key)
    )

  @mock.patch.object(cloud_pathways_helper, "read_arrays")
  def test_deserialize_standard_array(self, mock_read_arrays):
    sharding = jax.sharding.NamedSharding(
        self.mesh, jax.sharding.PartitionSpec("x")
    )
    arr = jax.device_put(np.arange(32, dtype=np.float32), sharding)

    fut = mock.MagicMock()
    fut.result.return_value = None
    mock_read_arrays.return_value = ([arr], fut)

    handler = cloud_pathways_array_handler.CloudPathwaysArrayHandler()
    info = types.ParamInfo(
        name="a", path=self.directory / "a", parent_dir=self.directory
    )
    restore_arg = jax_array_restore_args.ArrayRestoreArgs(
        dtype=arr.dtype, global_shape=arr.shape, sharding=sharding
    )

    async def run_deserialize():
      return await handler.deserialize([info], [restore_arg])

    [restored] = asyncio.run(run_deserialize())

    mock_read_arrays.assert_called_once()
    call_args = mock_read_arrays.call_args[0]
    read_dtypes = call_args[2]
    read_shapes = call_args[3]
    read_shardings = call_args[4]

    self.assertEqual(read_dtypes[0], np.float32)
    self.assertEqual(read_shapes[0], arr.shape)
    self.assertEqual(read_shardings[0].spec, jax.sharding.PartitionSpec("x"))
    np.testing.assert_array_equal(restored, arr)


if __name__ == "__main__":
  absltest.main()
