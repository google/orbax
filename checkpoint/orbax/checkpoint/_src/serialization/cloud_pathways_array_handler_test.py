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

  @mock.patch.object(cloud_pathways_helper, "write_arrays")
  def test_serialize_groups_by_location_and_device_assignment(
      self, mock_write_arrays
  ):
    fut = mock.MagicMock()
    fut.result.return_value = None
    mock_write_arrays.return_value = fut

    sharding1 = jax.sharding.NamedSharding(
        self.mesh, jax.sharding.PartitionSpec("x")
    )
    arr1 = jax.device_put(np.arange(16, dtype=np.int32), sharding1)
    arr2 = jax.device_put(np.arange(16, 32, dtype=np.int32), sharding1)
    subdir = self.directory / "subdir"
    subdir.mkdir(parents=True, exist_ok=True)
    arr3 = jax.device_put(np.arange(32, 48, dtype=np.int32), sharding1)

    infos = [
        types.ParamInfo(
            name="a1", path=self.directory / "a1", parent_dir=self.directory
        ),
        types.ParamInfo(
            name="a2", path=self.directory / "a2", parent_dir=self.directory
        ),
        types.ParamInfo(name="a3", path=subdir / "a3", parent_dir=subdir),
    ]
    args = [types.SaveArgs() for _ in infos]
    handler = cloud_pathways_array_handler.CloudPathwaysArrayHandler()

    async def run_serialize():
      futures = await handler.serialize([arr1, arr2, arr3], infos, args)
      for f in futures:
        f.result()

    asyncio.run(run_serialize())

    self.assertEqual(mock_write_arrays.call_count, 2)

    call1_args = mock_write_arrays.call_args_list[0][0]
    self.assertEqual(call1_args[0], str(self.directory))
    self.assertEqual(call1_args[1], ["a1", "a2"])

    call2_args = mock_write_arrays.call_args_list[1][0]
    self.assertEqual(call2_args[0], str(subdir))
    self.assertEqual(call2_args[1], ["a3"])

  @mock.patch.object(cloud_pathways_helper, "read_arrays")
  def test_deserialize_groups_by_location_and_mesh(self, mock_read_arrays):
    sharding = jax.sharding.NamedSharding(
        self.mesh, jax.sharding.PartitionSpec("x")
    )
    arr1 = jax.device_put(np.arange(16, dtype=np.float32), sharding)
    arr2 = jax.device_put(np.arange(16, 32, dtype=np.float32), sharding)

    fut = mock.MagicMock()
    fut.result.return_value = None
    mock_read_arrays.side_effect = [([arr1], fut), ([arr2], fut)]

    subdir = self.directory / "subdir"
    infos = [
        types.ParamInfo(
            name="a1", path=self.directory / "a1", parent_dir=self.directory
        ),
        types.ParamInfo(name="a2", path=subdir / "a2", parent_dir=subdir),
    ]
    restore_args = [
        jax_array_restore_args.ArrayRestoreArgs(
            dtype=arr1.dtype, global_shape=arr1.shape, sharding=sharding
        ),
        jax_array_restore_args.ArrayRestoreArgs(
            dtype=arr2.dtype, global_shape=arr2.shape, sharding=sharding
        ),
    ]
    handler = cloud_pathways_array_handler.CloudPathwaysArrayHandler()

    async def run_deserialize():
      return await handler.deserialize(infos, restore_args)

    restored = asyncio.run(run_deserialize())
    self.assertEqual(mock_read_arrays.call_count, 2)
    self.assertEqual(
        mock_read_arrays.call_args_list[0][0][0], str(self.directory)
    )
    self.assertEqual(mock_read_arrays.call_args_list[1][0][0], str(subdir))
    np.testing.assert_array_equal(restored[0], arr1)
    np.testing.assert_array_equal(restored[1], arr2)


if __name__ == "__main__":
  absltest.main()
