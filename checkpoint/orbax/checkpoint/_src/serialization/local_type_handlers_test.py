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

from typing import Any
import unittest

from absl import flags
from absl.testing import parameterized
from etils import epath
import jax
import numpy as np
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.serialization import ocdbt_utils
from orbax.checkpoint._src.serialization import serialization
from orbax.checkpoint._src.serialization import tensorstore_utils as ts_utils
from orbax.checkpoint._src.serialization import type_handlers
from orbax.checkpoint._src.testing import multiprocess_test
from orbax.checkpoint.testing import local_path as local_path_test_lib
import tensorstore as ts


mock = unittest.mock
PyTree = Any
ParamInfo = type_handlers.ParamInfo
ArrayRestoreArgs = type_handlers.ArrayRestoreArgs
PLACEHOLDER = type_handlers.PLACEHOLDER


FLAGS = flags.FLAGS

jax.config.update('jax_enable_x64', True)


class LocalTypeHandlersTest(
    unittest.IsolatedAsyncioTestCase,
    parameterized.TestCase,
):
  """Captures aspects of serialization relevant to type_handlers."""

  def setUp(self):
    super().setUp()
    self.base_directory = local_path_test_lib.create_local_path_base(self)

    test_utils.set_tensorstore_driver_for_test()
    self.validate_topology()

    test_utils.sync_global_processes('LocalTypeHandlersTest:setup_complete')

  def tearDown(self):
    test_utils.sync_global_processes('LocalTypeHandlersTest:tests_complete')
    super().tearDown()

  def validate_topology(self):
    self.assertEqual(jax.device_count(), 8)
    self.assertGreater(jax.process_count(), 1)

  def get_array_handler(self):
    return type_handlers.ArrayHandler(
        primary_host=None, replica_id=None, use_replica_parallel=False
    )

  @property
  def local_directory(self) -> epath.Path:
    return local_path_test_lib.LocalPath(self.base_directory)

  def validate_paths(self):
    # Array files should not exist at the global path level.
    self.assertFalse((self.base_directory / 'manifest.ocdbt').exists())
    self.assertTrue(self.local_directory.exists())

  def get_param_info(
      self,
      name: str,
      path: epath.Path,
      is_ocdbt: bool | None = False,
      ts_context: ts.Context | None = None,
      raise_array_data_missing_error: bool = True,
  ) -> ParamInfo:
    return ParamInfo(
        name=name,
        parent_dir=path,
        is_ocdbt_checkpoint=is_ocdbt,
        ts_context=ts_context,
        raise_array_data_missing_error=raise_array_data_missing_error,
    )

  async def finalize_save(
      self, *, ts_context: ts.Context, use_zarr3: bool, use_ocdbt: bool
  ):
    if use_ocdbt:
      await ocdbt_utils.merge_ocdbt_per_process_files(
          self.local_directory, ts_context=ts_context, use_zarr3=use_zarr3
      )
      test_utils.sync_global_processes(
          'local_serialization:merge_ocdbt_complete'
      )

  @parameterized.product(
      use_ocdbt=(True, False),
      use_zarr3=(True, False),
  )
  async def test_local_serialization(self, use_ocdbt, use_zarr3):
    handler = self.get_array_handler()
    sharding = jax.sharding.NamedSharding(
        mesh=jax.sharding.Mesh(
            devices=np.asarray(jax.devices()),
            axis_names=('x',),
        ),
        spec=jax.sharding.PartitionSpec('x'),
    )
    # 8 shards, each of length 4.
    arr = jax.device_put(np.arange(32, dtype=np.int32), sharding)
    ts_context = ts_utils.get_ts_context(use_ocdbt=use_ocdbt)
    info = self.get_param_info(
        'a',
        self.local_directory,
        is_ocdbt=use_ocdbt,
        ts_context=ts_context,
    )
    futures = await handler.serialize([arr], [info])
    for f in futures:
      f.result()
    test_utils.sync_global_processes('test_array_serialization:serialized')
    await self.finalize_save(
        ts_context=ts_context, use_zarr3=use_zarr3, use_ocdbt=use_ocdbt
    )

    restore_arg = ArrayRestoreArgs(
        global_shape=arr.shape, dtype=arr.dtype, sharding=sharding
    )
    test_utils.print_directory(self.base_directory)
    restored = await handler.deserialize([info], [restore_arg])
    test_utils.assert_array_equal(self, arr, restored[0])

  @parameterized.product(
      use_ocdbt=(True, False),
      raise_array_data_missing_error=(True, False),
      use_zarr3=(True, False),
  )
  async def test_local_serialization_shuffled_devices(
      self, use_ocdbt, raise_array_data_missing_error, use_zarr3
  ):
    if multihost.is_pathways_backend():
      self.skipTest('Pathways does not support shuffling devices.')
    handler = self.get_array_handler()
    sharding = jax.sharding.NamedSharding(
        mesh=jax.sharding.Mesh(
            devices=np.asarray(jax.devices()),
            axis_names=('x',),
        ),
        spec=jax.sharding.PartitionSpec('x'),
    )
    # 8 shards, each of length 4.
    arr = jax.device_put(np.arange(32, dtype=np.int32), sharding)
    zeros_arr = jax.device_put(np.zeros((32,), dtype=np.int32), sharding)
    ts_context = ts_utils.get_ts_context(use_ocdbt=use_ocdbt)
    info = self.get_param_info(
        'a',
        self.local_directory,
        is_ocdbt=use_ocdbt,
        ts_context=ts_context,
        raise_array_data_missing_error=raise_array_data_missing_error,
    )
    futures = await handler.serialize([arr], [info])
    for f in futures:
      f.result()
    test_utils.sync_global_processes('test_array_serialization:serialized')
    await self.finalize_save(
        ts_context=ts_context, use_zarr3=use_zarr3, use_ocdbt=use_ocdbt
    )

    restore_arg = ArrayRestoreArgs(
        global_shape=arr.shape, dtype=arr.dtype, sharding=sharding
    )

    orig_get_device_to_index_map = serialization._get_device_to_index_map

    def shuffled_get_device_to_index_map(global_shape, sharding):
      device_to_index_map = orig_get_device_to_index_map(global_shape, sharding)
      processes = [d.process_index for d in device_to_index_map.keys()]
      assert processes == sorted(processes)
      devices = list(device_to_index_map.keys())
      devices.reverse()
      return dict(zip(devices, device_to_index_map.values()))

    with mock.patch.object(
        serialization,
        '_get_device_to_index_map',
        new=shuffled_get_device_to_index_map,
    ):
      if raise_array_data_missing_error:
        with self.assertRaisesRegex(
            Exception, 'Encountered error while reading array index'
        ):
          await handler.deserialize([info], [restore_arg])
      else:
        restored = await handler.deserialize([info], [restore_arg])
        test_utils.assert_array_equal(self, zeros_arr, restored[0])


if __name__ == '__main__':
  multiprocess_test.main()
