# Copyright 2025 The Orbax Authors.
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

import unittest
from absl.testing import flagsaver
from absl.testing import parameterized
from etils import epath
import jax
import numpy as np
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.arrays import types
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.serialization import serialization
from orbax.checkpoint._src.serialization import tensorstore_utils as ts_utils
from orbax.checkpoint._src.serialization import type_handlers
from orbax.checkpoint._src.testing import multiprocess_test
from orbax.checkpoint.experimental.emergency import local_checkpoint_data_debugging


Index = types.Index
Shape = types.Shape
ChunkId = tuple[int, ...]

index_to_chunk_id = local_checkpoint_data_debugging.index_to_chunk_id
get_chunk_ids_from_tensorstore = (
    local_checkpoint_data_debugging.get_chunk_ids_from_tensorstore
)
open_tensorstore = local_checkpoint_data_debugging.open_tensorstore
get_present_and_missing_chunks = (
    local_checkpoint_data_debugging.get_present_and_missing_chunks
)


class LocalCheckpointDataValidatorTest(
    unittest.IsolatedAsyncioTestCase,
    parameterized.TestCase,
    multiprocess_test.MultiProcessTest,
):

  def make_global_mesh(self) -> jax.sharding.Mesh:
    self.assertEqual(jax.device_count(), 8)
    self.assertEqual(jax.process_count(), 4)
    self.assertEqual(jax.local_device_count(), 2)
    return jax.sharding.Mesh(jax.devices(), ('data',))

  def setUp(self):
    super().setUp()
    self.enter_context(
        flagsaver.flagsaver(experimental_orbax_use_distributed_process_id=True)
    )
    if not multihost.is_runtime_to_distributed_ids_initialized():
      multihost.initialize_runtime_to_distributed_ids()

    self.global_mesh = self.make_global_mesh()

    # make sure each process is working on different directories
    self.local_directory = epath.Path(
        self.create_tempdir(
            name=self._local_directory_for_process(multihost.process_index())
        ).full_path
    )
    test_utils.set_tensorstore_driver_for_test()
    test_utils.sync_global_processes('CheckpointManagerTest:setup_complete')

  def tearDown(self):
    super().tearDown()
    test_utils.sync_global_processes('CheckpointManagerTest:teardown_complete')

  def _local_directory_for_process(self, process_index: int) -> epath.Path:
    return f'local_checkpointing_test_pid_{process_index}'

  async def _write_array(
      self,
      array: jax.Array,
      param_name: str,
      *,
      use_ocdbt: bool,
      use_zarr3: bool,
  ):
    tspec = ts_utils.ArrayWriteSpec(
        self.local_directory.as_posix(),
        param_name,
        global_shape=array.shape,
        write_shape=array.sharding.shard_shape(array.shape),
        dtype=array.dtype,
        use_ocdbt=use_ocdbt,
        use_zarr3=use_zarr3,
    ).json
    replica_id = array.addressable_shards[0].replica_id
    await serialization.async_serialize(
        array,
        tspec,
        context=type_handlers.get_ts_context(use_ocdbt=use_ocdbt),
        primary_host=None,
        replica_id=replica_id,
    )

  @parameterized.product(use_ocdbt=[False, True], use_zarr3=[False, True])
  async def test_main(self, use_ocdbt: bool, use_zarr3: bool):
    self.assertEqual(multihost.process_count(), 4)
    param_name = 'array'
    array = test_utils.create_sharded_array(
        np.arange(16),
        self.global_mesh,
        jax.sharding.PartitionSpec('data'),
    )

    await self._write_array(
        array, param_name, use_ocdbt=use_ocdbt, use_zarr3=use_zarr3
    )
    test_utils.sync_global_processes('sync_after_write_array')

    # Rearrange two local directories to simulate restart.
    if multihost.process_index() == 0:
      process_0_directory = self.local_directory
      process_1_directory = (
          self.local_directory.parent / self._local_directory_for_process(1)
      )
      tmp_directory = self.local_directory.parent / 'tmp'
      process_0_directory.rename(tmp_directory)
      process_1_directory.rename(process_0_directory)
      tmp_directory.rename(process_1_directory)
    test_utils.sync_global_processes('sync_after_local_dir_rearrange')

    present_chunk_ids, missing_chunk_ids = await get_present_and_missing_chunks(
        self.local_directory,
        param_name,
        array,
        use_ocdbt=use_ocdbt,
        use_zarr3=use_zarr3,
    )

    if multihost.process_index() == 0:
      self.assertSameElements(present_chunk_ids, ((4,), (5,)))
      self.assertSameElements(missing_chunk_ids, ((0,), (1,)))
    elif multihost.process_index() == 1:
      self.assertSameElements(present_chunk_ids, ((0,), (1,)))
      self.assertSameElements(missing_chunk_ids, ((4,), (5,)))
    elif multihost.process_index() == 2:
      self.assertSameElements(present_chunk_ids, ((2,), (3,)))
      self.assertEmpty(missing_chunk_ids)
    elif multihost.process_index() == 3:
      self.assertSameElements(present_chunk_ids, ((6,), (7,)))
      self.assertEmpty(missing_chunk_ids)


if __name__ == '__main__':
  multiprocess_test.main()
