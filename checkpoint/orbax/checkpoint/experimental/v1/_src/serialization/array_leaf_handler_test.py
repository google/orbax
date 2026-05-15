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

import unittest

from absl import flags
from absl.testing import parameterized
from etils import epath
import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint import test_utils
from orbax.checkpoint import utils
from orbax.checkpoint._src.serialization import ocdbt_utils
from orbax.checkpoint._src.serialization import tensorstore_utils as ts_utils
from orbax.checkpoint._src.tree import utils as tree_utils
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.context import options as options_lib
from orbax.checkpoint.experimental.v1._src.serialization import array_leaf_handler
from orbax.checkpoint.experimental.v1._src.serialization import types
from orbax.checkpoint.experimental.v1._src.synchronization import multihost
from orbax.checkpoint.experimental.v1._src.testing import array_utils as array_test_utils
from orbax.checkpoint.experimental.v1._src.testing import path_utils as path_test_utils

from orbax.checkpoint._src.testing.oss import multiprocess_test

FLAGS = flags.FLAGS
jax.config.update('jax_enable_x64', True)


def _get_serialization_params(pytree):
  return [
      array_leaf_handler.ArraySerializationParam(
          keypath=keypath,
          value=array,
      )
      for keypath, array in jax.tree.flatten_with_path(pytree)[0]
  ]


def _get_deserialization_params(pytree):
  ret = []
  for keypath, array in jax.tree.flatten_with_path(pytree)[0]:
    shapedtype = tree_utils.to_shape_dtype_struct(array)
    assert isinstance(shapedtype, jax.ShapeDtypeStruct)
    ret.append(
        array_leaf_handler.ArrayDeserializationParam(
            keypath=keypath,
            value=shapedtype,
        )
    )
  return ret


def _get_metadata_params(pytree):
  return [
      types.DeserializationParam[None](
          keypath=keypath,
      )
      for keypath, _ in jax.tree.flatten_with_path(pytree)[0]
  ]


class ArrayLeafHandlerTest(
    unittest.IsolatedAsyncioTestCase, parameterized.TestCase
):

  def setUp(self):
    super().setUp()
    mesh = jax.sharding.Mesh(
        jax.devices(),
        ('x',),
        axis_types=(jax.sharding.AxisType.Auto,) * len(('x',)),
    )
    sharded = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('x'))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    self.pytree = {
        'replicated': array_test_utils.create_sharded_array(
            np.arange(16), replicated
        ),
        'sharded': array_test_utils.create_sharded_array(
            np.arange(32), sharded
        ),
    }

    if not utils.is_pathways_backend():
      self.pytree.update({
          'rand0': jax.random.key(
              jnp.array(1, device=replicated), impl='threefry2x32'
          ),
          'rand1': jax.random.key(jnp.array(2, device=replicated), impl='rbg'),
      })

    test_utils.sync_global_processes('setUp')

  def tearDown(self):
    test_utils.sync_global_processes('tearDown')
    super().tearDown()

  async def _test_simple_checkpoint_impl(
      self,
      use_ocdbt: bool = True,
      use_zarr3: bool = True,
      use_replica_parallel: bool = True,
      enable_replica_parallel_separate_folder: bool = False,
      enable_pinned_host_transfer: bool = False,
      save_concurrent_bytes: int | None = None,
      load_concurrent_bytes: int | None = None,
      use_compression: bool = True,
      min_slice_bytes_for_replica_parallel: int | None = None,
      max_replicas_for_replica_parallel: int | None = None,
  ):
    # make unit with different tests
    parent_dir = epath.Path(
        self.create_tempdir(f'tmp_{self._testMethodName}').full_path
    )

    init_context = context_lib.Context(
        array_options=options_lib.ArrayOptions(
            saving=options_lib.ArrayOptions.Saving(
                use_ocdbt=use_ocdbt,
                use_zarr3=use_zarr3,
                use_replica_parallel=use_replica_parallel,
                enable_replica_parallel_separate_folder=enable_replica_parallel_separate_folder,
                enable_pinned_host_transfer=enable_pinned_host_transfer,
                use_compression=use_compression,
                min_slice_bytes_for_replica_parallel=min_slice_bytes_for_replica_parallel,
                max_replicas_for_replica_parallel=max_replicas_for_replica_parallel,
            ),
            loading=options_lib.ArrayOptions.Loading(),
        ),
        memory_options=options_lib.MemoryOptions(
            write_concurrent_bytes=save_concurrent_bytes,
            read_concurrent_bytes=load_concurrent_bytes,
        ),
    )

    with context_lib.get_context(init_context) as context:

      handler = array_leaf_handler.ArrayLeafHandler()

      # serialization
      use_ocdbt = context.array_options.saving.use_ocdbt
      serialization_context = types.SerializationContext(
          parent_dir=path_test_utils.PathAwaitingCreationWrapper(parent_dir),
          ts_context=ts_utils.get_ts_context(use_ocdbt=use_ocdbt),
      )
      serialization_params = _get_serialization_params(self.pytree)
      task = await handler.serialize(
          params=serialization_params,
          serialization_context=serialization_context,
      )
      await task
      test_utils.sync_global_processes(
          f'{self._testMethodName}_serialize_complete'
      )

      # try finalize
      if use_ocdbt and multihost.is_primary_host(
          context.multiprocessing_options.primary_host
      ):
        await ocdbt_utils.merge_ocdbt_per_process_files(
            parent_dir,
            ts_context=ts_utils.get_ts_context(use_ocdbt=use_ocdbt),
            use_zarr3=context.array_options.saving.use_zarr3,
        )

      test_utils.sync_global_processes(
          f'{self._testMethodName}_finalize_complete'
      )

      # deserialization
      deserialization_context = types.DeserializationContext(
          parent_dir=parent_dir,
          ts_context=ts_utils.get_ts_context(use_ocdbt=use_ocdbt),
          ocdbt_checkpoint=use_ocdbt,
          zarr3_checkpoint=use_zarr3,
      )
      deserialization_task = await handler.deserialize(
          _get_deserialization_params(self.pytree),
          deserialization_context=deserialization_context,
      )

      restored = await deserialization_task

      for p, restored_array in zip(serialization_params, restored):
        test_utils.assert_array_equal(self, p.value, restored_array)

      # validate whether compression used
      self.assertEqual(
          test_utils.is_compression_used(
              parent_dir,
              serialization_params[0].name,
              use_zarr3,
              use_ocdbt,
          ),
          use_compression,
      )

  @parameterized.product(
      use_ocdbt=(True, False),
      use_zarr3=(True, False),
      enable_pinned_host_transfer=(True, False),
      save_concurrent_bytes=(None, 64),
      load_concurrent_bytes=(None, 64),
      use_compression=(True, False),
  )
  async def test_simple_checkpoint(
      self,
      use_ocdbt: bool,
      use_zarr3: bool,
      enable_pinned_host_transfer: bool,
      save_concurrent_bytes: int | None,
      load_concurrent_bytes: int | None,
      use_compression: bool,
  ):
    await self._test_simple_checkpoint_impl(
        use_ocdbt=use_ocdbt,
        use_zarr3=use_zarr3,
        enable_pinned_host_transfer=enable_pinned_host_transfer,
        save_concurrent_bytes=save_concurrent_bytes,
        load_concurrent_bytes=load_concurrent_bytes,
        use_compression=use_compression,
    )

  @parameterized.product(
      use_replica_parallel=(True, False),
      enable_replica_parallel_separate_folder=(True, False),
      min_slice_bytes_for_replica_parallel=(None, 1024),
      max_replicas_for_replica_parallel=(None, 2),
  )
  async def test_simple_checkpoint_for_replica_parallel(
      self,
      use_replica_parallel: bool,
      enable_replica_parallel_separate_folder: bool,
      min_slice_bytes_for_replica_parallel: int | None,
      max_replicas_for_replica_parallel: int | None,
  ):
    await self._test_simple_checkpoint_impl(
        use_replica_parallel=use_replica_parallel,
        enable_replica_parallel_separate_folder=enable_replica_parallel_separate_folder,
        min_slice_bytes_for_replica_parallel=min_slice_bytes_for_replica_parallel,
        max_replicas_for_replica_parallel=max_replicas_for_replica_parallel,
    )

  async def test_metadata(self):
    # make unit with different tests
    parent_dir = epath.Path(self.create_tempdir('test_metadata').full_path)

    with context_lib.get_context() as context:

      handler = array_leaf_handler.ArrayLeafHandler()

      # serialization
      use_ocdbt = context.array_options.saving.use_ocdbt
      serialization_context = types.SerializationContext(
          parent_dir=path_test_utils.PathAwaitingCreationWrapper(parent_dir),
          ts_context=ts_utils.get_ts_context(use_ocdbt=use_ocdbt),
      )
      serialization_params = _get_serialization_params(self.pytree)
      task = await handler.serialize(
          params=serialization_params,
          serialization_context=serialization_context,
      )
      await task
      test_utils.sync_global_processes(
          f'{self._testMethodName}_serialize_complete'
      )

      # try finalize
      if use_ocdbt and multihost.is_primary_host(
          context.multiprocessing_options.primary_host
      ):
        await ocdbt_utils.merge_ocdbt_per_process_files(
            parent_dir,
            ts_context=ts_utils.get_ts_context(use_ocdbt=use_ocdbt),
            use_zarr3=context.array_options.saving.use_zarr3,
        )

      test_utils.sync_global_processes(
          f'{self._testMethodName}_finalize_complete'
      )
      test_utils.print_directory(parent_dir)

      # load the metadata
      use_ocdbt = context.array_options.saving.use_ocdbt
      use_zarr3 = context.array_options.saving.use_zarr3
      deserialization_context = types.DeserializationContext(
          parent_dir=parent_dir,
          ts_context=ts_utils.get_ts_context(use_ocdbt=use_ocdbt),
          ocdbt_checkpoint=use_ocdbt,
          zarr3_checkpoint=use_zarr3,
      )

      metadata = await handler.metadata(
          _get_metadata_params(self.pytree),
          deserialization_context=deserialization_context,
      )

      # validate metadata
      for p, m in zip(serialization_params, metadata):
        expected_v = p.value
        if jax.dtypes.issubdtype(p.value.dtype, jax.dtypes.prng_key):
          expected_v = jax.random.key_data(p.value)

        self.assertEqual(expected_v.shape, m.shape)
        self.assertEqual(expected_v.dtype, m.dtype)
        expected_sharding = expected_v.sharding
        self.assertEqual(expected_sharding, m.sharding)


if __name__ == '__main__':
  multiprocess_test.main()
