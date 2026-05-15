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

"""Unittests for NumpyLeafHandler."""

import unittest

from absl import flags
from absl.testing import parameterized
from etils import epath
import jax
import numpy as np
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.serialization import ocdbt_utils
from orbax.checkpoint._src.serialization import tensorstore_utils as ts_utils
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.context import options as options_lib
from orbax.checkpoint.experimental.v1._src.serialization import numpy_leaf_handler
from orbax.checkpoint.experimental.v1._src.serialization import types
from orbax.checkpoint.experimental.v1._src.synchronization import multihost
from orbax.checkpoint.experimental.v1._src.testing import path_utils as path_test_utils

from orbax.checkpoint._src.testing.oss import multiprocess_test

FLAGS = flags.FLAGS


def _get_serialization_params(pytree):
  return [
      numpy_leaf_handler.NumpySerializationParam(
          keypath=keypath,
          value=nparray,
      )
      for keypath, nparray in jax.tree.flatten_with_path(pytree)[0]
  ]


def _get_deserialization_params(pytree):
  ret = []
  for keypath, nparray in jax.tree.flatten_with_path(pytree)[0]:

    ret.append(
        numpy_leaf_handler.NumpyDeserializationParam(
            keypath=keypath,
            value=numpy_leaf_handler.NumpyShapeDtype(
                shape=nparray.shape, dtype=nparray.dtype
            ),
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


class NumpyLeafHandlerTest(
    unittest.IsolatedAsyncioTestCase, parameterized.TestCase
):

  def setUp(self):
    super().setUp()
    self.pytree = {
        'np1': np.arange(16, dtype=np.float32),
        'np2': np.arange(32, dtype=np.int32),
        'np3': np.arange(1, dtype=np.float64),
    }

    test_utils.sync_global_processes('setUp')

  def tearDown(self):
    test_utils.sync_global_processes('tearDown')
    super().tearDown()

  @parameterized.product(
      use_ocdbt=(True, False),
      use_zarr3=(True, False),
      save_concurrent_bytes=(None, 64),
      load_concurrent_bytes=(None, 64),
      use_compression=(True, False),
  )
  async def test_simple_checkpoint(
      self,
      use_ocdbt: bool,
      use_zarr3: bool,
      save_concurrent_bytes: int | None,
      load_concurrent_bytes: int | None,
      use_compression: bool,
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
                use_compression=use_compression,
            ),
            loading=options_lib.ArrayOptions.Loading(),
        ),
        memory_options=options_lib.MemoryOptions(
            write_concurrent_bytes=save_concurrent_bytes,
            read_concurrent_bytes=load_concurrent_bytes,
        ),
    )

    with context_lib.get_context(init_context) as context:

      handler = numpy_leaf_handler.NumpyLeafHandler()

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

  async def test_metadata(self):
    # make unit with different tests
    parent_dir = epath.Path(self.create_tempdir('test_metadata').full_path)

    with context_lib.get_context() as context:

      handler = numpy_leaf_handler.NumpyLeafHandler()

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
        self.assertEqual(expected_v.shape, m.shape)
        self.assertEqual(expected_v.dtype, m.dtype)


if __name__ == '__main__':
  multiprocess_test.main()
