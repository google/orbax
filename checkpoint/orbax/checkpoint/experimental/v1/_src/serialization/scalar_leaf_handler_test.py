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

"""Unittests for ScalarLeafHandler."""

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
from orbax.checkpoint.experimental.v1._src.serialization import scalar_leaf_handler
from orbax.checkpoint.experimental.v1._src.serialization import types
from orbax.checkpoint.experimental.v1._src.synchronization import multihost
from orbax.checkpoint.experimental.v1._src.testing import path_utils as path_test_utils

from orbax.checkpoint._src.testing.oss import multiprocess_test

FLAGS = flags.FLAGS


def _get_serialization_params(pytree):
  return [
      scalar_leaf_handler.ScalarSerializationParam(
          keypath=keypath,
          value=scalar,
      )
      for keypath, scalar in jax.tree.flatten_with_path(pytree)[0]
  ]


def _get_deserialization_params(
    pytree, cast_to: type[int | float] | None = None, pass_scalar=False
):
  ret = []
  for keypath, scalar in jax.tree.flatten_with_path(pytree)[0]:

    ret.append(
        scalar_leaf_handler.ScalarDeserializationParam(
            keypath=keypath,
            value=scalar if pass_scalar else cast_to,
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


class ScalarLeafHandlerTest(
    unittest.IsolatedAsyncioTestCase, parameterized.TestCase
):

  def setUp(self):
    super().setUp()
    self.pytree = {
        'int_value': 0,
        'float_value': 1.1,
        'np_int32_value': np.int32(2),
        'np_int64_value': np.int64(3),
        'np_float32_value': np.float32(4.4),
        'np_float64_value': np.float64(5.5),
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
      cast_to=(None, int, float, 'scalar'),
      use_compression=(True, False),
  )
  async def test_simple_checkpoint(
      self,
      use_ocdbt: bool,
      use_zarr3: bool,
      save_concurrent_bytes: int | None,
      load_concurrent_bytes: int | None,
      cast_to: type[int | float] | str | None,
      use_compression: bool,
  ):
    # make unit with different tests
    parent_dir = epath.Path(
        self.create_tempdir(f'tmp_{self._testMethodName}').full_path
    )

    context = context_lib.Context()
    context.array.saving.use_ocdbt = use_ocdbt
    context.array.saving.use_zarr3 = use_zarr3
    context.array.saving.use_compression = use_compression
    context.memory.write_concurrent_bytes = save_concurrent_bytes
    context.memory.read_concurrent_bytes = load_concurrent_bytes

    with context:

      handler = scalar_leaf_handler.ScalarLeafHandler()

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
          _get_deserialization_params(
              self.pytree, cast_to=cast_to, pass_scalar=(cast_to == 'scalar')
          ),
          deserialization_context=deserialization_context,
      )

      restored = await deserialization_task

      self._validate(serialization_params, restored, cast_to=cast_to)

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

  def _validate(self, serialization_params, restored, cast_to=None):
    for p, restored_scalar in zip(serialization_params, restored):
      expected_value = p.value
      if cast_to in (int, float):
        expected_value = cast_to(p.value)
      self.assertEqual(expected_value, restored_scalar)

  async def test_metadata(self):
    # make unit with different tests
    parent_dir = epath.Path(self.create_tempdir('test_metadata').full_path)

    with context_lib.get_context() as context:

      handler = scalar_leaf_handler.ScalarLeafHandler()

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
        if isinstance(expected_v, (int, np.integer)):
          expected_type = int
        elif isinstance(expected_v, (float, np.floating)):
          expected_type = float
        else:
          raise ValueError(f'Unsupported type: {type(expected_v)}')
        self.assertIsInstance(m, expected_type)


if __name__ == '__main__':
  multiprocess_test.main()
