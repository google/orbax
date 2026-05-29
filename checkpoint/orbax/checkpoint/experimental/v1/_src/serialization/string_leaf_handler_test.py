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

"""Unittests for StringLeafHandler."""

from typing import Type
import unittest

from absl import flags
from absl.testing import parameterized
from etils import epath
import jax
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.serialization import ocdbt_utils
from orbax.checkpoint._src.serialization import tensorstore_utils as ts_utils
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.serialization import string_leaf_handler
from orbax.checkpoint.experimental.v1._src.serialization import types
from orbax.checkpoint.experimental.v1._src.synchronization import multihost
from orbax.checkpoint.experimental.v1._src.testing import path_utils as path_test_utils

from orbax.checkpoint._src.testing.oss import multiprocess_test

FLAGS = flags.FLAGS


def _get_serialization_params(pytree):
  return [
      string_leaf_handler.StringSerializationParam(
          keypath=keypath,
          value=string,
      )
      for keypath, string in jax.tree.flatten_with_path(pytree)[0]
  ]


def _get_deserialization_params(pytree, abstract_leaf=None):
  return [
      string_leaf_handler.StringDeserializationParam(
          keypath=keypath,
          value=abstract_leaf,
      )
      for keypath, _ in jax.tree.flatten_with_path(pytree)[0]
  ]


def _get_metadata_params(pytree):
  return [
      types.DeserializationParam[None](
          keypath=keypath,
      )
      for keypath, _ in jax.tree.flatten_with_path(pytree)[0]
  ]


class StringLeafHandlerTest(
    unittest.IsolatedAsyncioTestCase, parameterized.TestCase
):

  def setUp(self):
    super().setUp()
    self.pytree = {
        'a': 'some_string1',
        'b': 'some_string2',
        'c': '123',
    }

    test_utils.sync_global_processes('setUp')

  def tearDown(self):
    test_utils.sync_global_processes('tearDown')
    super().tearDown()

  @parameterized.product(
      abstract_leaf=(None, str),  # should have no effects.
  )
  async def test_simple_checkpoint(
      self,
      abstract_leaf: Type[str] | None,
  ):
    # Use different tests path for each test case.
    parent_dir = epath.Path(
        self.create_tempdir(f'tmp_{self._testMethodName}').full_path
    )

    context = context_lib.Context()
    with context:

      handler = string_leaf_handler.StringLeafHandler()

      # Serialize the self.pytree.
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

      # Try finalize the checkpoint.
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

      # Deserialize the self.pytree from the stored checkpoint.
      deserialization_context = types.DeserializationContext(
          parent_dir=parent_dir,
          ts_context=ts_utils.get_ts_context(use_ocdbt=use_ocdbt),
          ocdbt_checkpoint=use_ocdbt,
          zarr3_checkpoint=False,
      )

      deserialization_task = await handler.deserialize(
          _get_deserialization_params(self.pytree, abstract_leaf=abstract_leaf),
          deserialization_context=deserialization_context,
      )

      restored = await deserialization_task

      for p, restored_string in zip(serialization_params, restored):
        expected_value = p.value
        self.assertEqual(expected_value, restored_string)

  async def test_metadata(self):
    # make unit with different tests
    parent_dir = epath.Path(self.create_tempdir('test_metadata').full_path)

    with context_lib.get_context() as context:

      handler = string_leaf_handler.StringLeafHandler()

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

      self.assertEqual(metadata, ['string'] * len(self.pytree))


if __name__ == '__main__':
  multiprocess_test.main()
