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

"""Tests for `array_metadata_store` module."""

from typing import List
import unittest
from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import jax
import numpy as np
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.metadata import array_metadata as array_metadata_lib
from orbax.checkpoint._src.metadata import array_metadata_store as array_metadata_store_lib
from orbax.checkpoint._src.serialization import type_handler_registry
from orbax.checkpoint._src.serialization import type_handlers


class StoreTest(parameterized.TestCase, unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    super().setUp()
    self.checkpoint_dir = epath.Path(self.create_tempdir().full_path)
    self.store = array_metadata_store_lib.Store()

  async def test_non_existing_checkpoint_dir(self):
    with self.assertRaisesRegex(
        ValueError, 'Checkpoint directory does not exist'
    ):
      _ = await self.store.read(self.checkpoint_dir / 'unknown_dir')

  async def test_non_existing_metadata_files(self):
    self.assertIsNone(await self.store.read(self.checkpoint_dir))

    (self.checkpoint_dir / 'array_metadatas').mkdir(
        parents=True, exist_ok=False
    )
    self.assertIsNone(await self.store.read(self.checkpoint_dir))

  async def test_write_and_read_single_process(self):
    process_index = 0
    array_metadatas = [
        array_metadata_lib.ArrayMetadata(
            param_name='a',
            shape=(10, 20, 30),
            dtype=np.dtype(int),
            write_shape=(10, 20, 30),
            chunk_shape=(1, 2, 3),
            use_ocdbt=False,
            use_zarr3=False,
        ),
        array_metadata_lib.ArrayMetadata(
            param_name='b',
            shape=(1, 1, 1),
            dtype=np.dtype(int),
            write_shape=(1, 1, 1),
            chunk_shape=(1, 1, 1),
            use_ocdbt=False,
            use_zarr3=False,
        ),
    ]
    await self.store.write(
        self.checkpoint_dir, array_metadatas, process_index=process_index
    )

    self.assertEqual(
        await self.store.read(self.checkpoint_dir, process_index=process_index),
        [
            array_metadata_lib.SerializedArrayMetadata(
                param_name='a',
                write_shape=(10, 20, 30),
                chunk_shape=(1, 2, 3),
            ),
            array_metadata_lib.SerializedArrayMetadata(
                param_name='b',
                write_shape=(1, 1, 1),
                chunk_shape=(1, 1, 1),
            ),
        ],
    )

  async def test_append_to_existing_metadata_file(self):
    process_index = 0
    initial_array_metadatas = [
        array_metadata_lib.ArrayMetadata(
            param_name='a',
            shape=(10, 20, 30),
            dtype=np.dtype(int),
            write_shape=(10, 20, 30),
            chunk_shape=(1, 2, 3),
            use_ocdbt=False,
            use_zarr3=False,
        ),
    ]
    await self.store.write(
        self.checkpoint_dir,
        initial_array_metadatas,
        process_index=process_index,
    )

    appended_array_metadatas = [
        array_metadata_lib.ArrayMetadata(
            param_name='b',
            shape=(1, 1, 1),
            dtype=np.dtype(int),
            write_shape=(1, 1, 1),
            chunk_shape=(1, 1, 1),
            use_ocdbt=False,
            use_zarr3=False,
        ),
    ]
    await self.store.write(
        self.checkpoint_dir,
        appended_array_metadatas,
        process_index=process_index,
    )

    self.assertEqual(
        await self.store.read(self.checkpoint_dir, process_index=process_index),
        [
            array_metadata_lib.SerializedArrayMetadata(
                param_name='a',
                write_shape=(10, 20, 30),
                chunk_shape=(1, 2, 3),
            ),
            array_metadata_lib.SerializedArrayMetadata(
                param_name='b',
                write_shape=(1, 1, 1),
                chunk_shape=(1, 1, 1),
            ),
        ],
    )

  async def test_write_and_read_multiple_process(self):
    for process_index in [0, 1, 2]:
      array_metadatas = [
          array_metadata_lib.ArrayMetadata(
              param_name=f'a_{process_index}',
              shape=(10, 20, 30),
              dtype=np.dtype(int),
              write_shape=(10, 20, 30),
              chunk_shape=(1, 2, 3),
              use_ocdbt=False,
              use_zarr3=False,
          ),
      ]
      await self.store.write(
          self.checkpoint_dir, array_metadatas, process_index=process_index
      )

    self.assertEqual(
        await self.store.read(self.checkpoint_dir, process_index=None),
        {
            0: [
                array_metadata_lib.SerializedArrayMetadata(
                    param_name='a_0',
                    write_shape=(10, 20, 30),
                    chunk_shape=(1, 2, 3),
                )
            ],
            1: [
                array_metadata_lib.SerializedArrayMetadata(
                    param_name='a_1',
                    write_shape=(10, 20, 30),
                    chunk_shape=(1, 2, 3),
                )
            ],
            2: [
                array_metadata_lib.SerializedArrayMetadata(
                    param_name='a_2',
                    write_shape=(10, 20, 30),
                    chunk_shape=(1, 2, 3),
                )
            ],
        },
    )


class ResolveArrayMetadataStoreTest(parameterized.TestCase):

  @parameterized.product(
      array_metadata_store=(None, array_metadata_store_lib.Store()),
  )
  def test_resolve_array_metadata_store_with_array_metadata_store(
      self, array_metadata_store
  ):
    array_handler = type_handlers.ArrayHandler(
        array_metadata_store=array_metadata_store
    )
    ty = jax.Array
    fn = lambda ty: issubclass(ty, jax.Array)
    with test_utils.register_type_handler(ty, array_handler, fn):
      store = array_metadata_store_lib.resolve_array_metadata_store(
          type_handler_registry.GLOBAL_TYPE_HANDLER_REGISTRY
      )
      self.assertIs(store, array_metadata_store)

  def test_no_jax_array_type_handler(self):
    type_handler_reg = type_handler_registry.create_type_handler_registry(
        (int, type_handlers.ScalarHandler())
    )

    store = array_metadata_store_lib.resolve_array_metadata_store(
        type_handler_reg
    )

    self.assertIsNone(store)

  def test_no_array_metadata_store_attribute_in_jax_array_type_handler(self):
    class WithoutArrayMetadataStore(type_handlers.ArrayHandler):

      def __init__(self):
        super().__init__()
        del self._array_metadata_store

    self.assertFalse(
        hasattr(WithoutArrayMetadataStore(), '_array_metadata_store')
    )
    type_handler_reg = type_handler_registry.create_type_handler_registry(
        (jax.Array, WithoutArrayMetadataStore())
    )

    store = array_metadata_store_lib.resolve_array_metadata_store(
        type_handler_reg
    )

    self.assertIsNone(store)


class ValidatorTest(parameterized.TestCase):

  @parameterized.named_parameters([
      dict(
          testcase_name='empty_array_metadatas',
          array_metadatas={0: []},
          expected_error_regex=(
              'ArrayMetadata Store contains no metadata for process_index=0'
          ),
      ),
      dict(
          testcase_name='different_number_of_params',
          array_metadatas={
              0: [
                  array_metadata_lib.SerializedArrayMetadata(
                      param_name='a', write_shape=(1,), chunk_shape=(1,)
                  )
              ],
              1: [
                  array_metadata_lib.SerializedArrayMetadata(
                      param_name='a', write_shape=(1,), chunk_shape=(1,)
                  ),
                  array_metadata_lib.SerializedArrayMetadata(
                      param_name='b', write_shape=(1,), chunk_shape=(1,)
                  ),
              ],
          },
          expected_error_regex=(
              'ArrayMetadata Store contains different number of params'
          ),
      ),
      dict(
          testcase_name='different_params',
          array_metadatas={
              0: [
                  array_metadata_lib.SerializedArrayMetadata(
                      param_name='a', write_shape=(1,), chunk_shape=(1,)
                  ),
                  array_metadata_lib.SerializedArrayMetadata(
                      param_name='c', write_shape=(1,), chunk_shape=(1,)
                  ),
              ],
              1: [
                  array_metadata_lib.SerializedArrayMetadata(
                      param_name='a', write_shape=(1,), chunk_shape=(1,)
                  ),
                  array_metadata_lib.SerializedArrayMetadata(
                      param_name='b', write_shape=(1,), chunk_shape=(1,)
                  ),
              ],
          },
          expected_error_regex='ArrayMetadata Store contains different params',
      ),
      dict(
          testcase_name='different_chunk_shapes',
          array_metadatas={
              0: [
                  array_metadata_lib.SerializedArrayMetadata(
                      param_name='a', write_shape=(1,), chunk_shape=(1,)
                  ),
                  array_metadata_lib.SerializedArrayMetadata(
                      param_name='b', write_shape=(1,), chunk_shape=(2,)
                  ),
              ],
              1: [
                  array_metadata_lib.SerializedArrayMetadata(
                      param_name='a', write_shape=(1,), chunk_shape=(1,)
                  ),
                  array_metadata_lib.SerializedArrayMetadata(
                      param_name='b', write_shape=(1,), chunk_shape=(3,)
                  ),
              ],
          },
          expected_error_regex=(
              'ArrayMetadata Store contains different chunk_shape'
          ),
      ),
      dict(
          testcase_name='different_write_shapes',
          array_metadatas={
              0: [
                  array_metadata_lib.SerializedArrayMetadata(
                      param_name='a', write_shape=(1,), chunk_shape=(1,)
                  ),
                  array_metadata_lib.SerializedArrayMetadata(
                      param_name='b', write_shape=(2,), chunk_shape=(3,)
                  ),
              ],
              1: [
                  array_metadata_lib.SerializedArrayMetadata(
                      param_name='a', write_shape=(1,), chunk_shape=(1,)
                  ),
                  array_metadata_lib.SerializedArrayMetadata(
                      param_name='b', write_shape=(3,), chunk_shape=(3,)
                  ),
              ],
          },
          expected_error_regex=(
              'ArrayMetadata Store contains different write_shape'
          ),
      ),
      dict(
          testcase_name='single_process_array_metadatas',
          array_metadatas={
              0: [
                  array_metadata_lib.SerializedArrayMetadata(
                      param_name='a', write_shape=(1,), chunk_shape=(1,)
                  )
              ]
          },
          expected_error_regex=None,
      ),
      dict(
          testcase_name='valid_array_metadatas',
          array_metadatas={
              0: [
                  array_metadata_lib.SerializedArrayMetadata(
                      param_name='a', write_shape=(1,), chunk_shape=(1,)
                  ),
                  array_metadata_lib.SerializedArrayMetadata(
                      param_name='b', write_shape=(2,), chunk_shape=(2,)
                  ),
                  array_metadata_lib.SerializedArrayMetadata(
                      param_name='c', write_shape=(3,), chunk_shape=(3,)
                  ),
              ],
              1: [
                  array_metadata_lib.SerializedArrayMetadata(
                      param_name='a', write_shape=(1,), chunk_shape=(1,)
                  ),
                  array_metadata_lib.SerializedArrayMetadata(
                      param_name='b', write_shape=(2,), chunk_shape=(2,)
                  ),
                  array_metadata_lib.SerializedArrayMetadata(
                      param_name='c', write_shape=(3,), chunk_shape=(3,)
                  ),
              ],
              2: [
                  array_metadata_lib.SerializedArrayMetadata(
                      param_name='a', write_shape=(1,), chunk_shape=(1,)
                  ),
                  array_metadata_lib.SerializedArrayMetadata(
                      param_name='b', write_shape=(2,), chunk_shape=(2,)
                  ),
                  array_metadata_lib.SerializedArrayMetadata(
                      param_name='c', write_shape=(3,), chunk_shape=(3,)
                  ),
              ],
          },
          expected_error_regex=None,
      ),
      dict(
          testcase_name='same_ext_metadata',
          array_metadatas={
              0: [
                  array_metadata_lib.SerializedArrayMetadata(
                      param_name='a',
                      write_shape=(1,),
                      chunk_shape=(1,),
                      ext_metadata={array_metadata_lib.RANDOM_KEY_IMPL: 'rbg'},
                  ),
              ],
              1: [
                  array_metadata_lib.SerializedArrayMetadata(
                      param_name='a',
                      write_shape=(1,),
                      chunk_shape=(1,),
                      ext_metadata={array_metadata_lib.RANDOM_KEY_IMPL: 'rbg'},
                  ),
              ],
          },
          expected_error_regex=None,
      ),
      dict(
          testcase_name='different_ext_metadata_dict_vs_none',
          array_metadatas={
              0: [
                  array_metadata_lib.SerializedArrayMetadata(
                      param_name='a',
                      write_shape=(1,),
                      chunk_shape=(1,),
                      ext_metadata={array_metadata_lib.RANDOM_KEY_IMPL: 'rbg'},
                  ),
              ],
              1: [
                  array_metadata_lib.SerializedArrayMetadata(
                      param_name='a',
                      write_shape=(1,),
                      chunk_shape=(1,),
                  ),
              ],
          },
          expected_error_regex='Different ext_metadata types',
      ),
      dict(
          testcase_name='different_ext_metadata_random_key_impl',
          array_metadatas={
              0: [
                  array_metadata_lib.SerializedArrayMetadata(
                      param_name='a',
                      write_shape=(1,),
                      chunk_shape=(1,),
                      ext_metadata={array_metadata_lib.RANDOM_KEY_IMPL: 'rbg'},
                  ),
              ],
              1: [
                  array_metadata_lib.SerializedArrayMetadata(
                      param_name='a',
                      write_shape=(1,),
                      chunk_shape=(1,),
                      ext_metadata={
                          array_metadata_lib.RANDOM_KEY_IMPL: 'unsafe_rbg'
                      },
                  ),
              ],
          },
          expected_error_regex=(
              'ArrayMetadata Store contains different ext_metadata'
          ),
      ),
  ])
  def test_validate_all_array_metadatas(
      self,
      array_metadatas: dict[
          int, List[array_metadata_lib.SerializedArrayMetadata]
      ],
      expected_error_regex: str | None,
  ):
    validator = array_metadata_store_lib.Validator()
    if expected_error_regex is None:
      validator.validate_all_array_metadatas(array_metadatas)
    else:
      with self.assertRaisesRegex(ValueError, expected_error_regex):
        validator.validate_all_array_metadatas(array_metadatas)


if __name__ == '__main__':
  absltest.main()
