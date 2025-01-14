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

"""Tests for `array_metadata_store` module."""

import unittest
from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import numpy as np
from orbax.checkpoint._src.metadata import array_metadata as array_metadata_lib
from orbax.checkpoint._src.metadata import array_metadata_store


class StoreTest(parameterized.TestCase, unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    super().setUp()
    self.checkpoint_dir = epath.Path(self.create_tempdir().full_path)
    self.store = array_metadata_store.Store()

  def test_non_existing_checkpoint_dir(self):
    with self.assertRaisesRegex(
        ValueError, 'Checkpoint directory does not exist'
    ):
      _ = self.store.read(self.checkpoint_dir / 'unknown_dir')

  def test_non_existing_metadata_files(self):
    self.assertIsNone(self.store.read(self.checkpoint_dir))

    (self.checkpoint_dir / 'array_metadatas').mkdir(
        parents=True, exist_ok=False
    )
    self.assertIsNone(self.store.read(self.checkpoint_dir))

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
        self.store.read(self.checkpoint_dir, process_index=process_index),
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
        self.store.read(self.checkpoint_dir, process_index=None),
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


if __name__ == '__main__':
  absltest.main()
