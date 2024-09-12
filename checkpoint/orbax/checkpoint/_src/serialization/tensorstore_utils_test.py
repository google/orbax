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

from typing import Optional

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from orbax.checkpoint._src.arrays import types
from orbax.checkpoint._src.serialization import tensorstore_utils as ts_utils


GIB = 1024**3


class AddOcdbtWriteOptionsTest(parameterized.TestCase):

  def test_ocdbt_target_data_file_size_none(self):
    kvstore_tspec = {}
    ts_utils.add_ocdbt_write_options(kvstore_tspec, target_data_file_size=None)
    self.assertNotIn('target_data_file_size', kvstore_tspec)

  def test_ocdbt_target_data_file_size_none_is_default(self):
    kvstore_tspec = {}
    ts_utils.add_ocdbt_write_options(kvstore_tspec)
    self.assertNotIn('target_data_file_size', kvstore_tspec)

  def test_ocdbt_target_data_file_size_rejects_negative_value(self):
    with self.assertRaises(ValueError):
      ts_utils.add_ocdbt_write_options({}, target_data_file_size=-13)

  @parameterized.product(target_data_file_size=[0, 1024**3])
  def test_ocdbt_target_data_file_size_sets_value(
      self,
      target_data_file_size: int,
  ):
    kvstore_tspec = {}
    ts_utils.add_ocdbt_write_options(
        kvstore_tspec, target_data_file_size=target_data_file_size
    )
    self.assertEqual(
        kvstore_tspec['target_data_file_size'], target_data_file_size
    )


class AdjustChunkByteSizeTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='no_sizes_small_shape_returns_none',
          shape=(10, 6),
          dtype=np.dtype(np.int32),
          chunk_byte_size=None,
          target_data_file_size=None,
          expected_chunk_byte_size=None,
      ),
      dict(
          testcase_name='no_sizes_too_large_shape_returns_ts_default',
          shape=(10, 1024**3),
          dtype=np.dtype(np.int32),
          chunk_byte_size=None,
          target_data_file_size=None,
          expected_chunk_byte_size=(
              ts_utils._DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE
          ),
      ),
      dict(
          testcase_name='no_target_file_size_returns_chunk_size',
          shape=(10, 6),
          dtype=np.dtype(np.int32),
          chunk_byte_size=1024,
          target_data_file_size=None,
          expected_chunk_byte_size=1024,
      ),
      dict(
          testcase_name='no_target_file_size_large_shape_returns_ts_default',
          shape=(10, 1024**3),
          dtype=np.dtype(np.int32),
          chunk_byte_size=(8 * GIB),
          target_data_file_size=None,
          expected_chunk_byte_size=(
              ts_utils._DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE
          ),
      ),
      dict(
          testcase_name='no_chunk_size_small_write_shape_returns_none',
          shape=(10, 6),
          dtype=np.dtype(np.int32),
          chunk_byte_size=None,
          target_data_file_size=(1 * GIB),
          expected_chunk_byte_size=None,
      ),
      dict(
          testcase_name=(
              'no_chunk_size_large_write_shape_returns_target_file_size'
          ),
          shape=(10, 1024**3),
          dtype=np.dtype(np.int32),
          chunk_byte_size=None,
          target_data_file_size=(1 * GIB),
          expected_chunk_byte_size=(1 * GIB),
      ),
      dict(
          testcase_name='unlimited_target_file_size_returns_chunk_size_none',
          shape=(10, 1024**3),
          dtype=np.dtype(np.int32),
          chunk_byte_size=None,
          target_data_file_size=0,
          expected_chunk_byte_size=None,
      ),
      dict(
          testcase_name=(
              'unlimited_target_file_size_returns_chunk_size_not_none'
          ),
          shape=(10, 1024**3),
          dtype=np.dtype(np.int32),
          chunk_byte_size=(4 * GIB),
          target_data_file_size=0,
          expected_chunk_byte_size=(4 * GIB),
      ),
      dict(
          testcase_name='returns_min_chunk_size',
          shape=(10, 1024**3),
          dtype=np.dtype(np.int32),
          chunk_byte_size=(1 * GIB),
          target_data_file_size=(4 * GIB),
          expected_chunk_byte_size=(1 * GIB),
      ),
      dict(
          testcase_name='returns_min_target_file_size',
          shape=(10, 1024**3),
          dtype=np.dtype(np.int32),
          chunk_byte_size=(4 * GIB),
          target_data_file_size=(1 * GIB),
          expected_chunk_byte_size=(1 * GIB),
      ),
  )
  def test_adjust_chunk_byte_size(
      self,
      shape: types.Shape,
      dtype: np.dtype,
      chunk_byte_size: Optional[int],
      target_data_file_size: Optional[int],
      expected_chunk_byte_size: Optional[int],
  ):
    actual_chunk_byte_size = ts_utils.adjust_chunk_byte_size(
        write_shape=shape,
        dtype=dtype,
        chunk_byte_size=chunk_byte_size,
        ocdbt_target_data_file_size=target_data_file_size,
    )
    self.assertEqual(actual_chunk_byte_size, expected_chunk_byte_size)


if __name__ == '__main__':
  absltest.main()
