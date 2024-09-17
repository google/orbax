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

import dataclasses
import math
import os
from typing import Optional

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from orbax.checkpoint._src.arrays import subchunking
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


class BuildArrayTSpecForWriteTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.shape = (10, 6, 32)
    self.write_shape = (5, 6, 8)
    self.dtype = np.dtype(np.int32)
    self.directory = self.create_tempdir().full_path
    self.param_name = 'params/a'

  def array_metadata(
      self,
      use_zarr3: bool = False,
  ) -> ts_utils.ArrayWriteMetadata:
    return ts_utils.ArrayWriteMetadata(
        global_shape=self.shape,
        write_shape=self.write_shape,
        dtype=self.dtype,
        use_zarr3=use_zarr3
    )

  @parameterized.product(use_zarr3=[True, False])
  def test_respects_zarr_version(self, use_zarr3: bool):
    with self.subTest('with_ocdbt'):
      tspec = ts_utils.build_array_tspec_for_write(
          directory=self.directory,
          relative_array_filename=self.param_name,
          array_metadata=self.array_metadata(use_zarr3=use_zarr3),
          use_ocdbt=True,
          process_id=0,
      )
      self.assertEqual(tspec['driver'], 'zarr3' if use_zarr3 else 'zarr')
    with self.subTest('without_ocdbt'):
      tspec = ts_utils.build_array_tspec_for_write(
          directory=self.directory,
          relative_array_filename=self.param_name,
          array_metadata=self.array_metadata(use_zarr3=use_zarr3),
          use_ocdbt=False,
      )
      self.assertEqual(tspec['driver'], 'zarr3' if use_zarr3 else 'zarr')

  @parameterized.named_parameters(
      dict(
          testcase_name='local_fs_path',
          directory='/tmp/local_path',
          expected_driver=ts_utils.DEFAULT_DRIVER,
      ),
  )
  def test_file_kvstore(
      self,
      directory: str,
      expected_driver: str,
  ):
    array_metadata = self.array_metadata()
    tspec = ts_utils.build_array_tspec_for_write(
        directory=directory,
        relative_array_filename=self.param_name,
        array_metadata=array_metadata,
        use_ocdbt=False,
    )
    self.assertEqual(tspec['kvstore']['driver'], expected_driver)
    self.assertEqual(
        tspec['kvstore']['path'], os.path.join(directory, self.param_name)
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='regular_path',
          directory='gs://gcs_bucket/object_path',
      ),
      dict(
          testcase_name='path_with_single_slash',
          directory='gs:/gcs_bucket/object_path',
      ),
  )
  def test_file_kvstore_with_gcs_path(
      self,
      directory: str,
  ):
    array_metadata = self.array_metadata()
    tspec = ts_utils.build_array_tspec_for_write(
        directory=directory,
        relative_array_filename=self.param_name,
        array_metadata=array_metadata,
        use_ocdbt=False,
    )
    self.assertEqual(
        tspec['kvstore'],
        {
            'driver': 'gcs',
            'bucket': 'gcs_bucket',
            'path': f'object_path/{self.param_name}',
        },
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='local_fs_path',
          directory='/tmp/local_path',
          expected_base_driver=ts_utils.DEFAULT_DRIVER,
      ),
  )
  def test_ocdbt_kvstore(
      self,
      directory: str,
      expected_base_driver: str,
  ):
    array_metadata = self.array_metadata()
    tspec = ts_utils.build_array_tspec_for_write(
        directory=directory,
        relative_array_filename=self.param_name,
        array_metadata=array_metadata,
        use_ocdbt=True,
        process_id=13,
    )
    self.assertEqual(tspec['kvstore']['driver'], 'ocdbt')
    base_spec = tspec['kvstore']['base']
    self.assertEqual(base_spec['driver'], expected_base_driver)
    self.assertEqual(
        base_spec['path'],
        os.path.join(directory, 'ocdbt.process_13'),
    )
    self.assertEqual(tspec['kvstore']['path'], self.param_name)

  @parameterized.named_parameters(
      dict(
          testcase_name='regular_path',
          directory='gs://gcs_bucket/object_path',
          expected_directory=None,
      ),
      dict(
          testcase_name='path_with_single_slash',
          directory='gs:/gcs_bucket/object_path',
          expected_directory='gs://gcs_bucket/object_path',
      ),
  )
  def test_ocdbt_kvstore_with_gcs_path(
      self,
      directory: str,
      expected_directory: Optional[str],
  ):
    array_metadata = self.array_metadata()
    tspec = ts_utils.build_array_tspec_for_write(
        directory=directory,
        relative_array_filename=self.param_name,
        array_metadata=array_metadata,
        use_ocdbt=True,
        process_id=0,
    )
    self.assertEqual(tspec['kvstore']['driver'], 'ocdbt')
    self.assertEqual(
        tspec['kvstore']['base'],
        os.path.join(expected_directory or directory, 'ocdbt.process_0'),
    )
    self.assertEqual(tspec['kvstore']['path'], self.param_name)

  def test_ocdbt_kvstore_default_target_data_file_size(self):
    tspec = ts_utils.build_array_tspec_for_write(
        directory=self.directory,
        relative_array_filename=self.param_name,
        array_metadata=self.array_metadata(),
        use_ocdbt=True,
        process_id=13
    )
    self.assertEqual(tspec['kvstore']['driver'], 'ocdbt')
    self.assertNotIn('target_data_file_size', tspec['kvstore'])

  @parameterized.named_parameters(
      dict(testcase_name='none', target_data_file_size=None),
      dict(testcase_name='unlimited', target_data_file_size=0),
      dict(testcase_name='custom_limit', target_data_file_size=1024),
  )
  def test_ocdbt_kvstore_target_data_file_size(
      self,
      target_data_file_size: Optional[int],
  ):
    tspec = ts_utils.build_array_tspec_for_write(
        directory=self.directory,
        relative_array_filename=self.param_name,
        array_metadata=self.array_metadata(),
        use_ocdbt=True,
        process_id=13,
        ocdbt_target_data_file_size=target_data_file_size,
    )
    self.assertEqual(tspec['kvstore']['driver'], 'ocdbt')
    if target_data_file_size is None:
      self.assertNotIn('target_data_file_size', tspec['kvstore'])
    else:
      self.assertEqual(
          tspec['kvstore']['target_data_file_size'], target_data_file_size
      )

  @parameterized.product(
      use_zarr3=[True, False],
      use_ocdbt=[True, False],
  )
  def test_data_recheck_disabled(self, use_zarr3: bool, use_ocdbt: bool):
    tspec = ts_utils.build_array_tspec_for_write(
        directory=self.directory,
        relative_array_filename=self.param_name,
        array_metadata=self.array_metadata(use_zarr3=use_zarr3),
        use_ocdbt=use_ocdbt,
        process_id=13,
    )
    self.assertFalse(tspec['recheck_cached_metadata'])
    self.assertFalse(tspec['recheck_cached_data'])

  @parameterized.product(
      use_zarr3=[True, False],
      use_ocdbt=[True, False],
  )
  def test_dtype(self, use_zarr3: bool, use_ocdbt: bool):
    array_metadata = self.array_metadata(use_zarr3=use_zarr3)
    self.assertIsNone(array_metadata.target_dtype)
    tspec = ts_utils.build_array_tspec_for_write(
        directory=self.directory,
        relative_array_filename=self.param_name,
        array_metadata=array_metadata,
        use_ocdbt=use_ocdbt,
        process_id=42,
    )
    self.assertEqual(tspec['dtype'], 'int32')

  @parameterized.product(
      use_zarr3=[True, False],
      use_ocdbt=[True, False],
  )
  def test_no_casting_if_target_dtype_matches_source(
      self,
      use_zarr3: bool,
      use_ocdbt: bool,
  ):
    array_metadata = self.array_metadata(use_zarr3=use_zarr3)
    array_metadata = dataclasses.replace(
        array_metadata,
        target_dtype=array_metadata.dtype,
    )
    tspec = ts_utils.build_array_tspec_for_write(
        directory=self.directory,
        relative_array_filename=self.param_name,
        array_metadata=array_metadata,
        use_ocdbt=use_ocdbt,
        process_id=42,
    )
    self.assertEqual(tspec['dtype'], 'int32')
    self.assertEqual(tspec['driver'], 'zarr3' if use_zarr3 else 'zarr')

  @parameterized.product(
      use_zarr3=[True, False],
      use_ocdbt=[True, False],
  )
  def test_casts_to_target_dtype(
      self,
      use_zarr3: bool,
      use_ocdbt: bool,
  ):
    array_metadata = dataclasses.replace(
        self.array_metadata(use_zarr3=use_zarr3),
        target_dtype=np.dtype(np.float32),
    )
    assert array_metadata.target_dtype != array_metadata.dtype
    tspec = ts_utils.build_array_tspec_for_write(
        directory=self.directory,
        relative_array_filename=self.param_name,
        array_metadata=array_metadata,
        use_ocdbt=use_ocdbt,
        process_id=42,
    )
    self.assertEqual(tspec['driver'], 'cast')
    self.assertEqual(tspec['dtype'], 'int32')
    self.assertEqual(tspec['base']['dtype'], 'float32')

  def _get_chunk_shape_from_tspec(
      self,
      tspec: ts_utils.JsonSpec,
      use_zarr3: bool,
  ) -> types.Shape:
    try:
      if use_zarr3:
        return tspec['metadata']['codecs'][0]['configuration']['chunk_shape']
      else:
        return tspec['metadata']['chunks']
    except Exception as e:
      raise ValueError(tspec) from e

  @parameterized.product(
      (
          dict(use_ocdbt=False, process_id=None),
          dict(use_ocdbt=True, process_id=13),
      ),
      use_zarr3=[True, False],
      chunk_byte_size=[None, 256, 512, 1024, 1024**2],
  )
  def test_chunk_byte_size(
      self,
      use_ocdbt: bool,
      process_id: Optional[int],
      use_zarr3: bool,
      chunk_byte_size: Optional[int],
  ):
    array_metadata = dataclasses.replace(
        self.array_metadata(use_zarr3=use_zarr3),
        chunk_byte_size=chunk_byte_size,
    )
    tspec = ts_utils.build_array_tspec_for_write(
        directory=self.directory,
        relative_array_filename=self.param_name,
        array_metadata=array_metadata,
        use_ocdbt=use_ocdbt,
        process_id=process_id,
    )
    chunk_shape = self._get_chunk_shape_from_tspec(tspec, use_zarr3)
    if chunk_byte_size is None:
      np.testing.assert_array_equal(chunk_shape, self.write_shape)
    else:
      self.assertTrue(
          subchunking.validate_divisible_shapes(self.write_shape, chunk_shape)
      )
      self.assertLessEqual(
          math.prod(chunk_shape) * array_metadata.dtype.itemsize,
          chunk_byte_size,
      )

  @parameterized.product(
      (
          dict(use_ocdbt=False, process_id=None),
          dict(use_ocdbt=True, process_id=13),
      ),
      use_zarr3=[True, False],
  )
  def test_chunk_byte_size_accounts_for_target_dtype(
      self,
      use_ocdbt: bool,
      process_id: Optional[int],
      use_zarr3: bool,
  ):
    self.global_shape = (8, 64, 32)
    self.write_shape = (4, 64, 16)
    chunk_byte_size = 100  # In-between of two exact powers of 2.
    array_metadata = dataclasses.replace(
        self.array_metadata(use_zarr3=use_zarr3),
        target_dtype=np.dtype(np.int16),
        chunk_byte_size=chunk_byte_size,
    )
    assert array_metadata.target_dtype is not None  # Make type checker happy.
    tspec = ts_utils.build_array_tspec_for_write(
        directory=self.directory,
        relative_array_filename=self.param_name,
        array_metadata=array_metadata,
        use_ocdbt=use_ocdbt,
        process_id=process_id,
    )
    chunk_shape = self._get_chunk_shape_from_tspec(tspec['base'], use_zarr3)
    self.assertTrue(
        subchunking.validate_divisible_shapes(self.write_shape, chunk_shape),
        f'Write shape {self.write_shape} is not divisible by chunk shape'
        f' {chunk_shape}.',
    )
    self.assertLessEqual(
        math.prod(chunk_shape) * array_metadata.target_dtype.itemsize,
        chunk_byte_size,
    )
    self.assertGreater(
        math.prod(chunk_shape) * array_metadata.dtype.itemsize,
        chunk_byte_size,
    )

  @parameterized.product(
      (
          dict(
              chunk_byte_size=None,
              target_data_file_size=None,
              expected_chunk_byte_size_limit=(
                  ts_utils._DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE
              ),
          ),
          dict(
              chunk_byte_size=(5 * GIB),
              target_data_file_size=None,
              expected_chunk_byte_size_limit=(
                  ts_utils._DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE
              ),
          ),
          dict(
              chunk_byte_size=(
                  3 * ts_utils._DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE
              ),
              target_data_file_size=0,
              expected_chunk_byte_size_limit=(
                  3 * ts_utils._DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE
              ),
          ),
          dict(
              chunk_byte_size=(3 * GIB),
              target_data_file_size=(1 * GIB),
              expected_chunk_byte_size_limit=(1 * GIB),
          ),
      ),
      use_zarr3=[True, False],
      target_dtype=[None, np.dtype(np.int16)],
  )
  def test_chunk_byte_size_is_adjusted_for_target_data_file_size(
      self,
      chunk_byte_size: Optional[int],
      target_data_file_size: Optional[int],
      expected_chunk_byte_size_limit: int,
      use_zarr3: bool,
      target_dtype: Optional[np.dtype],
  ):
    self.global_shape = (8 * 1024, 2 * 1024, 4 * 1024)
    self.write_shape = (2 * 1024, 1024, 2 * 1024)
    array_metadata = dataclasses.replace(
        self.array_metadata(use_zarr3=use_zarr3),
        target_dtype=target_dtype,
        chunk_byte_size=chunk_byte_size,
    )
    if target_dtype is not None:
      assert array_metadata.target_dtype is not None
      assert array_metadata.target_dtype == target_dtype
      storage_dtype = array_metadata.target_dtype
    else:
      storage_dtype = array_metadata.dtype

    tspec = ts_utils.build_array_tspec_for_write(
        directory=self.directory,
        relative_array_filename=self.param_name,
        array_metadata=array_metadata,
        use_ocdbt=True,
        process_id='w13',
        ocdbt_target_data_file_size=target_data_file_size,
    )
    chunk_shape = self._get_chunk_shape_from_tspec(
        tspec if target_dtype is None else tspec['base'],
        use_zarr3,
    )
    self.assertTrue(
        subchunking.validate_divisible_shapes(self.write_shape, chunk_shape),
        f'Write shape {self.write_shape} is not divisible by chunk shape'
        f' {chunk_shape}.',
    )

    self.assertLessEqual(
        math.prod(chunk_shape) * storage_dtype.itemsize,
        expected_chunk_byte_size_limit,
    )
    if storage_dtype != array_metadata.dtype:
      assert storage_dtype.itemsize < array_metadata.dtype.itemsize
      self.assertGreater(
          math.prod(chunk_shape) * array_metadata.dtype.itemsize,
          expected_chunk_byte_size_limit,
      )


if __name__ == '__main__':
  absltest.main()
