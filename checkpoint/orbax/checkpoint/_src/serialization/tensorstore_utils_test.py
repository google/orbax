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

import functools
import math
import os

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

    self.array_write_spec_constructor = functools.partial(
        ts_utils.ArrayWriteSpec,
        global_shape=self.shape,
        write_shape=self.write_shape,
        dtype=self.dtype,
    )

  @parameterized.product(
      use_zarr3=(True, False),
      use_ocdbt=(True, False),
  )
  def test_metadata(self, use_zarr3: bool, use_ocdbt: bool):
    if use_ocdbt:
      tspec = self.array_write_spec_constructor(
          directory=self.directory,
          relative_array_filename=self.param_name,
          use_zarr3=use_zarr3,
          use_ocdbt=True,
          process_id=0,
      )
    else:
      tspec = self.array_write_spec_constructor(
          directory=self.directory,
          relative_array_filename=self.param_name,
          use_zarr3=use_zarr3,
          use_ocdbt=False,
      )
    self.assertEqual(tspec.metadata.shape, self.shape)
    self.assertEqual(tspec.metadata.write_shape, self.write_shape)
    self.assertEqual(tspec.metadata.chunk_shape, self.write_shape)
    self.assertEqual(tspec.metadata.dtype, self.dtype)
    self.assertEqual(tspec.metadata.use_ocdbt, use_ocdbt)
    self.assertEqual(tspec.metadata.use_zarr3, use_zarr3)
    self.assertEqual(tspec.json['driver'], 'zarr3' if use_zarr3 else 'zarr')

  @parameterized.product(use_ocdbt=(True, False))
  def test_default_zarr_driver_version(self, use_ocdbt: bool):
    tspec = self.array_write_spec_constructor(
        directory=self.directory,
        relative_array_filename=self.param_name,
        use_ocdbt=use_ocdbt,
    )
    self.assertEqual(tspec.metadata.use_ocdbt, use_ocdbt)
    self.assertFalse(tspec.metadata.use_zarr3)
    self.assertEqual(tspec.json['driver'], 'zarr')

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
    tspec = self.array_write_spec_constructor(
        directory=directory,
        relative_array_filename=self.param_name,
        use_zarr3=False,
        use_ocdbt=False,
    )
    self.assertFalse(tspec.metadata.use_ocdbt)
    json_tspec = tspec.json
    self.assertEqual(json_tspec['kvstore']['driver'], expected_driver)
    self.assertEqual(
        json_tspec['kvstore']['path'], os.path.join(directory, self.param_name)
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
    tspec = self.array_write_spec_constructor(
        directory=directory,
        relative_array_filename=self.param_name,
        use_zarr3=False,
        use_ocdbt=False,
    )
    self.assertFalse(tspec.metadata.use_ocdbt)
    self.assertEqual(
        tspec.json['kvstore'],
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
    tspec = self.array_write_spec_constructor(
        directory=directory,
        relative_array_filename=self.param_name,
        use_zarr3=False,
        use_ocdbt=True,
        process_id=13,
    )
    self.assertTrue(tspec.metadata.use_ocdbt)
    json_tspec = tspec.json
    self.assertEqual(json_tspec['kvstore']['driver'], 'ocdbt')
    base_spec = json_tspec['kvstore']['base']
    self.assertEqual(base_spec['driver'], expected_base_driver)
    self.assertEqual(
        base_spec['path'],
        os.path.join(directory, 'ocdbt.process_13'),
    )
    self.assertEqual(json_tspec['kvstore']['path'], self.param_name)

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
      expected_directory: str | None,
  ):
    tspec = self.array_write_spec_constructor(
        directory=directory,
        relative_array_filename=self.param_name,
        use_zarr3=False,
        use_ocdbt=True,
        process_id=0,
    )
    self.assertTrue(tspec.metadata.use_ocdbt)
    kvstore_tspec = tspec.json['kvstore']
    self.assertEqual(kvstore_tspec['driver'], 'ocdbt')
    self.assertEqual(
        kvstore_tspec['base'],
        os.path.join(expected_directory or directory, 'ocdbt.process_0'),
    )
    self.assertEqual(kvstore_tspec['path'], self.param_name)

  @parameterized.product(use_zarr3=(True, False))
  def test_ocdbt_kvstore_default_target_data_file_size(self, use_zarr3: bool):
    tspec = self.array_write_spec_constructor(
        directory=self.directory,
        relative_array_filename=self.param_name,
        use_zarr3=use_zarr3,
        use_ocdbt=True,
        process_id=13
    )
    self.assertEqual(tspec.metadata.use_zarr3, use_zarr3)
    self.assertTrue(tspec.metadata.use_ocdbt)
    self.assertEqual(tspec.json['kvstore']['driver'], 'ocdbt')
    self.assertNotIn('target_data_file_size', tspec.json['kvstore'])

  @parameterized.named_parameters(
      dict(testcase_name='none', target_data_file_size=None),
      dict(testcase_name='unlimited', target_data_file_size=0),
      dict(testcase_name='custom_limit', target_data_file_size=1024),
  )
  def test_ocdbt_kvstore_target_data_file_size(
      self,
      target_data_file_size: int | None,
  ):
    tspec = self.array_write_spec_constructor(
        directory=self.directory,
        relative_array_filename=self.param_name,
        use_zarr3=False,
        use_ocdbt=True,
        process_id=13,
        ocdbt_target_data_file_size=target_data_file_size,
    )
    self.assertTrue(tspec.metadata.use_ocdbt)
    kvstore_tspec = tspec.json['kvstore']
    self.assertEqual(kvstore_tspec['driver'], 'ocdbt')
    if target_data_file_size is None:
      self.assertNotIn('target_data_file_size', kvstore_tspec)
    else:
      self.assertEqual(
          kvstore_tspec['target_data_file_size'], target_data_file_size
      )

  @parameterized.product(
      use_zarr3=(True, False),
      use_ocdbt=(True, False),
  )
  def test_data_recheck_disabled(self, use_zarr3: bool, use_ocdbt: bool):
    json_tspec = self.array_write_spec_constructor(
        directory=self.directory,
        relative_array_filename=self.param_name,
        use_zarr3=use_zarr3,
        use_ocdbt=use_ocdbt,
        process_id=13,
    ).json
    self.assertFalse(json_tspec['recheck_cached_metadata'])
    self.assertFalse(json_tspec['recheck_cached_data'])

  @parameterized.product(
      use_zarr3=(True, False),
      use_ocdbt=(True, False),
  )
  def test_dtype(self, use_zarr3: bool, use_ocdbt: bool):
    tspec = self.array_write_spec_constructor(
        directory=self.directory,
        relative_array_filename=self.param_name,
        use_zarr3=use_zarr3,
        use_ocdbt=use_ocdbt,
        process_id=42,
    )
    self.assertEqual(tspec.metadata.dtype, self.dtype)
    self.assertEqual(tspec.json['dtype'], 'int32')

  @parameterized.product(
      use_zarr3=(True, False),
      use_ocdbt=(True, False),
  )
  def test_no_casting_if_target_dtype_matches_source(
      self,
      use_zarr3: bool,
      use_ocdbt: bool,
  ):
    tspec = self.array_write_spec_constructor(
        directory=self.directory,
        relative_array_filename=self.param_name,
        use_zarr3=use_zarr3,
        target_dtype=self.dtype,
        use_ocdbt=use_ocdbt,
        process_id=42,
    )
    self.assertEqual(tspec.metadata.dtype, self.dtype)
    self.assertEqual(tspec.json['dtype'], 'int32')
    self.assertEqual(tspec.json['driver'], 'zarr3' if use_zarr3 else 'zarr')

  @parameterized.product(
      use_zarr3=(True, False),
      use_ocdbt=(True, False),
  )
  def test_casts_to_target_dtype(
      self,
      use_zarr3: bool,
      use_ocdbt: bool,
  ):
    target_dtype = np.dtype(np.float32)
    assert target_dtype != self.dtype
    tspec = self.array_write_spec_constructor(
        directory=self.directory,
        relative_array_filename=self.param_name,
        target_dtype=target_dtype,
        use_zarr3=use_zarr3,
        use_ocdbt=use_ocdbt,
        process_id=42,
    )
    self.assertEqual(tspec.metadata.dtype, target_dtype)
    self.assertEqual(tspec.json['driver'], 'cast')
    self.assertEqual(tspec.json['dtype'], 'int32')
    self.assertEqual(tspec.json['base']['dtype'], 'float32')

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
      use_zarr3=(True, False),
      chunk_byte_size=(None, 256, 512, 1024, 1024**2),
  )
  def test_chunk_byte_size(
      self,
      use_ocdbt: bool,
      process_id: int | None,
      use_zarr3: bool,
      chunk_byte_size: int | None,
  ):
    tspec = self.array_write_spec_constructor(
        directory=self.directory,
        relative_array_filename=self.param_name,
        chunk_byte_size=chunk_byte_size,
        use_zarr3=use_zarr3,
        use_ocdbt=use_ocdbt,
        process_id=process_id,
    )
    chunk_shape = self._get_chunk_shape_from_tspec(tspec.json, use_zarr3)
    np.testing.assert_array_equal(chunk_shape, tspec.metadata.chunk_shape)
    if chunk_byte_size is None:
      np.testing.assert_array_equal(chunk_shape, self.write_shape)
    else:
      self.assertTrue(
          subchunking.validate_divisible_shapes(self.write_shape, chunk_shape)
      )
      self.assertLessEqual(
          math.prod(chunk_shape) * self.dtype.itemsize,
          chunk_byte_size,
      )

  @parameterized.product(
      (
          dict(use_ocdbt=False, process_id=None),
          dict(use_ocdbt=True, process_id=13),
      ),
      use_zarr3=(True, False),
  )
  def test_chunk_byte_size_accounts_for_target_dtype(
      self,
      use_ocdbt: bool,
      process_id: int | None,
      use_zarr3: bool,
  ):
    self.shape = (8, 64, 32)
    self.write_shape = (4, 64, 16)
    chunk_byte_size = 100  # In-between of two exact powers of 2.
    target_dtype = np.dtype(np.int16)
    assert target_dtype != self.dtype
    tspec = ts_utils.ArrayWriteSpec(
        directory=self.directory,
        relative_array_filename=self.param_name,
        global_shape=self.shape,
        write_shape=self.write_shape,
        dtype=self.dtype,
        target_dtype=target_dtype,
        chunk_byte_size=chunk_byte_size,
        use_zarr3=use_zarr3,
        use_ocdbt=use_ocdbt,
        process_id=process_id,
    )
    chunk_shape = self._get_chunk_shape_from_tspec(
        tspec.json['base'], use_zarr3
    )
    self.assertEqual(tspec.metadata.dtype, target_dtype)
    np.testing.assert_array_equal(chunk_shape, tspec.metadata.chunk_shape)
    self.assertTrue(
        subchunking.validate_divisible_shapes(self.write_shape, chunk_shape),
        f'Write shape {self.write_shape} is not divisible by chunk shape'
        f' {chunk_shape}.',
    )
    self.assertLessEqual(
        math.prod(chunk_shape) * target_dtype.itemsize,
        chunk_byte_size,
    )
    self.assertGreater(
        math.prod(chunk_shape) * self.dtype.itemsize,
        chunk_byte_size,
    )

  @parameterized.product(
      (
          dict(use_ocdbt=False, process_id=None),
          dict(use_ocdbt=True, process_id=13),
      ),
      use_zarr3=(True, False),
      target_dtype=(None, np.dtype(np.int16)),
      shard_axes=(
          (1, 2),  # both unsharded
          (0, 2),  # one already sharded and another unsharded
      ),
  )
  def test_chunk_byte_size_with_shard_axes(
      self,
      use_ocdbt: bool,
      process_id: int | None,
      use_zarr3: bool,
      target_dtype: np.dtype | None,
      shard_axes: tuple[int, ...],
  ):
    self.shape = (8, 64, 32)
    self.write_shape = (2, 64, 32)
    storage_dtype = target_dtype or self.dtype

    # Value in-between of two exact powers of 2 to check that the chunk shape is
    # adjusted for the target dtype, and small enough to be subchunked on all of
    # the requested shard axes.
    chunk_byte_size = 100
    assert chunk_byte_size < (
        math.prod(self.write_shape) * storage_dtype.itemsize
    ) // 2 ** len(shard_axes)

    tspec = ts_utils.ArrayWriteSpec(
        directory=self.directory,
        relative_array_filename=self.param_name,
        global_shape=self.shape,
        write_shape=self.write_shape,
        dtype=self.dtype,
        target_dtype=target_dtype,
        chunk_byte_size=chunk_byte_size,
        shard_axes=shard_axes,
        use_zarr3=use_zarr3,
        use_ocdbt=use_ocdbt,
        process_id=process_id,
    )
    chunk_shape = self._get_chunk_shape_from_tspec(
        tspec.json['base'] if target_dtype is not None else tspec.json,
        use_zarr3,
    )
    np.testing.assert_array_equal(chunk_shape, tspec.metadata.chunk_shape)
    self.assertTrue(
        subchunking.validate_divisible_shapes(self.write_shape, chunk_shape),
        f'Write shape {self.write_shape} is not divisible by chunk shape'
        f' {chunk_shape}.',
    )
    self.assertEqual(tspec.metadata.dtype, storage_dtype)
    # Byte size within requested limit.
    self.assertLessEqual(
        math.prod(chunk_shape) * storage_dtype.itemsize,
        chunk_byte_size,
    )
    # Write shape subchunked on both of requested axes.
    for shard_axis in shard_axes:
      self.assertLess(chunk_shape[shard_axis], self.write_shape[shard_axis])
    # If storage dtype is different from the dtype, it should be accounted for.
    if storage_dtype != self.dtype:
      assert storage_dtype.itemsize < self.dtype.itemsize
      self.assertGreater(
          math.prod(chunk_shape) * self.dtype.itemsize,
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
      chunk_byte_size: int | None,
      target_data_file_size: int | None,
      expected_chunk_byte_size_limit: int,
      use_zarr3: bool,
      target_dtype: np.dtype | None,
  ):
    self.shape = (8 * 1024, 2 * 1024, 4 * 1024)
    self.write_shape = (2 * 1024, 1024, 2 * 1024)
    storage_dtype = target_dtype or self.dtype

    tspec = ts_utils.ArrayWriteSpec(
        directory=self.directory,
        relative_array_filename=self.param_name,
        global_shape=self.shape,
        write_shape=self.write_shape,
        dtype=self.dtype,
        target_dtype=target_dtype,
        chunk_byte_size=chunk_byte_size,
        use_zarr3=use_zarr3,
        use_ocdbt=True,
        process_id='w13',
        ocdbt_target_data_file_size=target_data_file_size,
    )
    self.assertEqual(tspec.metadata.dtype, storage_dtype)
    chunk_shape = self._get_chunk_shape_from_tspec(
        tspec.json if target_dtype is None else tspec.json['base'],
        use_zarr3,
    )
    np.testing.assert_array_equal(chunk_shape, tspec.metadata.chunk_shape)
    self.assertTrue(
        subchunking.validate_divisible_shapes(self.write_shape, chunk_shape),
        f'Write shape {self.write_shape} is not divisible by chunk shape'
        f' {chunk_shape}.',
    )

    self.assertLessEqual(
        math.prod(chunk_shape) * storage_dtype.itemsize,
        expected_chunk_byte_size_limit,
    )
    if storage_dtype != self.dtype:
      assert storage_dtype.itemsize < self.dtype.itemsize
      self.assertGreater(
          math.prod(chunk_shape) * self.dtype.itemsize,
          expected_chunk_byte_size_limit,
      )


if __name__ == '__main__':
  absltest.main()
