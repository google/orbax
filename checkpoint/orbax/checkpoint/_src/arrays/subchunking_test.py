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

import math

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import numpy as np
from orbax.checkpoint._src.arrays import subchunking
from orbax.checkpoint._src.arrays import types


Shape = types.Shape


class FindDivisorsTest(absltest.TestCase):

  def test_find_divisors(self):
    for n in range(1, 17):
      self.assertListEqual(
          subchunking._find_divisors(n),
          [d for d in range(1, n + 1) if n % d == 0],
      )


class ChooseChunkShapeTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.ckpt_dir = epath.Path(self.create_tempdir('ckpt').full_path)

  @parameterized.named_parameters(
      dict(
          testcase_name='allow_only_one_element',
          shape=(10, 100, 200),
          target_element_count=1,
          expected_shape=(1, 1, 1),
      ),
      dict(
          testcase_name='allow_3_elements',
          shape=(10, 100, 200),
          target_element_count=5**3,
          expected_shape=(5, 5, 5),
      ),
      dict(
          testcase_name='allow_4_elements',
          shape=(10, 100, 200),
          target_element_count=5**4,
          expected_shape=(5, 10, 10),
      ),
  )
  def test_write_shape_equals_global_shape(
      self,
      shape: Shape,
      target_element_count: int,
      expected_shape: Shape,
  ):
    dtype = np.dtype('float32')

    chosen_shape = subchunking.choose_chunk_shape(
        global_shape=shape,
        write_shape=shape,
        dtype=dtype,
        target_byte_size=target_element_count * dtype.itemsize,
    )
    np.testing.assert_array_equal(chosen_shape, expected_shape)

  def test_target_byte_size_not_a_multiple_of_dtype_size(self):
    dtype = np.dtype('float32')
    shape = (10, 100, 200)

    # Should still result in a correct shape.
    chosen_shape = subchunking.choose_chunk_shape(
        global_shape=shape,
        write_shape=shape,
        dtype=dtype,
        target_byte_size=5**4 * dtype.itemsize + 3,
    )
    np.testing.assert_array_equal(chosen_shape, (5, 10, 10))

  def test_sharded_array(self):
    dtype = np.dtype('float32')

    with self.subTest('allows_to_split_on_the_sharded_axis'):
      chosen_shape = subchunking.choose_chunk_shape(
          global_shape=(10, 500, 200),
          write_shape=(10, 100, 200),
          dtype=dtype,
          target_byte_size=10 * 5 * 200 * dtype.itemsize,
      )
      np.testing.assert_array_equal(chosen_shape, (10, 5, 200))

    with self.subTest('prefers_a_sharded_axis_to_an_unsharded_axis'):
      chosen_shape = subchunking.choose_chunk_shape(
          global_shape=(10, 100, 200),
          write_shape=(10, 100, 100),
          dtype=dtype,
          target_byte_size=10 * 100 * 25 * dtype.itemsize,
      )
      np.testing.assert_array_equal(chosen_shape, (10, 100, 25))

    with self.subTest('forced_to_split_on_unsharded_axis_when_target_is_small'):
      chosen_shape = subchunking.choose_chunk_shape(
          global_shape=(10, 500, 200),
          write_shape=(10, 100, 200),
          dtype=dtype,
          target_byte_size=10 * 1 * 100 * dtype.itemsize,
      )
      np.testing.assert_array_equal(chosen_shape, (10, 1, 100))


class ChooseChunkShapeWithShardAxesTest(parameterized.TestCase):

  def test_maximizes_number_of_axes_to_shard(self):
    dtype = np.dtype('float32')
    global_shape = (8, 10, 9)
    target_elements = 180
    chosen_shape = subchunking.choose_chunk_shape(
        global_shape=global_shape,
        write_shape=global_shape,
        dtype=dtype,
        target_byte_size=target_elements * dtype.itemsize,
        shard_axes=(0, 1, 2),
    )
    np.testing.assert_array_equal(chosen_shape, (4, 5, 9))

  @parameterized.named_parameters(
      dict(
          testcase_name='shards_on_one_axis',
          global_shape=(4, 90, 6),
          shard_axes=(0,),
          target_elements=2 * 90 * 6,
          expected_shape=(2, 90, 6),
      ),
      dict(
          testcase_name='shards_multiple_times_on_one_axis',
          global_shape=(4, 90, 6),
          shard_axes=(1,),
          target_elements=4 * 9 * 6,
          expected_shape=(4, 9, 6),
      ),
      dict(
          testcase_name='shards_on_all_requested_axes_if_possible',
          global_shape=(4, 90, 6, 128),
          shard_axes=(0, 2, 3),
          target_elements=2 * 90 * 3 * 64,
          expected_shape=(2, 90, 3, 64),
      ),
      dict(
          testcase_name='skips_indivisible_axes',
          global_shape=(1, 90, 1, 128),
          shard_axes=(0, 2, 3),
          target_elements=1 * 90 * 1 * 32,
          expected_shape=(1, 90, 1, 32),
      ),
      dict(
          testcase_name='shards_multiple_times_on_several_axes',
          global_shape=(4, 90, 6, 128),
          shard_axes=(0, 3),
          target_elements=1 * 90 * 6 * 32,
          expected_shape=(1, 90, 6, 32),
      ),
      dict(
          testcase_name=(
              'shards_multiple_times_on_several_axes_exhausting_one_axis'
          ),
          global_shape=(4, 90, 6, 128),
          shard_axes=(0, 3),
          target_elements=1 * 90 * 6 * 8,
          expected_shape=(1, 90, 6, 8),
      ),
      dict(
          testcase_name=(
              'shards_multiple_times_exhausting_several_axes'
          ),
          global_shape=(4, 90, 6, 128),
          shard_axes=(0, 2, 3),
          target_elements=1 * 90 * 1 * 32,
          expected_shape=(1, 90, 1, 32),
      ),
      dict(
          testcase_name='exhausts_one_axis_and_falls_back_to_greedy_sharding',
          global_shape=(4, 90, 6),
          shard_axes=(0,),
          target_elements=1 * 45 * 6,
          expected_shape=(1, 45, 6),
      ),
      dict(
          testcase_name=(
              'exhausts_several_axes_and_falls_back_to_greedy_sharding'
          ),
          global_shape=(4, 90, 6, 128),
          shard_axes=(0, 2),
          target_elements=1 * 45 * 1 * 32,
          expected_shape=(1, 45, 1, 32),
      ),
  )
  def test_sharding_when_write_shape_equals_global_shape(
      self,
      global_shape: Shape,
      shard_axes: tuple[int, ...],
      target_elements: int,
      expected_shape: Shape,
  ):
    dtype = np.dtype('float32')
    chosen_shape = subchunking.choose_chunk_shape(
        global_shape=global_shape,
        write_shape=global_shape,
        dtype=dtype,
        target_byte_size=target_elements * dtype.itemsize,
        shard_axes=shard_axes,
    )
    np.testing.assert_array_equal(chosen_shape, expected_shape)

  @parameterized.named_parameters(
      dict(
          testcase_name='prefers_unsharded_axis_if_requested',
          global_shape=(4, 80, 12),
          write_shape=(4, 40, 6),
          shard_axes=(0,),
          target_elements=1 * 40 * 6,
          expected_shape=(1, 40, 6),
      ),
      dict(
          testcase_name='ensures_all_requested_axes_are_sharded',
          global_shape=(4, 80, 12, 16),
          write_shape=(4, 40, 6, 16),
          shard_axes=(0, 2, 3),
          target_elements=2 * 40 * 6 * 8,
          expected_shape=(2, 40, 6, 8),
      ),
      dict(
          # Splits both axes which have minimal available divisors (2).
          testcase_name='ensures_max_number_of_requested_axes_are_sharded',
          global_shape=(4, 80, 12, 16, 9),
          write_shape=(4, 40, 6, 16, 9),
          shard_axes=(0, 3, 4),
          target_elements=2 * 40 * 6 * 8 * 9,
          expected_shape=(2, 40, 6, 8, 9),
      ),
      dict(
          testcase_name='skips_indivisible_axes',
          global_shape=(4, 12, 80, 16),
          write_shape=(1, 1, 40, 16),
          shard_axes=(0, 1, 3),
          target_elements=1 * 1 * 40 * 4,
          expected_shape=(1, 1, 40, 4),
      ),
  )
  def test_shard_axes_with_an_already_sharded_array(
      self,
      global_shape: Shape,
      write_shape: Shape,
      shard_axes: tuple[int, ...],
      target_elements: int,
      expected_shape: Shape,
  ):
    dtype = np.dtype('float32')
    chosen_shape = subchunking.choose_chunk_shape(
        global_shape=global_shape,
        write_shape=write_shape,
        dtype=dtype,
        target_byte_size=target_elements * dtype.itemsize,
        shard_axes=shard_axes,
    )
    np.testing.assert_array_equal(chosen_shape, expected_shape)

  def test_with_target_byte_size_not_a_multiple_of_dtype_itemsize(self):
    dtype = np.dtype('float32')
    global_shape = (8, 2, 18, 12)
    write_shape = (4, 1, 6, 3)
    shard_axes = (0, 2)
    for target_byte_size in (10, 25, 50):
      assert target_byte_size % dtype.itemsize != 0
      chosen_shape = subchunking.choose_chunk_shape(
          global_shape,
          write_shape,
          dtype,
          target_byte_size,
          shard_axes=shard_axes,
      )
      self.assertTrue(
          subchunking.validate_divisible_shapes(write_shape, chosen_shape)
      )
      self.assertLess(
          math.prod(chosen_shape) * dtype.itemsize, target_byte_size
      )
      # We also should have sharded at least once on both of the requested axes.
      for i in shard_axes:
        self.assertLess(chosen_shape[i], write_shape[i])

  def test_with_target_byte_size_not_a_divisor_of_the_total_size(self):
    dtype = np.dtype('float32')
    global_shape = (8, 10, 270, 2048, 6, 30)
    write_shape = (4, 10, 135, 1024, 6, 15)
    total_bytes = math.prod(write_shape) * dtype.itemsize
    shard_axes = (1, 4)
    for target_byte_size in (s * subchunking._MIB for s in (15, 25, 50, 80)):
      assert total_bytes % target_byte_size != 0
      chosen_shape = subchunking.choose_chunk_shape(
          global_shape,
          write_shape,
          dtype,
          target_byte_size,
          shard_axes=shard_axes,
      )
      self.assertTrue(
          subchunking.validate_divisible_shapes(write_shape, chosen_shape)
      )
      self.assertLess(
          math.prod(chosen_shape) * dtype.itemsize, target_byte_size
      )
      # We also should have sharded at least once on both of the requested axes.
      for i in shard_axes:
        self.assertLess(chosen_shape[i], write_shape[i])


if __name__ == '__main__':
  absltest.main()
