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


if __name__ == '__main__':
  absltest.main()
