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
from etils import epath
import numpy as np
from orbax.checkpoint._src.arrays import subchunking


class ChooseChunkShapeTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.ckpt_dir = epath.Path(self.create_tempdir('ckpt').full_path)

  def test_choose_chunk_shape_equal_global_shape(self):
    shape = (10, 100, 200)
    dtype = np.dtype('float32')

    # allow only 1 element
    chosen_shape = subchunking.choose_chunk_shape(
        global_shape=shape,
        write_shape=shape,
        dtype=dtype,
        target_byte_size=dtype.itemsize,
    )
    np.testing.assert_array_equal(chosen_shape, (1, 1, 1))

    # allow 3 elements
    chosen_shape = subchunking.choose_chunk_shape(
        global_shape=shape,
        write_shape=shape,
        dtype=dtype,
        target_byte_size=5**3 * dtype.itemsize,
    )
    np.testing.assert_array_equal(chosen_shape, (5, 5, 5))

    # allow 4 elements
    chosen_shape = subchunking.choose_chunk_shape(
        global_shape=shape,
        write_shape=shape,
        dtype=dtype,
        target_byte_size=5**4 * dtype.itemsize,
    )
    np.testing.assert_array_equal(chosen_shape, (5, 10, 10))

    # not divisble target_byte_size should still result a correct shape
    chosen_shape = subchunking.choose_chunk_shape(
        global_shape=shape,
        write_shape=shape,
        dtype=dtype,
        target_byte_size=5**4 * dtype.itemsize + 3,
    )
    np.testing.assert_array_equal(chosen_shape, (5, 10, 10))

  def test_choose_chunk_shape_for_sharded_array(self):
    local_shape = (10, 100, 200)
    dtype = np.dtype('float32')

    # allow to split on at the sharded axis
    chosen_shape = subchunking.choose_chunk_shape(
        global_shape=(10, 500, 200),
        write_shape=local_shape,
        dtype=dtype,
        target_byte_size=10 * 5 * 200 * dtype.itemsize,
    )
    np.testing.assert_array_equal(chosen_shape, (10, 5, 200))

    # forced to split on unsharded axis when the target_byte_size is small
    chosen_shape = subchunking.choose_chunk_shape(
        global_shape=(10, 500, 200),
        write_shape=local_shape,
        dtype=dtype,
        target_byte_size=10 * 1 * 100 * dtype.itemsize,
    )
    np.testing.assert_array_equal(chosen_shape, (10, 1, 100))
