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
import jax
import numpy as np
from orbax.checkpoint._src.arrays import abstract_arrays
from orbax.checkpoint._src.serialization import type_handlers

to_shape_dtype_struct = abstract_arrays.to_shape_dtype_struct


class AbstractArraysTest(parameterized.TestCase):

  @parameterized.parameters((None,), (np.float64,))
  def test_jax_array(self, target_dtype):
    target_dtype = target_dtype or np.float32
    arr = jax.device_put(
        np.arange(4, dtype=np.float32),
        jax.sharding.SingleDeviceSharding(jax.devices()[0]),
    )
    self.assertIsInstance(arr, jax.Array)
    sds = to_shape_dtype_struct(arr, dtype=target_dtype)
    self.assertIsInstance(sds, jax.ShapeDtypeStruct)
    self.assertEqual(sds.shape, (4,))
    self.assertEqual(sds.dtype, target_dtype)
    self.assertEqual(
        sds.sharding, jax.sharding.SingleDeviceSharding(jax.devices()[0])
    )

  @parameterized.parameters((None,), (np.float64,))
  def test_shape_dtype_struct(self, target_dtype):
    target_dtype = target_dtype or np.float32
    arr = jax.ShapeDtypeStruct(
        (4,),
        np.float32,
        sharding=jax.sharding.SingleDeviceSharding(jax.devices()[0]),
    )
    sds = to_shape_dtype_struct(arr, dtype=target_dtype)
    self.assertIsInstance(sds, jax.ShapeDtypeStruct)
    self.assertEqual(sds.shape, (4,))
    self.assertEqual(sds.dtype, target_dtype)
    self.assertEqual(
        sds.sharding, jax.sharding.SingleDeviceSharding(jax.devices()[0])
    )

  @parameterized.parameters((None,), (np.float64,))
  def test_restore_args(self, target_dtype):
    target_dtype = target_dtype or np.float32
    args = type_handlers.ArrayRestoreArgs(
        global_shape=(4,),
        dtype=np.float32,
        sharding=jax.sharding.SingleDeviceSharding(jax.devices()[0]),
    )
    sds = to_shape_dtype_struct(args, dtype=target_dtype)
    self.assertIsInstance(sds, jax.ShapeDtypeStruct)
    self.assertEqual(sds.shape, (4,))
    self.assertEqual(sds.dtype, target_dtype)
    self.assertEqual(
        sds.sharding, jax.sharding.SingleDeviceSharding(jax.devices()[0])
    )

  @parameterized.parameters((None,), (np.float64,))
  def test_numpy_array(self, target_dtype):
    target_dtype = target_dtype or np.float32
    arr = np.arange(4, dtype=np.float32)
    sds = to_shape_dtype_struct(arr, dtype=target_dtype)
    self.assertIsInstance(sds, jax.ShapeDtypeStruct)
    self.assertEqual(sds.shape, (4,))
    self.assertEqual(sds.dtype, target_dtype)
    self.assertIsNone(sds.sharding, None)

  @parameterized.product(
      value=(5, np.int32(5)),
      scalar_dtype=(None, int, np.int32, np.int64, float, np.float32),
  )
  def test_int(self, value, scalar_dtype):
    sds = to_shape_dtype_struct(value, scalar_dtype=scalar_dtype)
    target_dtype = scalar_dtype or type(value)
    self.assertEqual(type(sds), target_dtype)
    self.assertEqual(sds, target_dtype(value))

  @parameterized.product(
      value=(5.5, np.float32(5.5)),
      scalar_dtype=(None, float, np.float32, np.float64, int, np.int32),
  )
  def test_float(self, value, scalar_dtype):
    sds = to_shape_dtype_struct(value, scalar_dtype=scalar_dtype)
    target_dtype = scalar_dtype or type(value)
    self.assertEqual(type(sds), target_dtype)
    self.assertEqual(sds, target_dtype(value))


if __name__ == "__main__":
  absltest.main()
