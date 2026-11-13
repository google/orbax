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

import datetime
import json
from typing import cast
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from orbax.checkpoint._src.serialization import cloud_pathways_helper


class CloudPathwaysHelperTest(parameterized.TestCase):

  location = "/path/to/location"
  name = "name"
  dtype = np.dtype(np.int32)
  shape = [8, 4]
  timeout = datetime.timedelta(seconds=3)

  def setUp(self):
    super().setUp()
    self.devices = jax.devices()
    self.mesh = jax.sharding.Mesh(self.devices, "batch")
    self.sharding = jax.sharding.NamedSharding(
        self.mesh, jax.sharding.PartitionSpec("batch")
    )

  @parameterized.named_parameters(
      ("bool", "bool", "PRED"),
      ("bfloat16", "bfloat16", "BF16"),
      ("float16", "float16", "F16"),
      ("float32", "float32", "F32"),
      ("float64", "float64", "F64"),
      ("int8", "int8", "S8"),
      ("int16", "int16", "S16"),
      ("int32", "int32", "S32"),
      ("int64", "int64", "S64"),
      ("uint8", "uint8", "U8"),
      ("uint16", "uint16", "U16"),
      ("uint32", "uint32", "U32"),
      ("uint64", "uint64", "U64"),
  )
  def test_dtype_to_xla_primitive_type_str(
      self, dtype_name: str, expected: str
  ):
    self.assertEqual(
        cloud_pathways_helper.dtype_to_xla_primitive_type_str(
            np.dtype(dtype_name)
        ),
        expected,
    )

  def test_dtype_to_xla_primitive_type_str_unsupported_raises_value_error(self):
    with self.assertRaisesRegex(ValueError, "Unsupported dtype: complex64"):
      cloud_pathways_helper.dtype_to_xla_primitive_type_str(
          np.dtype("complex64")
      )

  @parameterized.named_parameters(
      ("bool", "bool", "PRED"),
      ("float16", "float16", "F16"),
      ("int32", "int32", "S32"),
  )
  def test_get_read_request(self, dtype_name: str, expected_primitive: str):
    read_request = cloud_pathways_helper.get_read_request(
        location_path=self.location,
        name=self.name,
        dtype=np.dtype(dtype_name),
        shape=self.shape,
        sharding=self.sharding,
        devices=self.devices,
        timeout=self.timeout,
    )
    self.assertIsInstance(read_request, str)
    parsed = json.loads(read_request)
    self.assertEqual(
        parsed["persistenceReadRequest"]["shape"],
        {
            "xla_primitive_type_str": expected_primitive,
            "dimensions": self.shape,
        },
    )

  def test_get_bulk_read_request(self):
    bulk_read_request = cloud_pathways_helper.get_bulk_read_request(
        location_path=self.location,
        names=[self.name, "flag"],
        dtypes=[self.dtype, np.dtype("bool")],
        shapes=[self.shape, [4]],
        shardings=[self.sharding, self.sharding],
        devices=self.devices,
        timeout=self.timeout,
    )
    parsed = json.loads(bulk_read_request)
    read_requests = parsed["bulk_persistence_read_request"]["read_requests"]
    self.assertLen(read_requests, 2)
    self.assertEqual(
        read_requests[0]["shape"]["xla_primitive_type_str"], "S32"
    )
    self.assertEqual(
        read_requests[1]["shape"]["xla_primitive_type_str"], "PRED"
    )

  def test_get_write_request(self):
    jax_array = jax.device_put(
        np.arange(np.prod(self.shape), dtype=self.dtype).reshape(self.shape),
        self.sharding,
    )
    write_request = cloud_pathways_helper.get_write_request(
        location_path=self.location,
        name=self.name,
        jax_array=jax_array,
        timeout=self.timeout,
    )
    self.assertIsInstance(write_request, str)
    parsed = json.loads(write_request)
    self.assertEqual(parsed["persistenceWriteRequest"]["shape"], self.shape)

  def test_get_bulk_write_request(self):
    jax_array = jax.device_put(
        np.arange(np.prod(self.shape), dtype=self.dtype).reshape(self.shape),
        self.sharding,
    )
    bulk_write_request = cloud_pathways_helper.get_bulk_write_request(
        location_path=self.location,
        names=[self.name, self.name],
        jax_arrays=[jax_array, jax_array],
        timeout=self.timeout,
    )
    parsed = json.loads(bulk_write_request)
    self.assertLen(
        parsed["bulk_persistence_write_request"]["write_requests"], 2
    )

  def test_get_write_request_invalid_sharding_raises_value_error(self):
    fake_array = mock.MagicMock(spec=jax.Array)
    fake_array.sharding = "not_a_sharding"
    with self.assertRaisesRegex(ValueError, "Expected jax.sharding.Sharding"):
      cloud_pathways_helper.get_write_request(
          location_path=self.location,
          name=self.name,
          jax_array=cast(jax.Array, fake_array),
          timeout=self.timeout,
      )


if __name__ == "__main__":
  absltest.main()
