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

"""TensorProto util tests."""

from absl.testing import parameterized
import numpy as np
from orbax.experimental.model.core.python import concrete_function
from orbax.experimental.model.core.python.saved_model_proto import tensor_proto

from absl.testing import absltest


class TensorProtoTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('string', np.array([['abc', 'defgh', 'i'], ['123', 'qwe', 'rty']])),
      (
          'int32',
          np.array([[[1], [2], [5]], [[53], [32], [41]]], dtype=np.int32),
      ),
      (
          'int64',
          np.array([[[1], [-2], [5]], [[-53], [32], [41]]], dtype=np.int64),
      ),
      (
          'float32',
          np.array([[[1.52], [5.5334]], [[53.2], [41]]], dtype=np.int64),
      ),
      (
          'uint32',
          np.array([[[1], [2], [5]], [[53], [32], [41]]], dtype=np.uint32),
      ),
      ('scalar', np.array(1234)),
  )
  def test_round_trip(self, ndarray):
    tensor = concrete_function.Tensor(ndarray)
    proto = tensor_proto.to_tensor_proto(tensor)
    back_tensor = tensor_proto.to_tensor(proto)
    self.assertEqual(tensor.spec.dtype, back_tensor.spec.dtype)
    self.assertEqual(tensor.spec.shape, back_tensor.spec.shape)
    np.testing.assert_array_equal(ndarray, back_tensor.np_array)

  def test_bytes_round_trip(self):
    ndarray = np.array([[b'abc', b'defgh', b'i'], [b'123', b'qwe', b'rty']])
    tensor = concrete_function.Tensor(ndarray)
    proto = tensor_proto.to_tensor_proto(tensor)
    back_tensor = tensor_proto.to_tensor(proto)
    self.assertEqual(tensor.spec.dtype, back_tensor.spec.dtype)
    self.assertEqual(tensor.spec.shape, back_tensor.spec.shape)

    # The expected gets converted to string.
    expected = np.array([['abc', 'defgh', 'i'], ['123', 'qwe', 'rty']])
    np.testing.assert_array_equal(expected, back_tensor.np_array)

  def test_string_list(self):
    string_list = ['abc', 'defgh', 'i', '123', 'qwe', 'rty']
    proto = tensor_proto.string_list_to_proto(string_list)
    back_tensor = tensor_proto.to_tensor(proto)
    for n, s in enumerate(string_list):
      self.assertEqual(s, back_tensor.np_array[n])

if __name__ == '__main__':
  absltest.main()
