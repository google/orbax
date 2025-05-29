# Copyright 2025 The Orbax Authors.
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

"""Utilities for converting values to/from TensorProto."""

import numpy as np
from orbax.experimental.model.core.protos.saved_model import tensor_pb2
from orbax.experimental.model.core.protos.saved_model import tensor_shape_pb2
from orbax.experimental.model.core.protos.saved_model import types_pb2
from orbax.experimental.model.core.python import concrete_function
from orbax.experimental.model.core.python.concrete_function import Shape
from orbax.experimental.model.core.python.concrete_function import Tensor
from orbax.experimental.model.core.python.util import compat
from orbax.experimental.model.core.python.util import dtypes


# Code forked from tensorflow/python/tensor_util.py

_TENSOR_CONTENT_TYPES = frozenset([
    types_pb2.DT_HALF,
    types_pb2.DT_FLOAT,
    types_pb2.DT_DOUBLE,
    types_pb2.DT_INT32,
    types_pb2.DT_UINT8,
    types_pb2.DT_INT16,
    types_pb2.DT_INT8,
    types_pb2.DT_INT64,
    types_pb2.DT_QINT8,
    types_pb2.DT_QUINT8,
    types_pb2.DT_QINT16,
    types_pb2.DT_QUINT16,
    types_pb2.DT_QINT32,
    types_pb2.DT_UINT32,
    types_pb2.DT_UINT64,
    types_pb2.DT_FLOAT8_E5M2,
    types_pb2.DT_FLOAT8_E4M3FN,
    types_pb2.DT_BFLOAT16,
    # int4/uint4 intentionally not listed, since their binary representation
    # is implementation-dependent.
])


def string_list_to_proto(
    string_list, shape: Shape = None
) -> tensor_pb2.TensorProto:
  if not shape:
    shape = [len(string_list)]
  tensor_proto = tensor_pb2.TensorProto(
      dtype=types_pb2.DT_STRING, tensor_shape=shape_to_proto(shape)
  )

  for s in string_list:
    tensor_proto.string_val.append(compat.as_bytes(s))
  return tensor_proto


def to_tensor_proto(tensor: Tensor) -> tensor_pb2.TensorProto:
  """Converts a Tensor to a TensorProto.

  Args:
    tensor: A `Tensor`.

  Returns:
    A `TensorProto` with the contents of `Tensor`.
  """
  dtype = tensor.spec.dtype
  shape = tensor.spec.shape

  tensor_proto = tensor_pb2.TensorProto(
      dtype=dtype, tensor_shape=shape_to_proto(shape)
  )

  nparray = tensor.np_array
  shape_size = nparray.size

  if dtype in _TENSOR_CONTENT_TYPES and shape_size > 1:
    if nparray.size * nparray.itemsize >= (1 << 31):
      raise ValueError(
          "Cannot create a tensor proto whose content is larger than 2GB."
      )
    tensor_proto.tensor_content = nparray.tobytes()
    return tensor_proto

  # TensorFlow expects C order (a.k.a., eigen row major).
  proto_values = nparray.ravel()

  append_fn = _get_numpy_append_fn(proto_values.dtype)
  if append_fn is None:
    raise TypeError(f"Element type not supported in TensorProto: {dtype}.")
  append_fn(tensor_proto, proto_values)

  return tensor_proto


def to_tensor(proto: tensor_pb2.TensorProto) -> Tensor:
  """Create a Tensor from a TensorProto.

  Create a numpy ndarray with the same shape and data as the TensorProto, and
  returns as a Tensor.

  For example:

  ```python
  # Tensor a has shape (2,3)
  a = tf.constant([[1,2,3],[4,5,6]])
  proto_tensor = tf.make_tensor_proto(a)  # convert `tensor a` to a proto tensor
  tf.make_ndarray(proto_tensor) # output: array([[1, 2, 3],
  #                                              [4, 5, 6]], dtype=int32)
  # output has shape (2,3)
  ```

  Args:
    proto: A TensorProto.

  Returns:
    A `Tensor` with the contents from the proto.

  Raises:
    TypeError: if tensor has unsupported type.
  """
  shape = [d.size for d in proto.tensor_shape.dim]
  num_elements = np.prod(shape, dtype=np.int64)
  tensor_dtype = proto.dtype
  dtype = dtypes.tf_to_numpy_dtype(tensor_dtype)

  if proto.tensor_content:
    return concrete_function.Tensor(
        np.frombuffer(proto.tensor_content, dtype=dtype).copy().reshape(shape)
    )

  if tensor_dtype == types_pb2.DT_STRING:
    # np.pad throws on these arrays of type np.object_.
    values = list(proto.string_val)
    padding = num_elements - len(values)
    if padding > 0:
      last = values[-1] if values else ""
      values.extend([last] * padding)
    return concrete_function.Tensor(
        np.array(values, dtype=dtype).reshape(shape)
    )

  if tensor_dtype == types_pb2.DT_HALF or tensor_dtype == types_pb2.DT_BFLOAT16:
    # the half_val field of the TensorProto stores the binary representation
    # of the fp16: we need to reinterpret this as a proper float16
    values = np.fromiter(proto.half_val, dtype=np.uint16)
    values.dtype = dtype
  elif tensor_dtype in [
      types_pb2.DT_FLOAT8_E5M2,
      types_pb2.DT_FLOAT8_E4M3FN,
  ]:
    values = np.fromiter(proto.float8_val, dtype=np.uint8)
    values.dtype = dtype
  elif tensor_dtype == types_pb2.DT_FLOAT:
    values = np.fromiter(proto.float_val, dtype=dtype)
  elif tensor_dtype == types_pb2.DT_DOUBLE:
    values = np.fromiter(proto.double_val, dtype=dtype)
  elif tensor_dtype in [
      types_pb2.DT_INT32,
      types_pb2.DT_UINT8,
      types_pb2.DT_UINT16,
      types_pb2.DT_INT16,
      types_pb2.DT_INT8,
      types_pb2.DT_QINT32,
      types_pb2.DT_QUINT8,
      types_pb2.DT_QINT8,
      types_pb2.DT_QINT16,
      types_pb2.DT_QUINT16,
      types_pb2.DT_INT4,
      types_pb2.DT_UINT4,
  ]:
    values = np.fromiter(proto.int_val, dtype=dtype)
  elif tensor_dtype == types_pb2.DT_INT64:
    values = np.fromiter(proto.int64_val, dtype=dtype)
  elif tensor_dtype == types_pb2.DT_UINT32:
    values = np.fromiter(proto.uint32_val, dtype=dtype)
  elif tensor_dtype == types_pb2.DT_UINT64:
    values = np.fromiter(proto.uint64_val, dtype=dtype)
  elif tensor_dtype == types_pb2.DT_COMPLEX64:
    it = iter(proto.scomplex_val)
    values = np.array([complex(x[0], x[1]) for x in zip(it, it)], dtype=dtype)
  elif tensor_dtype == types_pb2.DT_COMPLEX128:
    it = iter(proto.dcomplex_val)
    values = np.array([complex(x[0], x[1]) for x in zip(it, it)], dtype=dtype)
  elif tensor_dtype == types_pb2.DT_BOOL:
    values = np.fromiter(proto.bool_val, dtype=dtype)
  else:
    raise TypeError(
        f"Unsupported tensor type: {proto.dtype}. See "
        "https://www.tensorflow.org/api_docs/python/tf/dtypes "
        "for supported TF dtypes."
    )

  if values.size == 0:
    return concrete_function.Tensor(np.zeros(shape, dtype))

  if values.size != num_elements:
    values = np.pad(values, (0, num_elements - values.size), "edge")

  return concrete_function.Tensor(values.reshape(shape))


def shape_to_proto(shape: Shape) -> tensor_shape_pb2.TensorShapeProto:
  shape_proto = tensor_shape_pb2.TensorShapeProto()
  if shape is None:
    shape_proto.unknown_rank = True
    return shape_proto
  for dim in shape:
    if dim is None:
      shape_proto.dim.add(size=-1)
    else:
      shape_proto.dim.add(size=dim)
  return shape_proto


def _get_numpy_append_fn(dtype):
  # numpy dtype for strings are variable length. We can not compare
  # dtype with a single constant (np.string does not exist) to decide
  # dtype is a "string" type. We need to compare the dtype.type to be
  # sure it's a string type.
  if dtype.type == np.bytes_ or dtype.type == np.str_:
    return _slow_append_object_array_to_tensor_proto
  return _get_from_numpy_dtype_dict(_NP_TO_APPEND_FN, dtype)


def _extract_bits_from_float(x):
  return np.asarray(x, dtype=np.float16).view(np.uint16).item()


def _slow_append_float16_array_to_tensor_proto(tensor_proto, proto_values):
  tensor_proto.half_val.extend(
      [_extract_bits_from_float(x) for x in proto_values]
  )


def _slow_append_float32_array_to_tensor_proto(tensor_proto, proto_values):
  tensor_proto.float_val.extend([x.item() for x in proto_values])


def _slow_append_float64_array_to_tensor_proto(tensor_proto, proto_values):
  tensor_proto.double_val.extend([x.item() for x in proto_values])


def _slow_append_int_array_to_tensor_proto(tensor_proto, proto_values):
  tensor_proto.int_val.extend([x.item() for x in proto_values])


def _slow_append_int64_array_to_tensor_proto(tensor_proto, proto_values):
  tensor_proto.int64_val.extend([x.item() for x in proto_values])


def _slow_append_uint32_array_to_tensor_proto(tensor_proto, proto_values):
  tensor_proto.uint32_val.extend([x.item() for x in proto_values])


def _slow_append_uint64_array_to_tensor_proto(tensor_proto, proto_values):
  tensor_proto.uint64_val.extend([x.item() for x in proto_values])


# pylint: disable=g-complex-comprehension
def _slow_append_complex64_array_to_tensor_proto(tensor_proto, proto_values):
  tensor_proto.scomplex_val.extend(
      [v.item() for x in proto_values for v in [x.real, x.imag]]
  )


def _slow_append_complex128_array_to_tensor_proto(tensor_proto, proto_values):
  tensor_proto.dcomplex_val.extend(
      [v.item() for x in proto_values for v in [x.real, x.imag]]
  )


# pylint: enable=g-complex-comprehension


def _slow_append_object_array_to_tensor_proto(tensor_proto, proto_values):
  tensor_proto.string_val.extend([compat.as_bytes(x) for x in proto_values])


def _slow_append_bool_array_to_tensor_proto(tensor_proto, proto_values):
  tensor_proto.bool_val.extend([x.item() for x in proto_values])


_NP_TO_APPEND_FN = {
    np.float16: _slow_append_float16_array_to_tensor_proto,
    np.float32: _slow_append_float32_array_to_tensor_proto,
    np.float64: _slow_append_float64_array_to_tensor_proto,
    np.int32: _slow_append_int_array_to_tensor_proto,
    np.int64: _slow_append_int64_array_to_tensor_proto,
    np.uint8: _slow_append_int_array_to_tensor_proto,
    np.uint16: _slow_append_int_array_to_tensor_proto,
    np.uint32: _slow_append_uint32_array_to_tensor_proto,
    np.uint64: _slow_append_uint64_array_to_tensor_proto,
    np.int8: _slow_append_int_array_to_tensor_proto,
    np.int16: _slow_append_int_array_to_tensor_proto,
    np.complex64: _slow_append_complex64_array_to_tensor_proto,
    np.complex128: _slow_append_complex128_array_to_tensor_proto,
    np.object_: _slow_append_object_array_to_tensor_proto,
    np.bool_: _slow_append_bool_array_to_tensor_proto,
}


def _get_from_numpy_dtype_dict(dtype_dict, dtype):
  # NOTE: dtype_dict.get(dtype) always returns None.
  for key, val in dtype_dict.items():
    if key == dtype:
      return val
  return None
