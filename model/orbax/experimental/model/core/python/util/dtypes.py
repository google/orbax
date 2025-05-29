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

"""Converts NumPy dtype to TensorFlow dtype."""

import numpy as np
from numpy import typing
from orbax.experimental.model.core.protos.saved_model import types_pb2

# Numpy to TF dtype map as defined in
# tensorflow/python/framework/dtypes.py

_NP_TO_TF: dict[typing.DTypeLike, types_pb2.DataType] = {
    np.float16: types_pb2.DT_HALF,
    np.float32: types_pb2.DT_FLOAT,
    np.float64: types_pb2.DT_DOUBLE,
    np.int32: types_pb2.DT_INT32,
    np.int64: types_pb2.DT_INT64,
    np.uint8: types_pb2.DT_UINT8,
    np.uint16: types_pb2.DT_UINT16,
    np.uint32: types_pb2.DT_UINT32,
    np.uint64: types_pb2.DT_UINT64,
    np.int16: types_pb2.DT_INT16,
    np.int8: types_pb2.DT_INT8,
    np.complex64: types_pb2.DT_COMPLEX64,
    np.complex128: types_pb2.DT_COMPLEX128,
    np.object_: types_pb2.DT_STRING,
    np.bytes_: types_pb2.DT_STRING,
    np.str_: types_pb2.DT_STRING,
    np.bool_: types_pb2.DT_BOOL,
}

# Map (some) NumPy platform dtypes to TF ones using their fixed-width
# synonyms. Note that platform dtypes are not always simples aliases,
# i.e. reference equality is not guaranteed. See e.g. numpy/numpy#9799.
for pdt in [
    np.intc,
    np.uintc,
    np.int_,
    np.uint,
    np.longlong,
    np.ulonglong,
]:
  if pdt not in _NP_TO_TF:
    _NP_TO_TF[pdt] = next(
        _NP_TO_TF[dt] for dt in _NP_TO_TF if dt == pdt().dtype
    )  # pylint: disable=no-value-for-parameter


def numpy_to_tf_dtype(dtype: typing.DTypeLike) -> types_pb2.DataType:
  try:
    return _NP_TO_TF[dtype.type]
  except KeyError:
    raise ValueError(  # pylint: disable=raise-missing-from
        f'Cannot convert dtype {dtype.type} to a TensorFlow dtype.'
    )


_TF_TO_NP: dict[types_pb2.DataType, typing.DTypeLike] = {
    v: k for k, v in _NP_TO_TF.items()
}


def tf_to_numpy_dtype(dtype: types_pb2.DataType) -> typing.DTypeLike:
  try:
    return _TF_TO_NP[dtype]
  except KeyError:
    raise ValueError(  # pylint: disable=raise-missing-from
        f'Cannot convert dtype {dtype} to a numpy dtype.'
    )
