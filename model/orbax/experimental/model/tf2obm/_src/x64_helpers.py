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

"""Helpers for suppressing 64-bit outputs from TF functions."""

from collections.abc import Callable
import functools
from typing import Any

import tensorflow as tf


def _is_x64(dtype: tf.DType) -> bool:
  return dtype in (tf.float64, tf.complex128, tf.int64, tf.uint64)


_X64_TO_X32 = {
    tf.float64: tf.float32,
    tf.complex128: tf.complex64,
    tf.int64: tf.int32,
    tf.uint64: tf.uint32,
}


def _suppress_x64(x: Any) -> Any:
  if not isinstance(x, tf.Tensor):
    return x
  new_dtype = _X64_TO_X32.get(x.dtype, None)
  if new_dtype is None:
    return x
  return tf.cast(x, new_dtype)


def has_x64_outputs(cf: tf.types.experimental.ConcreteFunction) -> bool:
  return any(
      isinstance(x, (tf.Tensor, tf.TensorSpec)) and _is_x64(x.dtype)
      for x in tf.nest.flatten(cf.structured_outputs)
  )


def suppress_x64_outputs(
    fn: Callable[..., Any],
) -> Callable[..., Any]:

  @functools.wraps(fn)
  def new_fn(*args_, **kwargs_):
    output = fn(*args_, **kwargs_)
    return tf.nest.map_structure(_suppress_x64, output)

  return new_fn
