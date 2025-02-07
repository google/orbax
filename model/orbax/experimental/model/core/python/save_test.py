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

"""Saving tests."""

from collections.abc import Sequence
import os
from typing import Tuple

import jax
from jax import export as jax_export
import jax.numpy as jnp
import numpy as np
from orbax.experimental.model.core.protos.saved_model import types_pb2
from orbax.experimental.model.core.python import concrete_function
from orbax.experimental.model.core.python import module
from orbax.experimental.model.core.python import save as save_lib
from orbax.experimental.model.core.python import signature
from orbax.experimental.model.core.python.concrete_function import dtype_from_np_dtype
from orbax.experimental.model.core.python.function import np_dtype_to_shlo_dtype
from orbax.experimental.model.core.python.function import ShloTensorSpec
from orbax.experimental.model.core.python.shlo_function import ShloFunction
import tensorflow as tf

from absl.testing import absltest

save = save_lib.save
Tensor = concrete_function.Tensor
Function = concrete_function.ConcreteFunction
Variable = concrete_function.Variable
TensorSpec = signature.TensorSpec


def read_checkpoint_values(
    prefix: str,
) -> dict[str, tuple[np.ndarray, types_pb2.DataType]]:
  loaded = tf.train.load_checkpoint(prefix)
  contents = {}
  for key in loaded.get_variable_to_dtype_map().keys():
    contents[key] = loaded.get_tensor(key)
  return contents


def jax_spec_from_aval(x: jax.core.AbstractValue) -> jax.ShapeDtypeStruct:
  assert isinstance(x, jax.core.ShapedArray)
  return jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype)


def jax_spec_to_shlo_spec(
    jax_spec: Sequence[jax.ShapeDtypeStruct],
) -> Tuple[ShloTensorSpec, ...]:
  return tuple(
      ShloTensorSpec(shape=x.shape, dtype=np_dtype_to_shlo_dtype(x.dtype))
      for x in jax_spec
  )


def jax_spec_to_tensor_spec(x: jax.ShapeDtypeStruct) -> TensorSpec:
  return TensorSpec(shape=x.shape, dtype=dtype_from_np_dtype(x.dtype))


class SaveTest(googletest.TestCase):
  # TODO(qidichen): We can move relevant parts of test from orbax/experimental/model/integration_tests/orbax_model_test.py here.
  def test_save(self):
    pass

if __name__ == '__main__':
  googletest.main()
