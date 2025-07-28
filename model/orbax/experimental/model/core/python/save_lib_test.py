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

"""Saving tests."""

from collections.abc import Sequence
import os
from typing import Tuple

from absl.testing import absltest
import jax
from jax import export as jax_export
import jax.numpy as jnp
from orbax.experimental.model.core.python import save_lib
from orbax.experimental.model.core.python.function import np_dtype_to_shlo_dtype
from orbax.experimental.model.core.python.function import ShloTensorSpec
from orbax.experimental.model.core.python.shlo_function import ShloFunction


save = save_lib.save


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


class SaveTest(absltest.TestCase):
  # TODO(wangpeng): We can move relevant parts of test from
  #   orbax/experimental/model/integration_tests/orbax_model_test.py
  #   here.
  def test_save(self):
    pass


if __name__ == '__main__':
  absltest.main()
