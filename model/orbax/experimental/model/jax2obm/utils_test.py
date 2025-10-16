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

"""Tests for utils."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from orbax.experimental.model import core as obm
from orbax.experimental.model.jax2obm import utils


class UtilsTest(parameterized.TestCase):

  @parameterized.parameters(
      (jax.float0, obm.ShloDType.bool),
      (jax.numpy.bfloat16, obm.ShloDType.bf16),
      (jax.numpy.bool_, obm.ShloDType.bool),
      (jax.numpy.int4, obm.ShloDType.i4),
      (jax.numpy.int8, obm.ShloDType.i8),
      (jax.numpy.int16, obm.ShloDType.i16),
      (jax.numpy.int32, obm.ShloDType.i32),
      (jax.numpy.int64, obm.ShloDType.i64),
      (jax.numpy.uint4, obm.ShloDType.ui4),
      (jax.numpy.uint8, obm.ShloDType.ui8),
      (jax.numpy.uint16, obm.ShloDType.ui16),
      (jax.numpy.uint32, obm.ShloDType.ui32),
      (jax.numpy.uint64, obm.ShloDType.ui64),
      (jax.numpy.float16, obm.ShloDType.f16),
      (jax.numpy.float32, obm.ShloDType.f32),
      (jax.numpy.float64, obm.ShloDType.f64),
      (jax.numpy.complex64, obm.ShloDType.c64),
      (jax.numpy.complex128, obm.ShloDType.c128),
  )
  def test_get_physical_dtype_returns_expected_shlo_dtype(
      self, jax_dtype, expected_shlo_dtype
  ):
    self.assertEqual(utils._get_physical_dtype(jax_dtype), expected_shlo_dtype)


if __name__ == '__main__':
  absltest.main()
