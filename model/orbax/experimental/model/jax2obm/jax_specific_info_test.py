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

from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from orbax.experimental.model import core as obm
from orbax.experimental.model.jax2obm import jax_specific_info
from orbax.experimental.model.jax2obm import jax_supplemental_pb2
from tensorflow.python.util.protobuf import compare
from google.protobuf import text_format
from absl.testing import absltest


class JaxSpecificInfoTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (f'_{idx}', jax_shape, expected_shlo_shape, expected_refinements)
      for idx, (
          jax_shape,
          expected_shlo_shape,
          expected_refinements,
      ) in enumerate([
          (
              (2, None, '3', 'a'),
              [2, None, 3, None],
              [None, None, None, 'a'],
          ),
          (
              (2, None, '3', '*'),
              [2, None, 3, None],
              [None, None, None, '*'],
          ),
          (
              (2, None, '3', '...'),
              [2, None, 3, None],
              [None, None, None, '...'],
          ),
          (
              [2, None, 3, '4'],
              [2, None, 3, 4],
              None,
          ),
          (
              (),
              [],
              None,
          ),
          (
              [],
              [],
              None,
          ),
      ])
  )
  def test_to_shlo_shape_and_refinement(
      self, jax_shape, expected_shlo_shape, expected_refinements
  ):
    if expected_refinements is not None:
      expected_refinements = jax_supplemental_pb2.ShapeRefinement(
          dimension_sizes=[
              jax_supplemental_pb2.DimensionSizeRefinement(size=r)
              for r in expected_refinements
          ]
      )
    self.assertEqual(
        jax_specific_info._to_shlo_shape_and_refinement(jax_shape),
        (expected_shlo_shape, expected_refinements),
    )

  def test_to_shape_dtype_refinements_proto(self):
    shape = jax_supplemental_pb2.ShapeRefinement(dimension_sizes=[])
    dtype = jax_supplemental_pb2.DTypeRefinement.f0
    input1 = [(None, None), (shape, None), (None, dtype), (shape, dtype)]
    expected_text1 = """
      list {
        refinements {
        }
        refinements {
          shape {
          }
        }
        refinements {
          dtype: f0
        }
        refinements {
          shape {
          }
          dtype: f0
        }
      }
    """
    compare.assertProtoEqual(
        self,
        jax_specific_info._to_shape_dtype_refinements_proto(input1),
        text_format.Parse(
            expected_text1, jax_supplemental_pb2.ShapeDTypeRefinements()
        ),
    )
    input2 = input1 + [(None, None)] * 4
    expected_text2 = """
      map {
        idx_to_refinement {
          key: 1
          value {
            shape {
            }
          }
        }
        idx_to_refinement {
          key: 2
          value {
            dtype: f0
          }
        }
        idx_to_refinement {
          key: 3
          value {
            shape {
            }
            dtype: f0
          }
        }
      }
    """
    compare.assertProtoEqual(
        self,
        jax_specific_info._to_shape_dtype_refinements_proto(input2),
        text_format.Parse(
            expected_text2, jax_supplemental_pb2.ShapeDTypeRefinements()
        ),
    )

  def test_to_shlo_dtype_and_refinement(self):
    jax_dtype = np.dtype(jnp.int32)
    self.assertEqual(
        jax_specific_info._to_shlo_dtype_and_refinement(jax_dtype),
        (obm.ShloDType.i32, None),
    )

  def test_to_shlo_dtype_and_refinement_f0(self):
    jax_dtype = jax.dtypes.float0
    self.assertEqual(
        jax_specific_info._to_shlo_dtype_and_refinement(jax_dtype),
        (obm.ShloDType.bool, jax_supplemental_pb2.DTypeRefinement.f0),
    )

  @parameterized.named_parameters(
      ('_str', 'int32'),
      # np.int32 is not allowed. It needs to be np.dtype(np.int32).
      ('_np_int32', np.int32),
      # Similarly for jnp.int32. It needs to be np.dtype(jnp.int32).
      ('_jnp_int32', jnp.int32),
  )
  def test_to_shlo_dtype_and_refinement_wrong_type(self, jax_dtype):
    with self.assertRaisesRegex(
        TypeError, r'jax_dtype must be an instance of np\.dtype, but'
    ):
      jax_specific_info._to_shlo_dtype_and_refinement(jax_dtype)


if __name__ == '__main__':
  absltest.main()
