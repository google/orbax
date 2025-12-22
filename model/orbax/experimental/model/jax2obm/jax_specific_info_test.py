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

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from orbax.experimental.model import core as obm
from orbax.experimental.model.jax2obm import jax_specific_info
from orbax.experimental.model.jax2obm import jax_supplemental_pb2
from tensorflow.python.util.protobuf import compare
from google.protobuf import text_format


def _get_spec():
  """Returns a dummy spec, creates a new instance each time."""
  return obm.ShloTensorSpec(shape=(1,), dtype=obm.ShloDType.f32)


class _CustomNode:

  def __init__(self, x, y):
    self.x = x
    self.y = y


jax.tree_util.register_pytree_node(
    _CustomNode,
    lambda node: ((node.x, node.y), None),
    lambda _, children: _CustomNode(*children),
)


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

  @parameterized.named_parameters(
      dict(
          testcase_name='dict_and_list',
          tree={'a': [_get_spec(), _get_spec()], 'b': {'c': _get_spec()}},
          expected_names=['a.0', 'a.1', 'b.c'],
      ),
      dict(
          testcase_name='tuple_and_nested_dict',
          tree=(_get_spec(), {'x': _get_spec()}),
          expected_names=['0', '1.x'],
      ),
      dict(
          testcase_name='list_of_tuples',
          tree=[(_get_spec(),), (_get_spec(), _get_spec())],
          expected_names=['0.0', '1.0', '1.1'],
      ),
      dict(
          testcase_name='custom_node',
          tree={'node': _CustomNode(_get_spec(), _get_spec())},
          expected_names=['node.0', 'node.1'],
      ),
      dict(
          testcase_name='single_spec',
          tree=_get_spec(),
          expected_names=[''],
      ),
      dict(
          testcase_name='list_with_one_spec',
          tree=[_get_spec()],
          expected_names=['0'],
      ),
  )
  def test_name_leaf(self, tree, expected_names):
    def _name_leaf_wrapper(tree):
      return jax.tree_util.tree_map_with_path(
          jax_specific_info._name_leaf, tree
      )

    named_tree = _name_leaf_wrapper(tree)
    leaves, treedef = jax.tree_util.tree_flatten(named_tree)
    with self.subTest('check_treedef'):
      self.assertEqual(treedef, jax.tree_util.tree_structure(tree))
    with self.subTest('check_leaf_count'):
      self.assertLen(leaves, len(expected_names))
    with self.subTest('check_leaves_names'):
      self.assertEqual([leaf.name for leaf in leaves], expected_names)

  def test_to_shlo_spec_tree_and_refinement_tuple_with_name_leaves(self):
    avals_tree = {
        'a': jax.core.ShapedArray((1,), jnp.float32),
        'b': [jax.core.ShapedArray((2,), jnp.int32)],
    }
    avals, tree_def = jax.tree_util.tree_flatten(avals_tree)
    shardings = [None] * len(avals)

    jax_tree, refinements = (
        jax_specific_info._to_shlo_spec_tree_and_refinement_tuple(
            avals, shardings, tree_def, name_leaves=True
        )
    )
    with self.subTest('check_refinements'):
      self.assertIsNone(refinements)
    with self.subTest('check_jax_tree_leaves_names'):
      self.assertIsInstance(jax_tree, dict)
      self.assertCountEqual(jax_tree.keys(), ['a', 'b'])
      self.assertEqual(jax_tree['a'].name, 'a')
      self.assertIsInstance(jax_tree['b'], list)
      self.assertEqual(jax_tree['b'][0].name, 'b.0')


if __name__ == '__main__':
  absltest.main()
