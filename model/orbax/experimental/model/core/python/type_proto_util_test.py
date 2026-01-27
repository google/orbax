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

import textwrap

from absl.testing import absltest
from absl.testing import parameterized
from google.protobuf import text_format
from orbax.experimental.model.core.protos import type_pb2
from orbax.experimental.model.core.python import test_utils
from orbax.experimental.model.core.python import type_proto_util
from orbax.experimental.model.core.python.function import Sharding
from orbax.experimental.model.core.python.function import ShloDType
from orbax.experimental.model.core.python.function import ShloTensorSpec
from orbax.experimental.model.core.python.tree_util import Tree

from tensorflow.compiler.xla import xla_data_pb2  # pylint: disable=g-direct-tensorflow-import
# TODO(wangpeng): Replace all "manifest" with "type_proto" in this file.


class RoundtripBetweenShloShapeAndManifestShapeTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (f"_{idx}", shlo_shape, manifest_shape_proto_text)
      for idx, (shlo_shape, manifest_shape_proto_text) in enumerate([
          (
              [1, 2, 3],
              """
              shape_with_known_rank {
                dimension_sizes {
                  size: 1
                }
                dimension_sizes {
                  size: 2
                }
                dimension_sizes {
                  size: 3
                }
              }
              """,
          ),
          (
              [],
              """
              shape_with_known_rank {}
              """,
          ),
          (
              (),
              """
              shape_with_known_rank {}
              """,
          ),
          (
              None,
              "",
          ),
      ])
  )
  def test_roundtrip(
      self, shlo_shape, manifest_shape_proto_text
  ):
    # One way conversion.
    result_manifest_shape = type_proto_util.shlo_shape_to_manifest_shape(
        shlo_shape
    )
    expected_manifest_shape = text_format.Parse(
        manifest_shape_proto_text, type_pb2.Shape()
    )
    self.assertEqual(result_manifest_shape, expected_manifest_shape)

    # The other way conversion.
    result_shlo_shape = type_proto_util.manifest_shape_to_shlo_shape(
        expected_manifest_shape
    )
    if shlo_shape is None:
      self.assertIsNone(result_shlo_shape)
    else:
      self.assertEqual(result_shlo_shape, tuple(shlo_shape))


class ShloTensorSpecToManifestTensorTypeTest(
    test_utils.ObmTestCase, parameterized.TestCase
):

  def test_converts_spec(self):
    sharding_str = textwrap.dedent("""
      type: OTHER
      tile_assignment_dimensions: [2, 2]
      iota_reshape_dims: [4]
      iota_transpose_perm: [0]
    """)
    sharding = text_format.Parse(sharding_str, xla_data_pb2.OpSharding())
    layout_str = textwrap.dedent("""
      minor_to_major: [1, 0]
    """)
    layout = text_format.Parse(layout_str, xla_data_pb2.LayoutProto())
    spec = ShloTensorSpec(
        shape=(4, 2), dtype=ShloDType.f32, sharding=sharding, layout=layout
    )

    tensor_type_proto = (
        type_proto_util.shlo_tensor_spec_to_manifest_tensor_type(spec)
    )

    expected_proto = text_format.Parse(
        textwrap.dedent("""
        shape: {
          shape_with_known_rank: {
            dimension_sizes: { size: 4 }
            dimension_sizes: { size: 2 }
          }
        }
        dtype: f32
        """),
        type_pb2.TensorType(),
    )
    expected_proto.sharding.CopyFrom(sharding)
    expected_proto.layout.CopyFrom(layout)
    self.assertEqual(tensor_type_proto, expected_proto)

  def test_converts_spec_without_sharding_and_layout(self):
    spec = ShloTensorSpec(
        shape=(4, 2), dtype=ShloDType.f32, sharding=None, layout=None
    )

    tensor_type_proto = (
        type_proto_util.shlo_tensor_spec_to_manifest_tensor_type(spec)
    )

    expected_proto = text_format.Parse(
        textwrap.dedent("""
        shape: {
          shape_with_known_rank: {
            dimension_sizes: {
              size: 4
            }
            dimension_sizes: {
              size: 2
            }
          }
        }
        dtype: f32
        """),
        type_pb2.TensorType(),
    )
    self.assertEqual(tensor_type_proto, expected_proto)


class ManifestTypeToShloTensorSpecTreeTest(test_utils.ObmTestCase):

  def test_manifest_type_to_shlo_tensor_spec_tree(self):
    manifest_type_proto_str = textwrap.dedent("""
      tuple {
        elements {
          tuple {
            elements {
              dict {
                string_to_type {
                  key: "a"
                  value {
                    leaf {
                      tensor_type {
                        shape {
                          shape_with_known_rank {
                            dimension_sizes {
                              size: 4
                            }
                            dimension_sizes {
                              size: 4
                            }
                          }
                        }
                        dtype: si4
                        sharding {
                          type: OTHER
                          tile_assignment_dimensions: 4
                          tile_assignment_dimensions: 1
                          tile_assignment_dimensions: 2
                          replicate_on_last_tile_dim: true
                          iota_reshape_dims: 8
                          iota_transpose_perm: 0
                        }
                      }
                    }
                  }
                }
                string_to_type {
                  key: "b"
                  value {
                    leaf {
                      tensor_type {
                        shape {
                          shape_with_known_rank {
                            dimension_sizes {
                              size: 4
                            }
                            dimension_sizes {
                              size: 16
                            }
                          }
                        }
                        dtype: f32
                        sharding {
                          type: OTHER
                          tile_assignment_dimensions: 4
                          tile_assignment_dimensions: 2
                          iota_reshape_dims: 8
                          iota_transpose_perm: 0
                        }
                      }
                    }
                  }
                }
              }
            }
            elements {
              leaf {
                tensor_type {
                  shape {
                    shape_with_known_rank {
                      dimension_sizes {
                        size: 8
                      }
                      dimension_sizes {
                        size: 4
                      }
                    }
                  }
                  dtype: f32
                  sharding {
                    type: OTHER
                    tile_assignment_dimensions: 4
                    tile_assignment_dimensions: 1
                    tile_assignment_dimensions: 2
                    replicate_on_last_tile_dim: true
                    iota_reshape_dims: 8
                    iota_transpose_perm: 0
                  }
                }
              }
            }
          }
        }
        elements {
          dict {
          }
        }
      }
    """)

    manifest_type = text_format.Parse(manifest_type_proto_str, type_pb2.Type())

    shlo_tensor_spec_tree = (
        type_proto_util.manifest_type_to_shlo_tensor_spec_pytree(manifest_type)
    )

    a_sharding = Sharding()
    a_sharding.type = Sharding.OTHER
    a_sharding.tile_assignment_dimensions.extend([4, 1, 2])
    a_sharding.replicate_on_last_tile_dim = True
    a_sharding.iota_reshape_dims.append(8)
    a_sharding.iota_transpose_perm.append(0)

    b_sharding = Sharding()
    b_sharding.type = Sharding.OTHER
    b_sharding.tile_assignment_dimensions.extend([4, 2])
    b_sharding.iota_reshape_dims.append(8)
    b_sharding.iota_transpose_perm.append(0)

    yet_another_sharding = Sharding()
    yet_another_sharding.type = Sharding.OTHER
    yet_another_sharding.tile_assignment_dimensions.extend([4, 1, 2])
    yet_another_sharding.replicate_on_last_tile_dim = True
    yet_another_sharding.iota_reshape_dims.append(8)
    yet_another_sharding.iota_transpose_perm.append(0)

    expected_shlo_tensor_spec_tree: Tree[ShloTensorSpec] = (
        (
            {
                "a": ShloTensorSpec(
                    shape=(4, 4),
                    dtype=ShloDType.i4,
                    sharding=a_sharding,
                ),
                "b": ShloTensorSpec(
                    shape=(4, 16), dtype=ShloDType.f32, sharding=b_sharding
                ),
            },
            ShloTensorSpec(
                shape=(8, 4), dtype=ShloDType.f32, sharding=yet_another_sharding
            ),
        ),
        {},
    )

    self.assertTreeEquiv(shlo_tensor_spec_tree, expected_shlo_tensor_spec_tree)


class ShloTensorSpecPyTreeToManifestTypeTest(
    test_utils.ObmTestCase, parameterized.TestCase
):

  class MyCustomClass:

    def __init__(self, x):
      self.x = x

  @parameterized.named_parameters(
      ("int", 1, "Unsupported tree type: <class 'int'>"),
      ("str", "hello", "Unsupported tree type: <class 'str'>"),
      (
          "custom_class",
          MyCustomClass(1),
          "Unsupported tree type: <class '.*MyCustomClass'>",
      ),
  )
  def test_unsupported_leaf_type_raises_error(self, value, regex):
    with self.assertRaisesRegex(ValueError, regex):
      type_proto_util.shlo_tensor_spec_pytree_to_manifest_type(value)  # pytype: disable=wrong-arg-types


if __name__ == "__main__":
  absltest.main()
