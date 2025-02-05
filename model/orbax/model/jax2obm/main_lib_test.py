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

import os

from absl.testing import parameterized
import flax.linen as nn
import jax
from jax import export as jax_export
from jax.experimental import mesh_utils
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec
import orbax.checkpoint as ocp
from orbax.experimental.model import core as obm
from orbax.experimental.model.jax2obm import constants
from orbax.experimental.model.jax2obm import jax_specific_info
from orbax.experimental.model.jax2obm import jax_supplemental_pb2
from orbax.experimental.model.jax2obm import main_lib

from .net.proto2.contrib.pyutil import compare
from google.protobuf import text_format
from absl.testing import absltest


os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'


class MainLibTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (  # pylint: disable=g-complex-comprehension
          f'_{flatten_signature}_{native_serialization_platforms}',
          flatten_signature,
          native_serialization_platforms,
      )
      for flatten_signature in (True, False)
      for native_serialization_platforms in (
          [constants.OrbaxNativeSerializationType.CPU],
          None,
      )
  )
  def test_sharded_jax_e2e(
      self, flatten_signature, native_serialization_platforms
  ):

    def jax_model_fn(params, x):
      return jnp.dot(jnp.dot(x, params['a']), params['b'])

    def _create_mesh() -> jax.sharding.Mesh:
      os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'
      devices = mesh_utils.create_device_mesh((4, 2))
      return jax.sharding.Mesh(devices, ('x', 'y'))

    mesh = _create_mesh()
    pytree_shardings = {
        'a': NamedSharding(mesh, PartitionSpec('x')),
        'b': NamedSharding(mesh, PartitionSpec('x', 'y')),
    }

    def _generated_sharded_params():
      a = jnp.array(jax.random.uniform(jax.random.key(0), (4, 4)))
      b = jnp.array(jax.random.uniform(jax.random.key(1), (4, 16)))
      return {
          'a': jax.device_put(a, pytree_shardings['a']),
          'b': jax.device_put(b, pytree_shardings['b']),
      }

    jax_params = _generated_sharded_params()
    in_shardings_ = (pytree_shardings, NamedSharding(mesh, PartitionSpec('x')))
    jax_model_fn_sharded = jax.jit(jax_model_fn, in_shardings=in_shardings_)
    input_ = jnp.ones((8, 4), dtype=jnp.float64)
    input_args_spec = main_lib.get_shape_dtype_struct(input_)
    params_args_spec = main_lib.get_shape_dtype_struct(jax_params)
    em_shlo_fn = main_lib.convert_to_shlo_fn(
        jax_model_fn_sharded,
        (params_args_spec, input_args_spec),
        {},
        native_serialization_platforms=native_serialization_platforms,
        flatten_signature=flatten_signature,
    )
    obm_module = obm.Module()
    model_function_name = 'my_model_fn'
    setattr(obm_module, model_function_name, em_shlo_fn)

    save_dir_path = os.path.join(self.create_tempdir())

    weights_name = 'my_weights'
    checkpoint_path = 'my_checkpoint/'
    setattr(
        obm_module,
        weights_name,
        main_lib.convert_path_to_value(
            checkpoint_path,
            mime_type='orbax_checkpoint',
        ),
    )

    supplemental_filename = 'my_orchestration.pb'

    obm.save(
        obm_module,
        save_dir_path,
        obm.SaveOptions(
            version=2,
            supplemental_info=obm.SupplementalInfo(
                obm.simple_orchestration.create(
                    signature=obm.simple_orchestration.calculate_signature(
                        model_function_signature=em_shlo_fn.signature,
                        num_of_weights=None
                        if not flatten_signature
                        else len(obm.tree_util.flatten(params_args_spec)),
                    ),
                    model_function_name=model_function_name,
                    weights_name=weights_name,
                ),
                supplemental_filename,
            ),
        ),
    )

    manifest_proto = obm.manifest_pb2.Manifest()
    with open(os.path.join(save_dir_path, obm.MANIFEST_FILENAME), 'rb') as f:
      manifest_proto.ParseFromString(f.read())

    if flatten_signature:
      # Expected function signature (depending on flatten_signature)
      expected_signature_text = """
        signature {
          input {
            tuple {
              elements {
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
              elements {
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
          output {
            tuple {
              elements {
                leaf {
                  tensor_type {
                    shape {
                      shape_with_known_rank {
                        dimension_sizes {
                          size: 8
                        }
                        dimension_sizes {
                          size: 16
                        }
                      }
                    }
                    dtype: f32
                  }
                }
              }
            }
          }
        }
      """

      expected_orchestration_signature_text = """
        signature {
          input {
            tuple {
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
          output {
            tuple {
              elements {
                leaf {
                  tensor_type {
                    shape {
                      shape_with_known_rank {
                        dimension_sizes {
                          size: 8
                        }
                        dimension_sizes {
                          size: 16
                        }
                      }
                    }
                    dtype: f32
                  }
                }
              }
            }
          }
        }
      """
    else:
      expected_signature_text = """
        signature {
          input {
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
          }
          output {
            leaf {
              tensor_type {
                shape {
                  shape_with_known_rank {
                    dimension_sizes {
                      size: 8
                    }
                    dimension_sizes {
                      size: 16
                    }
                  }
                }
                dtype: f32
              }
            }
          }
        }
      """

      expected_orchestration_signature_text = """
        signature {
          input {
            tuple {
              elements {
                tuple {
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
          }
          output {
            leaf {
              tensor_type {
                shape {
                  shape_with_known_rank {
                    dimension_sizes {
                      size: 8
                    }
                    dimension_sizes {
                      size: 16
                    }
                  }
                }
                dtype: f32
              }
            }
          }
        }
      """

    jax_supplemental_filename = f'{model_function_name}_supplemental.pb'

    expected_manifest_proto_text = (
        """
      objects {
        key: \""""
        + model_function_name
        + """\"
        value {
          function {"""
        + expected_signature_text
        + """
            body {
              stable_hlo_body {
                stable_hlo {
                  inlined_bytes: "ML StableHLO v0.9.0 ..."
                  mime_type: "mlir_stablehlo"
                  version: "1.0"
                }
                calling_convention_version: 9
                lowering_platforms: "cpu"
                module_kept_var_idx: 0
                module_kept_var_idx: 1
                module_kept_var_idx: 2
                supplemental_info {
                  file_system_location {
                    string_path: \""""
        + jax_supplemental_filename
        + """\"
                  }
                  mime_type: \""""
        + jax_specific_info.CURRENT_JAX_SUPPLEMENTAL_MIME_TYPE
        + """\"
                  version: \""""
        + jax_specific_info.CURRENT_JAX_SUPPLEMENTAL_VERSION
        + """\"
                }
              }
            }
            visibility: PUBLIC
          }
        }
      }
      objects {
        key: \""""
        + weights_name
        + """\"
        value {
          value {
            external {
              data {
                file_system_location {
                  string_path: \""""
        + checkpoint_path
        + """\"
                }
                mime_type: "orbax_checkpoint"
              }
            }
          }
        }
      }
      supplemental_info {
        single {
          file_system_location {
              string_path: \""""
        + supplemental_filename
        + """\"
          }
          mime_type: \""""
        + obm.CURRENT_SIMPLE_ORCHESTRATION_MIME_TYPE
        + """\"
          version: \""""
        + obm.CURRENT_SIMPLE_ORCHESTRATION_VERSION
        + """\"
        }
      }
    """
    )
    expected_manifest_proto = text_format.Parse(
        expected_manifest_proto_text, obm.manifest_pb2.Manifest()
    )
    compare.assertProto2Equal(
        self,
        manifest_proto,
        expected_manifest_proto,
        ignored_fields=[
            'objects.function.body.stable_hlo_body.stable_hlo.inlined_bytes'
        ],
    )
    self.assertIn(
        b'ML\xef',
        manifest_proto.objects[
            model_function_name
        ].function.body.stable_hlo_body.stable_hlo.inlined_bytes,
    )
    self.assertIn(
        b'StableHLO_v',
        manifest_proto.objects[
            model_function_name
        ].function.body.stable_hlo_body.stable_hlo.inlined_bytes,
    )

    orchestration_proto = obm.simple_orchestration_pb2.SimpleOrchestration()
    with open(os.path.join(save_dir_path, supplemental_filename), 'rb') as f:
      orchestration_proto.ParseFromString(f.read())
    expected_orchestration_proto_text = (
        expected_orchestration_signature_text + f"""
      model_function_name: "{model_function_name}"
      weights_name: "{weights_name}"
    """
    )
    expected_orchestration_proto = text_format.Parse(
        expected_orchestration_proto_text,
        obm.simple_orchestration_pb2.SimpleOrchestration(),
    )
    compare.assertProto2Equal(
        self,
        orchestration_proto,
        expected_orchestration_proto,
    )

    jax_supplemental_proto = jax_supplemental_pb2.Function()
    with open(
        os.path.join(save_dir_path, jax_supplemental_filename), 'rb'
    ) as f:
      jax_supplemental_proto.ParseFromString(f.read())
    expected_jax_supplemental_proto_text = """
      nr_devices: 8
      name: "jax_model_fn"
    """
    expected_jax_supplemental_proto = text_format.Parse(
        expected_jax_supplemental_proto_text,
        jax_supplemental_pb2.Function(),
    )
    compare.assertProto2Equal(
        self,
        jax_supplemental_proto,
        expected_jax_supplemental_proto,
    )

  def test_mnist_model(self):

    class JaxMnist(nn.Module):
      """Flax MNIST model."""

      @nn.compact
      def __call__(self, x):
        """See base class."""
        x = nn.Conv(features=32, kernel_size=(4, 4))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(4, 4))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        x = nn.log_softmax(x)
        return x

    model = JaxMnist()
    input_args_spec = jax.ShapeDtypeStruct((4, 28, 28, 1), jnp.float64)
    params = model.init(
        jax.random.PRNGKey(123),
        jnp.ones(shape=input_args_spec.shape, dtype=input_args_spec.dtype),
    )

    def get_mesh():
      devices = mesh_utils.create_device_mesh((2, 2, 2))
      return jax.sharding.Mesh(devices, ('b', 'x', 'y'))

    mesh = get_mesh()
    params_sharding_spec = jax.tree_util.tree_map(
        lambda _: NamedSharding(mesh, jax.sharding.PartitionSpec('y')), params
    )
    in_shardings_ = (
        params_sharding_spec,
        NamedSharding(mesh, PartitionSpec('b', 'x', None, None)),
    )

    model_apply_fn = jax.jit(
        model.apply,
        in_shardings=in_shardings_,
        out_shardings=NamedSharding(mesh, PartitionSpec('b', 'y')),
    )

    params_args_spec = main_lib.get_shape_dtype_struct(params)

    em_shlo_fn = main_lib.convert_to_shlo_fn(
        model_apply_fn,
        (params_args_spec, input_args_spec),
        {},
    )

    obm_module = obm.Module()

    model_function_name = 'mnist_forward_fn'
    setattr(obm_module, model_function_name, em_shlo_fn)
    save_dir_path = os.path.join(self.create_tempdir())

    checkpoint_path = 'my_checkpoint/'
    checkpoint_abs_path = os.path.join(save_dir_path, checkpoint_path)
    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    checkpointer.save(checkpoint_abs_path, params)
    obm_module.my_weights = main_lib.convert_path_to_value(
        checkpoint_path,
        mime_type='orbax_checkpoint',
    )

    obm.save(
        obm_module,
        save_dir_path,
        obm.SaveOptions(
            version=2,
        ),
    )

    manifest_proto = obm.manifest_pb2.Manifest()
    with open(os.path.join(save_dir_path, obm.MANIFEST_FILENAME), 'rb') as f:
      manifest_proto.ParseFromString(f.read())

    jax_supplemental_filename = f'{model_function_name}_supplemental.pb'

    expected_manifest_proto_text = (
        """
       objects {
          key: \""""
        + model_function_name
        + """\"
          value {
            function {
              signature {
                  input {
                    tuple {
                      elements {
                        tuple {
                          elements {
                            dict {
                              string_to_type {
                                key: "params"
                                value {
                                  dict {
                                    string_to_type {
                                      key: "Conv_0"
                                      value {
                                        dict {
                                          string_to_type {
                                            key: "bias"
                                            value {
                                              leaf {
                                                tensor_type {
                                                  shape {
                                                    shape_with_known_rank {
                                                      dimension_sizes {
                                                        size: 32
                                                      }
                                                    }
                                                  }
                                                  dtype: f32
                                                  sharding {
                                                    type: OTHER
                                                    tile_assignment_dimensions: 2
                                                    tile_assignment_dimensions: 4
                                                    replicate_on_last_tile_dim: true
                                                    iota_reshape_dims: 4
                                                    iota_reshape_dims: 2
                                                    iota_transpose_perm: 1
                                                    iota_transpose_perm: 0
                                                  }
                                                }
                                              }
                                            }
                                          }
                                          string_to_type {
                                            key: "kernel"
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
                                                      dimension_sizes {
                                                        size: 1
                                                      }
                                                      dimension_sizes {
                                                        size: 32
                                                      }
                                                    }
                                                  }
                                                  dtype: f32
                                                  sharding {
                                                    type: OTHER
                                                    tile_assignment_dimensions: 2
                                                    tile_assignment_dimensions: 1
                                                    tile_assignment_dimensions: 1
                                                    tile_assignment_dimensions: 1
                                                    tile_assignment_dimensions: 4
                                                    replicate_on_last_tile_dim: true
                                                    iota_reshape_dims: 4
                                                    iota_reshape_dims: 2
                                                    iota_transpose_perm: 1
                                                    iota_transpose_perm: 0
                                                  }
                                                }
                                              }
                                            }
                                          }
                                        }
                                      }
                                    }
                                    string_to_type {
                                      key: "Conv_1"
                                      value {
                                        dict {
                                          string_to_type {
                                            key: "bias"
                                            value {
                                              leaf {
                                                tensor_type {
                                                  shape {
                                                    shape_with_known_rank {
                                                      dimension_sizes {
                                                        size: 64
                                                      }
                                                    }
                                                  }
                                                  dtype: f32
                                                  sharding {
                                                    type: OTHER
                                                    tile_assignment_dimensions: 2
                                                    tile_assignment_dimensions: 4
                                                    replicate_on_last_tile_dim: true
                                                    iota_reshape_dims: 4
                                                    iota_reshape_dims: 2
                                                    iota_transpose_perm: 1
                                                    iota_transpose_perm: 0
                                                  }
                                                }
                                              }
                                            }
                                          }
                                          string_to_type {
                                            key: "kernel"
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
                                                      dimension_sizes {
                                                        size: 32
                                                      }
                                                      dimension_sizes {
                                                        size: 64
                                                      }
                                                    }
                                                  }
                                                  dtype: f32
                                                  sharding {
                                                    type: OTHER
                                                    tile_assignment_dimensions: 2
                                                    tile_assignment_dimensions: 1
                                                    tile_assignment_dimensions: 1
                                                    tile_assignment_dimensions: 1
                                                    tile_assignment_dimensions: 4
                                                    replicate_on_last_tile_dim: true
                                                    iota_reshape_dims: 4
                                                    iota_reshape_dims: 2
                                                    iota_transpose_perm: 1
                                                    iota_transpose_perm: 0
                                                  }
                                                }
                                              }
                                            }
                                          }
                                        }
                                      }
                                    }
                                    string_to_type {
                                      key: "Dense_0"
                                      value {
                                        dict {
                                          string_to_type {
                                            key: "bias"
                                            value {
                                              leaf {
                                                tensor_type {
                                                  shape {
                                                    shape_with_known_rank {
                                                      dimension_sizes {
                                                        size: 256
                                                      }
                                                    }
                                                  }
                                                  dtype: f32
                                                  sharding {
                                                    type: OTHER
                                                    tile_assignment_dimensions: 2
                                                    tile_assignment_dimensions: 4
                                                    replicate_on_last_tile_dim: true
                                                    iota_reshape_dims: 4
                                                    iota_reshape_dims: 2
                                                    iota_transpose_perm: 1
                                                    iota_transpose_perm: 0
                                                  }
                                                }
                                              }
                                            }
                                          }
                                          string_to_type {
                                            key: "kernel"
                                            value {
                                              leaf {
                                                tensor_type {
                                                  shape {
                                                    shape_with_known_rank {
                                                      dimension_sizes {
                                                        size: 3136
                                                      }
                                                      dimension_sizes {
                                                        size: 256
                                                      }
                                                    }
                                                  }
                                                  dtype: f32
                                                  sharding {
                                                    type: OTHER
                                                    tile_assignment_dimensions: 2
                                                    tile_assignment_dimensions: 1
                                                    tile_assignment_dimensions: 4
                                                    replicate_on_last_tile_dim: true
                                                    iota_reshape_dims: 4
                                                    iota_reshape_dims: 2
                                                    iota_transpose_perm: 1
                                                    iota_transpose_perm: 0
                                                  }
                                                }
                                              }
                                            }
                                          }
                                        }
                                      }
                                    }
                                    string_to_type {
                                      key: "Dense_1"
                                      value {
                                        dict {
                                          string_to_type {
                                            key: "bias"
                                            value {
                                              leaf {
                                                tensor_type {
                                                  shape {
                                                    shape_with_known_rank {
                                                      dimension_sizes {
                                                        size: 10
                                                      }
                                                    }
                                                  }
                                                  dtype: f32
                                                  sharding {
                                                    type: OTHER
                                                    tile_assignment_dimensions: 2
                                                    tile_assignment_dimensions: 4
                                                    replicate_on_last_tile_dim: true
                                                    iota_reshape_dims: 4
                                                    iota_reshape_dims: 2
                                                    iota_transpose_perm: 1
                                                    iota_transpose_perm: 0
                                                  }
                                                }
                                              }
                                            }
                                          }
                                          string_to_type {
                                            key: "kernel"
                                            value {
                                              leaf {
                                                tensor_type {
                                                  shape {
                                                    shape_with_known_rank {
                                                      dimension_sizes {
                                                        size: 256
                                                      }
                                                      dimension_sizes {
                                                        size: 10
                                                      }
                                                    }
                                                  }
                                                  dtype: f32
                                                  sharding {
                                                    type: OTHER
                                                    tile_assignment_dimensions: 2
                                                    tile_assignment_dimensions: 1
                                                    tile_assignment_dimensions: 4
                                                    replicate_on_last_tile_dim: true
                                                    iota_reshape_dims: 4
                                                    iota_reshape_dims: 2
                                                    iota_transpose_perm: 1
                                                    iota_transpose_perm: 0
                                                  }
                                                }
                                              }
                                            }
                                          }
                                        }
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
                                      size: 4
                                    }
                                    dimension_sizes {
                                      size: 28
                                    }
                                    dimension_sizes {
                                      size: 28
                                    }
                                    dimension_sizes {
                                      size: 1
                                    }
                                  }
                                }
                                dtype: f32
                                sharding {
                                  type: OTHER
                                  tile_assignment_dimensions: 2
                                  tile_assignment_dimensions: 2
                                  tile_assignment_dimensions: 1
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
                  }
                  output {
                    leaf {
                      tensor_type {
                        shape {
                          shape_with_known_rank {
                            dimension_sizes {
                              size: 4
                            }
                            dimension_sizes {
                              size: 10
                            }
                          }
                        }
                        dtype: f32
                        sharding {
                          type: OTHER
                          tile_assignment_dimensions: 2
                          tile_assignment_dimensions: 2
                          tile_assignment_dimensions: 2
                          replicate_on_last_tile_dim: true
                          iota_reshape_dims: 2
                          iota_reshape_dims: 2
                          iota_reshape_dims: 2
                          iota_transpose_perm: 0
                          iota_transpose_perm: 2
                          iota_transpose_perm: 1
                        }
                      }
                    }
                  }
                }
              body {
                stable_hlo_body {
                  stable_hlo {
                    inlined_bytes:"ML\357R\rStableHLO_v...."
                    mime_type: "mlir_stablehlo"
                    version: "1.0"
                  }
                  calling_convention_version: 9
                  lowering_platforms: "cpu"
                  module_kept_var_idx: 0
                  module_kept_var_idx: 1
                  module_kept_var_idx: 2
                  module_kept_var_idx: 3
                  module_kept_var_idx: 4
                  module_kept_var_idx: 5
                  module_kept_var_idx: 6
                  module_kept_var_idx: 7
                  module_kept_var_idx: 8
                  supplemental_info {
                    file_system_location {
                      string_path: \""""
        + jax_supplemental_filename
        + """\"
                    }
                    mime_type: \""""
        + jax_specific_info.CURRENT_JAX_SUPPLEMENTAL_MIME_TYPE
        + """\"
                    version: \""""
        + jax_specific_info.CURRENT_JAX_SUPPLEMENTAL_VERSION
        + """\"
                  }
                }
              }
              visibility: PUBLIC
            }
          }
        }
        objects {
          key: "my_weights"
          value {
            value {
              external {
                data {
                  file_system_location {
                    string_path: "my_checkpoint/"
                  }
                  mime_type: "orbax_checkpoint"
                }
              }
            }
          }
        }
    """
    )
    expected_manifest_proto = text_format.Parse(
        expected_manifest_proto_text, obm.manifest_pb2.Manifest()
    )

    compare.assertProto2Equal(
        self,
        manifest_proto,
        expected_manifest_proto,
        ignored_fields=[
            'objects.function.body.stable_hlo_body.stable_hlo.inlined_bytes'
        ],
    )
    self.assertNotEmpty(
        os.listdir(checkpoint_abs_path),
        f"Checkpoint directory '{checkpoint_abs_path}' is empty.",
    )

  # TODO(qidichen): Merge this test with `def test_sharded_jax_e2e` above
  # with extra param like `use_polymorphic_shape`.
  def test_polymorphic_shape_sharded_jax(
      self,
  ):

    def jax_model_fn(params, x):
      return jnp.dot(jnp.dot(x, params['a']), params['b'])

    def _create_mesh() -> jax.sharding.Mesh:
      os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'
      devices = mesh_utils.create_device_mesh((4, 2))
      return jax.sharding.Mesh(devices, ('x', 'y'))

    mesh = _create_mesh()
    pytree_shardings = {
        'a': NamedSharding(mesh, PartitionSpec('x')),
        'b': NamedSharding(mesh, PartitionSpec('x', 'y')),
    }

    def _generated_sharded_params():
      a = jnp.array(jax.random.uniform(jax.random.key(0), (4, 4)))
      b = jnp.array(jax.random.uniform(jax.random.key(1), (4, 16)))
      return {
          'a': jax.device_put(a, pytree_shardings['a']),
          'b': jax.device_put(b, pytree_shardings['b']),
      }

    jax_params = _generated_sharded_params()
    in_shardings_ = (
        pytree_shardings,
        NamedSharding(mesh, PartitionSpec(None, 'x')),
    )
    jax_model_fn_sharded = jax.jit(jax_model_fn, in_shardings=in_shardings_)

    # Use symbolic shapes for the model input.
    a = jax_export.symbolic_shape('a')
    input_args_spec = jax_export.symbolic_args_specs(
        jax.ShapeDtypeStruct(
            (a, 4),
            jnp.float64,
        ),
        shapes_specs='a, 4',
    )
    params_args_spec = main_lib.get_shape_dtype_struct(jax_params)

    em_shlo_fn = main_lib.convert_to_shlo_fn(
        jax_model_fn_sharded,
        (params_args_spec, input_args_spec),
        {},
        flatten_signature=True,
    )
    obm_module = obm.Module()
    model_function_name = 'my_model_fn'
    setattr(obm_module, model_function_name, em_shlo_fn)

    save_dir_path = os.path.join(self.create_tempdir())

    obm.save(
        obm_module,
        save_dir_path,
        obm.SaveOptions(
            version=2,
        ),
    )

    manifest_proto = obm.manifest_pb2.Manifest()
    with open(os.path.join(save_dir_path, obm.MANIFEST_FILENAME), 'rb') as f:
      manifest_proto.ParseFromString(f.read())

    expected_manifest_proto_text = """
        objects {
          key: "my_model_fn"
          value {
            function {
              signature {
                input {
                  tuple {
                    elements {
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
                    elements {
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
                    elements {
                      leaf {
                        tensor_type {
                          shape {
                            shape_with_known_rank {
                              dimension_sizes {
                              }
                              dimension_sizes {
                                size: 4
                              }
                            }
                          }
                          dtype: f32
                          sharding {
                            type: OTHER
                            tile_assignment_dimensions: 1
                            tile_assignment_dimensions: 4
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
                output {
                  tuple {
                    elements {
                      leaf {
                        tensor_type {
                          shape {
                            shape_with_known_rank {
                              dimension_sizes {
                              }
                              dimension_sizes {
                                size: 16
                              }
                            }
                          }
                          dtype: f32
                        }
                      }
                    }
                  }
                }
              }
              body {
                stable_hlo_body {
                  stable_hlo {
                    mime_type: "mlir_stablehlo"
                    version: "1.0"
                  }
                  calling_convention_version: 9
                  lowering_platforms: "cpu"
                  module_kept_var_idx: 0
                  module_kept_var_idx: 1
                  module_kept_var_idx: 2
                  supplemental_info {
                    file_system_location {
                      string_path: "my_model_fn_supplemental.pb"
                    }
                    mime_type: "orbax_model_jax_supplemental"
                    version: "0.0.1"
                  }
                }
              }
              visibility: PUBLIC
            }
          }
        }
    """
    expected_manifest_proto = text_format.Parse(
        expected_manifest_proto_text, obm.manifest_pb2.Manifest()
    )
    print('!!! \n', manifest_proto)
    compare.assertProto2Equal(
        self,
        expected_manifest_proto,
        manifest_proto,
        ignored_fields=[
            'objects.function.body.stable_hlo_body.stable_hlo.inlined_bytes'
        ],
    )
    self.assertIn(
        b'ML\xef',
        manifest_proto.objects[
            model_function_name
        ].function.body.stable_hlo_body.stable_hlo.inlined_bytes,
    )
    self.assertIn(
        b'StableHLO_v',
        manifest_proto.objects[
            model_function_name
        ].function.body.stable_hlo_body.stable_hlo.inlined_bytes,
    )


if __name__ == '__main__':
  googletest.main()
