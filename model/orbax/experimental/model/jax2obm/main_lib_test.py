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

import os
from typing import cast

from absl.testing import absltest
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
from orbax.experimental.model.jax2obm import jax_specific_info
from orbax.experimental.model.jax2obm import jax_supplemental_pb2
from orbax.experimental.model.jax2obm import main_lib
from orbax.experimental.model.test_utils import simple_orchestration
from orbax.experimental.model.test_utils import simple_orchestration_pb2

from tensorflow.python.util.protobuf import compare
from google.protobuf import text_format


os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'


class MainLibTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='_default',
          native_serialization_platforms=None,
          polymorphic_shape=False,
      ),
      dict(
          testcase_name='_native_cpu_serialization',
          native_serialization_platforms=[obm.manifest_pb2.Platform.CPU],
          polymorphic_shape=False,
      ),
      dict(
          testcase_name='_polymorphic_shape',
          native_serialization_platforms=None,
          polymorphic_shape=True,
      ),
  )
  def test_sharded_jax_e2e(
      self, native_serialization_platforms, polymorphic_shape
  ):

    def _jax_model_fn(params, x):
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

    params = _generated_sharded_params()

    in_shardings = (
        pytree_shardings,
        NamedSharding(
            mesh,
            PartitionSpec(None, 'x')
            if polymorphic_shape
            else PartitionSpec('x'),
        ),
    )
    model_fn = jax.jit(_jax_model_fn, in_shardings=in_shardings)

    if polymorphic_shape:
      input_args_spec = jax_export.symbolic_args_specs(
          jax.ShapeDtypeStruct(
              (jax_export.symbolic_shape('b'), 4),
              jnp.float64,
          ),
          shapes_specs='b, 4',
      )
    else:
      input_args_spec = jax.ShapeDtypeStruct((8, 4), jnp.float64)
    params_args_spec = main_lib.get_shape_dtype_struct(params)
    obm_shlo_fn = main_lib.convert(
        model_fn,
        (params_args_spec, input_args_spec),
        {},
        platforms=native_serialization_platforms,
    )
    obm_module = dict()
    model_function_name = 'my_model_fn'
    obm_module[model_function_name] = obm_shlo_fn

    save_dir_path = os.path.join(self.create_tempdir())
    ckpt_subdir = 'my_checkpoint/'
    ckpt_path = os.path.join(save_dir_path, ckpt_subdir)
    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    checkpointer.save(ckpt_path, params)
    weights_name = 'my_weights'
    obm_module[weights_name] = main_lib.convert_path_to_value(
        ckpt_subdir,
        mime_type='application/x.orbax-checkpoint',
    )

    supplemental_filename = 'my_orchestration.pb'

    obm.save(
        obm_module,
        save_dir_path,
        obm.SaveOptions(
            version=2,
            supplemental_info={
                simple_orchestration.TEST_ORCHESTRATION_SUPPLEMENTAL_NAME: (
                    obm.GlobalSupplemental(
                        simple_orchestration.create(
                            model_function_name=model_function_name,
                            weights_name=weights_name,
                        ),
                        supplemental_filename,
                    )
                )
            },
        ),
    )

    ckpt_path = os.path.join(save_dir_path, ckpt_subdir)
    self.assertNotEmpty(
        os.listdir(ckpt_path),
        f'Checkpoint directory {ckpt_subdir} is empty.',
    )

    manifest_proto = obm.manifest_pb2.Manifest()
    with open(os.path.join(save_dir_path, obm.MANIFEST_FILENAME), 'rb') as f:
      manifest_proto.ParseFromString(f.read())

    if polymorphic_shape:
      batch_size = ''
      sharding_dims = """
        tile_assignment_dimensions: 1
        tile_assignment_dimensions: 4
        tile_assignment_dimensions: 2
        """
    else:
      sharding_dims = """
        tile_assignment_dimensions: 4
        tile_assignment_dimensions: 1
        tile_assignment_dimensions: 2
      """
      batch_size = 'size: 8'

    expected_signature_text = (
        """
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
                            """
        + batch_size
        + """
                          }
                          dimension_sizes {
                            size: 4
                          }
                        }
                      }
                      dtype: f32
                      sharding {
                        type: OTHER
                        """
        + sharding_dims
        + """
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
                    """
        + batch_size
        + """
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
    )

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
                  mime_type: "application/x.mlir-stablehlo"
                  version: "1.0"
                }
                calling_convention_version: 10
                lowering_platforms: "cpu"
                module_kept_var_idx: 0
                module_kept_var_idx: 1
                module_kept_var_idx: 2
                supplemental_info {
                  key: "jax_specific_info"
                  value {
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
        + ckpt_subdir
        + """\"
                }
                mime_type: "application/x.orbax-checkpoint"
              }
            }
          }
        }
      }
      supplemental_info {
          key: \""""
        + simple_orchestration.TEST_ORCHESTRATION_SUPPLEMENTAL_NAME
        + """\"
          value {
            file_system_location {
                string_path: \""""
        + supplemental_filename
        + """\"
            }
            mime_type: \""""
        + simple_orchestration.TEST_ORCHESTRATION_MIME_TYPE
        + """\"
            version: \""""
        + simple_orchestration.TEST_ORCHESTRATION_VERSION
        + """\"
          }
        }
      """
    )
    expected_manifest_proto = text_format.Parse(
        expected_manifest_proto_text, obm.manifest_pb2.Manifest()
    )
    compare.assertProtoEqual(
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

    pipeline_proto = simple_orchestration_pb2.Pipeline()
    with open(os.path.join(save_dir_path, supplemental_filename), 'rb') as f:
      pipeline_proto.ParseFromString(f.read())
    expected_orchestration_proto_text = f"""
      model_function_name: "{model_function_name}"
      weights_name: "{weights_name}"
    """
    expected_orchestration_proto = text_format.Parse(
        expected_orchestration_proto_text,
        simple_orchestration_pb2.Pipeline(),
    )
    compare.assertProtoEqual(
        self, pipeline_proto, expected_orchestration_proto
    )

    jax_supplemental_proto = jax_supplemental_pb2.Function()
    with open(
        os.path.join(save_dir_path, jax_supplemental_filename), 'rb'
    ) as f:
      jax_supplemental_proto.ParseFromString(f.read())
    expected_jax_supplemental_proto_text = """
      nr_devices: 8
      name: "_jax_model_fn"
    """
    if polymorphic_shape:
      expected_jax_supplemental_proto_text += """
      uses_shape_polymorphism: true
      input_spec_refinements {
        map {
          idx_to_refinement {
            key: 2
            value {
              shape {
                dimension_sizes {
                  size: "b"
                }
                dimension_sizes {
                }
              }
            }
          }
        }
      }
      output_spec_refinements {
        list {
          refinements {
            shape {
              dimension_sizes {
                size: "b"
              }
              dimension_sizes {
              }
            }
          }
        }
      }
      """

    expected_jax_supplemental_proto = text_format.Parse(
        expected_jax_supplemental_proto_text,
        jax_supplemental_pb2.Function(),
    )
    compare.assertProtoEqual(
        self, jax_supplemental_proto, expected_jax_supplemental_proto
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

    em_shlo_fn = main_lib.convert(
        model_apply_fn,
        (params_args_spec, input_args_spec),
        {},
    )

    obm_module = dict()

    model_function_name = 'mnist_forward_fn'
    obm_module[model_function_name] = em_shlo_fn
    save_dir_path = os.path.join(self.create_tempdir())

    checkpoint_path = 'my_checkpoint/'
    checkpoint_abs_path = os.path.join(save_dir_path, checkpoint_path)
    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    checkpointer.save(checkpoint_abs_path, params)
    obm_module['my_weights'] = main_lib.convert_path_to_value(
        checkpoint_path,
        mime_type='application/x.orbax-checkpoint',
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
                    mime_type: "application/x.mlir-stablehlo"
                    version: "1.0"
                  }
                  calling_convention_version: 10
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
                    key: "jax_specific_info"
                    value {
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
                  mime_type: "application/x.orbax-checkpoint"
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

    compare.assertProtoEqual(
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


if __name__ == '__main__':
  absltest.main()
