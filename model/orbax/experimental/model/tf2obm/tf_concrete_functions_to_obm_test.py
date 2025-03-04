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
from orbax.experimental.model import core as obm
from orbax.experimental.model.tf2obm import tf_concrete_function_handle_pb2
from orbax.experimental.model.tf2obm.tf_concrete_functions_to_obm import _generate_names
from orbax.experimental.model.tf2obm.tf_concrete_functions_to_obm import save_tf_concrete_functions
from orbax.experimental.model.tf2obm.tf_concrete_functions_to_obm import SAVED_MODEL_MIME_TYPE
from orbax.experimental.model.tf2obm.tf_concrete_functions_to_obm import SAVED_MODEL_VERSION
from orbax.experimental.model.tf2obm.tf_concrete_functions_to_obm import TF_CONCRETE_FUNCTION_HANDLE_MIME_TYPE
from orbax.experimental.model.tf2obm.tf_concrete_functions_to_obm import TF_CONCRETE_FUNCTION_HANDLE_VERSION
from orbax.experimental.model.tf2obm.tf_concrete_functions_to_obm import tf_concrete_function_name_to_obm_function
from orbax.experimental.model.tf2obm.tf_concrete_functions_to_obm import tf_saved_model_as_obm_supplemental
from orbax.experimental.model.tf2obm.tf_concrete_functions_to_obm import TF_SAVED_MODEL_SUPPLEMENTAL_NAME
import tensorflow as tf

from tensorflow.python.util.protobuf import compare
from google.protobuf import text_format
from absl.testing import absltest


class TfConcreteFunctionsToObmTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      (
          "tuple",
          (
              tf.TensorSpec((2, 3), tf.float32),
              tf.TensorSpec((4, 5), tf.float64),
              tf.TensorSpec((6,), tf.float32),
          ),
          ("_0", "_1", "_2"),
      ),
      (
          "dict",
          {
              "a": tf.TensorSpec((2, 3), tf.float32),
              "b": tf.TensorSpec((4, 5), tf.float64),
          },
          None,
      ),
  )
  def test_generate_names(self, signature, expected_names):
    prefix = "my_prefix"
    if expected_names is not None:
      expected_names = tuple(prefix + name for name in expected_names)
    names, _ = _generate_names(signature, prefix=prefix)
    self.assertEqual(
        names,
        expected_names,
    )

  def test_e2e(self):
    var = tf.Variable(100.0)
    vocab = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=[
                "v",
            ],
            values=[
                200.0,
            ],
        ),
        default_value=10000.0,
    )

    @tf.function
    def tf_fn(a):
      degree = 2 if a.dtype == tf.float32 else 3
      return tf.cast(
          tf.cast(a**degree, tf.float32) + var + vocab.lookup(tf.constant("v")),
          a.dtype,
      )

    # pre-processing function
    input_arg_spec = tf.TensorSpec((2, 3, 4), tf.float32)
    tf_pre_processor = tf_fn.get_concrete_function(a=input_arg_spec)

    # post-processing function
    post_processor_input_spec = tf.TensorSpec((5, 6), tf.float64)
    tf_post_processor = tf_fn.get_concrete_function(
        a=post_processor_input_spec,
    )

    save_dir_path = os.path.join(self.create_tempdir())

    # Saves saved_model.pb
    pre_processor_name_in_tf = "my_pre_processor_in_tf"
    post_processor_name_in_tf = "my_post_processor_in_tf"

    pre_processor = tf_concrete_function_name_to_obm_function(
        pre_processor_name_in_tf, fn=tf_pre_processor
    )
    post_processor = tf_concrete_function_name_to_obm_function(
        post_processor_name_in_tf, fn=tf_post_processor
    )
    saved_model_rel_path = "tf_saved_model/"
    saved_model_abs_path = os.path.join(save_dir_path, saved_model_rel_path)
    tf_global_supplemental = tf_saved_model_as_obm_supplemental(
        saved_model_rel_path
    )
    save_tf_concrete_functions(
        saved_model_abs_path,
        {
            pre_processor_name_in_tf: tf_pre_processor,
            post_processor_name_in_tf: tf_post_processor,
        },
        (var, {"vocab": vocab}),
    )

    # Saves manifest.pb
    em_module = obm.Module()

    pre_processor_name = "my_pre_processor"
    setattr(em_module, pre_processor_name, pre_processor)

    post_processor_name = "my_post_processor"
    setattr(em_module, post_processor_name, post_processor)

    obm.save(
        em_module,
        save_dir_path,
        obm.SaveOptions(
            version=2,
            supplemental_info={
                TF_SAVED_MODEL_SUPPLEMENTAL_NAME: obm.SupplementalInfo(
                    tf_global_supplemental, None
                ),
            },
        ),
    )

    # Check resulting manifest proto.
    manifest_proto = obm.manifest_pb2.Manifest()
    with open(os.path.join(save_dir_path, obm.MANIFEST_FILENAME), "rb") as f:
      manifest_proto.ParseFromString(f.read())

    pre_processor_filename = f"{pre_processor_name}.pb"
    post_processor_filename = f"{post_processor_name}.pb"
    expected_manifest_proto_text = (
        """
      objects {
        key: \""""
        + pre_processor_name
        + """\"
        value {
          function {
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
                                  size: 2
                                }
                                dimension_sizes {
                                  size: 3
                                }
                                dimension_sizes {
                                  size: 4
                                }
                              }
                            }
                            dtype: f32
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
                          size: 2
                        }
                        dimension_sizes {
                          size: 3
                        }
                        dimension_sizes {
                          size: 4
                        }
                      }
                    }
                    dtype: f32
                  }
                }
              }
            }
            body {
              other {
                file_system_location {
                  string_path: \""""
        + pre_processor_filename
        + """\"
                }
                mime_type: \""""
        + TF_CONCRETE_FUNCTION_HANDLE_MIME_TYPE
        + """\"
                version: \""""
        + TF_CONCRETE_FUNCTION_HANDLE_VERSION
        + """\"
              }
            }
            visibility: PUBLIC
          }
        }
      }
      objects {
        key: \""""
        + post_processor_name
        + """\"
        value {
          function {
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
                                  size: 5
                                }
                                dimension_sizes {
                                  size: 6
                                }
                              }
                            }
                            dtype: f64
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
                          size: 5
                        }
                        dimension_sizes {
                          size: 6
                        }
                      }
                    }
                    dtype: f64
                  }
                }
              }
            }
            body {
              other {
                file_system_location {
                  string_path: \""""
        + post_processor_filename
        + """\"
                }
                mime_type: \""""
        + TF_CONCRETE_FUNCTION_HANDLE_MIME_TYPE
        + """\"
                version: \""""
        + TF_CONCRETE_FUNCTION_HANDLE_VERSION
        + """\"
              }
            }
            visibility: PUBLIC
          }
        }
      }
      supplemental_info {
        multiple {
          map {
            key: \""""
        + TF_SAVED_MODEL_SUPPLEMENTAL_NAME
        + """\"
            value {
              file_system_location {
                string_path: \""""
        + saved_model_rel_path
        + """\"
              }
              mime_type: \""""
        + SAVED_MODEL_MIME_TYPE
        + """\"
              version: \""""
        + SAVED_MODEL_VERSION
        + """\"
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
    )

    pre_processor_proto = (
        tf_concrete_function_handle_pb2.TfConcreteFunctionHandle()
    )
    with open(os.path.join(save_dir_path, pre_processor_filename), "rb") as f:
      pre_processor_proto.ParseFromString(f.read())
    expected_pre_processor_proto_text = (
        """
      fn_name: \""""
        + pre_processor_name_in_tf
        + """\"
      input_names {
        elements: "input_0"
      }
      output_names {
        elements: "output_0"
      }
    """
    )
    expected_pre_processor_proto = text_format.Parse(
        expected_pre_processor_proto_text,
        tf_concrete_function_handle_pb2.TfConcreteFunctionHandle(),
    )
    compare.assertProtoEqual(
        self,
        pre_processor_proto,
        expected_pre_processor_proto,
    )

    post_processor_proto = (
        tf_concrete_function_handle_pb2.TfConcreteFunctionHandle()
    )
    with open(os.path.join(save_dir_path, post_processor_filename), "rb") as f:
      post_processor_proto.ParseFromString(f.read())
    expected_post_processor_proto_text = (
        """
      fn_name: \""""
        + post_processor_name_in_tf
        + """\"
      input_names {
        elements: "input_0"
      }
      output_names {
        elements: "output_0"
      }
    """
    )
    expected_post_processor_proto = text_format.Parse(
        expected_post_processor_proto_text,
        tf_concrete_function_handle_pb2.TfConcreteFunctionHandle(),
    )
    compare.assertProtoEqual(
        self,
        post_processor_proto,
        expected_post_processor_proto,
    )

    loaded_tf_module = tf.saved_model.load(saved_model_abs_path)
    tf_input = (
        tf.ones(shape=input_arg_spec.shape, dtype=input_arg_spec.dtype) * 2
    )
    self.assertAllClose(
        tf.nest.flatten(
            loaded_tf_module.signatures[pre_processor_name_in_tf](tf_input)
        ),
        [tf_pre_processor(tf_input)],
    )
    post_processor_input = (
        tf.ones(
            shape=post_processor_input_spec.shape,
            dtype=post_processor_input_spec.dtype,
        )
        * 2
    )
    self.assertAllClose(
        tf.nest.flatten(
            loaded_tf_module.signatures[post_processor_name_in_tf](
                post_processor_input
            )
        ),
        [tf_post_processor(post_processor_input)],
    )


if __name__ == "__main__":
  googletest.main()
