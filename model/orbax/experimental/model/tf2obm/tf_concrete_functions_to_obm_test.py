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
from typing import Any, Sequence

from absl.testing import parameterized
from orbax.experimental.model import core as obm
from orbax.experimental.model.tf2obm import tf_concrete_function_handle_pb2
from orbax.experimental.model.tf2obm import utils
from orbax.experimental.model.tf2obm.tf_concrete_functions_to_obm import _generate_names
from orbax.experimental.model.tf2obm.tf_concrete_functions_to_obm import _is_dict_only
from orbax.experimental.model.tf2obm.tf_concrete_functions_to_obm import save_tf_concrete_functions
from orbax.experimental.model.tf2obm.tf_concrete_functions_to_obm import SAVED_MODEL_MIME_TYPE
from orbax.experimental.model.tf2obm.tf_concrete_functions_to_obm import SAVED_MODEL_VERSION
from orbax.experimental.model.tf2obm.tf_concrete_functions_to_obm import TF_CONCRETE_FUNCTION_HANDLE_MIME_TYPE
from orbax.experimental.model.tf2obm.tf_concrete_functions_to_obm import TF_CONCRETE_FUNCTION_HANDLE_VERSION
from orbax.experimental.model.tf2obm.tf_concrete_functions_to_obm import tf_concrete_function_name_to_obm_function
from orbax.experimental.model.tf2obm.tf_concrete_functions_to_obm import tf_saved_model_as_obm_supplemental
from orbax.experimental.model.tf2obm.tf_concrete_functions_to_obm import TF_SAVED_MODEL_SUPPLEMENTAL_NAME
from orbax.experimental.model.tf2obm.tf_concrete_functions_to_obm import to_keyword_only_fn
import tensorflow as tf

from tensorflow.python.util.protobuf import compare
from google.protobuf import text_format
from absl.testing import absltest


_TUPLE = (
    tf.TensorSpec((2, 3), tf.float32),
    tf.TensorSpec((4, 5), tf.float64),
    tf.TensorSpec((6,), tf.float32),
)


_DICT = {
    "a": tf.TensorSpec((2, 3), tf.float32),
    "b": tf.TensorSpec((4, 5), tf.float64),
}


def _dict_from_seq(prefix: str, seq: Sequence[Any]):
  return {f"{prefix}{i}": elem for i, elem in enumerate(seq)}


def _as_output_signature(tree):
  @tf.function(autograph=False)
  def f():
    return obm.tree_util.tree_map(
        lambda spec: tf.zeros(shape=spec.shape, dtype=spec.dtype), tree
    )

  return utils.get_output_signature(f.get_concrete_function())


class TfConcreteFunctionsToObmTest(
    tf.test.TestCase,
    obm.ObmTestCase,
):

  def setUp(self):
    super().setUp()
    base_path = os.path.dirname(os.path.abspath(__file__))
    self._testdata_dir = os.path.join(base_path, "testdata")

  def _get_testdata_path(self, filename: str) -> str:
    """Returns the full path to a file in the testdata directory."""
    return os.path.join(self._testdata_dir, filename)

  @parameterized.named_parameters(
      (  # pylint: disable=g-complex-comprehension
          f"_{idx}_{as_output_signature}",
          signature,
          expected,
          as_output_signature,
      )
      for as_output_signature in (False, True)
      for idx, (signature, expected) in enumerate((
          (
              _TUPLE,
              False,
          ),
          (
              _DICT,
              True,
          ),
          (
              ((), _DICT),
              True,
          ),
          (
              ((_DICT,), {}),
              True,
          ),
      ))
  )
  def test_is_dict_only(self, signature, expected, as_output_signature):
    if as_output_signature:
      signature = _as_output_signature(signature)
    self.assertEqual(
        _is_dict_only(signature),
        expected,
    )

  @parameterized.named_parameters(
      (  # pylint: disable=g-complex-comprehension
          f"_{name}_{as_output_signature}",
          signature,
          expected_names,
          as_output_signature,
      )
      for as_output_signature in (False, True)
      for name, signature, expected_names in (
          (
              "tuple",
              _TUPLE,
              ("my_prefix_0", "my_prefix_1", "my_prefix_2"),
          ),
          (
              "dict",
              _DICT,
              ("my_prefix_0", "my_prefix_1"),
          ),
      )
  )
  def test_generate_names(self, signature, expected_names, as_output_signature):
    if as_output_signature:
      signature = _as_output_signature(signature)
    prefix = "my_prefix"

    names, _ = _generate_names(signature, prefix=prefix)
    self.assertEqual(
        names,
        expected_names,
    )

  @parameterized.named_parameters(
      (  # pylint: disable=g-complex-comprehension
          f"_{input_case_id}_{output_case_id}",
          input_sig,
          expected_input_sig,
          output_sig,
          expected_output_sig,
      )
      for input_case_id, (input_sig, expected_input_sig) in enumerate((
          (
              (_TUPLE, {}),
              ((), _dict_from_seq("input_", _TUPLE)),
          ),
          (
              ((), _DICT),
              ((), _dict_from_seq("input_", obm.tree_util.flatten(_DICT))),
          ),
          (
              (_TUPLE, _DICT),
              (
                  (),
                  _dict_from_seq(
                      "input_", obm.tree_util.flatten((_TUPLE, _DICT))
                  ),
              ),
          ),
          (
              ((_DICT,), {}),
              ((), _dict_from_seq("input_", obm.tree_util.flatten(_DICT))),
          ),
      ))
      for output_case_id, (output_sig, expected_output_sig) in enumerate((
          (
              _TUPLE,
              _dict_from_seq("output_", _TUPLE),
          ),
          (
              _TUPLE[0],
              _dict_from_seq("output_", _TUPLE[0:1]),
          ),
          (
              _DICT,
              (_dict_from_seq("output_", obm.tree_util.flatten(_DICT))),
          ),
          (
              (_DICT,),
              _dict_from_seq("output_", obm.tree_util.flatten(_DICT)),
          ),
          (
              (_TUPLE, _DICT),
              _dict_from_seq("output_", obm.tree_util.flatten((_TUPLE, _DICT))),
          ),
      ))
  )
  def test_to_keyword_only_fn(
      self, input_sig, expected_input_sig, output_sig, expected_output_sig
  ):

    @tf.function(autograph=False)
    def f(*args, **kwargs):
      del args, kwargs
      return obm.tree_util.tree_map(
          lambda spec: tf.zeros(shape=spec.shape, dtype=spec.dtype), output_sig
      )

    args, kwargs = input_sig
    cf = f.get_concrete_function(*args, **kwargs)
    new_cf = to_keyword_only_fn(cf)

    def is_spec_equiv(a, b):
      self.assertEqual(a.shape, b.shape)
      self.assertEqual(a.dtype, b.dtype)
      return True

    self.assertTreeEquiv(
        utils.get_input_signature(new_cf),
        expected_input_sig,
        is_spec_equiv,
    )
    self.assertTreeEquiv(
        utils.get_output_signature(new_cf),
        expected_output_sig,
        is_spec_equiv,
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
    em_module = dict()

    pre_processor_name = "my_pre_processor"
    em_module[pre_processor_name] = pre_processor

    post_processor_name = "my_post_processor"
    em_module[post_processor_name] = post_processor

    obm.save(
        em_module,
        save_dir_path,
        obm.SaveOptions(
            version=2,
            supplemental_info={
                TF_SAVED_MODEL_SUPPLEMENTAL_NAME: obm.GlobalSupplemental(
                    tf_global_supplemental, None
                ),
            },
        ),
    )

    # Check resulting manifest proto.
    manifest_proto = obm.load(save_dir_path)

    pre_processor_filename = f"{pre_processor_name}.pb"
    post_processor_filename = f"{post_processor_name}.pb"
    expected_manifest_proto_path = self._get_testdata_path(
        "manifest_with_concrete_function.textproto"
    )
    with open(expected_manifest_proto_path, "r") as f:
      expected_manifest_proto_text = f.read()
    manifest_replace_dict = {
        "__PRE_PROCESSOR_NAME__": pre_processor_name,
        "__PRE_PROCESSOR_PATH__": pre_processor_filename,
        "__POST_PROCESSOR_NAME__": post_processor_name,
        "__POST_PROCESSOR_PATH__": post_processor_filename,
        "__TF_CONCRETE_FUNCTION_HANDLE_MIME_TYPE__": (
            TF_CONCRETE_FUNCTION_HANDLE_MIME_TYPE
        ),
        "__TF_CONCRETE_FUNCTION_HANDLE_VERSION__": (
            TF_CONCRETE_FUNCTION_HANDLE_VERSION
        ),
        "__TF_SAVED_MODEL_SUPPLEMENTAL_NAME__": (
            TF_SAVED_MODEL_SUPPLEMENTAL_NAME
        ),
        "__SAVED_MODEL_PATH__": saved_model_rel_path,
        "__SAVED_MODEL_MIME_TYPE__": SAVED_MODEL_MIME_TYPE,
        "__SAVED_MODEL_VERSION__": SAVED_MODEL_VERSION,
    }
    for k, v in manifest_replace_dict.items():
      expected_manifest_proto_text = expected_manifest_proto_text.replace(k, v)

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
    expected_pre_processor_proto_text = f"""
        fn_name: "{pre_processor_name_in_tf}"
        input_names: "input_0"
        output_names: "output_0"
        """
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
    expected_post_processor_proto_text = f"""
        fn_name: "{post_processor_name_in_tf}"
        input_names: "input_0"
        output_names: "output_0"
        """
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
  absltest.main()
