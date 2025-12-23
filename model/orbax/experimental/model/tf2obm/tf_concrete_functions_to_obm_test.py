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

import dataclasses
import itertools
import os

from absl.testing import absltest
from absl.testing import parameterized
import chex
from jax import tree_util as jax_tree_util
from orbax.experimental.model import core as obm
from orbax.experimental.model.tf2obm import tf_concrete_function_handle_pb2
from orbax.experimental.model.tf2obm import tf_concrete_functions_to_obm as tf_obm
import tensorflow as tf

from tensorflow.python.util.protobuf import compare
from google.protobuf import text_format


_T1 = tf.TensorSpec((2, 3), tf.float32)
_T2 = tf.TensorSpec((4, 5), tf.float64)
_T3 = tf.TensorSpec((6,), tf.float32)


@dataclasses.dataclass
class _Dataclass:
  a: tf.TensorSpec
  b: tf.TensorSpec


@chex.dataclass
class _ChexDataclass:
  a: tf.TensorSpec
  b: tf.TensorSpec

  def tree_flatten(self):
    return (self.a, self.b), None

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    del aux_data
    return cls(*children)


def _as_output_signature(tree):
  @tf.function(autograph=False)
  def f():
    return jax_tree_util.tree_map(
        lambda spec: tf.zeros(shape=spec.shape, dtype=spec.dtype), tree
    )

  return tf_obm.get_output_signature(f.get_concrete_function())


_INPUT_SIGNATURES = (
    (
        "positional_only",
        (
            (_T1, _T2, _T3),
            {},
        ),
        (
            (),
            {
                "args_0": _T1,
                "args_1": _T2,
                "args_2": _T3,
            },
        ),
    ),
    (
        "kwargs_only",
        ((), {"a": _T1, "b": _T2}),
        ((), {"a": _T1, "b": _T2}),
    ),
    (
        "positional_and_kwargs",
        ((_T1, _T2, _T3), {"a": _T1, "b": _T2, "c": _T3}),
        (
            (),
            {
                "args_0": _T1,
                "args_1": _T2,
                "args_2": _T3,
                "a": _T1,
                "b": _T2,
                "c": _T3,
            },
        ),
    ),
    (
        "dict_positional_and_kwargs",
        # A nested dict passed in a positional argument
        # if flattened but TF SavedModel loses track
        # of the original key names in this case.
        (({"a": _T1, "b": _T2},), {"c": _T3}),
        ((), {"args_0": _T1, "args_0_1": _T2, "c": _T3}),
    ),
    (
        "registered_dataclass",
        # Registered dataclasses are flattened.
        ((_ChexDataclass(a=_T1, b=_T2),), {}),
        (
            (),
            {"args_0": _T1, "args_0_1": _T2},
        ),
    ),
)
_OUTPUT_SIGNATURES = (
    (
        "tuple",
        (_T1, _T2, _T3),
        {"output_0": _T1, "output_1": _T2, "output_2": _T3},
    ),
    (
        "scalar",
        _T1,
        {"output_0": _T1},
    ),
    (
        "tuple_one_element",
        (_T1,),
        {"output_0": _T1},
    ),
    (
        "list",
        [_T2, _T1],
        {"output_0": _T2, "output_1": _T1},
    ),
    (
        "dict",
        {"a": _T1, "b": _T2},
        {"a": _T1, "b": _T2},
    ),
    (
        "registered_dataclass",
        # Registered dataclasses are flattened.
        (_ChexDataclass(a=_T1, b=_T2)),
        {"a": _T1, "b": _T2},
    ),
    (
        "nested_dict",
        {"a": {"b": _T1, "c": _T2}, "d": {"e": _T3}},
        {"a.b": _T1, "a.c": _T2, "d.e": _T3},
    ),
    (
        "mixed_types",
        (_T1, _T2, {"a": _T2}, {"b": [_T1, {"c": _T3}]}),
        {
            "output_0": _T1,
            "output_1": _T2,
            "output_2.a": _T2,
            "output_3.b.output_0": _T1,
            "output_3.b.output_1.c": _T3,
        },
    ),
)


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
      list(
          dict(
              testcase_name=f"_in_{input_case}_out_{output_case}",
              old_input_sig=old_input_sig,
              new_input_sig=new_input_sig,
              old_output_sig=old_output_sig,
              new_output_sig=new_output_sig,
          )
          for (
              (input_case, old_input_sig, new_input_sig),
              (output_case, old_output_sig, new_output_sig),
          ) in itertools.product(_INPUT_SIGNATURES, _OUTPUT_SIGNATURES)
      )
  )
  def test_to_keyword_only_fn(
      self,
      old_input_sig,
      new_input_sig,
      old_output_sig,
      new_output_sig,
  ):

    @tf.function(autograph=False)
    def f(*args, **kwargs):
      del args, kwargs
      return jax_tree_util.tree_map(
          lambda spec: tf.zeros(shape=spec.shape, dtype=spec.dtype),
          old_output_sig,
      )

    args, kwargs = old_input_sig
    cf = f.get_concrete_function(*args, **kwargs)
    new_cf = tf_obm.to_keyword_only_fn(cf)

    def is_spec_equiv(a, b):
      self.assertEqual(a.shape, b.shape)
      self.assertEqual(a.dtype, b.dtype)
      return True

    self.assertTreeEquiv(
        new_cf.structured_input_signature,
        new_input_sig,
        is_spec_equiv,
    )
    self.assertTreeEquiv(
        tf_obm.get_output_signature(new_cf),
        new_output_sig,
        is_spec_equiv,
    )

  def test_to_keyword_only_fn_fails_with_unregistered_dataclass_input(self):
    @tf.function(autograph=False)
    def f(*args, **kwargs):
      del args, kwargs
      return ()

    args, kwargs = ((_Dataclass(a=_T1, b=_T2),), {})
    with self.assertRaises(TypeError):
      f.get_concrete_function(*args, **kwargs)

  def test_to_keyword_only_fn_fails_with_unregistered_dataclass_output(self):
    output_sig = _Dataclass(a=_T1, b=_T2)

    @tf.function(autograph=False)
    def f():
      return jax_tree_util.tree_map(
          lambda spec: tf.zeros(shape=spec.shape, dtype=spec.dtype),
          output_sig,
      )

    with self.assertRaises(AttributeError):
      f.get_concrete_function()

  @parameterized.named_parameters(
      dict(
          testcase_name="_allows_mixed_types",
          get_return_value=lambda x, y: (x, {"a": y}),
      ),
      dict(
          testcase_name="_allows_nested_dicts",
          get_return_value=lambda x, y: {
              "sc_features": {
                  "client_sequential": x,
                  "nested": {"client_platform": y},
              }
          },
      ),
  )
  def test_to_keyword_only_return_value_discrepancies_with_tfsm(
      self,
      get_return_value,
  ):
    @tf.function(autograph=False)
    def f(x, y):
      return get_return_value(x, y)

    cf = f.get_concrete_function(_T1, _T2)

    tf_obm.save_tf_concrete_functions(
        self.create_tempdir().full_path,
        {"f": tf_obm.to_keyword_only_fn(cf)},
    )

    with self.assertRaisesRegex(ValueError, "Got a non-Tensor value"):
      tf.saved_model.save(
          f,
          self.create_tempdir().full_path,
          signatures={"f": cf},
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

    pre_processor = tf_obm.tf_concrete_function_name_to_obm_function(
        pre_processor_name_in_tf, fn=tf_pre_processor
    )
    post_processor = tf_obm.tf_concrete_function_name_to_obm_function(
        post_processor_name_in_tf, fn=tf_post_processor
    )
    saved_model_rel_path = "tf_saved_model/"
    saved_model_abs_path = os.path.join(save_dir_path, saved_model_rel_path)
    tf_global_supplemental = tf_obm.tf_saved_model_as_obm_supplemental(
        saved_model_rel_path
    )
    tf_obm.save_tf_concrete_functions(
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
            supplementals={
                tf_obm.TF_SAVED_MODEL_SUPPLEMENTAL_NAME: obm.GlobalSupplemental(
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
            tf_obm.TF_CONCRETE_FUNCTION_HANDLE_MIME_TYPE
        ),
        "__TF_CONCRETE_FUNCTION_HANDLE_VERSION__": (
            tf_obm.TF_CONCRETE_FUNCTION_HANDLE_VERSION
        ),
        "__TF_SAVED_MODEL_SUPPLEMENTAL_NAME__": (
            tf_obm.TF_SAVED_MODEL_SUPPLEMENTAL_NAME
        ),
        "__SAVED_MODEL_PATH__": saved_model_rel_path,
        "__SAVED_MODEL_MIME_TYPE__": tf_obm.SAVED_MODEL_MIME_TYPE,
        "__SAVED_MODEL_VERSION__": tf_obm.SAVED_MODEL_VERSION,
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
        input_names: "a"
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
        input_names: "a"
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
