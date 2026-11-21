# Copyright 2026 The Orbax Authors.
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

import gc

from absl.testing import absltest
from orbax.export.bfloat16_toolkit import converter_options_pb2
from orbax.export.bfloat16_toolkit import converter_options_v2_pb2
from orbax.export.bfloat16_toolkit.python import bfloat16
from orbax.export.bfloat16_toolkit.python import function_tree
import tensorflow as tf

from pybind11_abseil import status


def _build_xla_graph_def(
    *, include_tpu: bool, include_gpu: bool
) -> tf.compat.v1.GraphDef:
  graph_def = tf.compat.v1.GraphDef()
  if include_tpu:
    tpu_fn = graph_def.library.function.add()
    tpu_fn.signature.name = "tpu_fn"
    ordinal = graph_def.node.add(name="ordinal", op="Placeholder")
    ordinal.attr["dtype"].type = tf.int32.as_datatype_enum
    tpu_call = graph_def.node.add(
        name="tpu_call", op="TPUPartitionedCall", input=["ordinal"]
    )
    tpu_call.attr["f"].func.name = "tpu_fn"
    tpu_call.attr["Tin"].list.SetInParent()
    tpu_call.attr["Tout"].list.SetInParent()
  if include_gpu:
    gpu_fn = graph_def.library.function.add()
    gpu_fn.signature.name = "gpu_fn"
    gpu_call = graph_def.node.add(
        name="gpu_call",
        op="StatefulPartitionedCall",
        device="/device:GPU:0",
    )
    gpu_call.attr["f"].func.name = "gpu_fn"
    gpu_call.attr["Tin"].list.SetInParent()
    gpu_call.attr["Tout"].list.SetInParent()
    gpu_call.attr["_XlaMustCompile"].b = True
  return graph_def


class BFloat16Test(absltest.TestCase):

  # Only add a simple test to check the binding can be called.
  def test_apply_bfloat16_optimization_raise_execption(self):
    bfloat16_func = function_tree.FunctionInfo("root")
    options = converter_options_pb2.ConverterOptions()
    with self.assertRaisesRegex(
        status.StatusNotOk,
        "Cannot apply the bfloat16 optimization to the BatchFunction when there"
        " is no BatchFunction in the model. Either add the BatchFunction or"
        " change the BFloat16Scope.",
    ):
      bfloat16.apply_bfloat16_optimization(
          bfloat16.BATCH, options, bfloat16_func
      )

  # Only add a simple test to check the binding can be called.
  def test_apply_bfloat16_optimization_v2(self):
    xla_function_info = function_tree.FunctionInfo("root")
    options = converter_options_v2_pb2.BFloat16OptimizationOptions()
    options.scope = converter_options_v2_pb2.BFloat16OptimizationOptions.TPU
    options.preserve_signature = True
    # The safety check fails in the test environment due to unsupported CPU
    # instructions.
    options.skip_safety_checks = True
    with self.assertRaisesRegex(
        status.StatusNotOk,
        "The preserve_signature flag is only supported for the ALL scope. All"
        " other scopes preserve the signature by default. Set"
        " preserve_signature to false, or change the scope of the bfloat16"
        " optimization.",
    ):
      bfloat16.apply_bfloat16_optimization_v2(options, xla_function_info, {})

  def test_get_function_info_from_graph_def(self):
    graph_def = tf.compat.v1.GraphDef()
    function_info = bfloat16.get_function_info_from_graph_def(graph_def)
    self.assertIsInstance(function_info, function_tree.FunctionInfo)

  def test_bindings_reject_none_function_info(self):
    with self.assertRaises(TypeError):
      bfloat16.apply_bfloat16_optimization(
          bfloat16.ALL, converter_options_pb2.ConverterOptions(), None
      )
    with self.assertRaises(TypeError):
      bfloat16.apply_bfloat16_optimization_v2(
          converter_options_v2_pb2.BFloat16OptimizationOptions(), None
      )
    with self.assertRaises(TypeError):
      bfloat16.update_graph_def_using_function_info(None)
    with self.assertRaises(TypeError):
      bfloat16.get_xla_function_info(None)

  def test_apply_bfloat16_optimization_v2_updates_signature_and_graph_def(self):
    model = tf.Module()
    model.f = tf.function(
        lambda x, y: {"output": tf.matmul(x, y)},
        input_signature=[
            tf.TensorSpec([2, 2], tf.float32, name="x"),
            tf.TensorSpec([2, 2], tf.float32, name="y"),
        ],
    )
    model_dir = self.create_tempdir().full_path
    tf.saved_model.save(model, model_dir)
    meta_graph = tf.__internal__.saved_model.parse_saved_model(
        model_dir
    ).meta_graphs[0]

    root_info = bfloat16.get_function_info_from_graph_def(meta_graph.graph_def)
    options = converter_options_v2_pb2.BFloat16OptimizationOptions(
        scope=converter_options_v2_pb2.BFloat16OptimizationOptions.ALL,
        skip_safety_checks=True,
    )
    updated_signatures = bfloat16.apply_bfloat16_optimization_v2(
        options, root_info, dict(meta_graph.signature_def)
    )
    serving_sig = updated_signatures["serving_default"]
    self.assertEqual(
        serving_sig.inputs["x"].dtype, tf.bfloat16.as_datatype_enum
    )
    self.assertEqual(
        serving_sig.outputs["output"].dtype, tf.bfloat16.as_datatype_enum
    )

    updated_graph_def = bfloat16.update_graph_def_using_function_info(root_info)
    matmul_dtypes = []
    for fn in updated_graph_def.library.function:
      if fn.signature.name.endswith("_optim"):
        for node in fn.node_def:
          if node.op == "MatMul":
            matmul_dtypes.append(node.attr["T"].type)
    self.assertNotEmpty(matmul_dtypes)
    self.assertCountEqual(set(matmul_dtypes), [tf.bfloat16.as_datatype_enum])

    del root_info
    gc.collect()

  def test_get_xla_function_info_inspects_gpu_and_tpu_functions(self):
    gpu_graph_def = _build_xla_graph_def(include_tpu=False, include_gpu=True)
    gpu_func = bfloat16.get_xla_function_info(
        function_info=bfloat16.get_function_info_from_graph_def(gpu_graph_def)
    )[0]
    gc.collect()
    self.assertEqual(gpu_func.name(), "gpu_fn")
    self.assertTrue(gpu_func.is_top_level_gpu_function())
    self.assertFalse(gpu_func.is_top_level_tpu_function())

    tpu_root = bfloat16.get_function_info_from_graph_def(
        _build_xla_graph_def(include_tpu=True, include_gpu=False)
    )
    children = tpu_root.get_children()
    self.assertLen(children, 1)
    self.assertEqual(children[0].name(), "tpu_fn")
    xla_funcs = bfloat16.get_xla_function_info(tpu_root)
    self.assertLen(xla_funcs, 1)
    self.assertTrue(xla_funcs[0].is_top_level_tpu_function())
    self.assertFalse(xla_funcs[0].is_top_level_gpu_function())


if __name__ == "__main__":
  absltest.main()
