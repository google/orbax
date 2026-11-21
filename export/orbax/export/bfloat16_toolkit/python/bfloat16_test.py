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
import os
import pathlib

from absl.testing import absltest
from absl.testing import parameterized
from orbax.export.bfloat16_toolkit import converter_options_pb2
from orbax.export.bfloat16_toolkit import converter_options_v2_pb2
from orbax.export.bfloat16_toolkit.python import bfloat16
from orbax.export.bfloat16_toolkit.python import converter
from orbax.export.bfloat16_toolkit.python import function_tree
import tensorflow as tf

from pybind11_abseil import status

# pylint: disable=g-direct-tensorflow-import
from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import saved_model_pb2

# pylint: enable=g-direct-tensorflow-import


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


def _read_saved_model_proto(model_dir: str) -> saved_model_pb2.SavedModel:
  saved_model_pb = os.path.join(model_dir, "saved_model.pb")
  with tf.io.gfile.GFile(saved_model_pb, "rb") as f:
    return saved_model_pb2.SavedModel.FromString(f.read())


def _library_matmul_dtypes(saved_proto: saved_model_pb2.SavedModel) -> set[int]:
  dtypes = set()
  for func in saved_proto.meta_graphs[0].graph_def.library.function:
    for node in func.node_def:
      if node.op == "MatMul":
        dtypes.add(node.attr["T"].type)
  return dtypes


class BFloat16Test(parameterized.TestCase):

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

  def test_convert_saved_model_multiple_signatures_preserve_signature_e2e(self):
    module = tf.Module()
    module.w = tf.Variable(
        tf.constant(1.5, dtype=tf.float32, shape=[4, 2]), name="w"
    )
    module.scale = tf.Variable(tf.constant(2.0, dtype=tf.float32), name="scale")

    @tf.function(input_signature=[tf.TensorSpec([1, 4], tf.float32, name="x")])
    def serve_linear(x):
      return {"out": tf.matmul(x, module.w)}

    @tf.function(input_signature=[tf.TensorSpec([1, 4], tf.float32, name="x")])
    def serve_scaled(x):
      return {"out": tf.matmul(x, module.w) * module.scale}

    input_dir = self.create_tempdir().full_path
    output_dir = self.create_tempdir().full_path
    tf.saved_model.save(
        module,
        input_dir,
        signatures={
            "serve_linear": serve_linear.get_concrete_function(),
            "serve_scaled": serve_scaled.get_concrete_function(),
        },
    )

    options = converter_options_v2_pb2.ConverterOptionsV2(
        bfloat16_optimization=converter_options_v2_pb2.ConverterOptionsV2.ENABLED,
        bfloat16_optimization_options=converter_options_v2_pb2.BFloat16OptimizationOptions(
            scope=converter_options_v2_pb2.BFloat16OptimizationOptions.ALL,
            preserve_signature=True,
            skip_safety_checks=True,
            experimental=converter_options_v2_pb2.BFloat16OptimizationOptions.Experimental(
                convert_checkpoint=True
            ),
        ),
    )
    converter.convert_saved_model(input_dir, output_dir, options)

    loaded = tf.saved_model.load(output_dir)
    x = tf.ones([1, 4], dtype=tf.float32)
    linear_out = loaded.signatures["serve_linear"](x=x)["out"]
    scaled_out = loaded.signatures["serve_scaled"](x=x)["out"]
    self.assertEqual(linear_out.dtype, tf.float32)
    self.assertEqual(scaled_out.dtype, tf.float32)
    self.assertEqual(linear_out.numpy().tolist(), [[6.0, 6.0]])
    self.assertEqual(scaled_out.numpy().tolist(), [[12.0, 12.0]])

  def test_convert_saved_model_zero_variables_with_convert_checkpoint(self):
    module = tf.Module()

    @tf.function(input_signature=[tf.TensorSpec([2, 3], tf.float32, name="x")])
    def infer(x):
      weights = tf.ones([3, 2], dtype=tf.float32)
      return {"y": tf.matmul(x, weights)}

    input_dir = self.create_tempdir().full_path
    output_dir = self.create_tempdir().full_path
    tf.saved_model.save(
        module,
        input_dir,
        signatures={"serving_default": infer.get_concrete_function()},
    )

    options = converter_options_v2_pb2.ConverterOptionsV2(
        bfloat16_optimization=converter_options_v2_pb2.ConverterOptionsV2.ENABLED,
        bfloat16_optimization_options=converter_options_v2_pb2.BFloat16OptimizationOptions(
            scope=converter_options_v2_pb2.BFloat16OptimizationOptions.ALL,
            preserve_signature=True,
            skip_safety_checks=True,
            experimental=converter_options_v2_pb2.BFloat16OptimizationOptions.Experimental(
                convert_checkpoint=True
            ),
        ),
    )
    converter.convert_saved_model(input_dir, output_dir, options)

    loaded = tf.saved_model.load(output_dir)
    out = loaded.signatures["serving_default"](
        x=tf.constant(2.0, dtype=tf.float32, shape=[2, 3])
    )["y"]
    self.assertEqual(out.dtype, tf.float32)
    self.assertEqual(out.numpy().tolist(), [[6.0, 6.0], [6.0, 6.0]])

  def test_convert_saved_model_disabled_bfloat16_copies_variables_and_assets(
      self,
  ):
    asset_src = os.path.join(self.create_tempdir().full_path, "vocab.txt")
    with tf.io.gfile.GFile(asset_src, "w") as f:
      f.write("alpha\nbeta\n")

    module = tf.Module()
    module.w = tf.Variable(
        tf.constant(3.0, dtype=tf.float32, shape=[2, 2]), name="w"
    )
    module.asset = tf.saved_model.Asset(asset_src)

    @tf.function(input_signature=[tf.TensorSpec([1, 2], tf.float32, name="x")])
    def infer(x):
      return {"y": tf.matmul(x, module.w)}

    input_dir = self.create_tempdir().full_path
    output_dir = self.create_tempdir().full_path
    tf.saved_model.save(
        module,
        input_dir,
        signatures={"serving_default": infer.get_concrete_function()},
    )

    options = converter_options_v2_pb2.ConverterOptionsV2(
        bfloat16_optimization=converter_options_v2_pb2.ConverterOptionsV2.DISABLED,
        bfloat16_optimization_options=converter_options_v2_pb2.BFloat16OptimizationOptions(
            experimental=converter_options_v2_pb2.BFloat16OptimizationOptions.Experimental(
                convert_checkpoint=True
            ),
        ),
    )
    converter.convert_saved_model(input_dir, output_dir, options)

    self.assertTrue(
        tf.io.gfile.exists(os.path.join(output_dir, "fingerprint.pb"))
    )
    copied_asset = os.path.join(output_dir, "assets", "vocab.txt")
    self.assertTrue(tf.io.gfile.exists(copied_asset))
    with tf.io.gfile.GFile(copied_asset, "r") as f:
      self.assertEqual(f.read(), "alpha\nbeta\n")

    ckpt_dtypes = tf.train.load_checkpoint(
        os.path.join(output_dir, "variables", "variables")
    ).get_variable_to_dtype_map()
    self.assertIn(tf.float32, ckpt_dtypes.values())
    self.assertNotIn(tf.bfloat16, ckpt_dtypes.values())

    loaded = tf.saved_model.load(output_dir)
    out = loaded.signatures["serving_default"](
        x=tf.ones([1, 2], dtype=tf.float32)
    )["y"]
    self.assertEqual(out.dtype, tf.float32)
    self.assertEqual(out.numpy().tolist(), [[6.0, 6.0]])

  def test_convert_saved_model_selective_xla_functions_e2e(self):
    module = tf.Module()

    @tf.function(input_signature=[tf.TensorSpec([1, 4], tf.float32, name="x")])
    def fast_fn(x):
      return {"y": tf.matmul(x, tf.ones([4, 2], dtype=tf.float32))}

    @tf.function(input_signature=[tf.TensorSpec([1, 4], tf.float32, name="x")])
    def precise_fn(x):
      return {
          "y": tf.matmul(x, tf.constant(2.0, dtype=tf.float32, shape=[4, 2]))
      }

    @tf.function(jit_compile=True)
    def jit_block(x):
      return tf.matmul(x, tf.constant(3.0, dtype=tf.float32, shape=[4, 2]))

    @tf.function(input_signature=[tf.TensorSpec([1, 4], tf.float32, name="x")])
    def mixed_fn(x):
      jit_part = jit_block(x)
      cpu_part = tf.matmul(x, tf.ones([4, 2], dtype=tf.float32))
      return {"y": jit_part + cpu_part}

    input_dir = self.create_tempdir().full_path
    fast_concrete = fast_fn.get_concrete_function()
    tf.saved_model.save(
        module,
        input_dir,
        signatures={
            "fast_sig": fast_concrete,
            "precise_sig": precise_fn.get_concrete_function(),
            "mixed_sig": mixed_fn.get_concrete_function(),
        },
        options=tf.saved_model.SaveOptions(
            function_aliases={"fast_alias": fast_concrete}
        ),
    )

    with self.subTest("by_signature_name"):
      out_dir = self.create_tempdir().full_path
      opts = converter_options_v2_pb2.ConverterOptionsV2(
          bfloat16_optimization=converter_options_v2_pb2.ConverterOptionsV2.ENABLED,
          bfloat16_optimization_options=converter_options_v2_pb2.BFloat16OptimizationOptions(
              scope=converter_options_v2_pb2.BFloat16OptimizationOptions.DEFAULT,
              skip_safety_checks=True,
          ),
          xla_functions=[
              converter_options_v2_pb2.XlaFunctionSpec(
                  signature_name="fast_sig"
              )
          ],
      )
      converter.convert_saved_model(input_dir, out_dir, opts)
      saved_proto = _read_saved_model_proto(out_dir)
      self.assertEqual(
          _library_matmul_dtypes(saved_proto),
          {types_pb2.DT_BFLOAT16, types_pb2.DT_FLOAT},
      )
      loaded = tf.saved_model.load(out_dir)
      x = tf.ones([1, 4], dtype=tf.float32)
      self.assertEqual(
          loaded.signatures["fast_sig"](x=x)["y"].numpy().tolist(), [[4.0, 4.0]]
      )
      self.assertEqual(
          loaded.signatures["precise_sig"](x=x)["y"].numpy().tolist(),
          [[8.0, 8.0]],
      )

    with self.subTest("by_function_alias"):
      out_dir = self.create_tempdir().full_path
      opts = converter_options_v2_pb2.ConverterOptionsV2(
          bfloat16_optimization=converter_options_v2_pb2.ConverterOptionsV2.ENABLED,
          bfloat16_optimization_options=converter_options_v2_pb2.BFloat16OptimizationOptions(
              scope=converter_options_v2_pb2.BFloat16OptimizationOptions.DEFAULT,
              skip_safety_checks=True,
          ),
          xla_functions=[
              converter_options_v2_pb2.XlaFunctionSpec(
                  function_alias="fast_alias"
              )
          ],
      )
      converter.convert_saved_model(input_dir, out_dir, opts)
      saved_proto = _read_saved_model_proto(out_dir)
      alias_concrete_names = {
          name
          for name, alias in saved_proto.meta_graphs[
              0
          ].meta_info_def.function_aliases.items()
          if alias == "fast_alias"
      }
      self.assertNotEmpty(alias_concrete_names)
      bf16_matmuls = 0
      fp32_matmuls = 0
      for func in saved_proto.meta_graphs[0].graph_def.library.function:
        is_aliased_fn = any(
            func.signature.name.startswith(prefix)
            for prefix in alias_concrete_names
        )
        for node in func.node_def:
          if node.op == "MatMul":
            if is_aliased_fn:
              self.assertEqual(node.attr["T"].type, types_pb2.DT_BFLOAT16)
              bf16_matmuls += 1
            else:
              self.assertEqual(node.attr["T"].type, types_pb2.DT_FLOAT)
              fp32_matmuls += 1
      self.assertGreater(bf16_matmuls, 0)
      self.assertGreater(fp32_matmuls, 0)
      loaded = tf.saved_model.load(out_dir)
      x = tf.ones([1, 4], dtype=tf.float32)
      self.assertEqual(
          loaded.signatures["fast_sig"](x=x)["y"].numpy().tolist(), [[4.0, 4.0]]
      )
      self.assertEqual(
          loaded.signatures["precise_sig"](x=x)["y"].numpy().tolist(),
          [[8.0, 8.0]],
      )

    with self.subTest("by_jit_compile_functions"):
      out_dir = self.create_tempdir().full_path
      opts = converter_options_v2_pb2.ConverterOptionsV2(
          bfloat16_optimization=converter_options_v2_pb2.ConverterOptionsV2.ENABLED,
          bfloat16_optimization_options=converter_options_v2_pb2.BFloat16OptimizationOptions(
              scope=converter_options_v2_pb2.BFloat16OptimizationOptions.DEFAULT,
              skip_safety_checks=True,
          ),
          xla_functions=[
              converter_options_v2_pb2.XlaFunctionSpec(
                  jit_compile_functions=True
              )
          ],
      )
      converter.convert_saved_model(input_dir, out_dir, opts)
      saved_proto = _read_saved_model_proto(out_dir)
      self.assertEqual(
          _library_matmul_dtypes(saved_proto),
          {types_pb2.DT_BFLOAT16, types_pb2.DT_FLOAT},
      )
      loaded = tf.saved_model.load(out_dir)
      out = loaded.signatures["mixed_sig"](x=tf.ones([1, 4], dtype=tf.float32))[
          "y"
      ]
      self.assertEqual(out.numpy().tolist(), [[16.0, 16.0]])

  def test_convert_saved_model_validation_and_overwrite_semantics(self):
    module = tf.Module()

    @tf.function(input_signature=[tf.TensorSpec([1, 2], tf.float32, name="x")])
    def infer(x):
      return {"y": x + 1.0}

    input_dir = self.create_tempdir().full_path
    tf.saved_model.save(
        module,
        input_dir,
        signatures={"serving_default": infer.get_concrete_function()},
    )
    default_opts = converter_options_v2_pb2.ConverterOptionsV2(
        bfloat16_optimization=converter_options_v2_pb2.ConverterOptionsV2.DISABLED
    )

    with self.assertRaises(FileExistsError):
      converter.convert_saved_model(input_dir, input_dir, default_opts)

    with self.assertRaises(TypeError):
      converter.convert_saved_model(
          input_dir,
          self.create_tempdir().full_path,
          options="invalid_options_type",
      )

    user_internal_opts = converter_options_v2_pb2.ConverterOptionsV2(
        internal_options=converter_options_v2_pb2.InternalOptions(
            input_model_dir="should_not_be_set_by_caller"
        )
    )
    with self.assertRaises(ValueError):
      converter.convert_saved_model(
          input_dir, self.create_tempdir().full_path, user_internal_opts
      )

    non_empty_out = self.create_tempdir().full_path
    stale_file = os.path.join(non_empty_out, "stale.txt")
    with tf.io.gfile.GFile(stale_file, "w") as f:
      f.write("stale")

    converter.convert_saved_model(
        pathlib.Path(input_dir),
        pathlib.Path(non_empty_out),
        default_opts,
    )
    self.assertFalse(tf.io.gfile.exists(stale_file))
    self.assertTrue(
        tf.io.gfile.exists(os.path.join(non_empty_out, "saved_model.pb"))
    )

    multi_meta_dir = self.create_tempdir().full_path
    multi_proto = _read_saved_model_proto(input_dir)
    multi_proto.meta_graphs.add().CopyFrom(multi_proto.meta_graphs[0])
    with tf.io.gfile.GFile(
        os.path.join(multi_meta_dir, "saved_model.pb"), "wb"
    ) as f:
      f.write(multi_proto.SerializeToString())
    with self.assertRaisesRegex(ValueError, "Expected exactly 1 MetaGraphDef"):
      converter.convert_saved_model(
          multi_meta_dir, self.create_tempdir().full_path, default_opts
      )

  def test_convert_saved_model_rejects_missing_gpu_or_tpu_scope_functions(self):
    module = tf.Module()

    @tf.function(input_signature=[tf.TensorSpec([1, 2], tf.float32, name="x")])
    def infer(x):
      return {"y": x + 1.0}

    input_dir = self.create_tempdir().full_path
    tf.saved_model.save(
        module,
        input_dir,
        signatures={"serving_default": infer.get_concrete_function()},
    )

    bf16_cls = converter_options_v2_pb2.BFloat16OptimizationOptions
    gpu_opts = converter_options_v2_pb2.ConverterOptionsV2(
        bfloat16_optimization=converter_options_v2_pb2.ConverterOptionsV2.ENABLED,
        bfloat16_optimization_options=bf16_cls(scope=bf16_cls.GPU),
    )
    with self.assertRaisesRegex(ValueError, "Could not find any GPU functions"):
      converter.convert_saved_model(
          input_dir, self.create_tempdir().full_path, gpu_opts
      )

    tpu_opts = converter_options_v2_pb2.ConverterOptionsV2(
        bfloat16_optimization=converter_options_v2_pb2.ConverterOptionsV2.ENABLED,
        bfloat16_optimization_options=bf16_cls(scope=bf16_cls.TPU),
    )
    with self.assertRaisesRegex(ValueError, "Could not find any TPU functions"):
      converter.convert_saved_model(
          input_dir, self.create_tempdir().full_path, tpu_opts
      )

  def test_reshard_saved_model_variables_rejects_dropped_or_empty_keys(self):
    module = tf.Module()
    module.w = tf.Variable(tf.ones([2, 2], dtype=tf.float32), name="w")
    model_dir = self.create_tempdir().full_path
    tf.saved_model.save(module, model_dir)

    with self.assertRaisesRegex(ValueError, "must keep every checkpoint key"):
      converter.reshard_saved_model_variables(model_dir, lambda _: [{}])
    with self.assertRaisesRegex(ValueError, "must keep every checkpoint key"):
      converter.reshard_saved_model_variables(
          model_dir,
          lambda tensors: [{st.checkpoint_key: {} for st in tensors}],
      )


  def test_convert_saved_model_preserves_signature_with_ckpt_and_float_table(
      self,
  ):
    module = tf.Module()
    module.w_used = tf.Variable(
        tf.constant([[1.5, -0.5], [2.0, 3.0]], dtype=tf.float32),
        name="w_used",
    )
    module.w_unused = tf.Variable(
        tf.constant([4.25, -1.75], dtype=tf.float32), name="w_unused"
    )
    module.table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            tf.constant(["scale"]), tf.constant([2.0], dtype=tf.float32)
        ),
        default_value=1.0,
    )

    @tf.function(input_signature=[tf.TensorSpec([2, 2], tf.float32, name="x")])
    def serving(x):
      scale = module.table.lookup(tf.constant("scale"))
      return {"y": tf.matmul(x, module.w_used) * scale + 1.0}

    input_dir = self.create_tempdir().full_path
    output_dir = self.create_tempdir().full_path
    tf.saved_model.save(
        module,
        input_dir,
        signatures={"serving_default": serving.get_concrete_function()},
    )

    bf16_opts_cls = converter_options_v2_pb2.BFloat16OptimizationOptions
    options = converter_options_v2_pb2.ConverterOptionsV2(
        bfloat16_optimization=(
            converter_options_v2_pb2.ConverterOptionsV2.ENABLED
        ),
        bfloat16_optimization_options=bf16_opts_cls(
            scope=bf16_opts_cls.ALL,
            preserve_signature=True,
            skip_safety_checks=True,
            experimental=bf16_opts_cls.Experimental(convert_checkpoint=True),
        ),
    )

    converter.convert_saved_model(input_dir, output_dir, options)

    ckpt = tf.train.load_checkpoint(
        os.path.join(output_dir, "variables", "variables")
    )
    dtype_map = ckpt.get_variable_to_dtype_map()
    self.assertEqual(
        dtype_map["w_used/.ATTRIBUTES/VARIABLE_VALUE"], tf.bfloat16
    )
    self.assertEqual(
        dtype_map["w_unused/.ATTRIBUTES/VARIABLE_VALUE"], tf.float32
    )

    loaded = tf.saved_model.load(output_dir)
    sample_x = tf.constant([[1.0, 2.0], [-1.0, 0.5]], dtype=tf.float32)
    out_y = loaded.signatures["serving_default"](x=sample_x)["y"]
    self.assertEqual(out_y.dtype, tf.float32)
    self.assertEqual(out_y.numpy().tolist(), [[12.0, 12.0], [0.0, 5.0]])

  def test_convert_saved_model_with_save_options_shards_and_updates_fingerprint(
      self,
  ):
    module = tf.Module()
    module.w1 = tf.Variable(
        tf.constant(1.5, dtype=tf.float32, shape=[16, 16]), name="w1"
    )
    module.w2 = tf.Variable(
        tf.constant(2.0, dtype=tf.float32, shape=[16, 16]), name="w2"
    )
    module.unused = tf.Variable(
        tf.constant(3.25, dtype=tf.float32, shape=[8, 8]), name="unused"
    )

    @tf.function(input_signature=[tf.TensorSpec([1, 16], tf.float32, name="x")])
    def infer(x):
      return {"y": tf.matmul(tf.matmul(x, module.w1), module.w2)}

    input_dir = self.create_tempdir().full_path
    output_dir = self.create_tempdir().full_path
    save_options = tf.saved_model.SaveOptions(
        experimental_sharding_callback=(
            tf.train.experimental.MaxShardSizePolicy(max_shard_size=600)
        )
    )
    tf.saved_model.save(
        module,
        input_dir,
        signatures={"serving_default": infer.get_concrete_function()},
        options=save_options,
    )

    options = converter_options_v2_pb2.ConverterOptionsV2(
        bfloat16_optimization=(
            converter_options_v2_pb2.ConverterOptionsV2.ENABLED
        ),
        bfloat16_optimization_options=(
            converter_options_v2_pb2.BFloat16OptimizationOptions(
                scope=converter_options_v2_pb2.BFloat16OptimizationOptions.ALL,
                preserve_signature=True,
                skip_safety_checks=True,
                experimental=(
                    converter_options_v2_pb2.BFloat16OptimizationOptions.Experimental(
                        convert_checkpoint=True
                    )
                ),
            )
        ),
    )
    converter.convert_saved_model(
        input_dir, output_dir, options, save_options=save_options
    )

    variables_prefix = os.path.join(output_dir, "variables", "variables")
    shards = tf.io.gfile.glob(f"{variables_prefix}.data-*")
    self.assertGreaterEqual(len(shards), 2)
    dtype_map = tf.train.load_checkpoint(
        variables_prefix
    ).get_variable_to_dtype_map()
    self.assertEqual(dtype_map["w1/.ATTRIBUTES/VARIABLE_VALUE"], tf.bfloat16)
    self.assertEqual(dtype_map["w2/.ATTRIBUTES/VARIABLE_VALUE"], tf.bfloat16)
    self.assertEqual(dtype_map["unused/.ATTRIBUTES/VARIABLE_VALUE"], tf.float32)

    fingerprint = tf.saved_model.experimental.read_fingerprint(output_dir)
    self.assertNotEqual(fingerprint.saved_model_checksum, 0)

    loaded = tf.saved_model.load(output_dir)
    out = loaded.signatures["serving_default"](
        x=tf.ones([1, 16], dtype=tf.float32)
    )["y"]
    self.assertEqual(out.dtype, tf.float32)
    self.assertEqual(out.numpy().tolist(), [[768.0] * 16])

  def test_convert_saved_model_handles_empty_var_and_sliced_one_shard(self):
    module = tf.Module()
    module.w = tf.Variable(
        tf.constant(2.0, dtype=tf.float32, shape=[4, 4]), name="w"
    )
    module.empty_var = tf.Variable(
        tf.zeros([0, 4], dtype=tf.float32), name="empty_var"
    )

    @tf.function(input_signature=[tf.TensorSpec([1, 4], tf.float32, name="x")])
    def infer(x):
      return {"y": tf.matmul(x, module.w)}

    input_dir = self.create_tempdir().full_path
    output_dir = self.create_tempdir().full_path
    tf.saved_model.save(
        module,
        input_dir,
        signatures={"serving_default": infer.get_concrete_function()},
    )

    def _single_shard_slicing_callback(shardable_tensors):
      single_shard = {}
      for st in shardable_tensors:
        if st.shape.rank and st.shape[0] == 4:
          single_shard[st.checkpoint_key] = {
              "4 4 0,2:0,4": st.tensor[:2, :],
              "4 4 2,2:0,4": st.tensor[2:, :],
          }
        else:
          single_shard[st.checkpoint_key] = {st.slice_spec: st.tensor}
      return [single_shard]

    # Pre-slice into a single shard (`len(old_data_files) == 1` with non-empty
    # slice specs) to verify resharding does NOT skip re-merging sliced single
    # shards.
    converter.reshard_saved_model_variables(
        input_dir, _single_shard_slicing_callback
    )

    options = converter_options_v2_pb2.ConverterOptionsV2(
        bfloat16_optimization=(
            converter_options_v2_pb2.ConverterOptionsV2.ENABLED
        ),
        bfloat16_optimization_options=(
            converter_options_v2_pb2.BFloat16OptimizationOptions(
                scope=converter_options_v2_pb2.BFloat16OptimizationOptions.ALL,
                preserve_signature=True,
                skip_safety_checks=True,
                experimental=(
                    converter_options_v2_pb2.BFloat16OptimizationOptions.Experimental(
                        convert_checkpoint=True
                    )
                ),
            )
        ),
    )
    save_options = tf.saved_model.SaveOptions(
        experimental_sharding_callback=tf.train.experimental.MaxShardSizePolicy(
            max_shard_size=1024 * 1024
        )
    )
    converter.convert_saved_model(
        pathlib.Path(input_dir),
        pathlib.Path(output_dir),
        options,
        save_options=save_options,
    )

    ckpt = tf.train.load_checkpoint(
        os.path.join(output_dir, "variables", "variables")
    )
    shape_map = ckpt.get_variable_to_shape_map()
    self.assertEqual(shape_map["empty_var/.ATTRIBUTES/VARIABLE_VALUE"], [0, 4])
    loaded = tf.saved_model.load(output_dir)
    out = loaded.signatures["serving_default"](
        x=tf.ones([1, 4], dtype=tf.float32)
    )["y"]
    self.assertEqual(out.numpy().tolist(), [[8.0, 8.0, 8.0, 8.0]])

  def test_convert_saved_model_retains_reachable_gradients_and_prunes_unused(
      self,
  ):
    @tf.custom_gradient
    def scaled_square(x):
      def grad(dy):
        return dy * 2.0 * x

      return x * x, grad

    module = tf.Module()
    module.w = tf.Variable(tf.ones([2, 2], dtype=tf.float32), name="w")

    @tf.function(input_signature=[tf.TensorSpec([1, 2], tf.float32, name="x")])
    def infer(x):
      with tf.GradientTape() as tape:
        tape.watch(x)
        y = tf.matmul(scaled_square(x), module.w)
      return {"y": tape.gradient(y, x)}

    input_dir = self.create_tempdir().full_path
    tf.saved_model.save(
        module,
        input_dir,
        signatures={"serving_default": infer.get_concrete_function()},
    )
    in_proto = _read_saved_model_proto(input_dir)
    in_lib = in_proto.meta_graphs[0].graph_def.library
    reg_fn = in_lib.registered_gradients[0].gradient_func
    helper_fn = in_lib.function.add()
    helper_fn.CopyFrom(
        next(f for f in in_lib.function if f.signature.name == reg_fn)
    )
    helper_fn.signature.name = "kept_grad_helper"
    unused_fn = in_lib.function.add()
    unused_fn.CopyFrom(helper_fn)
    unused_fn.signature.name = "unused_grad_helper"
    in_lib.gradient.add(function_name=reg_fn, gradient_func="kept_grad_helper")
    in_lib.gradient.add(
        function_name="unreachable_fn", gradient_func="unused_grad_helper"
    )
    with tf.io.gfile.GFile(
        os.path.join(input_dir, tf.saved_model.SAVED_MODEL_FILENAME_PB), "wb"
    ) as f:
      f.write(in_proto.SerializeToString())

    out_dir = self.create_tempdir().full_path
    opts = converter_options_v2_pb2.ConverterOptionsV2(
        bfloat16_optimization=converter_options_v2_pb2.ConverterOptionsV2.ENABLED,
        bfloat16_optimization_options=converter_options_v2_pb2.BFloat16OptimizationOptions(
            scope=converter_options_v2_pb2.BFloat16OptimizationOptions.ALL,
            preserve_signature=True,
            skip_safety_checks=True,
        ),
    )
    converter.convert_saved_model(input_dir, out_dir, opts)
    out_meta = _read_saved_model_proto(out_dir).meta_graphs[0]
    out_fn_names = {
        f.signature.name for f in out_meta.graph_def.library.function
    }
    self.assertNotEmpty(out_meta.graph_def.library.registered_gradients)
    self.assertEqual(
        [g.gradient_func for g in out_meta.graph_def.library.gradient],
        ["kept_grad_helper"],
    )
    self.assertIn("kept_grad_helper", out_fn_names)
    self.assertNotIn("unused_grad_helper", out_fn_names)
    var_entries = out_meta.collection_def["variables"].bytes_list.value
    self.assertEqual(len(var_entries), len(set(var_entries)))
    loaded = tf.saved_model.load(out_dir)
    self.assertEqual(
        loaded.signatures["serving_default"](
            x=tf.constant([[3.0, 4.0]], dtype=tf.float32)
        )["y"]
        .numpy()
        .tolist(),
        [[12.0, 16.0]],
    )


if __name__ == "__main__":
  absltest.main()
