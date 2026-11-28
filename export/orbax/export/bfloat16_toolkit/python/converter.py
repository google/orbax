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

"""SavedModel bfloat16 conversion helper for Orbax and OBM."""

import os
from typing import Any

from orbax.export.bfloat16_toolkit import converter_options_v2_pb2
from orbax.export.bfloat16_toolkit.python import bfloat16
import tensorflow as tf

# pylint: disable=g-direct-tensorflow-import
from tensorflow.core.framework import variable_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.saved_model import fingerprinting_utils
from tensorflow.python.saved_model import loader_impl

# pylint: enable=g-direct-tensorflow-import

_INIT_OP_KEY = '__saved_model_init_op'
_MAIN_OP_KEY = 'saved_model_main_op'


def _copy_dir(src_dir: str, dst_dir: str) -> None:
  """Recursively copies `src_dir` to `dst_dir` via `tf.io.gfile`."""
  src_prefix = src_dir.rstrip('/')
  for dir_name, _, filenames in tf.io.gfile.walk(src_dir):
    rel = dir_name.rstrip('/')[len(src_prefix) :].lstrip('/')
    dst_sub = tf.io.gfile.join(dst_dir, rel) if rel else dst_dir
    tf.io.gfile.makedirs(dst_sub)
    for name in filenames:
      tf.io.gfile.copy(
          tf.io.gfile.join(dir_name, name),
          tf.io.gfile.join(dst_sub, name),
          overwrite=True,
      )


def _iter_graph_nodes(graph_def: Any):
  yield from graph_def.node
  for fn in graph_def.library.function:
    yield from fn.node_def


def _convert_meta_graph(
    meta_graph: meta_graph_pb2.MetaGraphDef,
    options: converter_options_v2_pb2.ConverterOptionsV2,
) -> None:
  """Applies bfloat16 conversion and post-conversion cleanup in-place."""
  bf16_enabled = options.bfloat16_optimization == options.ENABLED or (
      options.bfloat16_optimization == options.DEFAULT
      and not options.disable_default_optimizations
      and not any(
          n.op == 'XlaCallModule'
          for n in _iter_graph_nodes(meta_graph.graph_def)
      )
  )
  if bf16_enabled:
    scope = options.bfloat16_optimization_options.scope
    if scope != converter_options_v2_pb2.BFloat16OptimizationOptions.ALL:
      raise ValueError(
          f'Only ALL scope is supported without XLA target resolution: {scope}.'
      )
    root_info = bfloat16.get_function_info_from_graph_def(meta_graph.graph_def)
    targets = [root_info]
    sig_map = dict(meta_graph.signature_def)
    for target in targets:
      sig_map = bfloat16.apply_bfloat16_optimization_v2(
          options.bfloat16_optimization_options,
          target,
          sig_map,
          options.internal_options,
      )
    if targets:
      meta_graph.signature_def.clear()
      for k, v in sig_map.items():
        meta_graph.signature_def[k].CopyFrom(v)
      reg_grads = list(meta_graph.graph_def.library.registered_gradients)
      meta_graph.graph_def.CopyFrom(
          bfloat16.update_graph_def_using_function_info(root_info)
      )
      meta_graph.graph_def.library.registered_gradients.extend(reg_grads)
      fn_names = {
          f.signature.name for f in meta_graph.graph_def.library.function
      }
      aliases = dict(meta_graph.meta_info_def.function_aliases)
      meta_graph.meta_info_def.function_aliases.clear()
      for fn_name, alias in aliases.items():
        opt = f'{fn_name}_optim' if f'{fn_name}_optim' in fn_names else fn_name
        if opt in fn_names:
          meta_graph.meta_info_def.function_aliases[opt] = alias

  restore_op = meta_graph.saver_def.restore_op_name.split(':', 1)[0]
  if restore_op and any(
      n.name == restore_op for n in meta_graph.graph_def.node
  ):
    var_list = meta_graph.collection_def['variables'].bytes_list.value
    del var_list[:]
    seen_vars = set()
    for n in _iter_graph_nodes(meta_graph.graph_def):
      if n.op == 'VarHandleOp' and n.name not in seen_vars:
        seen_vars.add(n.name)
        var_list.append(
            variable_pb2.VariableDef(
                variable_name=f'{n.name}:0',
                initializer_name=restore_op,
                trainable=True,
                is_resource=True,
            ).SerializeToString()
        )
  meta_graph.ClearField('object_graph_def')
  if _INIT_OP_KEY in meta_graph.signature_def:
    init_sig = meta_graph.signature_def.pop(_INIT_OP_KEY)
    main_ops = meta_graph.collection_def[_MAIN_OP_KEY].node_list.value
    if not main_ops and init_sig.outputs[_INIT_OP_KEY].name:
      main_ops.append(init_sig.outputs[_INIT_OP_KEY].name)


def convert_saved_model(
    input_model_dir: str | os.PathLike[str],
    output_model_dir: str | os.PathLike[str],
    options: Any,
) -> None:
  """Loads a SavedModel and converts it to bfloat16 per ConverterOptionsV2."""
  in_dir, out_dir = os.fspath(input_model_dir), os.fspath(output_model_dir)
  if in_dir == out_dir or (
      '://' not in in_dir
      and '://' not in out_dir
      and os.path.realpath(in_dir) == os.path.realpath(out_dir)
  ):
    raise FileExistsError('The input and output path cannot be the same.')

  if not isinstance(options, converter_options_v2_pb2.ConverterOptionsV2):
    raise TypeError(
        f'Expected ConverterOptionsV2 proto, got {type(options).__name__}.'
    )
  opts = converter_options_v2_pb2.ConverterOptionsV2()
  opts.CopyFrom(options)
  if opts.HasField('internal_options'):
    raise ValueError('Internal options must not be set by users.')

  if tf.io.gfile.isdir(out_dir) and tf.io.gfile.listdir(out_dir):
    tf.io.gfile.rmtree(out_dir)
  tf.io.gfile.makedirs(out_dir)

  saved_model = loader_impl.parse_saved_model(in_dir)
  if len(saved_model.meta_graphs) != 1:
    raise ValueError(
        f'Expected exactly 1 MetaGraphDef, got {len(saved_model.meta_graphs)}.'
    )
  if opts.bfloat16_optimization_options.experimental.convert_checkpoint:
    opts.internal_options.input_model_dir = in_dir
    opts.internal_options.output_model_dir = out_dir
  _convert_meta_graph(saved_model.meta_graphs[0], opts)

  for folder in ('assets', 'assets.extra', 'variables'):
    if folder == 'variables' and tf.io.gfile.exists(
        tf.io.gfile.join(out_dir, 'variables', 'variables.index')
    ):
      continue
    src = tf.io.gfile.join(in_dir, folder)
    if tf.io.gfile.exists(src):
      _copy_dir(src, tf.io.gfile.join(out_dir, folder))

  with tf.io.gfile.GFile(
      tf.io.gfile.join(out_dir, tf.saved_model.SAVED_MODEL_FILENAME_PB), 'wb'
  ) as f:
    f.write(saved_model.SerializeToString(deterministic=True))
  fingerprinting_utils.write_fingerprint(out_dir)
