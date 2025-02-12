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

"""Emits TF nodes from/for an `em.ConcreteFunction`."""

import math
import operator
import threading
from typing import List, Optional, Sequence, Tuple

from absl import logging
import numpy as np
from orbax.experimental.model.core.protos import xla_data_pb2
from orbax.experimental.model.core.python import shlo_function
from orbax.experimental.model.core.python import tracing
from orbax.experimental.model.core.python import tree_util
from orbax.experimental.model.core.python.concrete_function import ConcreteFunction
from orbax.experimental.model.core.python.function import ShloDimSize
from orbax.experimental.model.core.python.function import ShloDType
from orbax.experimental.model.core.python.function import ShloTensorSpec
from orbax.experimental.model.core.python.saved_model_proto import node_builder
from orbax.experimental.model.core.python.saved_model_proto import nodes as nodes_lib
from orbax.experimental.model.core.python.shlo_function import ShloFunction
from orbax.experimental.model.core.python.signature import DType
from orbax.experimental.model.core.python.signature import float32
from orbax.experimental.model.core.python.signature import float64
from orbax.experimental.model.core.python.signature import int32
from orbax.experimental.model.core.python.signature import Shape
from orbax.experimental.model.core.python.signature import TensorSpec
from orbax.experimental.model.core.python.tree_util import flatten

OpSharding = xla_data_pb2.OpSharding
TfSymbolicTensor = tracing.GraphEdgeTensor
TfVal = TfSymbolicTensor
NodeBuilder = node_builder.NodeBuilder


# The node-builder stack, from the outer-most to the inner-most
_node_builders = threading.local()
_node_builders._value: List[NodeBuilder] = []


def is_emitting() -> bool:
  return bool(_node_builders._value)  # pylint: disable=protected-access


def assert_emitting() -> NodeBuilder:
  """Asserts that the node-builder stack is not empty.

  Returns:
    The inner-most node builder.
  """
  assert is_emitting()
  return _node_builders._value[-1]  # pylint: disable=protected-access


def start_emitting(builder: NodeBuilder) -> None:
  _node_builders._value.append(builder)  # pylint: disable=protected-access


def finish_emitting() -> None:
  assert_emitting()
  _node_builders._value.pop()  # pylint: disable=protected-access


def tf_identity(x: TfVal, name: str) -> TfVal:
  nodes = assert_emitting()
  node = nodes_lib.Identity(
      inputs=[x.graph_edge],
      dtype=x.dtype,
      name=name,
  )
  nodes.add_node(node)
  return TfVal(x.spec, node.get_output(0))


def tf_PreventGradient(input_: TfVal, message: str) -> TfVal:  # pylint: disable=invalid-name
  nodes = assert_emitting()
  node = nodes_lib.PreventGradient(
      inputs=[input_.graph_edge],
      dtype=input_.dtype,
      message=message,
      name="PreventGradient",
  )
  nodes.add_node(node)
  return TfVal(input_.spec, node.get_output(0))


def tf_cast(x: TfVal, dtype: DType) -> TfVal:
  nodes = assert_emitting()
  node = nodes_lib.Cast(
      inputs=[x.graph_edge],
      src_dtype=x.dtype,
      dst_dtype=dtype,
      name="Cast",
  )
  nodes.add_node(node)
  return TfVal(x.spec.with_dtype(dtype), node.get_output(0))


def call_module(  # pylint: disable=missing-function-docstring
    inputs: Sequence[TfVal],
    *,
    version: int,
    module: bytes,
    Sout: Sequence[Shape],
    Tout: Sequence[DType],
    platforms: Sequence[str] = (),
    has_token_input_output: bool = False,
    disabled_checks: Sequence[str] = (),
) -> Tuple[Sequence[TfVal], str]:
  nodes = assert_emitting()
  node = nodes_lib.XlaCallModule(
      inputs=[inp.graph_edge for inp in inputs],
      Tin=[inp.dtype for inp in inputs],
      version=version,
      module=module,
      Sout=Sout,
      Tout=Tout,
      platforms=platforms,
      has_token_input_output=has_token_input_output,
      disabled_checks=disabled_checks,
      name="XlaCallModule",
  )
  nodes.add_node(node)
  control_output = node.name
  assert control_output is not None
  return (
      tuple(
          map(
              lambda i, s, t: TfVal(TensorSpec(s, t), node.get_output(0, i)),
              range(len(Sout)),
              Sout,
              Tout,
          )
      ),
      control_output,
  )


# TODO(b/332765622): Properly implement `xla_sharding` by forking
#   `tensorflow/python/compiler/xla/experimental/xla_sharding.py/Sharding/apply_to_tensor(use_sharding_op=True)`.
def xla_sharding(x: TfVal, proto: OpSharding) -> TfVal:
  del proto
  nodes = assert_emitting()
  node = nodes_lib.XlaSharding(
      inputs=[x.graph_edge],
      dtype=x.dtype,
      name="XlaSharding",
  )
  nodes.add_node(node)
  return TfVal(x.spec, node.get_output(0))


def emit_nodes_for_concrete_function(
    nodes: NodeBuilder, fn: ConcreteFunction, args_tf: Sequence[TfVal]
) -> Tuple[Sequence[TfVal], str]:  # pylint: disable=g-doc-args
  """Emits TF nodes from a `ConcreteFunction`.

  Returns:
    The flattened tuple of data outputs and the control output.
  """

  # We push the NodeBuilder into a background context, so that the emitting
  # experience is closer to using TF Python ops such as tf.cast(...) .
  start_emitting(nodes)
  shlo_fn = fn.base_fn
  supplemental_info = shlo_fn.supplemental_info
  args_flat_tf: Sequence[TfVal] = args_tf
  # Somehow the typechecker needs this superfluous annotation.
  shlo_fn.input_signature: tree_util.Tree[ShloTensorSpec]
  specs_flat_shlo: List[ShloTensorSpec] = flatten(shlo_fn.input_signature)

  args_flat_tf = tuple(
      map(
          lambda i, a, s, p: preprocess_arg_tf(i, a, s.dtype, p),
          range(len(args_flat_tf)),
          args_flat_tf,
          specs_flat_shlo,
          shlo_fn.physical_in_dtypes,
      )
  )

  outs_tf, control_output = _run_shlo_fn_as_tf(
      shlo_fn, args_flat_tf, supplemental_info
  )
  message = "This function does not support gradients."
  # We use PreventGradient, which is propagated through a SavedModel.
  outs_flat_tf = [
      tf_PreventGradient(input_=o, message=message) for o in outs_tf
  ]

  outs_flat_tf = [tf_identity(x, "em_out") for x in outs_flat_tf]
  # TODO(b/329306885): Support structured output.
  out_tf = outs_flat_tf
  finish_emitting()
  return out_tf, control_output


def preprocess_arg_tf(
    arg_idx: int,  # pylint: disable=missing-function-docstring
    arg_tf: TfVal,
    shlo_dtype: ShloDType,
    physical_dtype: Optional[ShloDType],
) -> TfVal:
  assert isinstance(
      shlo_dtype, ShloDType
  ), f"Wrong shlo_dtype type: {type(shlo_dtype)}"
  if physical_dtype is not None:
    assert isinstance(
        physical_dtype, ShloDType
    ), f"Wrong physical_dtype type: {type(physical_dtype)}"
  # May cast the args_flat to SHLO types.
  arg_tf = _convert_tf_value(arg_tf, shlo_dtype, physical_dtype)
  # Name input tensors; do this after we have cast the arguments
  arg_tf = tf_identity(arg_tf, f"em_arg_{arg_idx}")
  return arg_tf


def is_constant_dim(d: ShloDimSize) -> bool:
  # Whether the dimension is a static integer constant.
  try:
    operator.index(d)
    return True
  except Exception:  # pylint: disable=broad-exception-caught
    return False


def _run_shlo_fn_as_tf(
    shlo_fn: ShloFunction,
    args_flat_tf: Sequence[TfVal],
    supplemental_info: Optional[shlo_function.ShloFunctionSupplementalInfo],
) -> Tuple[Sequence[TfVal], str]:  # pylint: disable=g-doc-args
  """Runs the `ShloFunction` as an XlaCallModule TF op (plus some auxiliary TF ops).

  Returns:
    The flattened tuple of data outputs and the control output.
  """
  shlo_fn.input_signature: tree_util.Tree[ShloTensorSpec]
  shlo_fn.output_signature: tree_util.Tree[ShloTensorSpec]

  in_avals: List[ShloTensorSpec] = flatten(shlo_fn.input_signature)

  # TF values may be integer types for float0
  def _convert_value(val, aval, physical_dtype):
    # Check the shape
    assert all(
        d_aval == d_val
        for d_aval, d_val in zip(aval.shape, val.shape)
        if is_constant_dim(d_aval)
    ), (aval, val)
    conversion_dtype = _to_tf_dtype(aval.dtype, physical_dtype)
    if conversion_dtype != aval.dtype:
      return tf_cast(val, conversion_dtype)
    else:
      return val

  args_flat_tf = tuple(
      map(_convert_value, args_flat_tf, in_avals, shlo_fn.physical_in_dtypes)
  )

  out_avals = flatten(shlo_fn.output_signature)

  out_shapes_tf: Tuple[Shape, ...] = tuple(
      tuple(d if is_constant_dim(d) else None for d in out_aval.shape)
      for out_aval in out_avals
  )

  out_types = tuple(
      _to_tf_dtype(out_aval.dtype, physical_dtype)
      for out_aval, physical_dtype in zip(
          out_avals, shlo_fn.physical_out_dtypes
      )
  )

  kept_args_flat_tf = [
      atf
      for i, atf in enumerate(args_flat_tf)
      if i in shlo_fn.module_kept_var_idx
  ]

  version = shlo_fn.calling_convention_version

  max_supported_version = 9

  if version > max_supported_version:
    raise NotImplementedError(
        "XlaCallModule from your ML-Exported-Model (EM) installation "
        "supports up to "
        f"serialization version {max_supported_version} but the serialized "
        f"module needs version {version}. "
        "You should upgrade EM."
    )

  call_module_attrs = dict(
      version=version,
      Tout=out_types,
      Sout=out_shapes_tf,
      # We always set has_token_input_output because it requires real tokens
      # for versions less than 9 and is not used starting with version 9.
      has_token_input_output=False,
  )

  call_module_attrs["platforms"] = tuple(
      p.upper() for p in shlo_fn.lowering_platforms
  )
  if supplemental_info is not None:
    supplemental_info.process_xla_call_module_attrs(version, call_module_attrs)

  if logging.vlog_is_on(3):
    # We already logged the MLIR module when we exported it.
    logging.vlog(3, "XlaCallModule %s", str(call_module_attrs))

  call_module_attrs["module"] = shlo_fn.mlir_module_serialized

  # Apply the shardings on arguments and results for pjit. This is redundant
  # because the mlir_module_text will already contain the shardings, but it
  # makes it easier for tools like the TPU inference converter to see the
  # sharding without digging into the `module` attribute of the `XlaCallModule`
  # op, in the same way as it is done for the legacy jax2tf conversion.
  # Do not apply XlaSharding for REPLICATED, on inputs and outputs.
  # This is an agreed convention, and also improves usability under TF eager.
  # See b/255511660.
  kept_in_shardings = []
  for i in shlo_fn.module_kept_var_idx:
    kept_in_shardings.append(in_avals[i].sharding)
  args_flat_tf = tuple(map(_shard_value, kept_args_flat_tf, kept_in_shardings))
  res, control_output = call_module(args_flat_tf, **call_module_attrs)
  res = list(
      map(lambda val, aval: _shard_value(val, aval.sharding), res, out_avals)
  )
  res = tuple(map(_convert_value, res, out_avals, shlo_fn.physical_out_dtypes))
  return res, control_output


_SHLO_DTYPE_TO_DTYPE: dict[ShloDType, DType] = {
    ShloDType.i32: int32,
    ShloDType.f32: float32,
    ShloDType.f64: float64,
}


def shlo_dtype_to_dtype(dtype: ShloDType) -> DType:
  assert isinstance(
      dtype, ShloDType
  ), f"Wrong shlo_dtype type: type=[{type(dtype)}] repr=[{repr(dtype)}]"
  return _SHLO_DTYPE_TO_DTYPE[dtype]


def _to_tf_dtype(
    shlo_dtype: ShloDType, physical_dtype: Optional[ShloDType]
) -> DType:
  assert isinstance(
      shlo_dtype, ShloDType
  ), f"Wrong shlo_dtype type: {type(shlo_dtype)}"
  if physical_dtype is not None:
    assert isinstance(
        physical_dtype, ShloDType
    ), f"Wrong physical_dtype type: {type(physical_dtype)}"
    shlo_dtype = physical_dtype
  return shlo_dtype_to_dtype(shlo_dtype)


def _convert_tf_value(
    val: TfVal,
    shlo_dtype: ShloDType,
    physical_dtype: Optional[ShloDType],
) -> TfVal:
  conversion_dtype = _to_tf_dtype(shlo_dtype, physical_dtype)
  if conversion_dtype != val.dtype:  # May need to cast for 64-bit values
    return tf_cast(val, conversion_dtype)
  else:
    return val


def _shard_value(val: TfVal, sharding_proto: OpSharding | None) -> TfVal:  # pylint: disable=missing-function-docstring
  if sharding_proto is None:
    return val

  # Tensorflow heavily relies on tile_assignment_devices proto fields specific
  # to V1 sharding format, falling back to this format.
  if (
      not sharding_proto.tile_assignment_devices
      and sharding_proto.iota_reshape_dims
  ):
    tad = list(
        np.arange(math.prod(sharding_proto.tile_assignment_dimensions))
        .reshape(sharding_proto.iota_reshape_dims)
        .transpose(sharding_proto.iota_transpose_perm)
        .flat
    )
  else:
    tad = sharding_proto.tile_assignment_devices  # type: ignore

  # To use xla_sharding.py, we must have a OpSharding.
  xla_sharding_proto: OpSharding = OpSharding(
      type=sharding_proto.type,
      tile_assignment_dimensions=sharding_proto.tile_assignment_dimensions,
      tile_assignment_devices=tad,
      replicate_on_last_tile_dim=sharding_proto.replicate_on_last_tile_dim,
      last_tile_dims=sharding_proto.last_tile_dims,
  )

  return xla_sharding(val, proto=xla_sharding_proto)
