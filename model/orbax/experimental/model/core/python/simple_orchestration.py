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

"""Simple orchestration: pre-processor + model-function + post-processor."""

from typing import Any, Sequence, Tuple

from orbax.experimental.model.core.protos import simple_orchestration_pb2
from orbax.experimental.model.core.python import tree_util
from orbax.experimental.model.core.python import unstructured_data
from orbax.experimental.model.core.python.function import ShloTensorSpec
from orbax.experimental.model.core.python.tree_util import Tree
from orbax.experimental.model.core.python.type_proto_util import to_function_signature_proto


CURRENT_SIMPLE_ORCHESTRATION_MIME_TYPE: str = "simple_orchestration"
CURRENT_SIMPLE_ORCHESTRATION_VERSION: str = "0.0.1"


def serialize_proto(
    proto: simple_orchestration_pb2.SimpleOrchestration,
) -> unstructured_data.UnstructuredData:
  return unstructured_data.UnstructuredData(
      inlined_bytes=proto.SerializeToString(),
      mime_type=CURRENT_SIMPLE_ORCHESTRATION_MIME_TYPE,
      version=CURRENT_SIMPLE_ORCHESTRATION_VERSION,
  )


Signature = Tuple[Tree[ShloTensorSpec], Tree[ShloTensorSpec]]


def is_pair(tree: Tree[Any]) -> bool:
  return isinstance(tree, Sequence) and len(tree) == 2


def is_args_kwargs_pattern(tree: Tree[Any]) -> bool:
  return (
      is_pair(tree)
      and isinstance(tree[0], Sequence)
      and isinstance(tree[1], dict)
  )


def calculate_signature(
    *,
    model_function_signature: Signature,
    pre_processor_signature: Signature | None = None,
    post_processor_signature: Signature | None = None,
    num_of_weights: int | None = None,
) -> Signature:
  """Calculates the overall signature of the orchestration pipeline.

  The overall input signature will be the input signature of the
  pre-processor if present, otherwise the input signature of the model function
  minus the weights argument.

  The overall output signature will be the output signature of the
  post-processor if present, otherwise the output signature of the model
  function.

  Args:
    model_function_signature: The signature of the model function.
    pre_processor_signature: (Optional) The signature of the pre-processor
      function.
    post_processor_signature: (Optional) The signature of the post-processor
      function.
    num_of_weights: (Optional) The number of weights arguments in the model
      function. Normally the model function should have only one argument (the
      1st argument) representing the weights PyTree. When this parameter is set,
      we assume that the model function's signature has been flattened (e.g. via
      jax2obm's `flatten_signature`) so that the first `num_of_weights`
      arguments represent the leaves of the weights PyTree.

  Returns:
    A pair. The overall input and output signature of the orchestration
    pipeline.
  """
  if pre_processor_signature is not None:
    input_sig = pre_processor_signature[0]
  else:
    old_input_sig = model_function_signature[0]
    if is_args_kwargs_pattern(old_input_sig):
      pos_args = old_input_sig[0]
      kwargs = old_input_sig[1]
    else:
      pos_args = old_input_sig
      kwargs = None
    if not isinstance(pos_args, Sequence):
      raise ValueError(
          "The positional-arguments part of model_function_signature must be a"
          f" sequence. Got: {pos_args}"
      )
    if num_of_weights is None:
      if len(pos_args) != 2:
        raise ValueError(
            "The positional part of model_function_signature must have exactly"
            f" two elements. Got: {pos_args}"
        )
      # TODO(wangpeng): Check args[0] against the weight-tree's type/signature.
      pos_args = (pos_args[1],)
    else:
      if num_of_weights < 0:
        raise ValueError(
            f"num_of_weights must be non-negative. Got: {num_of_weights}"
        )
      if len(pos_args) < num_of_weights:
        raise ValueError(
            "The positional part of model_function_signature must have at"
            f" least {num_of_weights} elements. Got: {len(pos_args)}"
        )
      # TODO(wangpeng): Check args[:num_of_weights] against the weights' types.
      pos_args = pos_args[num_of_weights:]
    input_sig = (pos_args, kwargs if kwargs is not None else {})

  if post_processor_signature is not None:
    output_sig = post_processor_signature[1]
  else:
    output_sig = model_function_signature[1]

  if num_of_weights is not None:
    # We assume this condition indicates that JAX2OBM's
    # flatten_signature is True, so we need to flatten.
    input_sig = tuple(tree_util.flatten(input_sig))
    output_sig = tuple(tree_util.flatten(output_sig))

  return input_sig, output_sig


def create(
    *,
    signature: Signature,
    model_function_name: str,
    weights_name: str,
    pre_processor_name: str | None = None,
    post_processor_name: str | None = None,
) -> unstructured_data.UnstructuredData:
  """Returns an `UnstructuredData` containing a `SimpleOrchestration` proto.

  The `SimpleOrchestration` proto will be returned as `inlined_bytes`.

  Args:
    signature: The overall (input and output) signature of this orchestration
      pipeline.
    model_function_name: The name of the model function.
    weights_name: The name of the weights.
    pre_processor_name: The name of the pre-processor function.
    post_processor_name: The name of the post-processor function.

  Returns:
    An `UnstructuredData` containing in its `inlined_bytes` field a
    `SimpleOrchestration` proto.
  """
  orch = simple_orchestration_pb2.SimpleOrchestration()
  orch.signature.CopyFrom(to_function_signature_proto(*signature))
  orch.model_function_name = model_function_name
  orch.weights_name = weights_name
  if pre_processor_name is not None:
    orch.pre_processor_name = pre_processor_name
  if post_processor_name is not None:
    orch.post_processor_name = post_processor_name
  return serialize_proto(orch)


ORCHESTRATION_SUPPLEMENTAL_NAME = "orchestration"
