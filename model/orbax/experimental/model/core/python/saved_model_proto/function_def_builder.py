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

"""Constructs a FunctionDef proto from an em.ConcreteFunction."""

from orbax.experimental.model.core.protos.saved_model import function_pb2
from orbax.experimental.model.core.protos.saved_model import types_pb2
from orbax.experimental.model.core.python import signature
from orbax.experimental.model.core.python.concrete_function import ConcreteFunction
from orbax.experimental.model.core.python.saved_model_proto import node_builder
from orbax.experimental.model.core.python.saved_model_proto import nodes as nodes_lib
from orbax.experimental.model.core.python.saved_model_proto.em_function_to_tf_nodes import emit_nodes_for_concrete_function
from orbax.experimental.model.core.python.saved_model_proto.em_function_to_tf_nodes import TfSymbolicTensor
from orbax.experimental.model.core.python.util import naming


ResourceType = types_pb2.DT_RESOURCE


def to_function_def(
    fn: ConcreteFunction, name: str
) -> function_pb2.FunctionDef:
  """Constructs a FunctionDef proto with input signatures."""
  fdef = function_pb2.FunctionDef()
  nodes = node_builder.NodeBuilder(in_graph=False)

  fdef.signature.name = name
  fdef.signature.is_stateful = bool(fn.captured_vars)
  fdef.attr['_XlaMustCompile'].b = True

  # Handle inputs and captured vars.
  arg_index = 0
  module_inputs = []
  module_input_signature: list[signature.TensorSpec] = []
  fdef_control_outputs: list[str] = []
  for n, spec in enumerate(fn.input_signature):
    inp_name = f'input_{n}_arg_{arg_index}'
    fdef.signature.input_arg.add(name=inp_name, type=spec.dtype)
    module_input = inp_name
    module_inputs.append(module_input)
    module_input_signature.append(spec)
    arg_index += 1

  for n, v in enumerate(fn.captured_vars):
    dtype = v.value.spec.dtype
    inp_name = f'captured_var_{n}_arg_{arg_index}'
    fdef.signature.input_arg.add(name=inp_name, type=ResourceType)
    read_var = nodes_lib.ReadVariableOp(
        name=f'read_var_{n}_arg_{arg_index}', dtype=dtype, inputs=[inp_name]
    )
    nodes.add_node(read_var)
    # TODO(b/332769034): We have data-dependency on read_var, so do we
    #   really need to add it to fdef_control_outputs?
    assert read_var.name is not None
    fdef_control_outputs.append(read_var.name)
    module_input = read_var.get_output(0)
    module_inputs.append(module_input)
    module_input_signature.append(v.value.spec)
    arg_index += 1

  call_module_outputs, call_module_control_output = (
      emit_nodes_for_concrete_function(
          nodes,
          fn,
          tuple(
              map(
                  lambda name, spec: TfSymbolicTensor(
                      spec=spec, graph_edge=name
                  ),
                  module_inputs,
                  module_input_signature,
              )
          ),
      )
  )
  call_module_output_names = tuple(v.graph_edge for v in call_module_outputs)

  # Handle control inputs (variable reads and XLACallModule op)
  fdef_control_outputs.append(call_module_control_output)
  control_noop = nodes_lib.NoOp(
      control_inputs=[naming.control_input(out) for out in fdef_control_outputs]
  )
  nodes.add_node(control_noop)
  control_noop_name = naming.control_input(control_noop.name)
  for control_output in fdef_control_outputs:
    fdef.control_ret[control_output] = control_output

  # Run outputs through Identity w/ control input
  for n, spec in enumerate(fn.output_signature):
    control_identity = nodes_lib.Identity(
        inputs=[call_module_output_names[n]],
        dtype=spec.dtype,
        control_inputs=[control_noop_name],
    )
    nodes.add_node(control_identity)

    # Set output of final control identity op as the output of the function.
    fdef.signature.output_arg.add(name=control_identity.name, type=spec.dtype)
    fdef.ret[control_identity.name] = control_identity.get_output(0)

  fdef.node_def.extend(node.to_node_def() for node in nodes.nodes)
  return fdef
