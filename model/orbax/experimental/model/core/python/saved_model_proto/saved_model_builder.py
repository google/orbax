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

"""Constructs a GraphDef proto from an em.function.ConcreteFunction."""

from collections.abc import Mapping
from orbax.experimental.model.core.protos.saved_model import saved_model_pb2
from orbax.experimental.model.core.protos.saved_model import saver_pb2
from orbax.experimental.model.core.protos.saved_model import types_pb2
from orbax.experimental.model.core.python import constants
from orbax.experimental.model.core.python import signature
from orbax.experimental.model.core.python.concrete_function import ConcreteFunction
from orbax.experimental.model.core.python.concrete_function import Variable
from orbax.experimental.model.core.python.saved_model_proto import function_def_builder
from orbax.experimental.model.core.python.saved_model_proto import node_builder
from orbax.experimental.model.core.python.saved_model_proto import nodes as nodes_lib
from orbax.experimental.model.core.python.saved_model_proto import tensor_proto
from orbax.experimental.model.core.python.util import compat
from orbax.experimental.model.core.python.util import naming
from orbax.experimental.model.core.python.util import object_identity


OpType = nodes_lib.OpType
Node = nodes_lib.Node

ResourceType = types_pb2.DT_RESOURCE


class SavedModelProtoBuilder(node_builder.NodeBuilder):
  """Constructs a valid SavedModel/GraphDef."""

  def __init__(self):
    self.proto = saved_model_pb2.SavedModel()
    self.meta_graph = self.proto.meta_graphs.add()
    self.graph = self.meta_graph.graph_def

    self.functions_by_name: dict[str, ConcreteFunction] = {}
    self.variables_by_name: dict[str, Variable] = {}
    self.variable_names = object_identity.ObjectIdentityDictionary()
    self.built = False

    super().__init__()

  def add_variable(self, variable: Variable, name: str):
    if self.built:
      raise ValueError('SavedModelBuilder is already built.')

    # Create and add VarHandleOp to get a unique name for this variable.
    node = nodes_lib.VarHandleOp(
        dtype=variable.value.spec.dtype, shape=variable.value.spec.shape
    )
    self.add_node(node, name)
    assert node.name is not None
    self.variables_by_name[node.name] = variable
    self.variable_names[variable] = name

  def add_function(self, fn: ConcreteFunction, name: str):
    if self.built:
      raise ValueError('SavedModelBuilder is already built.')

    name = self.unique_name(name)
    self.functions_by_name[name] = fn

  def add_function_alias(
      self,
      input_function_aliases: Mapping[str, ConcreteFunction],
      named_functions: Mapping[ConcreteFunction, str],
  ):
    """Adds function aliases to the SavedModel meta graph info.

    Args:
      input_function_aliases: A mapping from user-chosen function alias name to
        the function that runs on TPU.
      named_functions: A mapping from function to the name of the function in
        the graph.
    """
    function_aliases = self.meta_graph.meta_info_def.function_aliases
    for alias, func in input_function_aliases.items():
      function_aliases[named_functions[func]] = alias

  @property
  def saved_model_proto(self) -> saved_model_pb2.SavedModel:
    if not self.built:
      raise ValueError(
          'GraphDef is not built. Please call .build() to finish building'
          ' the GraphDef'
      )
    return self.proto

  def build(self):
    """Finalizes the GraphDef."""
    # Add meta information
    meta_info = self.meta_graph.meta_info_def
    meta_info.tags.append(constants.SERVE_TAG)
    meta_info.tensorflow_version = 'fsm'
    meta_info.stripped_default_attrs = True

    # TODO(b/328687954): Add Stripped Op List when there is an Op Registry.

    # Add all functions to the function library and SignatureDef
    for name, fn in self.functions_by_name.items():
      self._build_function(fn, name)

    # Finish adding Variable RestoreOp and AssignVariableOps
    file_prefix_op = nodes_lib.Placeholder(
        name='saver_filename', dtype=types_pb2.DT_STRING, shape=()
    )
    self.add_node(file_prefix_op)

    num_vars = len(self.variables_by_name)
    shape_and_slices = nodes_lib.StringConst(
        name='RestoreV2/shape_and_slices',
        shape=(num_vars,),
        value=[b''] * num_vars,
    )
    self.add_node(shape_and_slices)

    ordered_names = [name for name in self.variables_by_name]

    tensor_names = nodes_lib.StringConst(
        name='RestoreV2/tensor_names',
        shape=(num_vars,),
        value=[compat.as_bytes(x) for x in ordered_names],
    )
    self.add_node(tensor_names)

    restore_op = nodes_lib.RestoreV2(
        dtypes=[
            self.variables_by_name[k].value.spec.dtype for k in ordered_names
        ],
        inputs=[
            file_prefix_op.get_output(0),
            tensor_names.get_output(0),
            shape_and_slices.get_output(0),
        ],
    )
    self.add_node(restore_op)

    control_inputs = []

    for n, name in enumerate(ordered_names):
      variable = self.variables_by_name[name]
      identity = nodes_lib.Identity(
          dtype=variable.value.spec.dtype,
          inputs=[restore_op.get_output(n)],
          device='/device:CPU:0',
      )
      self.add_node(identity)

      assign_op = nodes_lib.AssignVariableOp(
          dtype=variable.value.spec.dtype,
          inputs=[name, identity.get_output(0)],
          device='/device:CPU:0',
      )
      self.add_node(assign_op)
      control_inputs.append(naming.control_input(assign_op.name))

    restore_output = nodes_lib.Identity(
        dtype=types_pb2.DT_STRING,
        inputs=[file_prefix_op.get_output(0)],
        control_inputs=control_inputs,
        device='/device:CPU:0',
    )
    self.add_node(restore_output)

    # Create fake node for the save tensor (we skip exporting the checkpoint
    # save ops)
    save_op = nodes_lib.Identity(
        name='fake_save_op',
        dtype=types_pb2.DT_STRING,
        inputs=[file_prefix_op.get_output(0)],
        device='/device:CPU:0',
    )
    self.add_node(save_op)

    # Add SaverDef
    saver_def = self.meta_graph.saver_def
    saver_def.filename_tensor_name = naming.to_slice_notation(
        file_prefix_op.name, 0
    )
    saver_def.save_tensor_name = save_op.get_output(0)
    saver_def.restore_op_name = restore_output.name
    saver_def.version = saver_pb2.SaverDef.V2

    # Put all nodes in the GraphDef
    self.graph.node.extend(node.to_node_def() for node in self.nodes)

    self.built = True
    return self.proto

  def _build_function(self, fn: ConcreteFunction, name: str):
    """Adds function to the proto."""
    # Does the following:
    # 1. Adds FunctionDef to proto
    # 2. Adds StatefulPartitionedCall and Placeholder nodes
    # 3. Adds SignatureDef

    fdef = function_def_builder.to_function_def(fn, name)
    self.graph.library.function.append(fdef)

    signature_def = self.meta_graph.signature_def[name]

    fn_input_names = []
    fn_input_specs = []

    # TODO(b/328686396) Stop using "input_{n}" and "output_{n}" for the
    # signature names.
    for n, spec in enumerate(fn.input_signature):
      inp_name = f'input_{n}'
      placeholder_name = f'{name}_input_{n}'
      placeholder = nodes_lib.Placeholder(dtype=spec.dtype, shape=spec.shape)
      self.add_node(placeholder, placeholder_name)
      fn_input_names.append(placeholder.get_output(0))
      fn_input_specs.append(spec)
      signature_input = signature_def.inputs[inp_name]
      signature_input.name = placeholder.get_output(0)
      signature_input.dtype = spec.dtype
      signature_input.tensor_shape.CopyFrom(
          tensor_proto.shape_to_proto(spec.shape)
      )

    for v in fn.captured_vars:
      assert (
          v in self.variable_names
      ), f'Variable {v} was never added to SavedModelBuilder.'
      variable_name = self.variable_names[v]
      fn_input_names.append(variable_name)
      fn_input_specs.append(
          signature.TensorSpec(dtype=ResourceType, shape=v.value.spec.shape)
      )

    call_op = nodes_lib.StatefulPartitionedCall(
        function_name=name,
        flat_input_specs=fn_input_specs,
        flat_output_specs=fn.output_signature,
        inputs=fn_input_names,
    )
    self.add_node(call_op)

    for n, spec in enumerate(fn.output_signature):
      output = call_op.get_output(n)
      signature_output = signature_def.outputs[f'output_{n}']
      signature_output.name = output
      signature_output.dtype = spec.dtype
      signature_output.tensor_shape.CopyFrom(
          tensor_proto.shape_to_proto(spec.shape)
      )
