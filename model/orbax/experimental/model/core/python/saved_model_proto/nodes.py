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

"""Defines NodeDefs that may appear in the exported model."""

from collections.abc import Iterable
import dataclasses
import enum
from typing import Optional, Sequence

from orbax.experimental.model.core.protos.saved_model import attr_value_pb2
from orbax.experimental.model.core.protos.saved_model import node_def_pb2
from orbax.experimental.model.core.protos.saved_model import tensor_shape_pb2
from orbax.experimental.model.core.protos.saved_model import types_pb2
from orbax.experimental.model.core.python import signature
from orbax.experimental.model.core.python.saved_model_proto import tensor_proto
from orbax.experimental.model.core.python.signature import DType
from orbax.experimental.model.core.python.util import compat
from orbax.experimental.model.core.python.util import naming

Signature = signature.Signature
Shape = signature.Shape
shape_to_proto = tensor_proto.shape_to_proto


# The input/output names are different when in a FunctionDef than a GraphDef.
# In GraphDef, the names are always in the format "name:idx" while in a
# FunctionDef the name may be in the form of "node:out:idx"
class NodeContainer(enum.Enum):
  FUNCTION_DEF = enum.auto()
  GRAPH_DEF = enum.auto()


class OpType(enum.StrEnum):
  """Limited set of supported op types."""

  NOOP = 'NoOp'
  IDENTITY = 'Identity'
  CAST = 'Cast'
  READVARIABLEOP = 'ReadVariableOp'
  XLACALLMODULE = 'XlaCallModule'
  XLASHARDING = 'XlaSharding'
  PREVENTGRADIENT = 'PreventGradient'
  VARHANDLEOP = 'VarHandleOp'
  CONST = 'Const'
  PLACEHOLDER = 'Placeholder'
  STATEFULPARTITIONEDCALL = 'StatefulPartitionedCall'
  ASSIGNVARIABLEOP = 'AssignVariableOp'
  RESTOREV2 = 'RestoreV2'


# TODO(b/328687522): Use op registration for better input/output
# validation and attribute generation.
@dataclasses.dataclass(frozen=True, kw_only=True)
class Node:
  """Represents an operation in the GraphDef or FunctionDef.

  Attributes:
    op: The op type.
    outputs: The names of the outputs.
    name: Name of this node. Should be unique within the same context (GraphDef
      or FunctionDef).
    inputs: List of inputs for this node.
    control_inputs: List of names of nodes that must be executed before this
      node.
    device: String name of device to use to execute this op.
    output: (Overridden property) The name of the output(s) of this node.
  """

  op: OpType
  outputs: Sequence[str]
  name: str | None = None
  inputs: list[str] = dataclasses.field(default_factory=list)
  control_inputs: list[str] = dataclasses.field(default_factory=list)
  device: str | None = None
  _container: NodeContainer | None = None

  def check_name_set(self):
    if self.name is None:
      raise ValueError('Node name must be set.')
    if not naming.is_valid_node_name(self.name):
      raise ValueError(
          f'Provided name {self.name} does not fit the regex'
          ' ^[A-Za-z0-9.][A-Za-z0-9_.\\/>-]*$'
      )

  def set_name(self, name: str):
    object.__setattr__(self, 'name', name)

  def set_container(self, container: NodeContainer):
    object.__setattr__(self, '_container', container)

  def get_output(self, output_idx: int, list_idx: Optional[int] = None) -> str:  # pylint: disable=missing-function-docstring
    self.check_name_set()
    if self._container is NodeContainer.FUNCTION_DEF:
      # FunctionDef node input/output names must be {node:out:idx} .
      # See specification in
      # 
      return naming.to_slice_notation(
          f'{self.name}:{self.outputs[output_idx]}',
          list_idx)
    elif self._container is NodeContainer.GRAPH_DEF:
      # GraphDef node input/output names must be in {node:idx} or {node}
      assert list_idx is None
      return naming.to_slice_notation(self.name, output_idx)
    else:  # Container is not set.
      raise ValueError(
          'Cannot retrieve output from a node that has not been added to a node'
          ' builder.'
      )

  def to_node_def(self):
    self.check_name_set()
    node_def = node_def_pb2.NodeDef()
    node_def.name = self.name
    node_def.op = self.op
    node_def.input.extend(self.inputs)
    node_def.input.extend(self.control_inputs)
    if self.device:
      node_def.device = self.device
    return node_def


@dataclasses.dataclass(frozen=True, kw_only=True)
class NoOp(Node):
  op: OpType = OpType.NOOP
  outputs: Sequence[str] = ()


@dataclasses.dataclass(frozen=True, kw_only=True)
class Identity(Node):
  """Represents an Identity op."""

  op: OpType = OpType.IDENTITY
  outputs: Sequence[str] = ('output',)
  dtype: types_pb2.DataType

  def to_node_def(self):
    node_def = super().to_node_def()
    node_def.attr['T'].type = self.dtype
    return node_def


@dataclasses.dataclass(frozen=True, kw_only=True)
class Cast(Node):
  """Represents a Cast op."""

  op: OpType = OpType.CAST
  outputs: Sequence[str] = ('y',)
  src_dtype: types_pb2.DataType
  dst_dtype: types_pb2.DataType
  truncate: bool = False

  def to_node_def(self):
    node_def = super().to_node_def()
    node_def.attr['SrcT'].type = self.src_dtype
    node_def.attr['DstT'].type = self.dst_dtype
    node_def.attr['Truncate'].b = self.truncate
    return node_def


@dataclasses.dataclass(frozen=True, kw_only=True)
class ReadVariableOp(Node):
  """Represents a ReadVariableOp."""

  op: OpType = OpType.READVARIABLEOP
  outputs: Sequence[str] = ('value',)

  dtype: types_pb2.DataType

  def to_node_def(self):
    node_def = super().to_node_def()
    node_def.attr['dtype'].type = self.dtype
    return node_def


# pylint: disable=invalid-name
# pylint: disable=g-missing-from-attributes
@dataclasses.dataclass(frozen=True, kw_only=True)
class XlaCallModule(Node):
  """Represents an XlaCallModuleoOp.

  For documentation about this op, see
  

  Several attributes are hardcoded in this implementation.

  Attributes:
    inputs: List of names to be passed as inputs.
    module: A serialized computation, a text or bytecode representation of an
      mlir.Module. The return type must be a tuple if and only if the `Sout` is
      a list with 0 or more than 1 elements. The length of `Tout` and `Sout`
      must match. This op always returns a tuple of results, even if the module
      returns a single result.
    flat_input_specs: Flat input TensorSpecs.
    flat_output_specs: Flat output TensorSpecs.
    version: Tracks changes the semantics of the op, to support backwards
      compatibility.
    platforms: the list of platforms supported by `module`.
  """

  op: OpType = OpType.XLACALLMODULE
  outputs: Sequence[str] = ('output',)
  inputs: Sequence[str]
  version: int
  module: bytes
  Sout: Sequence[Shape]
  Tout: Sequence[DType]
  Tin: Sequence[DType]
  platforms: Sequence[str] = ()
  has_token_input_output: bool = False
  disabled_checks: Sequence[str] = ()

  def to_node_def(self):
    node_def = super().to_node_def()
    if len(self.inputs) != len(self.Tin):
      raise ValueError(
          '`inputs` must have the same length as `flat_input_specs`.'
      )

    node_def.attr['version'].i = self.version
    node_def.attr['module'].s = self.module

    node_def.attr['Sout'].list.shape.extend(
        [shape_to_proto(s) for s in self.Sout])
    node_def.attr['Tout'].list.type.extend(self.Tout)
    node_def.attr['Tin'].list.type.extend(self.Tin)

    platforms = self.platforms
    node_def.attr['platforms'].list.s.extend(
        compat.as_bytes(s) for s in platforms)

    # Some of the following attributes are hardcoded to be empty, may
    # need to modify for more complex functionality.

    # Copy the empty list to ensure that it passes the HasField('list') check.
    # This could maybe be removed but further testing is needed.
    empty_list = attr_value_pb2.AttrValue.ListValue()
    node_def.attr['dim_args_spec'].list.MergeFrom(empty_list)
    node_def.attr['disabled_checks'].list.s.extend(
        compat.as_bytes(s) for s in self.disabled_checks)
    node_def.attr['function_list'].list.MergeFrom(empty_list)
    node_def.attr['has_token_input_output'].b = self.has_token_input_output

    return node_def


# pylint: enable=g-missing-from-attributes
# pylint: enable=invalid-name
@dataclasses.dataclass(frozen=True, kw_only=True)
class XlaSharding(Node):
  """Represents an XlaShardingOp.

  An op which shards the input based on the given sharding attribute. It can
  selectively annotate a subset of tensor dimensions by skipping
  unspecified_dims, and the sharding annotation should be replicated in those
  dims.

  This is currently hardcoded to do nothing.
  """

  op: OpType = OpType.XLASHARDING
  outputs: Sequence[str] = ('output',)
  dtype: types_pb2.DataType

  def to_node_def(self):
    node_def = super().to_node_def()
    node_def.attr['T'].type = self.dtype
    node_def.attr['_XlaSharding'].s = b''
    node_def.attr['sharding'].s = b''
    empty_list = attr_value_pb2.AttrValue.ListValue()
    node_def.attr['unspecified_dims'].list.MergeFrom(empty_list)
    return node_def


@dataclasses.dataclass(frozen=True, kw_only=True)
class PreventGradient(Node):
  """Adds a PreventGradient op that raises an error if a gradient is computed."""

  op: OpType = OpType.PREVENTGRADIENT
  outputs: Sequence[str] = ('output',)
  dtype: types_pb2.DataType
  message: str

  def to_node_def(self):
    node_def = super().to_node_def()
    node_def.attr['T'].type = self.dtype
    node_def.attr['message'].s = compat.as_bytes(self.message)
    return node_def


@dataclasses.dataclass(frozen=True, kw_only=True)
class VarHandleOp(Node):
  """Represents a VarHandleOp Node."""

  op: OpType = OpType.VARHANDLEOP
  outputs: Sequence[str] = ('resource',)
  dtype: types_pb2.DataType
  shape: signature.Shape

  def to_node_def(self):
    node_def = super().to_node_def()
    node_def.attr['dtype'].type = self.dtype
    node_def.attr['shape'].shape.CopyFrom(shape_to_proto(self.shape))
    node_def.attr['_output_shapes'].list.shape.append(shape_to_proto(tuple()))
    node_def.attr['debug_name'].s = compat.as_bytes(self.name)
    node_def.attr['shared_name'].s = compat.as_bytes(self.name)
    return node_def


@dataclasses.dataclass(frozen=True, kw_only=True)
class StringConst(Node):
  """Represents a Const Node."""

  op: OpType = OpType.CONST
  outputs: Sequence[str] = ('output',)
  shape: signature.Shape
  value: list[bytes]

  def to_node_def(self):
    node_def = super().to_node_def()
    node_def.attr['dtype'].type = types_pb2.DataType.DT_STRING
    shape_proto = shape_to_proto(self.shape)
    node_def.attr['_output_shapes'].list.shape.append(shape_proto)
    node_def.attr['value'].tensor.CopyFrom(
        tensor_proto.string_list_to_proto(self.value)
    )
    return node_def


@dataclasses.dataclass(frozen=True, kw_only=True)
class Placeholder(Node):
  """Represents a Placeholder Node."""

  op: OpType = OpType.PLACEHOLDER
  outputs: Sequence[str] = ('output',)
  dtype: types_pb2.DataType
  shape: signature.Shape

  def to_node_def(self):
    node_def = super().to_node_def()
    node_def.attr['dtype'].type = self.dtype
    shape_proto = shape_to_proto(self.shape)
    node_def.attr['shape'].shape.CopyFrom(shape_proto)
    node_def.attr['_output_shapes'].list.shape.append(shape_proto)
    return node_def


@dataclasses.dataclass(frozen=True, kw_only=True)
class StatefulPartitionedCall(Node):
  """Represents a StatefulPartitionedCall Node."""

  op: OpType = OpType.STATEFULPARTITIONEDCALL
  outputs: Sequence[str] = ('output',)
  function_name: str
  flat_input_specs: Iterable[signature.TensorSpec]
  flat_output_specs: Iterable[signature.TensorSpec]

  def to_node_def(self):
    node_def = super().to_node_def()
    node_def.attr['Tin'].list.type.extend(
        [x.dtype for x in self.flat_input_specs]
    )
    node_def.attr['Tout'].list.type.extend(
        [x.dtype for x in self.flat_output_specs]
    )

    node_def.attr['_output_shapes'].list.shape.extend(
        shape_to_proto(x.shape) for x in self.flat_output_specs
    )

    node_def.attr['f'].func.name = self.function_name
    node_def.attr['_read_only_resource_inputs'].list.i.extend([
        i
        for i, x in enumerate(self.flat_input_specs)
        if x.dtype == types_pb2.DataType.DT_RESOURCE
    ])
    return node_def


@dataclasses.dataclass(frozen=True, kw_only=True)
class AssignVariableOp(Node):
  """Represents an AssignVariableOp Node."""

  op: OpType = OpType.ASSIGNVARIABLEOP
  outputs: Sequence[str] = ()
  dtype: types_pb2.DataType

  def to_node_def(self):
    node_def = super().to_node_def()
    node_def.attr['dtype'].type = self.dtype
    node_def.attr['_output_shapes'].list.shape.append(shape_to_proto(tuple()))
    node_def.attr['_has_manual_control_dependencies'].b = True
    return node_def


@dataclasses.dataclass(frozen=True, kw_only=True)
class RestoreV2(Node):
  """Represents a RestoreV2 Node."""

  op: OpType = OpType.RESTOREV2
  outputs: Sequence[str] = ('tensors',)
  dtypes: list[types_pb2.DataType]

  def to_node_def(self):
    node_def = super().to_node_def()
    unknown_rank_shape = tensor_shape_pb2.TensorShapeProto(unknown_rank=True)
    node_def.attr['_output_shapes'].list.shape.extend(
        [unknown_rank_shape] * len(self.dtypes)
    )
    node_def.attr['dtypes'].list.type.extend(self.dtypes)
    return node_def


@dataclasses.dataclass(frozen=True, kw_only=True)
class NodeDefWrapper(Node):  # pylint: disable=missing-class-docstring
  node_def: node_def_pb2.NodeDef

  def __init__(self, node_def: node_def_pb2.NodeDef, outputs):
    super().__init__(op=OpType(node_def.op), outputs=outputs)
    self.node_def = node_def

  def set_name(self, name) -> None:
    super().set_name(name)
    self.node_def.name = name

  def to_node_def(self):
    return self.node_def
