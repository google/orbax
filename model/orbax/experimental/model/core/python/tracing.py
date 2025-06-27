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

"""A simple tracing system."""

# pylint: disable=g-importing-member
from dataclasses import dataclass
from dataclasses import field
import threading
from typing import Any, List, Sequence, Tuple
from typing import Callable
from typing import Protocol

from orbax.experimental.model.core.python.concrete_function import ConcreteFunction
from orbax.experimental.model.core.python.concrete_function import is_tree_of_vars
from orbax.experimental.model.core.python.concrete_function import partial
from orbax.experimental.model.core.python.concrete_function import TreeOfVars
from orbax.experimental.model.core.python.concrete_function import vars_to_spec
from orbax.experimental.model.core.python.signature import DType
from orbax.experimental.model.core.python.signature import Shape
from orbax.experimental.model.core.python.signature import Signature
from orbax.experimental.model.core.python.signature import TensorSpec
from orbax.experimental.model.core.python.signature import TreeOfTensorSpecs


@dataclass
class SymbolicTensor:

  spec: TensorSpec

  @property
  def shape(self) -> Shape:
    return self.spec.shape

  @property
  def dtype(self) -> DType:
    return self.spec.dtype


@dataclass
class Placeholder(SymbolicTensor):

  idx: int


@dataclass
class GraphEdgeTensor(SymbolicTensor):
  """A symbolic tensor representing a graph edge.

  Note that an "input edge" (an "edge" whose destination is a placeholder) is
  not
  considered a graph edge (i.e. graph edges are only "internal edges" and
  "output edges").

  Attributes:
    graph_edge: the name of the graph edge, in the format
      `"source_node_name:output_index"`, such as `"Add_Node_32:0"`.
  """

  graph_edge: str


# TODO(b/329305005): Support nested structures.
TreeOfSymbolicTensors = SymbolicTensor
TreeOfPlaceholders = Placeholder
TreeOfGraphEdgeTensors = GraphEdgeTensor


def is_tree_of_symbolic_tensors(x: Any) -> bool:
  return isinstance(x, TreeOfSymbolicTensors)


def is_tree_of_placeholders(x: Any) -> bool:
  return isinstance(x, TreeOfPlaceholders)


def symbolic_tensors_to_spec(
    tree: TreeOfSymbolicTensors,
) -> TreeOfTensorSpecs:
  assert isinstance(tree, SymbolicTensor)
  tensor: SymbolicTensor = tree
  return tensor.spec


@dataclass
class Node:

  input_: Sequence[TreeOfSymbolicTensors]
  name: str
  op: ConcreteFunction


@dataclass
class Graph:

  nodes: List[Node] = field(default_factory=list)


@dataclass
class TracingContext:
  """The context needed for tracing."""

  graph: Graph = field(default_factory=Graph)
  next_placeholder_idx: int = 0
  next_node_id: int = 0

  def get_and_increase_next_placeholder_idx(self, delta: int = 1) -> int:
    result = self.next_placeholder_idx
    self.next_placeholder_idx += delta
    return result

  def get_and_increase_next_node_id(self, delta: int = 1) -> int:
    result = self.next_node_id
    self.next_node_id += delta
    return result


# The tracing-context stack, from the outer-most to the inner-most
_tracing_contexts = threading.local()
_tracing_contexts._value: List[TracingContext] = []


def is_tracing() -> bool:
  return bool(_tracing_contexts._value)  # pylint: disable=protected-access


def assert_tracing() -> TracingContext:
  """Asserts that the tracing-context stack is not empty.

  Returns:
    The inner-most tracing context.
  """
  assert is_tracing()
  return _tracing_contexts._value[-1]  # pylint: disable=protected-access


def start_tracing() -> None:
  _tracing_contexts._value.append(TracingContext())  # pylint: disable=protected-access


def finish_tracing() -> Graph:
  assert_tracing()
  ctx = _tracing_contexts._value.pop()  # pylint: disable=protected-access
  return ctx.graph


def _create_placeholders_from_signature(
    sig: Signature,
) -> Tuple[TreeOfPlaceholders, ...]:
  ctx = assert_tracing()
  next_idx = ctx.get_and_increase_next_placeholder_idx(len(sig))
  placeholders = []
  for spec in sig:
    placeholders.append(Placeholder(spec, idx=next_idx))
    next_idx += 1
  return tuple(placeholders)


def _assert_and_get_single_node(  # pylint: disable=g-doc-args,g-doc-return-or-yield
    graph: Graph, output: Tuple[TreeOfSymbolicTensors, ...]
) -> ConcreteFunction:
  """Asserts the `graph` has only one node, and return its `op`.

  Also asserts that the node's input is in the pattern
  `(Placeholder(0), Placeholder(1), ...)`, and `output` is in the pattern
  `(GraphEdgeTensor("node_name:0"), GraphEdgeTensor("node_name:1"), ...)` (where
  `node_name` stands for `node.name`).
  """
  assert len(graph.nodes) == 1
  node = graph.nodes[0]
  for i, tree in enumerate(node.input_):
    assert isinstance(tree, Placeholder)
    placeholder = tree
    assert placeholder.idx == i
  for i, tree in enumerate(output):
    assert isinstance(tree, GraphEdgeTensor)
    tensor = tree
    assert tensor.graph_edge == node.name + f":{i}"
  return node.op


TracingOutput = Tuple[TreeOfSymbolicTensors, ...]


class Tracable(Protocol):
  """A Python `Callable` that can be used for tracing.

  A `Tracable` is a Python `Callable` with some extra constraints:
  it should be able to accept several `SymbolicTensor`s (as positional
  arguments), and should return a tuple of `SymbolicTensor`s. (Of course,
  per the meaning of "contravariance", a `Tracable` can also choose to accept
  additional input types.)

  Note that a `Tracable` shouldn't assume that the inputs are `Placeholder`s.
  The inputs can be non-`Placeholder` `SymbolicTensor`s
  """

  def __call__(self, *args: TreeOfSymbolicTensors) -> TracingOutput:
    pass


def trace(f: Tracable, input_signature: Signature) -> ConcreteFunction:  # pylint: disable=g-doc-args
  """Creates a `ConcreteFunction` from a `Tracable` by tracing.

  The returned `ConcreteFunction` will have `input_signature` as its input
  signature.
  Its output signature will be determined by the `Tracable`'s output.

  Returns:
    A `ConcreteFunction`.
  """
  start_tracing()
  placeholders: Tuple[TreeOfPlaceholders, ...] = (
      _create_placeholders_from_signature(input_signature)
  )
  output: TracingOutput = f(*placeholders)
  graph = finish_tracing()
  fn = _assert_and_get_single_node(graph, output)
  return fn


def _create_node_output_tensors(
    node_name: str, output_spec: Signature
) -> Tuple[TreeOfGraphEdgeTensors, ...]:
  tensors = []
  for i, tree in enumerate(output_spec):
    assert isinstance(tree, TensorSpec)
    spec = tree
    tensors.append(GraphEdgeTensor(spec, graph_edge=node_name + f":{i}"))
  return tuple(tensors)


def add_node(
    input_: Sequence[TreeOfSymbolicTensors], op: ConcreteFunction
) -> Tuple[TreeOfGraphEdgeTensors, ...]:
  """Adds a node to the current tracing graph.

  Consecutively added nodes will have names `"node0"`, `"node1"`, etc.

  Returns `GraphEdgeTensor`s whose sources will be the new node, and whose specs
  will be given by `op.output_signature`. Note that those `GraphEdgeTensor`s
  will have
  edge names `"new_node:0"`, `"new_node:1"`, etc., so you need to make sure that
  `op` has enough outputs.

  Args:
    input_: the input of the new node.
    op: the op of the new node.

  Returns:
    A pytree of `GraphEdgeTensor`s, with the same tree pattern as
    `op.output_signature`.
  """
  ctx = assert_tracing()
  node_id = ctx.get_and_increase_next_node_id()
  node_name = f"node{node_id}"
  ctx.graph.nodes.append(Node(input_=input_, name=node_name, op=op))
  return _create_node_output_tensors(node_name, op.output_signature)


# TODO(b/332771015): Support variables in arbitrary positions.
def add_variable_support(
    f: Tracable,
) -> Callable[
    [TreeOfSymbolicTensors, TreeOfSymbolicTensors | TreeOfVars], TracingOutput
]:
  """Converts a `Tracable` to a `Tracable` that also accepts `Variable` inputs.

  The result `Tracable` can be used to create another `Tracable` that binds
  some arguments to some variables (i.e. captures those variables) and removes
  the arguments from the input signature. For example:

  ```
  # fn is a Tracable that expects two arguments, both of which need to be
  # tensors.
  def fn(x: SymbolicTensor, y: SymbolicTensor):
    ...

  fn = add_variable_support(fn)
  # fn now can accepts variables.
  v = Variable(...)
  fn2: Tracable = lambda x: fn(x, v)
  # fn2 is a Tracable that captures a variable (v) and expects only one
  # argument.
  ```

  Warning: Only two-argument `Tracable`s are supported, and the result
  `Tracable` only allows variable at the second argument.

  Args:
    f: a `Tracable`.

  Returns:
    a `Tracable` that also accepts `Variable` inputs.
  """

  def result_fn(
      arg0: TreeOfSymbolicTensors,
      arg1: TreeOfVars,
  ) -> TracingOutput:
    assert_tracing()
    if not is_tree_of_symbolic_tensors(arg0):
      raise TypeError(f"`arg0`[={arg0}] must be a tree of symbolic tensors.")

    if is_tree_of_symbolic_tensors(arg1):
      arg1: TreeOfSymbolicTensors
      return f(arg0, arg1)

    if not is_tree_of_vars(arg1):
      raise TypeError(
          f"`arg1`[={arg1}] must be a tree of symbolic tensors or a tree of "
          "variables."
      )

    arg0_spec = symbolic_tensors_to_spec(arg0)
    arg1_spec = vars_to_spec(arg1)
    free_fn = trace(f, (arg0_spec, arg1_spec))
    fn = partial(free_fn, arg1, bind_last_arg=True)
    output = add_node(input_=(arg0,), op=fn)
    return output

  return result_fn
