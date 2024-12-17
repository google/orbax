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

"""Base class for constructing a graph of nodes."""

from orbax.experimental.model.core.python.saved_model_proto import nodes
from orbax.experimental.model.core.python.util import object_identity

OpType = nodes.OpType
Node = nodes.Node


class NodeBuilder:
  """Constructs a valid list of nodes.

  Valid meaning: all nodes names are unique and no node appears twice in
  the `.nodes` list.
  """

  def __init__(self, in_graph=True):
    self.nodes: list[Node] = []
    self.unique_nodes = object_identity.ObjectIdentitySet()
    self.nodes_by_name: dict[str, Node] = {}
    # Maps a name used in the graph to the next id to use for that name.
    self._names_in_use = {}
    self._node_container = (
        nodes.NodeContainer.GRAPH_DEF
        if in_graph
        else nodes.NodeContainer.FUNCTION_DEF
    )

  def add_node(self, node: Node, name: str | None = None):
    """Adds a node to the graph, while ensuring the node has a unique name.

    This function may update the node's name if another node with the same name
    exists.

    Args:
      node: `Node` to add.
      name: Requested name. If empty, the default name will be the op name. The
        final node name may change if the name is already used.
    """
    if node in self.unique_nodes:
      raise ValueError(f"Node already exists: {node}")
    name = name or node.name or str(node.op)
    name = self.unique_name(name)
    node.set_name(name)
    node.set_container(self._node_container)
    self.nodes_by_name[name] = node
    self.nodes.append(node)
    self.unique_nodes.add(node)

  def unique_name(self, name, mark_as_used=True) -> str:
    """Return a unique operation name for `name`.

    Implementation forked from Tensorflow's `Graph.unique_name`.

    Note: You rarely need to call `unique_name()` directly.  Most of
    the time you just need to create `with g.name_scope()` blocks to
    generate structured names.

    `unique_name` is used to generate structured names, separated by
    `"/"`, to help identify operations when debugging a graph.
    Operation names are displayed in error messages reported by the
    TensorFlow runtime, and in various visualization tools such as
    TensorBoard.

    If `mark_as_used` is set to `True`, which is the default, a new
    unique name is created and marked as in use. If it's set to `False`,
    the unique name is returned without actually being marked as used.
    This is useful when the caller simply wants to know what the name
    to be created will be.

    Args:
      name: The name for an operation.
      mark_as_used: Whether to mark this name as being used.

    Returns:
      A string to be passed to `create_op()` that will be used
      to name the operation being created.
    """
    # For the sake of checking for names in use, we treat names as case
    # insensitive (e.g. foo = Foo).
    name_key = name.lower()
    i = self._names_in_use.get(name_key, 0)
    # Increment the number for "name_key".
    if mark_as_used:
      self._names_in_use[name_key] = i + 1
    if i > 0:
      base_name_key = name_key
      # Make sure the composed name key is not already used.
      while name_key in self._names_in_use:
        name_key = "%s_%d" % (base_name_key, i)
        i += 1
      # Mark the composed name_key as used in case someone wants
      # to call unique_name("name_1").
      if mark_as_used:
        self._names_in_use[name_key] = 1

      # Return the new name with the original capitalization of the given name.
      name = "%s_%d" % (name, i - 1)
    return name
