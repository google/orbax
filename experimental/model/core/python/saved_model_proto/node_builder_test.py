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

"""Tests NodeBuilder."""

from orbax.experimental.model.core.protos.saved_model import types_pb2
from orbax.experimental.model.core.python.saved_model_proto import node_builder
from orbax.experimental.model.core.python.saved_model_proto import nodes
from absl.testing import absltest

DataType = types_pb2.DataType


class NodeBuilderTest(googletest.TestCase):

  def test_add_nodes(self):
    builder = node_builder.NodeBuilder()

    handle1 = nodes.VarHandleOp(
        name='variable', dtype=DataType.DT_FLOAT, shape=(5, 10)
    )
    handle2 = nodes.VarHandleOp(
        name='variable', dtype=DataType.DT_FLOAT, shape=(3, 2, 1)
    )
    builder.add_node(handle1)
    builder.add_node(handle2)

    self.assertEqual(handle1.name, 'variable')
    self.assertEqual(handle2.name, 'variable_1')
    self.assertLen(builder.nodes, 2)
    self.assertEqual(builder.nodes_by_name.keys(), {'variable', 'variable_1'})
    self.assertIs(builder.nodes_by_name['variable'], handle1)
    self.assertIs(builder.nodes_by_name['variable_1'], handle2)

    # Test adding duplicate nodes (should fail).
    with self.assertRaisesRegex(ValueError, 'already exists'):
      builder.add_node(handle1)
    self.assertEqual(builder.nodes_by_name.keys(), {'variable', 'variable_1'})

    # Test adding a node with the same existing name (with index)
    handle3 = nodes.VarHandleOp(
        name='variable_1', dtype=DataType.DT_FLOAT, shape=(3, 2, 1)
    )
    builder.add_node(handle3)
    self.assertEqual(handle3.name, 'variable_1_1')
    self.assertEqual(
        builder.nodes_by_name.keys(), {'variable', 'variable_1', 'variable_1_1'}
    )
    self.assertIs(builder.nodes_by_name['variable_1_1'], handle3)


if __name__ == '__main__':
  googletest.main()
