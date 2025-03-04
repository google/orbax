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

"""Node tests."""

import dataclasses
from typing import Sequence

from orbax.experimental.model.core.python.saved_model_proto import node_builder
from orbax.experimental.model.core.python.saved_model_proto import nodes

from tensorflow.python.util.protobuf import compare
from absl.testing import absltest

Node = nodes.Node
OpType = nodes.OpType


class NodesTest(googletest.TestCase):

  def assertProtoEqual(self, a, b):
    compare.assertProtoEqual(self, a, b)

  def test_node(self):
    node = Node(
        op=OpType.VARHANDLEOP,
        outputs=(),
        inputs=['inp1', 'inp2'],
        control_inputs=['inp3', 'inp4', 'inp5'],
        device='device',
        name='valid_name',
    )
    self.assertProtoEqual(
        '''name: "valid_name"
op: "VarHandleOp"
input: "inp1"
input: "inp2"
input: "inp3"
input: "inp4"
input: "inp5"
device: "device"''',
        node.to_node_def(),
    )

    builder = node_builder.NodeBuilder(in_graph=True)
    builder.add_node(node)
    self.assertEqual(node.get_output(101), 'valid_name:101')

  def test_node_invalid_name(self):
    node = Node(
        op=OpType.VARHANDLEOP,
        outputs=(),
        inputs=['inp1', 'inp2'],
        control_inputs=['inp3', 'inp4', 'inp5'],
        device='device',
    )
    with self.assertRaisesRegex(ValueError, 'Node name must be set'):
      node.to_node_def()

    node.set_name('a name')
    with self.assertRaisesRegex(ValueError, 'name does not fit the regex'):
      node.to_node_def()

  def test_node_output(self):
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class FakeNode(Node):

      outputs: Sequence[str] = ('my_output',)

    node = FakeNode(
        op=OpType.NOOP,
    )

    with self.assertRaisesRegex(ValueError, 'Node name must be set'):
      node.get_output(111)

    node = FakeNode(
        op=OpType.NOOP,
        name='valid_name',
    )
    with self.assertRaisesRegex(ValueError, 'Cannot retrieve output'):
      node.get_output(111)

    builder = node_builder.NodeBuilder(in_graph=True)
    builder.add_node(node)
    self.assertEqual(node.get_output(111), 'valid_name:111')

    builder = node_builder.NodeBuilder(in_graph=False)
    builder.add_node(node)
    self.assertEqual(node.get_output(0, 111), 'valid_name:my_output:111')


if __name__ == '__main__':
  googletest.main()
