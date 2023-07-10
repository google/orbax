# Copyright 2023 The Orbax Authors.
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

"""Tests for ProtoCheckpointHandler."""

from absl import flags
from absl.testing import absltest
from etils import epath
import jax
from orbax.checkpoint import proto_checkpoint_handler
from orbax.checkpoint.proto.testing import foo_pb2


# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()

FLAGS = flags.FLAGS

ProtoCheckpointHandler = proto_checkpoint_handler.ProtoCheckpointHandler


class ProtoCheckpointHandlerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(
        self.create_tempdir(name='checkpointing_test').full_path
    )

  def test_save_restore(self):
    item = foo_pb2.Foo(bar='some_str', baz=32)
    handler = ProtoCheckpointHandler(filename='some_filename')
    handler.save(self.directory, item)
    restored = handler.restore(self.directory, foo_pb2.Foo)
    self.assertEqual(item, restored)
    self.assertTrue((self.directory / 'some_filename').exists())
    handler.close()

  def test_restore_with_none_item_throws_error(self):
    item = foo_pb2.Foo(bar='some_str', baz=32)
    handler = ProtoCheckpointHandler(filename='some_filename')
    handler.save(self.directory, item)

    with self.assertRaisesRegex(
        ValueError,
        (
            'Must provide `item` in order to deserialize proto to the correct'
            ' type.'
        ),
    ):
      # Call restore without passing the proto class.
      handler.restore(self.directory)
    handler.close()


if __name__ == '__main__':
  absltest.main()
