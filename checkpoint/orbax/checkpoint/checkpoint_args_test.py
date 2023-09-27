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

"""Tests CheckpointArg registration."""
import dataclasses
from absl.testing import absltest
from orbax.checkpoint import checkpoint_args
from orbax.checkpoint import standard_checkpoint_handler

StandardCheckpointHandler = (
    standard_checkpoint_handler.StandardCheckpointHandler
)


@checkpoint_args.register_with_handler(StandardCheckpointHandler)
@dataclasses.dataclass
class StandardSaveArgs(checkpoint_args.CheckpointArgs):
  pass


class CheckpointArgsTest(absltest.TestCase):

  def test_get_registered_handler_cls(self):
    self.assertIs(
        checkpoint_args.get_registered_handler_cls(StandardSaveArgs),
        StandardCheckpointHandler,
    )
    self.assertIs(
        checkpoint_args.get_registered_handler_cls(StandardSaveArgs()),
        StandardCheckpointHandler,
    )

  def test_duplicate_registration(self):
    @dataclasses.dataclass
    class StandardRestoreArgs(checkpoint_args.CheckpointArgs):
      pass

    checkpoint_args.register_with_handler(StandardCheckpointHandler)(
        StandardRestoreArgs
    )

    self.assertIs(
        checkpoint_args.get_registered_handler_cls(StandardRestoreArgs),
        StandardCheckpointHandler,
    )
    self.assertIs(
        checkpoint_args.get_registered_handler_cls(StandardRestoreArgs()),
        StandardCheckpointHandler,
    )

  def test_invalid_registration(self):
    @dataclasses.dataclass
    class MyInvalidArgs:  # Doesn't subclass CheckpointArgs
      pass

    with self.assertRaisesRegex(TypeError, 'must subclass'):
      checkpoint_args.register_with_handler(StandardCheckpointHandler)(
          MyInvalidArgs
      )


if __name__ == '__main__':
  absltest.main()
