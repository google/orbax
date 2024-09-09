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

"""Tests CheckpointArg registration."""

import dataclasses
from absl.testing import absltest
from orbax.checkpoint import checkpoint_args
from orbax.checkpoint._src.handlers import checkpoint_handler
from orbax.checkpoint._src.handlers import standard_checkpoint_handler

StandardCheckpointHandler = (
    standard_checkpoint_handler.StandardCheckpointHandler
)


@checkpoint_args.register_with_handler(StandardCheckpointHandler, for_save=True)
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

    checkpoint_args.register_with_handler(
        StandardCheckpointHandler,
        for_restore=True,
    )(
        StandardRestoreArgs,
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
      checkpoint_args.register_with_handler(
          StandardCheckpointHandler, for_save=True
      )(
          MyInvalidArgs,
      )

  def test_get_registered_args_cls(self):
    class MyCheckpointHandler(checkpoint_handler.CheckpointHandler):

      def save(self, *args, **kwargs):
        pass

      def restore(self, *args, **kwargs):
        pass

    @checkpoint_args.register_with_handler(MyCheckpointHandler, for_save=True)
    @dataclasses.dataclass
    class MySaveArgs(checkpoint_args.CheckpointArgs):
      pass

    with self.assertRaises(ValueError):
      checkpoint_args.get_registered_args_cls(MyCheckpointHandler)

    @checkpoint_args.register_with_handler(
        MyCheckpointHandler, for_restore=True
    )
    @dataclasses.dataclass
    class MyRestoreArgs(checkpoint_args.CheckpointArgs):
      pass

    save_args, restore_args = checkpoint_args.get_registered_args_cls(
        MyCheckpointHandler
    )
    self.assertIs(save_args, MySaveArgs)
    self.assertIs(restore_args, MyRestoreArgs)
    save_args, restore_args = checkpoint_args.get_registered_args_cls(
        MyCheckpointHandler()
    )
    self.assertIs(save_args, MySaveArgs)
    self.assertIs(restore_args, MyRestoreArgs)

  def test_get_registered_args_cls_save_and_restore(self):
    class MyCheckpointHandler(checkpoint_handler.CheckpointHandler):

      def save(self, *args, **kwargs):
        pass

      def restore(self, *args, **kwargs):
        pass

    @checkpoint_args.register_with_handler(
        MyCheckpointHandler, for_save=True, for_restore=True
    )
    @dataclasses.dataclass
    class MyHandlerArgs(checkpoint_args.CheckpointArgs):
      pass

    save_args, restore_args = checkpoint_args.get_registered_args_cls(
        MyCheckpointHandler
    )
    self.assertIs(save_args, MyHandlerArgs)
    self.assertIs(restore_args, MyHandlerArgs)
    save_args, restore_args = checkpoint_args.get_registered_args_cls(
        MyCheckpointHandler()
    )
    self.assertIs(save_args, MyHandlerArgs)
    self.assertIs(restore_args, MyHandlerArgs)

  def test_get_registered_args_cls_single_handler_multiple_args(self):
    class MyCheckpointHandler(checkpoint_handler.CheckpointHandler):

      def save(self, *args, **kwargs):
        pass

      def restore(self, *args, **kwargs):
        pass

    # Register first CheckpointArgs to MyCheckpointHandler.
    @checkpoint_args.register_with_handler(
        MyCheckpointHandler, for_save=True, for_restore=True
    )
    @dataclasses.dataclass
    class Args1(checkpoint_args.CheckpointArgs):
      pass

    # Register second CheckpointArgs to MyCheckpointHandler.
    @checkpoint_args.register_with_handler(
        MyCheckpointHandler, for_save=True, for_restore=True
    )
    @dataclasses.dataclass
    class Args2(checkpoint_args.CheckpointArgs):
      pass

    save_args, restore_args = checkpoint_args.get_registered_args_cls(
        MyCheckpointHandler
    )
    self.assertIs(save_args, Args1)
    self.assertIs(restore_args, Args1)

    save_args, restore_args = checkpoint_args.get_registered_args_cls(
        MyCheckpointHandler()
    )
    self.assertIsNot(save_args, Args2)
    self.assertIsNot(restore_args, Args2)


if __name__ == '__main__':
  absltest.main()
