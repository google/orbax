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

"""Tests for CheckpointerHandler type registry."""

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from orbax.checkpoint._src.handlers import checkpoint_handler
from orbax.checkpoint._src.handlers import handler_type_registry
from orbax.checkpoint._src.handlers import standard_checkpoint_handler


HandlerTypeRegistry = handler_type_registry.HandlerTypeRegistry


class TestHandler(checkpoint_handler.CheckpointHandler):
  def save(self, directory: epath.Path, *args, **kwargs):
    pass

  def restore(self, directory: epath.Path, *args, **kwargs):
    pass


class ParentHandler(checkpoint_handler.CheckpointHandler):
  class TestHandler(checkpoint_handler.CheckpointHandler):
    def save(self, directory: epath.Path, *args, **kwargs):
      pass

    def restore(self, directory: epath.Path, *args, **kwargs):
      pass


class StandardCheckpointHandler(checkpoint_handler.CheckpointHandler):
  def save(self, directory: epath.Path, *args, **kwargs):
    pass

  def restore(self, directory: epath.Path, *args, **kwargs):
    pass


class ChildStandardCheckpointHandler(
    standard_checkpoint_handler.StandardCheckpointHandler
):
  pass


class TypestrOverrideHandler(checkpoint_handler.CheckpointHandler):
  @classmethod
  def typestr(cls) -> str:
    return 'typestr_override'


class HandlerTypeRegistryTest(parameterized.TestCase):

  def test_register_and_get(self):
    registry = HandlerTypeRegistry()
    registry.add(TestHandler.typestr(), TestHandler)
    self.assertEqual(
        registry.get(TestHandler.typestr()),
        TestHandler,
    )
    registry.add(ParentHandler.TestHandler.typestr(), ParentHandler.TestHandler)
    self.assertEqual(
        registry.get(ParentHandler.TestHandler.typestr()),
        ParentHandler.TestHandler,
    )
    self.assertTrue(
        '__main__.TestHandler' in registry._registry
        or
        'handler_type_registry_test.TestHandler' in registry._registry
    )
    self.assertTrue(
        '__main__.ParentHandler.TestHandler' in registry._registry
        or
        'handler_type_registry_test.ParentHandler.TestHandler'
        in registry._registry
    )

  def test_register_different_modules(self):
    registry = HandlerTypeRegistry()
    registry.add(StandardCheckpointHandler.typestr(), StandardCheckpointHandler)
    self.assertEqual(
        registry.get(StandardCheckpointHandler.typestr()),
        StandardCheckpointHandler,
    )
    registry.add(
        standard_checkpoint_handler.StandardCheckpointHandler.typestr(),
        standard_checkpoint_handler.StandardCheckpointHandler
    )
    self.assertEqual(
        registry.get(
            standard_checkpoint_handler.StandardCheckpointHandler().typestr()
        ),
        standard_checkpoint_handler.StandardCheckpointHandler,
    )
    self.assertTrue(
        '__main__.StandardCheckpointHandler' in registry._registry
        or
        'handler_type_registry_test.StandardCheckpointHandler'
        in registry._registry
    )
    self.assertIn(
        'orbax.checkpoint._src.handlers.standard_checkpoint_handler.'
        'StandardCheckpointHandler',
        registry._registry
    )

  def test_register_duplicate_handler_type(self):
    registry = HandlerTypeRegistry()
    registry.add(TestHandler.typestr(), TestHandler)

    # Register the same handler and typestr again. OK.
    registry.add(TestHandler.typestr(), TestHandler)

    # Register a different handler with the same typestr. Error.
    with self.assertRaisesRegex(
        ValueError,
        'Handler type string '
        r'"(?:__main__|handler_type_registry_test)\.TestHandler"'
        ' already exists in the registry with type '
        r'<class \'(?:__main__|handler_type_registry_test)\.TestHandler\'>. '
        'Cannot add type '
        r'<class \'(?:__main__|handler_type_registry_test)\.'
        'ParentHandler.TestHandler\'>.',
    ):
      registry.add(TestHandler.typestr(), ParentHandler.TestHandler)

  def test_get_handler_type_not_found(self):
    registry = HandlerTypeRegistry()
    with self.assertRaisesRegex(
        ValueError,
        'Handler type string '
        r'"(?:__main__|handler_type_registry_test)\.TestHandler"'
        ' not found in the registry.',
    ):
      registry.get(TestHandler.typestr())

  def test_register_subclass_handler_type(self):
    registry = HandlerTypeRegistry()
    registry.add(
        standard_checkpoint_handler.StandardCheckpointHandler.typestr(),
        standard_checkpoint_handler.StandardCheckpointHandler
    )
    registry.add(
        ChildStandardCheckpointHandler.typestr(),
        ChildStandardCheckpointHandler
    )
    self.assertEqual(
        registry.get(
            standard_checkpoint_handler.StandardCheckpointHandler.typestr()
        ),
        standard_checkpoint_handler.StandardCheckpointHandler,
    )
    self.assertEqual(
        registry.get(ChildStandardCheckpointHandler.typestr()),
        ChildStandardCheckpointHandler,
    )

  def test_typestr_override(self):
    registry = HandlerTypeRegistry()
    registry.add(TypestrOverrideHandler.typestr(), TypestrOverrideHandler)
    self.assertEqual(
        registry.get(TypestrOverrideHandler.typestr()),
        TypestrOverrideHandler,
    )

if __name__ == '__main__':
  absltest.main()
