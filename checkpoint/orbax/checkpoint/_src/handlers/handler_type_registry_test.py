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


HANDLER_REGISTRY = handler_type_registry._GLOBAL_HANDLER_TYPE_REGISTRY
register_handler_type = handler_type_registry.register_handler_type
get_handler_type = handler_type_registry.get_handler_type


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


class HandlerTypeRegistryTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    HANDLER_REGISTRY._registry.clear()

  def test_register_and_get(self):
    register_handler_type(TestHandler)
    self.assertEqual(
        get_handler_type(TestHandler().typestr),
        TestHandler,
    )
    register_handler_type(ParentHandler.TestHandler)
    self.assertEqual(
        get_handler_type(ParentHandler.TestHandler().typestr),
        ParentHandler.TestHandler,
    )
    self.assertTrue(
        '__main__.TestHandler' in HANDLER_REGISTRY._registry
        or
        'handler_type_registry_test.TestHandler' in HANDLER_REGISTRY._registry
    )
    self.assertTrue(
        '__main__.ParentHandler.TestHandler' in HANDLER_REGISTRY._registry
        or
        'handler_type_registry_test.ParentHandler.TestHandler'
        in HANDLER_REGISTRY._registry
    )

  def test_register_different_modules(self):
    register_handler_type(StandardCheckpointHandler)
    self.assertEqual(
        get_handler_type(StandardCheckpointHandler().typestr),
        StandardCheckpointHandler,
    )
    register_handler_type(standard_checkpoint_handler.StandardCheckpointHandler)
    self.assertEqual(
        get_handler_type(
            standard_checkpoint_handler.StandardCheckpointHandler().typestr
        ),
        standard_checkpoint_handler.StandardCheckpointHandler,
    )
    self.assertTrue(
        '__main__.StandardCheckpointHandler' in HANDLER_REGISTRY._registry
        or
        'handler_type_registry_test.StandardCheckpointHandler'
        in HANDLER_REGISTRY._registry
    )
    self.assertIn(
        'orbax.checkpoint._src.handlers.standard_checkpoint_handler.'
        'StandardCheckpointHandler',
        HANDLER_REGISTRY._registry
    )

  def test_register_duplicate_handler_type(self):
    register_handler_type(TestHandler)
    with self.assertRaisesRegex(
        ValueError,
        'Handler type string '
        r'"(?:__main__|handler_type_registry_test)\.TestHandler"'
        ' already exists in the registry with associated type '
        r'<class \'(?:__main__|handler_type_registry_test)\.TestHandler\'>.',
    ):
      register_handler_type(TestHandler)

  def test_get_handler_type_not_found(self):
    with self.assertRaisesRegex(
        ValueError,
        'Handler type string '
        r'"(?:__main__|handler_type_registry_test)\.TestHandler"'
        ' not found in the registry.',
    ):
      get_handler_type(TestHandler().typestr)

  def test_register_with_decorator(self):
    @register_handler_type
    class DecoratedTestHandler(checkpoint_handler.CheckpointHandler):
      def save(self, directory: epath.Path, *args, **kwargs):
        pass

      def restore(self, directory: epath.Path, *args, **kwargs):
        pass

    self.assertEqual(
        get_handler_type(DecoratedTestHandler().typestr),
        DecoratedTestHandler,
    )

if __name__ == '__main__':
  absltest.main()
