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

from absl.testing import absltest
from absl.testing import parameterized
from orbax.checkpoint.experimental.v1._src.handlers import registration
from orbax.checkpoint.experimental.v1._src.handlers import types as handler_types
from orbax.checkpoint.experimental.v1._src.testing import handler_utils


class RegistrationTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    registration._GLOBAL_REGISTRY = (
        registration._DefaultCheckpointableHandlerRegistry()
    )
    registration.register_handler(handler_utils.BazHandler)

  def test_global_registry(self):
    expected_types = [
        handler
        for handler, _ in registration.global_registry().get_all_entries()
    ]
    self.assertSameElements(
        expected_types,
        [handler_utils.BazHandler],
    )

  @parameterized.product(include_global_registry=[True, False])
  def test_local_registry(self, include_global_registry):
    local_registry = registration.local_registry(
        include_global_registry=include_global_registry
    )
    local_registry.add(handler_utils.FooHandler, 'foo')
    local_registry.add(handler_utils.BarHandler)
    self.assertTrue(local_registry.has('foo'))
    self.assertEqual(local_registry.get('foo'), handler_utils.FooHandler)
    expected = [handler_utils.FooHandler, handler_utils.BarHandler]
    if include_global_registry:
      expected.append(handler_utils.BazHandler)
    self.assertSameElements(
        expected,
        [handler for handler, _ in local_registry.get_all_entries()],
    )

  def test_missing_handler(self):
    local_registry = registration.local_registry()
    local_registry.add(handler_utils.FooHandler, 'foo')
    local_registry.add(handler_utils.BarHandler)
    with self.assertRaises(registration.NoEntryError):
      local_registry.get('bar')

  @parameterized.parameters(
      (
          handler_utils.FooHandler,
          handler_utils.Foo(1, 'hi'),
          'checkpointable_name',
      ),
      (handler_utils.FooHandler, handler_utils.Foo(1, 'hi'), None),
      (
          handler_utils.BarHandler,
          handler_utils.Bar(2, 'bye'),
          'checkpointable_name',
      ),
      (handler_utils.BarHandler, handler_utils.Bar(2, 'bye'), None),
  )
  def test_resolve_handler_for_save(self, handler_type, checkpointable, name):
    local_registry = registration.local_registry()
    local_registry.add(handler_type, name)
    name = name or 'checkpointable_name'
    resolved_handler = registration.resolve_handler_for_save(
        local_registry, checkpointable, name=name
    )
    self.assertIsInstance(resolved_handler, handler_type)

  def test_resolve_handler_for_save_resolution_order(self):

    class HandlerOne(handler_utils.DictHandler):
      pass

    class HandlerTwo(handler_utils.DictHandler):
      pass

    handlers_to_register = [HandlerOne, HandlerTwo]

    with self.subTest('in_order'):
      local_registry = registration.local_registry()
      for handler in handlers_to_register:
        local_registry.add(handler)
      resolved_handler = registration.resolve_handler_for_save(
          local_registry, {'a': 1}, name='checkpointable_name'
      )
      self.assertIsInstance(resolved_handler, handlers_to_register[0])
    with self.subTest('reversed'):
      local_registry = registration.local_registry()
      for handler in reversed(handlers_to_register):
        local_registry.add(handler)
      resolved_handler = registration.resolve_handler_for_save(
          local_registry, {'a': 1}, name='checkpointable_name'
      )
      self.assertIsInstance(resolved_handler, handlers_to_register[-1])

  def test_resolve_handler_for_save_not_handleable(self):
    local_registry = registration.local_registry()
    local_registry.add(handler_utils.FooHandler)
    with self.assertRaises(registration.NoEntryError):
      registration.resolve_handler_for_save(
          local_registry, handler_utils.Bar(2, 'bye'), name='bar'
      )

  def test_resolve_handler_for_save_no_matching_name(self):
    local_registry = registration.local_registry()
    local_registry.add(handler_utils.FooHandler, 'foo')
    with self.assertRaises(registration.NoEntryError):
      registration.resolve_handler_for_save(
          local_registry, handler_utils.Foo(1, 'hi'), name='foo1'
      )

  def test_resolve_handler_for_save_abstract_checkpointable(self):
    local_registry = registration.local_registry()
    local_registry.add(handler_utils.FooHandler)
    with self.assertRaises(registration.NoEntryError):
      registration.resolve_handler_for_save(
          local_registry, handler_utils.AbstractFoo(), name='foo'
      )

  @parameterized.parameters(
      (
          handler_utils.FooHandler,
          handler_utils.AbstractFoo(),
          'checkpointable_name',
          handler_types.typestr(handler_utils.FooHandler),
      ),
      (
          handler_utils.FooHandler,
          handler_utils.AbstractFoo(),
          None,
          handler_types.typestr(handler_utils.FooHandler),
      ),
      (
          handler_utils.FooHandler,
          None,
          'checkpointable_name',
          handler_types.typestr(handler_utils.FooHandler),
      ),
      (
          handler_utils.FooHandler,
          None,
          None,
          handler_types.typestr(handler_utils.FooHandler),
      ),
      (
          handler_utils.BarHandler,
          handler_utils.AbstractBar(),
          'checkpointable_name',
          handler_types.typestr(handler_utils.BarHandler),
      ),
      # The typestr does not match, but that is ok.
      (
          handler_utils.BarHandler,
          handler_utils.AbstractBar(),
          None,
          handler_types.typestr(handler_utils.FooHandler),
      ),
  )
  def test_resolve_handler_for_load(
      self, handler_type, checkpointable, name, handler_typestr
  ):
    local_registry = registration.local_registry()
    local_registry.add(handler_type, name)
    name = name or 'checkpointable_name'
    resolved_handler = registration.resolve_handler_for_load(
        local_registry,
        checkpointable,
        name=name,
        handler_typestr=handler_typestr,
    )
    self.assertIsInstance(resolved_handler, handler_type)

  def test_resolve_handler_for_load_resolution_order(self):

    class HandlerOne(handler_utils.DictHandler):
      pass

    class HandlerTwo(handler_utils.DictHandler):
      pass

    handlers_to_register = [HandlerOne, HandlerTwo]

    with self.subTest('globally_registered'):
      resolved_handler = registration.resolve_handler_for_load(
          registration.local_registry(),
          None,
          name='checkpointable_name',
          handler_typestr='unknown_class',
      )
      self.assertIsInstance(resolved_handler, handler_utils.BazHandler)
    with self.subTest('in_order'):
      local_registry = registration.local_registry(
          include_global_registry=False
      )
      for handler in handlers_to_register:
        local_registry.add(handler)
      resolved_handler = registration.resolve_handler_for_load(
          local_registry,
          None,
          name='checkpointable_name',
          handler_typestr='unknown_class',
      )
      self.assertIsInstance(resolved_handler, handlers_to_register[0])
    with self.subTest('with_typestr'):
      local_registry = registration.local_registry()
      for handler in handlers_to_register:
        local_registry.add(handler)
      resolved_handler = registration.resolve_handler_for_load(
          local_registry,
          None,
          name='checkpointable_name',
          handler_typestr=handler_types.typestr(HandlerTwo),
      )
      self.assertIsInstance(resolved_handler, handlers_to_register[-1])
    with self.subTest('reversed'):
      local_registry = registration.local_registry(
          include_global_registry=False
      )
      for handler in reversed(handlers_to_register):
        local_registry.add(handler)
      resolved_handler = registration.resolve_handler_for_load(
          local_registry,
          None,
          name='checkpointable_name',
          handler_typestr='unknown_class',
      )
      self.assertIsInstance(resolved_handler, handlers_to_register[-1])

  def test_resolve_handler_for_load_not_handleable(self):
    local_registry = registration.local_registry()
    local_registry.add(handler_utils.FooHandler)
    with self.assertRaises(registration.NoEntryError):
      registration.resolve_handler_for_load(
          local_registry,
          handler_utils.AbstractBar(),
          name='bar',
          handler_typestr='unused',
      )

  def test_resolve_handler_for_load_no_matching_name(self):
    local_registry = registration.local_registry()
    local_registry.add(handler_utils.FooHandler, 'foo')
    with self.assertRaises(registration.NoEntryError):
      registration.resolve_handler_for_load(
          local_registry,
          handler_utils.AbstractBar(),
          name='foo1',
          handler_typestr='unused',
      )

  def test_resolve_handler_for_load_checkpointable(self):
    local_registry = registration.local_registry()
    local_registry.add(handler_utils.FooHandler)
    with self.assertRaises(registration.NoEntryError):
      registration.resolve_handler_for_load(
          local_registry,
          handler_utils.Foo(1, 'hi'),
          name='foo',
          handler_typestr='unused',
      )


if __name__ == '__main__':
  absltest.main()
