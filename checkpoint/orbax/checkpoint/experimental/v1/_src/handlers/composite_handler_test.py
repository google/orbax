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

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.metadata import checkpoint as checkpoint_metadata
from orbax.checkpoint._src.metadata import step_metadata_serialization
from orbax.checkpoint._src.tree import structure_utils as tree_structure_utils
from orbax.checkpoint.experimental.v1._src.handlers import composite_handler
from orbax.checkpoint.experimental.v1._src.handlers import pytree_handler
from orbax.checkpoint.experimental.v1._src.handlers import registration
from orbax.checkpoint.experimental.v1._src.handlers import types as handler_types
import orbax.checkpoint.experimental.v1._src.handlers.global_registration  # pylint: disable=unused-import
from orbax.checkpoint.experimental.v1._src.partial import path as partial_path_lib
from orbax.checkpoint.experimental.v1._src.partial import saving as partial_saving
from orbax.checkpoint.experimental.v1._src.synchronization import multihost
from orbax.checkpoint.experimental.v1._src.testing import handler_utils
from orbax.checkpoint.experimental.v1._src.testing import path_utils


CompositeHandler = composite_handler.CompositeHandler
CheckpointableHandler = handler_types.CheckpointableHandler
PyTreeHandler = pytree_handler.PyTreeHandler

FooHandler = handler_utils.FooHandler
BarHandler = handler_utils.BarHandler
BazHandler = handler_utils.BazHandler
Foo = handler_utils.Foo
Bar = handler_utils.Bar
Baz = handler_utils.Baz


class CompositeHandlerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(self.create_tempdir(name='test_dir')) / 'ckpt'
    self._mock_global_registry = registration.local_registry()
    self.enter_context(
        mock.patch.object(
            registration, '_GLOBAL_REGISTRY', new=self._mock_global_registry
        )
    )
    # Baz is registered globally, while the others are not.
    registration.register_handler(BazHandler)

  def save(
      self,
      handler: CompositeHandler,
      directory: epath.Path,
      checkpointables: dict[str, Any],
      *,
      partial_save: bool = False,
  ):
    if multihost.is_primary_host(0):
      directory.mkdir(parents=False, exist_ok=partial_save)
      for k in checkpointables:
        (directory / k).mkdir(parents=False, exist_ok=partial_save)

    handler_typestrs = {
        name: handler_types.typestr(
            type(
                registration.resolve_handler_for_save(
                    handler.handler_registry, checkpointables[name], name=name
                )
            )
        )
        for name in checkpointables.keys()
    }
    step_metadata_file = checkpoint_metadata.step_metadata_file_path(directory)
    if partial_save and step_metadata_file.exists():
      step_metadata = checkpoint_metadata.metadata_store(
          enable_write=True, blocking_write=True
      ).read(step_metadata_file)
      if step_metadata:
        old_handler_typestrs = step_metadata_serialization.deserialize(
            step_metadata
        ).item_handlers
        handler_typestrs = old_handler_typestrs | handler_typestrs

    # Metadata expected to be created outside the handler.
    checkpoint_metadata.metadata_store(
        enable_write=True, blocking_write=True
    ).write(
        file_path=step_metadata_file,
        metadata=step_metadata_serialization.serialize(
            checkpoint_metadata.StepMetadata(
                init_timestamp_nsecs=time.time_ns(),
                item_handlers=handler_typestrs,
            )
        ),
    )
    test_utils.sync_global_processes('CompositeHandlerTest:mkdir')

    async def _save():
      awaitable = await handler.save(
          path_utils.PathAwaitingCreationWrapper(directory), checkpointables
      )
      await awaitable

    asyncio.run(_save())
    test_utils.sync_global_processes('CompositeHandlerTest:save')

  def load(self, handler, directory, checkpointable):

    async def _load():
      awaitable = await handler.load(directory, checkpointable)
      return await awaitable

    result = asyncio.run(_load())
    test_utils.sync_global_processes('CompositeHandlerTest:load')
    return result

  def create_registry(
      self, include_global_registry: bool = True
  ) -> registration.CheckpointableHandlerRegistry:
    return registration.local_registry(
        include_global_registry=include_global_registry
    )

  def test_init(self):
    handler = CompositeHandler(
        self.create_registry().add(PyTreeHandler, 'pytree_foo')
    )
    self.assertTrue(handler._handler_registry.has('pytree_foo'))
    self.assertEqual(handler._handler_registry.get('pytree_foo'), PyTreeHandler)

    self.assertTrue(handler._handler_registry.has('pytree'))
    self.assertEqual(
        handler._handler_registry.get('pytree'), PyTreeHandler
    )

  @parameterized.product(
      save_checkpointables=({'foo': {'a': 1}, 'bar': {'x': 5}},),
      abstract_checkpointables=(
          None,
          {},
          {'foo': None, 'bar': None},
          {'foo': {'a': 0}, 'bar': {'x': 0}},
          {'foo': {'a': 0}},  # Skip loading 'bar'.
      ),
  )
  def test_save_load(
      self,
      save_checkpointables,
      abstract_checkpointables,
  ):
    registry = self.create_registry()
    for k in save_checkpointables:
      registry.add(PyTreeHandler, k)

    self.save(
        CompositeHandler(registry),
        self.directory,
        save_checkpointables,
    )
    for k in save_checkpointables:
      self.assertTrue((self.directory / k).exists())

    result = self.load(
        CompositeHandler(registry),
        self.directory,
        abstract_checkpointables,
    )
    if abstract_checkpointables:
      expected_result = {
          k: v
          for k, v in save_checkpointables.items()
          if k in abstract_checkpointables
      }
    else:
      expected_result = save_checkpointables
    self.assertDictEqual(expected_result, result)

  @parameterized.product(
      with_name=(True, False),
  )
  def test_save_load_checkpointables(
      self,
      with_name: bool,
  ):
    if with_name:
      pairs_to_register = [
          (PyTreeHandler, 'pytree'),
          (FooHandler, 'foo'),
      ]
    else:
      pairs_to_register = [
          (PyTreeHandler, None),
          (FooHandler, None),
      ]
    registry = self.create_registry(include_global_registry=False)
    for handler_type, checkpointable in pairs_to_register:
      registry.add(handler_type, checkpointable)

    checkpointables = {'pytree': {'a': 1}, 'foo': Foo(x=1, y='foo')}
    self.save(
        CompositeHandler(registry),
        self.directory,
        checkpointables,
    )
    for k in checkpointables:
      self.assertTrue((self.directory / k).exists())

    result = self.load(
        CompositeHandler(registry),
        self.directory,
        None,
    )
    self.assertDictEqual(checkpointables, result)

  def test_save_unregistered_checkpointable(self):
    checkpointables = {'foo': Foo(x=1, y='foo')}
    with self.assertRaises(registration.NoEntryError):
      self.save(
          CompositeHandler(self.create_registry()),
          self.directory,
          checkpointables,
      )

  def test_save_custom_object_with_global_registry(self):
    checkpointables = {'baz': Baz(int_val=2, str_val='baz')}
    self.save(
        CompositeHandler(self.create_registry()),
        self.directory,
        checkpointables,
    )
    result = self.load(
        CompositeHandler(self.create_registry()), self.directory, None
    )
    self.assertDictEqual(checkpointables, result)

  def test_save_and_load_with_different_handlers(self):
    checkpointables = {'foo': Foo(x=1, y='foo'), 'bar': Bar(a=5, b='bar')}

    registry = (
        self.create_registry().add(FooHandler, 'foo').add(BarHandler, 'bar')
    )
    self.save(CompositeHandler(registry), self.directory, checkpointables)
    for k in checkpointables:
      self.assertTrue((self.directory / k).exists())

    registry = (
        self.create_registry().add(FooHandler, 'bar').add(BarHandler, 'foo')
    )
    result = self.load(CompositeHandler(registry), self.directory, None)
    expected_result = {'foo': Bar(a=1, b='foo'), 'bar': Foo(x=5, y='bar')}
    self.assertDictEqual(expected_result, result)

  def test_orbax_identifier_file_exists(self):
    checkpointables = {'foo': Foo(x=1, y='foo')}
    registry = self.create_registry().add(FooHandler, 'foo')
    self.save(CompositeHandler(registry), self.directory, checkpointables)
    self.assertTrue((self.directory / 'orbax.checkpoint').exists())

  @parameterized.parameters(True, False)
  def test_partial_save_and_finalize(self, finalize_with_partial_path: bool):
    final_path = self.directory
    partial_path = partial_path_lib.add_partial_save_suffix(final_path)

    first_save_checkpointables = {
        'foo': {'a': 1},
        'bar': {'x': 5},
        # Dict elements are the only way to partial save lists.
        'foo_list': [{'a1': 1, 'b1': 2}, {'a2': 3}],
    }
    second_save_checkpointables = {
        'baz': {'a': 2},
        'foo': {'b': 3},
        'foo_list': [{}, {'b2': 4}],
    }
    merged_checkpointables = tree_structure_utils.merge_trees(
        first_save_checkpointables, second_save_checkpointables
    )
    registry = self.create_registry()
    for k in merged_checkpointables:
      registry.add(PyTreeHandler, k)
    handler = CompositeHandler(registry)

    self.save(
        handler, partial_path, first_save_checkpointables, partial_save=True
    )
    self.assertTrue(partial_path.exists())
    self.assertTrue((partial_path / 'orbax.checkpoint').exists())

    self.save(
        handler, partial_path, second_save_checkpointables, partial_save=True
    )
    self.assertTrue(partial_path.exists())
    self.assertTrue((partial_path / 'orbax.checkpoint').exists())

    restored_checkpointables = self.load(
        handler, partial_path, merged_checkpointables
    )
    test_utils.assert_tree_equal(
        self, restored_checkpointables, merged_checkpointables
    )

    partial_saving.finalize(
        partial_path if finalize_with_partial_path else final_path
    )
    self.assertTrue(final_path.exists())
    self.assertTrue((final_path / 'orbax.checkpoint').exists())

    restored_checkpointables = self.load(
        handler, final_path, merged_checkpointables
    )
    test_utils.assert_tree_equal(
        self, restored_checkpointables, merged_checkpointables
    )

  @parameterized.product(
      second_save_checkpointables=({'foo': {'a': 2}}, {'bar': {'x': 6}})
  )
  def test_partial_save_replacement_raises_error(
      self, second_save_checkpointables
  ):
    final_path = self.directory
    partial_path = partial_path_lib.add_partial_save_suffix(final_path)

    first_save_checkpointables = {'foo': {'a': 1}, 'bar': {'x': 5}}

    registry = self.create_registry()
    for k in first_save_checkpointables:
      registry.add(PyTreeHandler, k)
    handler = CompositeHandler(registry)

    self.save(
        handler, partial_path, first_save_checkpointables, partial_save=True
    )
    with self.assertRaisesRegex(
        ValueError,
        'Partial saving currently does not support REPLACEMENT.',
    ):
      self.save(
          handler, partial_path, second_save_checkpointables, partial_save=True
      )

  @parameterized.product(
      checkpointable_name=('foo', 'bar'),
      first_save_leaf_is_subtree=(True, False),
  )
  def test_partial_save_subtree_replacement_raises_error(
      self, checkpointable_name: str, first_save_leaf_is_subtree: bool
  ):
    final_path = self.directory
    partial_path = partial_path_lib.add_partial_save_suffix(final_path)

    if first_save_leaf_is_subtree:
      tree1 = {'a': {'b': 1}}
      tree2 = {'a': 2}
    else:
      tree1 = {'a': 2}
      tree2 = {'a': {'b': 1}}

    first_save_checkpointables = {checkpointable_name: tree1, 'other': {'c': 3}}
    second_save_checkpointables = {checkpointable_name: tree2}

    registry = self.create_registry()
    for k in first_save_checkpointables:
      registry.add(PyTreeHandler, k)
    handler = CompositeHandler(registry)

    self.save(
        handler, partial_path, first_save_checkpointables, partial_save=True
    )
    with self.assertRaisesRegex(
        ValueError, 'Partial saving currently does not support REPLACEMENT.'
    ):
      self.save(
          handler, partial_path, second_save_checkpointables, partial_save=True
      )

  def test_partial_save_with_mixed_handlers(self):
    final_path = self.directory
    partial_path = partial_path_lib.add_partial_save_suffix(final_path)

    # PyTreeHandler supports partial save, FooHandler does not.
    registry = self.create_registry(include_global_registry=False)
    registry.add(PyTreeHandler, 'pytree')
    registry.add(FooHandler, 'foo')
    handler = CompositeHandler(registry)

    first_save = {'pytree': {'a': 1}, 'foo': Foo(x=1, y='foo1')}
    self.save(handler, partial_path, first_save, partial_save=True)

    second_save = {'pytree': {'b': 2}, 'foo': Foo(x=2, y='foo2')}
    self.save(handler, partial_path, second_save, partial_save=True)

    partial_saving.finalize(final_path)

    # PyTreeHandler should have merged the results.
    # FooHandler should have overwritten.
    expected = {'pytree': {'a': 1, 'b': 2}, 'foo': Foo(x=2, y='foo2')}
    restored = self.load(handler, final_path, None)
    test_utils.assert_tree_equal(self, expected, restored)


if __name__ == '__main__':
  absltest.main()
