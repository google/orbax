# Copyright 2026 The Orbax Authors.
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

from absl.testing import parameterized
from etils import epath
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.metadata import step_metadata_serialization
from orbax.checkpoint._src.path import atomicity_types
from orbax.checkpoint._src.path.snapshot import snapshot
from orbax.checkpoint._src.testing import multiprocess_test
from orbax.checkpoint._src.tree import structure_utils as tree_structure_utils
from orbax.checkpoint.experimental.v1._src.handlers import pytree_handler
from orbax.checkpoint.experimental.v1._src.handlers import registration
from orbax.checkpoint.experimental.v1._src.handlers import stateful_checkpointable_handler
from orbax.checkpoint.experimental.v1._src.handlers import types as handler_types
import orbax.checkpoint.experimental.v1._src.handlers.global_registration  # pylint: disable=unused-import
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout

from orbax.checkpoint.experimental.v1._src.layout import orbax_layout
from orbax.checkpoint.experimental.v1._src.metadata import serialization as metadata_serialization
from orbax.checkpoint.experimental.v1._src.partial import path as partial_path_lib
from orbax.checkpoint.experimental.v1._src.partial import saving as partial_saving
from orbax.checkpoint.experimental.v1._src.synchronization import multihost
from orbax.checkpoint.experimental.v1._src.testing import handler_utils
from orbax.checkpoint.experimental.v1._src.testing import path_utils


STATE_CHECKPOINTABLE_KEY = checkpoint_layout.STATE_CHECKPOINTABLE_KEY
CHECKPOINT_METADATA = orbax_layout.CHECKPOINT_METADATA
ORBAX_CHECKPOINT_INDICATOR_FILE = orbax_layout.ORBAX_CHECKPOINT_INDICATOR_FILE
InternalCheckpointMetadata = (
    step_metadata_serialization.InternalCheckpointMetadata
)
PyTreeHandler = pytree_handler.PyTreeHandler
FooHandler = handler_utils.FooHandler
BarHandler = handler_utils.BarHandler
BazHandler = handler_utils.BazHandler
Foo = handler_utils.Foo
Bar = handler_utils.Bar
Baz = handler_utils.Baz
StatefulCheckpointableHandler = (
    stateful_checkpointable_handler.StatefulCheckpointableHandler
)
PartialSavePyTree = partial_saving._PartialSavePyTree


OrbaxLayout = orbax_layout.OrbaxLayout
InvalidLayoutError = orbax_layout.InvalidLayoutError


class OrbaxLayoutCompositeTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.directory = (
        epath.Path(
            self.create_tempdir(name='orbax_layout_multiprocess_test_dir')
        )
        / 'ckpt'
    )
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
      layout: OrbaxLayout,
      directory: epath.Path,
      checkpointables: dict[str, Any],
      *,
      partial_save: bool = False,
  ):
    if partial_save:
      final_dir = directory
      directory = (
          directory.parent / f'{directory.name}{atomicity_types.TMP_DIR_SUFFIX}'
      )

    test_utils.sync_global_processes('CompositeHandlerTest:save:start')
    if multihost.is_primary_host(0):
      directory.mkdir(parents=True, exist_ok=True)
      for k in checkpointables:
        (directory / k).mkdir(parents=True, exist_ok=True)
    test_utils.sync_global_processes('CompositeHandlerTest:save:mkdir')

    async def _save():
      handler_typestrs = {
          name: handler_types.typestr(
              type(
                  registration.resolve_handler_for_save(
                      layout._handler_registry, checkpointables[name], name=name
                  )
              )
          )
          for name in checkpointables.keys()
      }

      # For partial save in this test, we skip reading existing global metadata
      # here since it will be merged during finalize, just like real execution.

      # Metadata expected to be created outside the handler.
      if multihost.is_primary_host(0):
        internal_metadata = InternalCheckpointMetadata.create(
            handler_typestrs=handler_typestrs,
            init_timestamp_nsecs=time.time_ns(),
            commit_timestamp_nsecs=time.time_ns(),
            custom_metadata={},
        )
        await metadata_serialization.write(
            metadata_serialization.checkpoint_metadata_file_path(directory),
            internal_metadata.serialize(),
        )
      await multihost.sync_global_processes(
          'CompositeHandlerTest:save:checkpoint_metadata_write',
          operation_id='op',
          processes=None,
      )
      awaitable = await layout.save(
          path_utils.PathAwaitingCreationWrapper(directory),
          checkpointables=checkpointables,
      )
      await awaitable

      if partial_save and multihost.is_primary_host(0):
        final_dir.mkdir(parents=True, exist_ok=True)
        pending_dir = final_dir / snapshot.get_pending_dir_name(final_dir.name)
        directory.rename(pending_dir)

    asyncio.run(_save())
    test_utils.sync_global_processes('CompositeHandlerTest:save:complete')

  def load(self, layout, directory, checkpointable):
    test_utils.sync_global_processes('CompositeHandlerTest:load:start')

    async def _load():
      awaitable = await layout.load_checkpointables(directory, checkpointable)
      return await awaitable

    result = asyncio.run(_load())
    test_utils.sync_global_processes('CompositeHandlerTest:load:complete')
    return result

  def create_registry(
      self, include_global_registry: bool = True
  ) -> registration.CheckpointableHandlerRegistry:
    return registration.local_registry(
        include_global_registry=include_global_registry
    )

  def test_init(self):
    patch_registry = self.create_registry().add(
        PyTreeHandler, checkpointable_name='pytree_foo'
    )
    layout = OrbaxLayout()
    layout._handler_registry = patch_registry
    self.assertTrue(layout._handler_registry.has('pytree_foo'))
    self.assertEqual(layout._handler_registry.get('pytree_foo'), PyTreeHandler)

    self.assertTrue(layout._handler_registry.has(STATE_CHECKPOINTABLE_KEY))
    self.assertEqual(
        layout._handler_registry.get(STATE_CHECKPOINTABLE_KEY),
        PyTreeHandler,
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
    patch_registry = self.create_registry()
    for k in save_checkpointables:
      patch_registry.add(PyTreeHandler, checkpointable_name=k)
    layout = OrbaxLayout()
    layout._handler_registry = patch_registry

    self.save(
        layout,
        self.directory,
        save_checkpointables,
    )
    for k in save_checkpointables:
      self.assertTrue((self.directory / k).exists())

    result = self.load(
        layout,
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
          (PyTreeHandler, 'state'),
          (FooHandler, 'foo'),
      ]
    else:
      pairs_to_register = [
          (PyTreeHandler, None),
          (FooHandler, None),
      ]
    registry = self.create_registry(include_global_registry=False)
    for handler_type, checkpointable in pairs_to_register:
      registry.add(handler_type, checkpointable_name=checkpointable)
    layout = OrbaxLayout()
    layout._handler_registry = registry

    checkpointables = {'state': {'a': 1}, 'foo': Foo(x=1, y='foo')}
    self.save(
        layout,
        self.directory,
        checkpointables,
    )
    for k in checkpointables:
      self.assertTrue((self.directory / k).exists())

    result = self.load(
        layout,
        self.directory,
        None,
    )
    self.assertDictEqual(checkpointables, result)

  def test_save_unregistered_checkpointable(self):
    checkpointables = {'foo': Foo(x=1, y='foo')}
    registry = self.create_registry()
    layout = OrbaxLayout()
    layout._handler_registry = registry
    with self.assertRaises(registration.NoEntryError):
      self.save(
          layout,
          self.directory,
          checkpointables,
      )

  def test_save_custom_object_with_global_registry(self):
    checkpointables = {'baz': Baz(int_val=2, str_val='baz')}
    registry = self.create_registry()
    layout = OrbaxLayout()
    layout._handler_registry = registry
    registry.add(BazHandler, checkpointable_name='baz')

    self.save(
        layout,
        self.directory,
        checkpointables,
    )
    result = self.load(layout, self.directory, None)
    self.assertDictEqual(checkpointables, result)

  def test_save_and_load_with_different_handlers(self):
    checkpointables = {'foo': Foo(x=1, y='foo'), 'bar': Bar(a=5, b='bar')}

    registry = (
        self.create_registry()
        .add(FooHandler, checkpointable_name='foo')
        .add(BarHandler, checkpointable_name='bar')
    )
    layout = OrbaxLayout()
    layout._handler_registry = registry
    self.save(layout, self.directory, checkpointables)
    for k in checkpointables:
      self.assertTrue((self.directory / k).exists())

    registry = (
        self.create_registry()
        .add(FooHandler, checkpointable_name='bar')
        .add(BarHandler, checkpointable_name='foo')
    )
    layout = OrbaxLayout()
    layout._handler_registry = registry

    result = self.load(layout, self.directory, None)
    expected_result = {'foo': Bar(a=1, b='foo'), 'bar': Foo(x=5, y='bar')}
    self.assertDictEqual(expected_result, result)

  def test_orbax_identifier_file_exists(self):
    checkpointables = {'foo': Foo(x=1, y='foo')}
    registry = self.create_registry().add(FooHandler, checkpointable_name='foo')
    layout = OrbaxLayout()
    layout._handler_registry = registry
    self.save(layout, self.directory, checkpointables)
    self.assertTrue((self.directory / ORBAX_CHECKPOINT_INDICATOR_FILE).exists())
    test_utils.sync_global_processes(
        'CompositeHandlerTest:test_orbax_identifier_file_exists'
    )

  @parameterized.parameters(True, False)
  def test_partial_save_and_finalize(self, finalize_with_partial_path: bool):
    final_path = self.directory
    partial_path = partial_path_lib.add_partial_save_suffix(final_path)

    first_save_checkpointables = {
        'foo': PartialSavePyTree({'a': 1}),
        'bar': PartialSavePyTree({'x': 5}),
        'foo_list': PartialSavePyTree([{'a1': 1, 'b1': 2}, {'a2': 3}]),
    }
    second_save_checkpointables = {
        'baz': PartialSavePyTree({'a': 2}),
        'foo': PartialSavePyTree({'b': 3}),
        'foo_list': PartialSavePyTree([{}, {'b2': 4}]),
    }
    merged_checkpointables = tree_structure_utils.merge_trees(
        {k: v.state for k, v in first_save_checkpointables.items()},
        {k: v.state for k, v in second_save_checkpointables.items()},
    )
    registry = self.create_registry(include_global_registry=False)
    registry.add(StatefulCheckpointableHandler)
    registry.add(PyTreeHandler)
    layout = OrbaxLayout()
    layout._handler_registry = registry

    self.save(
        layout, partial_path, first_save_checkpointables, partial_save=True
    )
    self.assertTrue(partial_path.exists())

    self.save(
        layout,
        partial_path,
        second_save_checkpointables,
        partial_save=True,
    )
    self.assertTrue(partial_path.exists())

    partial_saving.finalize(
        partial_path if finalize_with_partial_path else final_path
    )
    self.assertTrue(final_path.exists())
    self.assertTrue((final_path / ORBAX_CHECKPOINT_INDICATOR_FILE).exists())

    restored_checkpointables = self.load(
        layout, final_path, merged_checkpointables
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

    first_save_checkpointables = {
        'foo': PartialSavePyTree({'a': 1}),
        'bar': PartialSavePyTree({'x': 5}),
    }

    registry = self.create_registry(include_global_registry=False)
    registry.add(StatefulCheckpointableHandler)
    registry.add(PyTreeHandler)
    layout = OrbaxLayout()
    layout._handler_registry = registry

    self.save(
        layout, partial_path, first_save_checkpointables, partial_save=True
    )

    wrapped_second_save = {
        k: PartialSavePyTree(v) for k, v in second_save_checkpointables.items()
    }
    with self.assertRaisesRegex(
        pytree_handler.PartialSaveReplacementError,
        'Partial saving currently does not support REPLACEMENT.',
    ):
      self.save(
          layout,
          partial_path,
          wrapped_second_save,
          partial_save=True,
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

    first_save_checkpointables = {
        checkpointable_name: PartialSavePyTree(tree1),
        'other': PartialSavePyTree({'c': 3}),
    }
    second_save_checkpointables = {
        checkpointable_name: PartialSavePyTree(tree2)
    }

    registry = self.create_registry(include_global_registry=False)
    registry.add(StatefulCheckpointableHandler)
    registry.add(PyTreeHandler)
    layout = OrbaxLayout()
    layout._handler_registry = registry

    self.save(
        layout, partial_path, first_save_checkpointables, partial_save=True
    )
    with self.assertRaisesRegex(
        pytree_handler.PartialSaveReplacementError,
        'Partial saving currently does not support REPLACEMENT.',
    ):
      self.save(
          layout,
          partial_path,
          second_save_checkpointables,
          partial_save=True,
      )

  def test_partial_save_with_mixed_handlers(self):
    final_path = self.directory
    partial_path = partial_path_lib.add_partial_save_suffix(final_path)

    # PyTreeHandler supports partial save, FooHandler does not.
    registry = self.create_registry(include_global_registry=False)
    registry.add(StatefulCheckpointableHandler, checkpointable_name='pytree')
    registry.add(FooHandler, checkpointable_name='foo')
    layout = OrbaxLayout()
    layout._handler_registry = registry

    first_save = {
        'pytree': PartialSavePyTree({'a': 1}),
        'foo': Foo(x=1, y='foo1'),
    }
    self.save(layout, partial_path, first_save, partial_save=True)

    second_save = {
        'pytree': PartialSavePyTree({'b': 2}),
        'foo': Foo(x=2, y='foo2'),
    }
    self.save(layout, partial_path, second_save, partial_save=True)

    # Since FooHandler does not support partial saving, the overlapping saves
    # to 'foo/foo.txt' will cause a FileExistsError during finalize().
    if multihost.is_primary_host(0):
      with self.assertRaisesRegex(
          FileExistsError,
          'File collision on foo.txt during finalize. Overwriting '
          'destination file is not allowed.'
      ):
        partial_saving.finalize(final_path)
    else:
      with self.assertRaisesRegex(
          OSError,
          'Partial checkpoint finalization failed.'
      ):
        partial_saving.finalize(final_path)


if __name__ == '__main__':
  multiprocess_test.main()
