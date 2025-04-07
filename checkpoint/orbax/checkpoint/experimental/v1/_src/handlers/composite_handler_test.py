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

from __future__ import annotations

import asyncio
import time
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.metadata import checkpoint as checkpoint_metadata
from orbax.checkpoint._src.metadata import step_metadata_serialization
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint.experimental.v1._src.handlers import composite_handler
from orbax.checkpoint.experimental.v1._src.handlers import pytree_handler
from orbax.checkpoint.experimental.v1._src.handlers import registration
from orbax.checkpoint.experimental.v1._src.handlers import types as handler_types
from orbax.checkpoint.experimental.v1._src.testing import handler_utils

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

    # Baz is registered globally, while the others are not.
    registration.register_handler(BazHandler)

  def save(
      self,
      handler: CompositeHandler,
      directory: epath.Path,
      checkpointables: dict[str, Any],
  ):
    if multihost.is_primary_host(0):
      directory.mkdir(parents=False, exist_ok=False)
      for k in checkpointables:
        (directory / k).mkdir(parents=False, exist_ok=False)

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
    # Metadata expected to be created outside the handler.
    checkpoint_metadata.metadata_store(
        enable_write=True, blocking_write=True
    ).write(
        file_path=checkpoint_metadata.step_metadata_file_path(directory),
        metadata=step_metadata_serialization.serialize(
            checkpoint_metadata.StepMetadata(
                init_timestamp_nsecs=time.time_ns(),
                item_handlers=handler_typestrs,
            )
        ),
    )
    test_utils.sync_global_processes('CompositeHandlerTest:mkdir')

    async def _save():
      awaitable = await handler.save(directory, checkpointables)
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
      self,
  ) -> registration.CheckpointableHandlerRegistry:
    return registration.local_registry()

  def test_init(self):
    handler = CompositeHandler(
        self.create_registry().add(PyTreeHandler, 'pytree')
    )
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
      use_save_registry=(True, False),
      use_load_registry=(True, False),
  )
  def test_save_load(
      self,
      save_checkpointables,
      abstract_checkpointables,
      use_save_registry,
      use_load_registry,
  ):
    registry = self.create_registry()
    for k in save_checkpointables:
      registry.add(PyTreeHandler, k)

    self.save(
        CompositeHandler(registry if use_save_registry else None),
        self.directory,
        save_checkpointables,
    )
    for k in save_checkpointables:
      self.assertTrue((self.directory / k).exists())

    result = self.load(
        CompositeHandler(registry if use_load_registry else None),
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
    registry = self.create_registry()
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
      self.save(CompositeHandler(), self.directory, checkpointables)

  def test_save_custom_object_with_global_registry(self):
    checkpointables = {'baz': Baz(int_val=2, str_val='baz')}
    self.save(CompositeHandler(), self.directory, checkpointables)
    result = self.load(CompositeHandler(), self.directory, None)
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


if __name__ == '__main__':
  absltest.main()
