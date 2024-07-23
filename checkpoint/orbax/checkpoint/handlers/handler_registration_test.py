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

import dataclasses
from typing import Optional, Type, Union

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from orbax.checkpoint import checkpoint_args
from orbax.checkpoint import checkpoint_handler
from orbax.checkpoint.handlers import handler_registration


CheckpointHandler = checkpoint_handler.CheckpointHandler
DefaultCheckpointHandlerRegistry = (
    handler_registration.DefaultCheckpointHandlerRegistry
)
AlreadyExistsError = handler_registration.AlreadyExistsError
NoEntryError = handler_registration.NoEntryError


class _TestCheckpointHandler(CheckpointHandler):
  """No-op checkpoint handler for testing."""

  def save(self, directory: epath.Path, *args, **kwargs) -> None:
    del directory, args, kwargs

  def restore(self, directory: epath.Path, *args, **kwargs) -> None:
    del directory, args, kwargs


@dataclasses.dataclass
class _TestArgs(checkpoint_args.CheckpointArgs):
  """No-op checkpoint args for testing."""

  ...


class HandlerRegistryTest(parameterized.TestCase):

  @parameterized.product(
      handler=(_TestCheckpointHandler, _TestCheckpointHandler()),
      item=(None, 'item'),
  )
  def test_add_and_get_entry(
      self,
      handler: Union[CheckpointHandler, Type[CheckpointHandler]],
      item: Optional[str],
  ):
    args_type = _TestArgs
    registry = DefaultCheckpointHandlerRegistry()

    registry.add(
        item,
        args_type,
        handler,
    )

    # Check that the entry is added to the registry.
    self.assertTrue(registry.has(item, args_type))
    # Check that the handler is returned and that it is initialized as an
    # object.
    self.assertIsInstance(
        registry.get(item, args_type),
        _TestCheckpointHandler,
    )

  def test_add_entry_with_existing_item_and_args_type_raises_error(self):
    item = 'item'
    args_type = _TestArgs
    registry = DefaultCheckpointHandlerRegistry()

    registry.add(item, args_type, _TestCheckpointHandler)

    with self.assertRaisesRegex(
        AlreadyExistsError, r'already exists in the registry'
    ):
      registry.add(item, args_type, _TestCheckpointHandler)

  def test_get_all_entries(self):
    item1 = 'item1'
    item2 = 'item2'
    args_type = _TestArgs
    handler = _TestCheckpointHandler
    registry = DefaultCheckpointHandlerRegistry()

    registry.add(item1, args_type, handler)
    registry.add(item2, args_type, handler)

    entries = registry.get_all_entries()
    self.assertLen(entries, 2)
    self.assertIsInstance(
        entries[(item1, args_type)],
        handler,
    )
    self.assertIsInstance(
        entries[(item2, args_type)],
        handler,
    )

  def test_instantiate_registry_from_another_registry(self):
    item1 = 'item1'
    item2 = 'item2'
    args_type = _TestArgs
    handler = _TestCheckpointHandler

    registry1 = DefaultCheckpointHandlerRegistry()
    registry1.add(item1, args_type, handler)
    registry2 = DefaultCheckpointHandlerRegistry(registry1)
    registry2.add(item2, args_type, handler)

    entries = registry2.get_all_entries()
    self.assertLen(entries, 2)
    self.assertIsInstance(
        entries[(item1, args_type)],
        handler,
    )
    self.assertIsInstance(
        entries[(item2, args_type)],
        handler,
    )

  @parameterized.product(
      item=(None, 'item'),
  )
  def test_raise_error_when_no_entry_found(self, item: Optional[str]):
    registry = DefaultCheckpointHandlerRegistry()

    with self.assertRaisesRegex(
        NoEntryError,
        r'No entry for item=.* and args_ty=.* in the registry',
    ):
      registry.get(item, _TestArgs)

  def test_concrete_item_takes_precedence_over_general_args_type(self):
    none_item = None
    item = 'item'
    args_type = _TestArgs

    class _TestCheckpointHandlerA(_TestCheckpointHandler):
      pass

    class _TestCheckpointHandlerB(_TestCheckpointHandler):
      pass

    registry = DefaultCheckpointHandlerRegistry()
    registry.add(none_item, args_type, _TestCheckpointHandlerA)
    registry.add(item, args_type, _TestCheckpointHandlerB)

    self.assertTrue(registry.has(none_item, args_type))
    self.assertTrue(registry.has(item, args_type))
    self.assertIsInstance(
        registry.get(none_item, args_type),
        _TestCheckpointHandlerA,
    )
    self.assertIsInstance(
        registry.get(item, args_type),
        _TestCheckpointHandlerB,
    )

  def test_falls_back_to_general_args_type(self):
    none_item = None
    registered_item = 'registered_item'
    item_without_registration = 'item_without_registration'
    args_type = _TestArgs

    class _TestCheckpointHandlerA(_TestCheckpointHandler):
      pass

    class _TestCheckpointHandlerB(_TestCheckpointHandler):
      pass

    registry = DefaultCheckpointHandlerRegistry()
    registry.add(none_item, args_type, _TestCheckpointHandlerA)
    registry.add(registered_item, args_type, _TestCheckpointHandlerB)

    self.assertTrue(registry.has(none_item, args_type))
    self.assertTrue(registry.has(registered_item, args_type))
    self.assertFalse(registry.has(item_without_registration, args_type))

    self.assertIsInstance(
        registry.get(none_item, args_type),
        _TestCheckpointHandlerA,
    )
    self.assertIsInstance(
        registry.get(item_without_registration, args_type),
        _TestCheckpointHandlerA,
    )
    self.assertIsInstance(
        registry.get(registered_item, args_type),
        _TestCheckpointHandlerB,
    )

  def test_multiple_handlers_for_same_item(self):
    item = 'item'

    class _TestArgsA(checkpoint_args.CheckpointArgs):
      pass

    class _TestArgsB(checkpoint_args.CheckpointArgs):
      pass

    registry = DefaultCheckpointHandlerRegistry()
    registry.add(item, _TestArgsA, _TestCheckpointHandler)
    registry.add(item, _TestArgsB, _TestCheckpointHandler)

    self.assertIsInstance(
        registry.get(item, _TestArgsA),
        _TestCheckpointHandler,
    )
    self.assertIsInstance(
        registry.get(item, _TestArgsB),
        _TestCheckpointHandler,
    )

if __name__ == '__main__':
  absltest.main()
