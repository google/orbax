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

"""Handler that combines other handlers.

Usage example:

```
import orbax.checkpoint as ocp

# A PyTree of jax.Arrays
my_state = ...
# A dictionary to be serialized as JSON
json_dict = ...

ckpter = ocp.Checkpointer(ocp.CompositeCheckpointHandler(
    'state', 'metadata'))
ckpter.save(
    path,
    args=ocp.args.Composite(
        state=ocp.args.PyTreeSave(my_state),
        metadata=ocp.args.JsonSave(json_dict)
    )
)

restored = ckpter.save(
    path,
    args=ocp.args.Composite(
        state=ocp.args.PyTreeRestore(),
        metadata=ocp.args.JsonRestore()
    )
)
my_state = restored.state
json_dict = restored.metadata
```
"""

import asyncio
from collections.abc import Collection, KeysView
from typing import AbstractSet, Any, List, Mapping, Optional, Tuple

from etils import epath
from orbax.checkpoint import async_checkpoint_handler
from orbax.checkpoint import checkpoint_args
from orbax.checkpoint import checkpoint_handler
from orbax.checkpoint import future
from orbax.checkpoint import utils

CheckpointArgs = checkpoint_args.CheckpointArgs
Future = future.Future
CheckpointArgs = checkpoint_args.CheckpointArgs
CheckpointHandler = checkpoint_handler.CheckpointHandler
AsyncCheckpointHandler = async_checkpoint_handler.AsyncCheckpointHandler


class CompositeArgs(CheckpointArgs):
  """Args for wrapping multiple checkpoint items together.

  For simplicity, this object is immutable (although objects attached to it
  may be mutable).
  """

  __items__: Mapping[str, CheckpointArgs]

  def __init__(self, **items: CheckpointArgs):
    super().__setattr__('__items__', items)

    reserved_keys = set(dir(self))

    for key, value in items.items():
      # Reserve and prevent users from setting keys that start with '__'. These
      # may be used later to define options for CompositeCheckpointManager.
      if key.startswith('__'):
        raise ValueError(f'Composiite keys cannot start with "__". Got: {key}')
      if key not in reserved_keys:
        # We do not raise an error if the user specifies a key that matches an
        # existing attribute (like 'keys', 'values', 'items'). These can be
        # accessed through self[key], but not self.key.
        super().__setattr__(key, value)

  def __getitem__(self, key: str) -> CheckpointArgs:
    return self.__items__[key]

  def __setattr__(self, key: str, value: Any):
    del key
    del value
    raise ValueError('CompositeArgs is immutable after initialization.')

  def __len__(self) -> int:
    return len(self.__items__)

  def keys(self) -> KeysView[str]:
    return self.__items__.keys()

  def values(self) -> Collection[CheckpointArgs]:
    return self.__items__.values()

  def items(self) -> AbstractSet[Tuple[str, CheckpointArgs]]:
    return self.__items__.items()

  def get(self, key: str, default=None) -> Optional[CheckpointArgs]:
    try:
      return self.__getitem__(key)
    except KeyError:
      return default


# Returned object of CompositeCheckpointHandler is an alias of CompositeArgs.
CompositeResults = CompositeArgs


class CompositeCheckpointHandler(AsyncCheckpointHandler):
  """CheckpointHandler for saving multiple items.

  As with all `CheckpointHandler` implementations, use only in conjunction with
  an instance of `AbstractCheckpointer`.

  `CompositeCheckpointHandler` allows dealing with multiple items of different
  types or logical distinctness, such as training state (PyTree), dataset
  iterator, JSON metadata, or anything else. The items managed by the
  `CompositeCheckpointHandler` must be specified at initialization.

  For an individual item, `CompositeCheckpointHandler` provides two mechanisms
  for ensuring that the object gets saved and restored as the correct type and
  with the correct logic. The item-specific handler can either be (1) specified
  when the `CompositeCheckpointHandler` is created, or (2) it can be deferred
  (you just need to give the item name up-front). When deferred, the handler
  will be determined from which `CheckpointArgs` are provided during the first
  call to `save` or `restore`.

  Usage::

    ckptr = ocp.Checkpointer(
        ocp.CompositeCheckpointHandler('state', 'metadata')
    )
    ckptr.save(directory,
        ocp.args.Composite(
            # The handler now knows `state` uses `StandardCheckpointHandler`
            # and `metadata` uses `JsonCheckpointHandler`. Any subsequent calls
            # will have to conform to this assumption.
            state=ocp.args.StandardSave(pytree),
            metadata=ocp.args.JsonSave(metadata),
        )
    )

    restored: ocp.args.Composite = ckptr.restore(directory,
        ocp.args.Composite(
            state=ocp.args.StandardSave(abstract_pytree),
            # Only provide the restoration arguments you actually need.
            metadata=ocp.args.JsonRestore(),
        )
    )
    restored.state ...
    restored.metadata ...

    # Skip restoring `metadata` (you can save a subset of items too, in a
    # similar fashion).
    restored: ocp.args.Composite = ckptr.restore(directory,
        ocp.args.Composite(
            state=ocp.args.StandardRestore(abstract_pytree),
        )
    )
    restored.state ...
    restored.metadata ... # Error

    # If the per-item handler doesn't require any extra information in order to
    # restore, in many cases you can use the following pattern if you just want
    # to restore everything:
    restored: ocp.args.Composite = ckptr.restore(directory)
    restored.state ...
    restored.metadata ...

    ckptr = ocp.Checkpointer(
        ocp.CompositeCheckpointHandler(
            'state',
            metadata=ocp.JsonCheckpointHandler()
        )
    )
    ckptr.save(directory,
        ocp.args.Composite(
            state=ocp.args.StandardSave(pytree),
            # Error because `metadata` was specified to use JSON save/restore
            # logic, not `StandardCheckpointHandler`.
            metadata=ocp.args.StandardSave(metadata),
        )
    )
  """

  _known_handlers: dict[str, Optional[CheckpointHandler]] = {}

  def __init__(
      self,
      *item_names: str,
      **items_and_handlers: CheckpointHandler,
  ):
    """Constructor.

    All items must be provided up-front, at initialization.

    Args:
      *item_names: A list of string item names that this handler will manage.
      **items_and_handlers: A mapping of item name to `CheckpointHandler`
        instance, which will be used as the handler for objects of the
        corresponding name.

    """
    self._known_handlers: dict[str, Optional[CheckpointHandler]] = (
        items_and_handlers
    )
    for item in item_names:
      if item not in self._known_handlers:
        self._known_handlers[item] = None

  def _get_or_set_handler(
      self, item_name: str, args: CheckpointArgs
  ) -> CheckpointHandler:
    if item_name not in self._known_handlers:
      raise ValueError(
          f'Unknown key {item_name}. Please make sure that this key was'
          ' specified during initialization.'
      )
    handler = self._known_handlers[item_name]
    registered_handler_cls_for_args = (
        checkpoint_args.get_registered_handler_cls(args)
    )
    if handler is None:
      handler = registered_handler_cls_for_args()  # pytype: disable=not-instantiable
      self._known_handlers[item_name] = handler
    if not isinstance(handler, registered_handler_cls_for_args):
      raise ValueError(
          f'Provided args of type: {type(args)}, which does not correspond to'
          ' the registered handler for these args:'
          f' {registered_handler_cls_for_args}.'
      )
    return handler

  def _get_item_directory(
      self, directory: epath.Path, item_name: str
  ) -> epath.Path:
    return directory / item_name

  async def async_save(
      self, directory: epath.Path, args: CompositeArgs
  ) -> Optional[List[Future]]:
    """Saves multiple items to individual subdirectories."""
    futures = []
    item_directories = [
        self._get_item_directory(directory, item_name)
        for item_name in args.keys()
    ]
    await asyncio.gather(*[
        utils.async_makedirs(path, parents=False, exist_ok=True)
        for path in item_directories
    ])
    for item_name, arg in args.items():
      item_directory = self._get_item_directory(directory, item_name)
      handler = self._get_or_set_handler(item_name, arg)
      if isinstance(handler, AsyncCheckpointHandler):
        futures.extend(await handler.async_save(item_directory, args=arg))
      else:
        handler.save(item_directory, args=arg)
    return futures

  def save(self, *args, **kwargs):
    """Saves synchronously."""

    async def async_save():
      commit_futures = await self.async_save(*args, **kwargs)
      if commit_futures:
        for f in commit_futures:
          f.result()

    asyncio.run(async_save())
    utils.sync_global_devices('CompositeCheckpointHandler:save')

  async def _items_exist(
      self, directory: epath.Path, item_names: List[str]
  ) -> Mapping[str, bool]:
    items_exist = await asyncio.gather(*[
        utils.async_exists(self._get_item_directory(directory, item_name))
        for item_name in item_names
    ])
    return {
        item_name: exists for item_name, exists in zip(item_names, items_exist)
    }

  def restore(
      self,
      directory: epath.Path,
      args: Optional[CompositeArgs] = None,
  ) -> CompositeResults:
    """Restores the provided item synchronously.

    Args:
      directory: Path to restore from.
      args: CompositeArgs object used to restore individual sub-items. May be
        None or "empty" (`CompositeArgs()`), which is used to indicate that the
        handler should restore everything. If an individual item was not
        specified during the save, `None` will be returned for that item's
        entry.

    Returns:
      A CompositeResults object with keys matching `CompositeArgs`, or with keys
      for all known items as specified at creation.
    """
    items_exist = asyncio.run(
        self._items_exist(directory, list(self._known_handlers.keys()))
    )
    if args is None or not args.items():
      composite_args_items = {}
      for item_name, handler in self._known_handlers.items():
        if not items_exist[item_name]:
          composite_args_items[item_name] = None
          continue
        if handler is None:
          raise ValueError(
              f'Item with name: {item_name} had an undetermined'
              ' `CheckpointHandler` when restoring. Please ensure the handler'
              ' was specified during initialization, or use the appropriate'
              ' `CheckpointArgs` subclass to indicate the item type.'
          )
        _, restore_ckpt_args_cls = checkpoint_args.get_registered_args_cls(
            handler
        )
        composite_args_items[item_name] = restore_ckpt_args_cls()
      args = CompositeArgs(**composite_args_items)

    restored = {}
    for item_name, arg in args.items():
      if not items_exist[item_name]:
        restored[item_name] = None
        continue
      handler = self._get_or_set_handler(item_name, arg)
      restored[item_name] = handler.restore(
          self._get_item_directory(directory, item_name), args=arg
      )
    return CompositeResults(**restored)

  def metadata(self, directory: epath.Path) -> CompositeResults:
    items_exist = asyncio.run(
        self._items_exist(directory, list(self._known_handlers.keys()))
    )
    metadata = {}
    for item_name, handler in self._known_handlers.items():
      if not items_exist[item_name]:
        metadata[item_name] = None
        continue
      if handler is not None:
        metadata[item_name] = handler.metadata(
            self._get_item_directory(directory, item_name)
        )
    return CompositeResults(**metadata)

  def finalize(self, directory: epath.Path):
    for item_name, handler in self._known_handlers.items():
      if handler is not None:
        handler.finalize(self._get_item_directory(directory, item_name))

  def close(self):
    for item_name, handler in self._known_handlers.items():
      if handler is not None:
        handler.close()
        self._known_handlers[item_name] = None
