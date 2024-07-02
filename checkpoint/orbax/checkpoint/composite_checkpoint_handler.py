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

"""Handler that combines other handlers.

Usage example:

```
import orbax.checkpoint as ocp

# A PyTree of jax.Arrays
my_state = ...
# A dictionary to be serialized as JSON
json_dict = ...

ckptr = ocp.Checkpointer(
    ocp.CompositeCheckpointHandler('state', 'metadata')
)
ckptr.save(
    path,
    args=ocp.args.Composite(
        state=ocp.args.StandardSave(my_state),
        metadata=ocp.args.JsonSave(json_dict)
    )
)

restored = ckptr.restore(
    path,
    args=ocp.args.Composite(
        state=ocp.args.StandardRestore(),
        metadata=ocp.args.JsonRestore()
    )
)
my_state = restored.state
json_dict = restored.metadata
```
"""

import asyncio
from collections.abc import Collection, KeysView
import concurrent.futures
import dataclasses
from typing import AbstractSet, Any, Dict, List, Mapping, Optional, Tuple, Type, Set
import uuid

from absl import logging
from etils import epath
import nest_asyncio
from orbax.checkpoint import async_checkpoint_handler
from orbax.checkpoint import checkpoint_args
from orbax.checkpoint import checkpoint_handler
from orbax.checkpoint import future
from orbax.checkpoint import options as options_lib
from orbax.checkpoint import proto_checkpoint_handler
from orbax.checkpoint.path import atomicity

CheckpointArgs = checkpoint_args.CheckpointArgs
Future = future.Future
CheckpointArgs = checkpoint_args.CheckpointArgs
CheckpointHandler = checkpoint_handler.CheckpointHandler
AsyncCheckpointHandler = async_checkpoint_handler.AsyncCheckpointHandler
register_with_handler = checkpoint_args.register_with_handler
ProtoCheckpointHandler = proto_checkpoint_handler.ProtoCheckpointHandler
ProtoSaveArgs = proto_checkpoint_handler.ProtoSaveArgs

_CONCURRENT_WORKERS = 3
RESERVED_ITEM_NAMES = []



# TODO(b/295899152) Clean up when users are all registering `CheckpointArgs`.
class _LegacyCheckpointHandlerWrapper(checkpoint_handler.CheckpointHandler):
  """Wrapper for `CheckpointHandler`s without registered `CheckpointArgs`."""

  def __init__(self, handler: checkpoint_handler.CheckpointHandler):
    self._handler = handler

  def save(self, directory: epath.Path, args: '_WrapperArgs'):
    return self._handler.save(directory, *args.args, **args.kwargs)

  def restore(self, directory: epath.Path, args: '_WrapperArgs'):
    return self._handler.restore(directory, *args.args, **args.kwargs)

  def metadata(self, directory: epath.Path) -> Optional[Any]:
    return self._handler.metadata(directory)

  def structure(self, directory: epath.Path) -> Optional[Any]:
    if hasattr(self._handler, 'structure'):
      return self._handler.structure(directory)
    raise AttributeError(
        f'CheckpointHandler of type: {type(self._handler)} has no method'
        ' `structure`.'
    )

  def finalize(self, directory: epath.Path):
    return self._handler.finalize(directory)

  def close(self):
    return self._handler.close()


@register_with_handler(
    _LegacyCheckpointHandlerWrapper, for_save=True, for_restore=True
)
class _WrapperArgs(CheckpointArgs):

  def __init__(self, *args, **kwargs):
    self.args = args
    self.kwargs = kwargs


# TODO(b/295899152) Clean up when users are all registering `CheckpointArgs`.
class _AsyncLegacyCheckpointHandlerWrapper(
    async_checkpoint_handler.AsyncCheckpointHandler
):
  """Wrapper for `CheckpointHandler`s without registered `CheckpointArgs`."""

  def __init__(self, handler: async_checkpoint_handler.AsyncCheckpointHandler):
    self._handler = handler

  def save(self, directory: epath.Path, args: '_AsyncWrapperArgs'):
    return self._handler.save(directory, *args.args, **args.kwargs)

  async def async_save(self, directory: epath.Path, args: '_AsyncWrapperArgs'):
    return await self._handler.async_save(directory, *args.args, **args.kwargs)

  def restore(self, directory: epath.Path, args: '_AsyncWrapperArgs'):
    return self._handler.restore(directory, *args.args, **args.kwargs)

  def metadata(self, directory: epath.Path) -> Optional[Any]:
    return self._handler.metadata(directory)

  def structure(self, directory: epath.Path) -> Optional[Any]:
    if hasattr(self._handler, 'structure'):
      return self._handler.structure(directory)
    raise AttributeError(
        f'CheckpointHandler of type: {type(self._handler)} has no method'
        ' `structure`.'
    )

  def finalize(self, directory: epath.Path):
    return self._handler.finalize(directory)

  def close(self):
    return self._handler.close()


@register_with_handler(
    _AsyncLegacyCheckpointHandlerWrapper, for_save=True, for_restore=True
)
class _AsyncWrapperArgs(CheckpointArgs):

  def __init__(self, *args, **kwargs):
    self.args = args
    self.kwargs = kwargs


def get_legacy_handler_wrapper(
    handler: checkpoint_handler.CheckpointHandler,
) -> checkpoint_handler.CheckpointHandler:
  if isinstance(handler, async_checkpoint_handler.AsyncCheckpointHandler):
    return _AsyncLegacyCheckpointHandlerWrapper(handler)
  return _LegacyCheckpointHandlerWrapper(handler)


def _maybe_raise_reserved_item_error(item_name: str):
  if item_name in RESERVED_ITEM_NAMES:
    raise ValueError(f'Cannot specify reserved item name: "{item_name}".')


@dataclasses.dataclass
class CompositeOptions:
  primary_host: Optional[int] = 0
  active_processes: Optional[Set[int]] = None
  barrier_sync_key_prefix: Optional[str] = None
  temporary_path_class: Optional[Type[atomicity.TemporaryPath]] = None


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

  _known_handlers: Dict[str, Optional[CheckpointHandler]] = {}
  _current_temporary_paths: Dict[str, atomicity.TemporaryPath] = {}

  def __init__(
      self,
      *item_names: str,
      composite_options: CompositeOptions = CompositeOptions(),
      **items_and_handlers: CheckpointHandler,
  ):
    """Constructor.

    All items must be provided up-front, at initialization.

    Args:
      *item_names: A list of string item names that this handler will manage.
      composite_options: Options.
      **items_and_handlers: A mapping of item name to `CheckpointHandler`
        instance, which will be used as the handler for objects of the
        corresponding name.
    """
    self._known_handlers: Dict[str, Optional[CheckpointHandler]] = (
        items_and_handlers
    )
    for item in item_names:
      if item not in self._known_handlers:
        self._known_handlers[item] = None
    for item_name, handler in self._known_handlers.items():
      _maybe_raise_reserved_item_error(item_name)
      if handler and not checkpoint_args.has_registered_args(handler):
        logging.warning(
            'No registered CheckpointArgs found for handler type: %s',
            type(handler),
        )
        self._known_handlers[item_name] = get_legacy_handler_wrapper(handler)
    self._primary_host = composite_options.primary_host
    self._active_processes = composite_options.active_processes
    self._barrier_sync_key_prefix = composite_options.barrier_sync_key_prefix
    self._temporary_path_class = composite_options.temporary_path_class
    logging.info(
        'Initialized item_names=%s, _known_handlers=%s',
        item_names,
        self._known_handlers,
    )

  def _get_or_set_handler(
      self,
      item_name: str,
      args: Optional[CheckpointArgs],
  ) -> CheckpointHandler:
    if item_name not in self._known_handlers:
      raise ValueError(
          f'Unknown key "{item_name}". Please make sure that this key was'
          ' specified during initialization.'
      )
    handler = self._known_handlers[item_name]
    if args is None:
      if handler is None:
        raise ValueError(
            'Provided `None` for `CheckpointArgs`, and the `CheckpointHandler`'
            f' for item "{item_name}" was not configured. If saving, this'
            f' indicates that "{item_name}" was saved with `None` (an instance'
            ' of `CheckpointArgs` was expected). If restoring, providing'
            ' `None` for the item is valid, but only if the `CheckpointHandler'
            ' was already configured. Provide a `CheckpointArgs` subclass so'
            ' that `CompositeCheckpointHandler` will know how to restore the'
            ' item.'
        )
      return handler

    registered_handler_cls_for_args = (
        checkpoint_args.get_registered_handler_cls(args)
    )
    if handler is None:
      handler = registered_handler_cls_for_args()  # pytype: disable=not-instantiable
      self._known_handlers[item_name] = handler
    if not isinstance(handler, registered_handler_cls_for_args):
      raise ValueError(
          f'For item, "{item_name}", CheckpointHandler {type(handler)} does not'
          f' match with registered handler {registered_handler_cls_for_args}'
          f' for provided args of type: {type(args)}'
      )
    return handler

  def _get_item_directory(
      self, directory: epath.Path, item_name: str
  ) -> epath.Path:
    return directory / item_name

  def _get_item_temporary_directory(
      self, directory: epath.Path, item_name: str
  ) -> atomicity.TemporaryPath:
    temporary_path_class = (
        self._temporary_path_class
        or atomicity.get_default_temporary_path_class(directory)
    )
    tmp_item_dir = temporary_path_class.from_final(
        self._get_item_directory(directory, item_name),
        multiprocessing_options=options_lib.MultiprocessingOptions(
            primary_host=self._primary_host,
            active_processes=self._active_processes,
            barrier_sync_key_prefix=self._barrier_sync_key_prefix,
        ),
    )
    return tmp_item_dir

  async def async_save(
      self, directory: epath.Path, args: 'CompositeArgs'
  ) -> Optional[List[Future]]:
    """Saves multiple items to individual subdirectories."""
    futures = []
    # Sort keys to maintain consistent ordering across processes, otherwise
    # we may hit timeouts if processes wait at different barriers in per-item
    # handlers.
    # TODO(b/295899152): Find a less brittle solution or support
    # async-compatible barrier function within handlers.
    # The main blocker here is that many users with custom CheckpointHandlers
    # use barriers in their implementations, which are usually not actually
    # needed.
    item_names = sorted(args.keys())
    item_temporary_paths = [
        self._get_item_temporary_directory(directory, item_name)
        for item_name in item_names
    ]
    self._current_temporary_paths = {
        item_name: item_directory
        for item_name, item_directory in zip(item_names, item_temporary_paths)
    }


    for path in self._current_temporary_paths.values():
      path.create()

    for item_name, item_directory in self._current_temporary_paths.items():
      arg = args[item_name]
      _maybe_raise_reserved_item_error(item_name)
      handler = self._get_or_set_handler(item_name, arg)
      if isinstance(handler, AsyncCheckpointHandler):
        futures.extend(await handler.async_save(item_directory.get(), args=arg))
      else:
        handler.save(item_directory.get(), args=arg)
    return futures

  def save(self, *args, **kwargs):
    """Saves synchronously."""

    async def async_save():
      # Needed for item handlers that also call `asyncio.run`.
      asyncio.get_running_loop()
      nest_asyncio.apply()
      commit_futures = await self.async_save(*args, **kwargs)
      if commit_futures:
        for f in commit_futures:
          f.result()

    asyncio.run(async_save())

  def _items_exist(
      self,
      directory: epath.Path,
      item_names: List[str],
  ) -> Mapping[str, bool]:
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=_CONCURRENT_WORKERS
    ) as executor:
      items_exist = executor.map(
          lambda item_name: self._get_item_directory(
              directory, item_name
          ).exists(),
          item_names,
      )
    return {
        item_name: exists for item_name, exists in zip(item_names, items_exist)
    }

  def restore(
      self,
      directory: epath.Path,
      args: Optional['CompositeArgs'] = None,
  ) -> 'CompositeResults':
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
    items_exist = self._items_exist(
        directory, list(self._known_handlers.keys())
    )
    if args is None or not args.items():
      composite_args_items = {}
      for item_name, handler in self._known_handlers.items():
        if not items_exist[item_name]:
          composite_args_items[item_name] = None
          continue
        # Skip reserved items unless specifically requested.
        if item_name in RESERVED_ITEM_NAMES:
          continue
        if handler is None:
          raise ValueError(
              f'Item with name: "{item_name}" had an undetermined'
              ' `CheckpointHandler` when restoring. Please ensure the handler'
              ' was specified during initialization, or use the appropriate'
              ' `CheckpointArgs` subclass to indicate the item type.'
          )
        _, restore_ckpt_args_cls = checkpoint_args.get_registered_args_cls(
            handler
        )
        try:
          composite_args_items[item_name] = restore_ckpt_args_cls()
        except TypeError as e:
          raise ValueError(
              'Attempted to construct default restoration arguments for item'
              f' "{item_name}", but the `CheckpointArgs` class of type'
              f' {restore_ckpt_args_cls} is not default constructible. Original'
              f' error: {e}.'
          ) from e
      args = CompositeArgs(**composite_args_items)

    restored = {}
    # Sort keys to maintain consistent ordering across processes, otherwise
    # we may hit timeouts if processes wait at different barriers in per-item
    # handlers.
    # TODO(b/295899152): Find a less brittle solution or support
    # async-compatible barrier function.
    for item_name in sorted(args.keys()):
      arg = args[item_name]
      if not items_exist[item_name]:
        restored[item_name] = None
        continue
      handler = self._get_or_set_handler(item_name, arg)
      restored[item_name] = handler.restore(
          self._get_item_directory(directory, item_name), args=arg
      )
    return CompositeResults(**restored)

  def metadata(self, directory: epath.Path) -> 'CompositeResults':
    items_exist = self._items_exist(
        directory, list(self._known_handlers.keys())
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
    if not self._current_temporary_paths:
      raise ValueError('finalize() called before any items were saved.')
    for item_name, handler in self._known_handlers.items():
      tmp_dir = self._current_temporary_paths.get(item_name, None)
      if tmp_dir is None or handler is None:
        # Not an error, as some items may not have been saved.
        continue
      handler.finalize(tmp_dir.get())
      tmp_dir.finalize()

  def close(self):
    for item_name, handler in self._known_handlers.items():
      if handler is not None:
        handler.close()
        self._known_handlers[item_name] = None


@register_with_handler(
    CompositeCheckpointHandler, for_save=True, for_restore=True
)
class CompositeArgs(CheckpointArgs):
  """Args for wrapping multiple checkpoint items together.

  For simplicity, this object is immutable (although objects attached to it
  may be mutable).

  Generally, this object can be treated as a key-value store similar to a dict.

  Usage examples::

    CompositeArgs(
        state=my_train_state,
        dataset=my_dataset,
        metadata=json_metadata,
    )

    CompositeArgs(
        **{
            'state': my_train_state,
            'dataset': my_dataset,
            'metadata': json_metadata,
          }
    )

    args = CompositeArgs(...)
    args.state
    args['state']
    'state' in args

    for key, value in args.items():
      ...
  """

  __items__: Mapping[str, CheckpointArgs]

  def __init__(self, **items: CheckpointArgs):
    super().__setattr__('__items__', items)

    reserved_keys = set(dir(self))

    for key, value in items.items():
      # Reserve and prevent users from setting keys that start with '__'. These
      # may be used later to define options for CompositeCheckpointManager.
      if key.startswith('__'):
        raise ValueError(f'Composite keys cannot start with "__". Got: {key}')
      if key not in reserved_keys:
        # We do not raise an error if the user specifies a key that matches an
        # existing attribute (like 'keys', 'values', 'items'). These can be
        # accessed through self[key], but not self.key.
        super().__setattr__(key, value)

  def __getitem__(self, key: str) -> CheckpointArgs:
    if key not in self.__items__:
      raise KeyError(
          f'Unknown key: {key}. Available keys: {self.__items__.keys()}'
      )
    return self.__items__[key]

  def __contains__(self, key: str) -> bool:
    return key in self.__items__

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

  def __and__(self, other: 'CompositeArgs') -> 'CompositeArgs':
    if isinstance(other, dict):
      other = CompositeArgs(**other)
    return CompositeArgs(**(self.__items__ & other.__items__))

  def __or__(self, other: 'CompositeArgs') -> 'CompositeArgs':
    if isinstance(other, dict):
      other = CompositeArgs(**other)
    return CompositeArgs(**(self.__items__ | other.__items__))

  def __repr__(self):
    return f'CompositeArgs({repr(self.__items__)})'


# Returned object of CompositeCheckpointHandler is an alias of CompositeArgs.
CompositeResults = CompositeArgs
