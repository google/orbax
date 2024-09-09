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

Usage example::

  import orbax.checkpoint as ocp

  # A PyTree of jax.Arrays
  my_state = ...
  # A dictionary to be serialized as JSON
  json_dict = ...

  checkpointer = ocp.Checkpointer(ocp.CompositeCheckpointHandler())
  checkpointer.save(
      path,
      args=ocp.args.Composite(
          state=ocp.args.StandardSave(my_state),
          metadata=ocp.args.JsonSave(json_dict)
      )
  )

  restored = checkpointer.restore(
      path,
      args=ocp.args.Composite(
          state=ocp.args.StandardRestore(),
          metadata=ocp.args.JsonRestore()
      )
  )
  my_state = restored.state
  json_dict = restored.metadata
"""

import asyncio
from collections.abc import Collection, KeysView
import concurrent.futures
import dataclasses
from typing import AbstractSet, Any, Coroutine, Dict, List, Mapping, MutableSet, Optional, Tuple, Type
import uuid

from absl import logging
from etils import epath
import jax
import nest_asyncio
from orbax.checkpoint import checkpoint_args
from orbax.checkpoint import future
from orbax.checkpoint import options as options_lib
from orbax.checkpoint._src.handlers import async_checkpoint_handler
from orbax.checkpoint._src.handlers import checkpoint_handler
from orbax.checkpoint._src.handlers import handler_registration
from orbax.checkpoint._src.handlers import proto_checkpoint_handler
from orbax.checkpoint.path import atomicity

CheckpointArgs = checkpoint_args.CheckpointArgs
Future = future.Future
CheckpointArgs = checkpoint_args.CheckpointArgs
CheckpointHandler = checkpoint_handler.CheckpointHandler
AsyncCheckpointHandler = async_checkpoint_handler.AsyncCheckpointHandler
register_with_handler = checkpoint_args.register_with_handler
ProtoCheckpointHandler = proto_checkpoint_handler.ProtoCheckpointHandler
ProtoSaveArgs = proto_checkpoint_handler.ProtoSaveArgs
ProtoRestoreArgs = proto_checkpoint_handler.ProtoRestoreArgs
AsyncSaveCoroutine = Coroutine[Any, Any, Optional[List[Future]]]


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
  multiprocessing_options: options_lib.MultiprocessingOptions = (
      dataclasses.field(default_factory=options_lib.MultiprocessingOptions)
  )
  temporary_path_class: Optional[Type[atomicity.TemporaryPath]] = None
  file_options: Optional[options_lib.FileOptions] = None


def _get_unique_registered_items_and_handlers(
    registry: handler_registration.CheckpointHandlerRegistry,
) -> List[Tuple[str, CheckpointHandler]]:
  """Returns unique items and handlers from the registry.

  Args:
    registry: The registry to get the items and handlers from.

  Returns:
    A list of unique `(item name, handler)` tuples.
  """
  item_and_handers = []
  for (
      item,
      _,
  ), handler in registry.get_all_entries().items():
    if item is not None and (item, handler) not in item_and_handers:
      item_and_handers.append((item, handler))
  return item_and_handers


def _add_to_handler_registry_from_global_registry(
    registry: handler_registration.CheckpointHandlerRegistry,
    registered_handler_for_args: CheckpointHandler,
    item_name: str,
) -> None:
  """Adds items and handlers to a registry from the global registry."""

  save_args, restore_args = checkpoint_args.get_registered_args_cls(
      registered_handler_for_args
  )

  logging.info(
      'Deferred registration for item: "%s". Adding handler `%s` for item "%s"'
      ' and save args `%s` and restore args `%s` to `_handler_registry`.',
      item_name,
      registered_handler_for_args,
      item_name,
      save_args,
      restore_args,
  )
  registry.add(item_name, save_args, registered_handler_for_args)

  # If the restore args are different from the save args, add them to the
  # registry as well.
  if restore_args != save_args:
    registry.add(item_name, restore_args, registered_handler_for_args)


class CompositeCheckpointHandler(AsyncCheckpointHandler):
  """CheckpointHandler for saving multiple items.

  As with all :py:class:`CheckpointHandler` implementations, use only in
  conjunction with an instance of `AbstractCheckpointer`.

  The :py:class:`CompositeCheckpointHandler` allows dealing with multiple items
  of different types or logical distinctness, such as training state (PyTree),
  dataset iterator, JSON metadata, or anything else. To specify the handler
  and args for each checkpointable item, use the `handler_registry` option at
  initialization (see :py:class:`CheckpointHandlerRegistry`).

  If a handler registry is provided, the :py:class:`CompositeCheckpointHandler`
  will use this registry to determine the :py:class:`CheckpointHandler` for each
  checkpointable item. If no registry is provided, an empty registry will be
  constructed by default internally. Additional items not registered in the
  handler registry can be specified when calling `save` or `restore`. The
  handler will be determined from the provided :py:class:`CheckpointArgs` during
  the first call to `save` or `restore`. Subsequent calls to `save` or `restore`
  will use the same handler and will require the same :py:class:`CheckpointArgs`
  to be provided.

  `item_names` and `items_and_handlers` are deprecated arguments for specifying
  the items and handlers. Please use `handler_registry` instead.

  Usage::
    # The simplest use-case, with no handler registry provided on construction.
    checkpointer = ocp.Checkpointer(
        ocp.CompositeCheckpointHandler()
    )

    checkpointer.save(directory,
        ocp.args.Composite(
            # The handler will be determined from the global registry using the
            # provided args.
            state=ocp.args.StandardSave(pytree),
        )
    )

    restored: ocp.args.Composite = checkpoint_handler.restore(directory,
        ocp.args.Composite(
            state=ocp.args.StandardRestore(abstract_pytree),
        )
    )
    restored.state ...

    # The "state" item's args was determined on the first call to `save`.
    # Trying to save "state" with a different set of args will raise an error.
    checkpointer.save(directory,
      ocp.args.Composite(
        # Will raise `ValueError: Item "state" and args "JsonSave [...]" does
        not match with any registered handler! ...`.
        state=ocp.args.JsonSave(pytree),
        )
    )

    # You can also register specific items and handlers using a handler
    # registry.
    handler_registry = (
        ocp.handler_registration.DefaultCheckpointHandlerRegistry()
    )
    # Some subclass of `CheckpointHandler` that handles the "state" item.
    handler = state_handler()
    # Multiple save args and handlers can be registered for the same item.
    handler_registry.add(
        'state,
        StateSaveArgs,
        handler,
    )
    handler_registry.add(
        'state',
        StateRestoreArgs,
        handler,
    )

    # The `CompositeCheckpointHandler` will use the handlers registered in the
    # registry when it encounters the item name `state`.
    checkpointer_with_registry = ocp.Checkpointer(
        ocp.CompositeCheckpointHandler(handler_registry=handler_registry)
    )

    checkpointer_with_registry.save(directory,
        ocp.args.Composite(
            # The handler knows that the "state" item should use the
            # `state_handler` specified in the registry. When an item has been
            # added to the handler registry, only save args that are registered
            # for that item are allowed.
            state=StateSaveArgs(pytree),
            # You can also provide other arguments that are not registered in
            # the handler registry as long as a handler has been globally
            # registered for the args.
            other_param=ocp.args.StandardSave(other_pytree),
        )
    )

    restored: ocp.args.Composite = checkpointer_with_registry.restore(directory,
        ocp.args.Composite(
            state=StateRestoreArgs(abstract_pytree),
            other_param=ocp.args.StandardRestore(abstract_other_pytree),
        )
    )
    restored.state ...
    restored.other_param...

    # Skip restoring `other_params` (you can save a subset of items too, in a
    # similar fashion).
    restored_state_only: ocp.args.Composite = (
      checkpointer_with_registry.restore(directory,
        ocp.args.Composite(
            state=StateRestoreArgs(abstract_pytree),
        )
      )
    )
    restored.state ...
    restored.other_param ... # None.
  """

  _current_temporary_paths: Dict[str, atomicity.TemporaryPath] = {}
  # Items that do not have a registered handler. This is set to `None` if the
  # user provided a `handler_registry` at initialization. If the user provided
  # `item_names` and `items_and_handlers` at initialization, this is set to a
  # set of the item names that have a `None` handler and will need to be
  # registered at the first call to `_get_or_set_handler`.
  _item_names_without_registered_handlers: Optional[MutableSet[str]] = None

  def __init__(
      self,
      *item_names: str,
      composite_options: CompositeOptions = CompositeOptions(),
      handler_registry: Optional[
          handler_registration.CheckpointHandlerRegistry
      ] = None,
      **items_and_handlers: CheckpointHandler,
  ):
    """Constructor.

    If you are `item_names` and/or `items_and_handlers`, all items must be
    provided up-front, at initialization. If you are using a `handler_registry`,
    you can register items at any time, even after the first call to `save` or
    `restore`.

    Args:
      *item_names: A list of string item names that this handler will manage.
        `item_names` is deprecated. Please use `handler_registry` instead.
      composite_options: Options.
      handler_registry: A :py:class:`CheckpointHandlerRegistry` instance. If
        provided, the :py:class:`CompositeCheckpointHandler` will use this
        registry to determine the :py:class:`CheckpointHandler` for each item.
        This option is mutually exclusive with `items_and_handlers` and
        `item_names`.
      **items_and_handlers: A mapping of item name to `CheckpointHandler`
        instance, which will be used as the handler for objects of the
        corresponding name. `items_and_handlers` is deprecated. Please use
        `handler_registry` instead.
    """
    if handler_registry is not None and items_and_handlers:
      raise ValueError(
          'Both `handler_registry` and `items_and_handlers` were provided. '
          'Please specify only one of the two.'
      )
    if handler_registry is not None and item_names:
      raise ValueError(
          'Both `handler_registry` and `item_names` were provided. '
          'Please specify only one of the two.'
      )
    if items_and_handlers or item_names:
      # This whole code branch will be removed once `item_and_handlers` and
      # `item_names` have been completely deprecated.
      self._item_names_without_registered_handlers: MutableSet[str] = set()
      self._handler_registry = (
          handler_registration.DefaultCheckpointHandlerRegistry()
      )

      if item_names:
        for item in item_names:
          _maybe_raise_reserved_item_error(item)
          if item not in items_and_handlers:
            # If an item is in `item_names`, but not in `items_and_handlers`,
            # there is no way to know what the corresponding handler and args
            # are. Instead, defer registration until the first call to
            # `_get_or_set_handler`.
            self._item_names_without_registered_handlers.add(item)

      # Register all the items and handlers in the handler registry.
      if items_and_handlers:
        for item, handler in items_and_handlers.items():
          _maybe_raise_reserved_item_error(item)
          if handler is not None and checkpoint_args.has_registered_args(
              handler
          ):
            _add_to_handler_registry_from_global_registry(
                self._handler_registry,
                handler,
                item,
            )
          elif handler is not None:
            logging.warning(
                'No registered CheckpointArgs found for handler type: %s',
                type(handler),
            )
            legacy_wrapped_handler = get_legacy_handler_wrapper(handler)
            self._handler_registry.add(
                item,
                _WrapperArgs,
                legacy_wrapped_handler,
            )
          else:
            # If the handler is `None`, the handler will be determined from the
            # global checkpoint args registry during the first call to
            # `_get_or_set_handler`.
            self._item_names_without_registered_handlers.add(item)

      self._items_with_deferred_registration = None
    elif handler_registry is not None:
      # Create a new handler registry to prevent the input from being mutated.
      self._handler_registry = (
          handler_registration.DefaultCheckpointHandlerRegistry(
              handler_registry
          )
      )
      self._item_names_without_registered_handlers = None
      self._items_with_deferred_registration = set()

      # Prevent the user from lazily registering items that are already
      # registered in the provided handler registry.
      for item, _ in _get_unique_registered_items_and_handlers(
          self._handler_registry
      ):
        self._items_with_deferred_registration.add(item)
    else:
      self._handler_registry = (
          handler_registration.DefaultCheckpointHandlerRegistry()
      )
      self._item_names_without_registered_handlers = None
      self._items_with_deferred_registration = set()

    self._primary_host = composite_options.multiprocessing_options.primary_host
    self._active_processes = (
        composite_options.multiprocessing_options.active_processes
    )
    self._barrier_sync_key_prefix = (
        composite_options.multiprocessing_options.barrier_sync_key_prefix
    )
    self._multiprocessing_options = composite_options.multiprocessing_options

    self._temporary_path_class = composite_options.temporary_path_class
    self._file_options = composite_options.file_options

    logging.info(
        'Initialized registry %s.',
        self._handler_registry,
    )

  # TODO: b/359524229 - Remove this property as it has been deprecated.
  @property
  def _known_handlers(self) -> Dict[str, Optional[CheckpointHandler]]:
    logging.info(
        '`_known_handlers` has been deprecated and should not be used.'
    )
    return dict(self._get_all_registered_and_unregistered_items_and_handlers())

  def _get_all_registered_and_unregistered_items_and_handlers(
      self,
  ) -> list[tuple[str, Optional[CheckpointHandler]]]:
    """Gets all items and corresponding handlers.

    Can be rmeoved once `item_names` and `items_and_handlers` have been
    completely deprecated.

    Returns:
      A tuple containing all uniuqe items and handlers registered in the handler
      registry. If `item_names` or `items_and_handlers` has been specified, the
      tuple may also contain item without a corresponding checkpoint handler.
    """
    items_and_handlers = _get_unique_registered_items_and_handlers(
        self._handler_registry
    )
    if self._item_names_without_registered_handlers is not None:
      for item in self._item_names_without_registered_handlers:
        items_and_handlers.append((item, None))
    return items_and_handlers

  def _get_or_set_handler(
      self,
      item_name: str,
      args: Optional[CheckpointArgs],
  ) -> CheckpointHandler:
    handler_registry = self._handler_registry
    assert handler_registry is not None
    if args is not None:
      # Check if registry contains a handler for the item name or the
      # the general argument type.
      if handler_registry.has(item_name, args):
        return handler_registry.get(item_name, args)
      elif handler_registry.has(None, args):
        handler = handler_registry.get(None, args)
        # There is a default handler for the args, but it is not registered
        # for the specific item. Register the handler for the item.
        handler_registry.add(item_name, args, handler)
        return handler
      else:
        logging.info(
            'No entry found in handler registry for item: %s and args with'
            ' type: %s. Falling back to global handler registry.',
            item_name,
            type(args),
        )
    else:
      # Try to find a handler for the item. If none is found, raise an error.
      items_and_handlers = _get_unique_registered_items_and_handlers(
          handler_registry
      )
      for item, handler in items_and_handlers:
        if item_name == item:
          assert handler is not None
          return handler
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

    # Retrieve and default initialize the handler from the global registry. This
    # handler will be added to the handler registry if it is not already
    # registered for the item.
    registered_handler_for_args = (
        checkpoint_args.get_registered_handler_cls(args)
    )()  # pytype: disable=not-instantiable

    # Add items from `item_names` to the handler registry if it is not already
    # registered. This happens on the first call of `_get_or_set_handler` for
    # each item. `self._item_names_without_registered_handlers` is only set if
    # the user provided `item_names` or `items_and_handlers` at initialization.
    item_names_without_registered_handlers = (
        self._item_names_without_registered_handlers
    )
    if item_names_without_registered_handlers is not None:
      if item_name in item_names_without_registered_handlers:
        item_names_without_registered_handlers.remove(item_name)
        _add_to_handler_registry_from_global_registry(
            handler_registry,
            registered_handler_for_args,
            item_name,
        )
        return registered_handler_for_args
    else:
      # This branch is only reached if the user provided a `handler_registry` at
      # initialization. Any item that is not registered in the provided handler
      # registry will be registered in a deferred manner. This allows the user
      # to provide a `handler_registry` at initialization that does not need to
      # contain all the items that will be checkpointed, as long as the
      # checkpoint args for the item have been registered globally with
      # a corresponding handler.
      assert self._items_with_deferred_registration is not None
      if item_name not in self._items_with_deferred_registration:
        # Add the item to the set of items with deferred registration to prevent
        # adding the same item multiple times to the handler registry.
        self._items_with_deferred_registration.add(item_name)
        _add_to_handler_registry_from_global_registry(
            handler_registry,
            registered_handler_for_args,
            item_name,
        )
        return registered_handler_for_args

    raise ValueError(
        f'Item "{item_name}" and args "{args}" does not match with any'
        ' registered handler! Registered handlers:'
        f' {self._handler_registry.get_all_entries().items()}'
    )

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
        multiprocessing_options=self._multiprocessing_options,
        file_options=self._file_options,
    )
    return tmp_item_dir

  def _get_item_temporary_paths(
      self, directory: epath.Path, args: 'CompositeArgs'
  ) -> Dict[str, atomicity.TemporaryPath]:
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
    result = {
        item_name: item_directory
        for item_name, item_directory in zip(item_names, item_temporary_paths)
    }

    return result


  async def async_save(
      self, directory: epath.Path, args: 'CompositeArgs'
  ) -> Optional[List[Future]]:
    """Saves multiple items to individual subdirectories."""
    self._current_temporary_paths = self._get_item_temporary_paths(
        directory, args
    )
    await atomicity.create_all(
        list(self._current_temporary_paths.values()),
        multiprocessing_options=self._multiprocessing_options,
    )
    save_ops = []
    for item_name, item_directory in self._current_temporary_paths.items():
      arg = args[item_name]
      _maybe_raise_reserved_item_error(item_name)
      handler = self._get_or_set_handler(item_name, arg)
      if isinstance(handler, AsyncCheckpointHandler):
        save_ops.append(handler.async_save(item_directory.get(), args=arg))
      else:
        # Blocking save.
        handler.save(item_directory.get(), args=arg)

    commit_futures = jax.tree.flatten(await asyncio.gather(*save_ops))[0]
    return commit_futures or []

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

    items_and_handlers = (
        self._get_all_registered_and_unregistered_items_and_handlers()
    )

    all_items = [item_name for item_name, _ in items_and_handlers]
    if args is not None:
      all_items.extend(args.keys())

    items_exist = self._items_exist(directory, all_items)

    if args is None or not args.items():
      if self._item_names_without_registered_handlers:
        raise ValueError(
            ' No `CheckpointArgs` were provided, and the following items were'
            ' not registered: %s. Providing `None` for the item is valid when'
            ' restoring, but only if the `CheckpointHandler` was already'
            ' configured. Provide a `CheckpointArgs` subclass so that'
            ' `CompositeCheckpointHandler` will know how to restore the item.'
            % self._item_names_without_registered_handlers
        )
      composite_args_items = {}
      for item_name, handler in items_and_handlers:
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
    items_and_handlers = (
        self._get_all_registered_and_unregistered_items_and_handlers()
    )
    items_exist = self._items_exist(
        directory, [item_name for item_name, _ in items_and_handlers]
    )

    metadata = {}

    for item_name, handler in items_and_handlers:
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
    for item_name, handler in _get_unique_registered_items_and_handlers(
        self._handler_registry
    ):
      tmp_dir = self._current_temporary_paths.get(item_name, None)
      if tmp_dir is None or handler is None:
        # Not an error, as some items may not have been saved.
        continue
      handler.finalize(tmp_dir.get())
      tmp_dir.finalize()

  def close(self):
    for _, handler in _get_unique_registered_items_and_handlers(
        self._handler_registry
    ):
      if handler is not None:
        handler.close()


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
