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

"""Defines `CompositeHandler`, a helper component for saving and loading."""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, List

from absl import logging
from etils import epath
from orbax.checkpoint._src import composite
from orbax.checkpoint._src.metadata import checkpoint as checkpoint_metadata
from orbax.checkpoint._src.metadata import step_metadata_serialization
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.handlers import registration
import orbax.checkpoint.experimental.v1._src.handlers.global_registration  # pylint: disable=unused-import
from orbax.checkpoint.experimental.v1._src.path import types as path_types


StepMetadata = checkpoint_metadata.StepMetadata
CompositeItemMetadata = checkpoint_metadata.CompositeItemMetadata
Composite = composite.Composite


class CompositeHandler:
  """CompositeHandler.

  This class is a helper component for `save_checkpointables` and
  `load_checkpointables`. It performs a few core functions:
    - Resolves handlers for saving and loading.
    - Saves and loads checkpointables to/from individual subdirectories by
    delegating to the resolved handlers.

  TODO(b/396190818): This code is only minimally used in the main save/load
  path. It should eventually be incorporated there when the dependency on V0
  implementations is removed.
  """

  def __init__(
      self,
      handler_registry: (
          registration.CheckpointableHandlerRegistry | None
      ) = None,
  ):
    self._context = context_lib.get_context()
    # Create a copy to prevent mutation by the caller.
    self._handler_registry = registration.local_registry(
        registration.global_registry()
    )
    if handler_registry is not None:
      for handler, checkpointable in handler_registry.get_all_entries():
        self._handler_registry.add(handler, checkpointable)

    logging.vlog(
        1,
        'Initialized CompositeHandler with registry: %s.',
        self._handler_registry,
    )

    self._metadata_store = checkpoint_metadata.metadata_store(enable_write=True)

  @property
  def handler_registry(
      self,
  ) -> registration.CheckpointableHandlerRegistry:
    return self._handler_registry

  def _get_checkpointable_directory(
      self, directory: path_types.Path, checkpointable_name: str
  ) -> path_types.Path:
    return directory / checkpointable_name

  async def save(
      self, directory: path_types.Path, checkpointables: dict[str, Any]
  ) -> Awaitable[None]:
    """Saves multiple checkpointables to individual subdirectories.

    The subdirectories are expected to already exist.

    Args:
      directory: The directory to save the checkpointables to. The
        checkpointables subdirectories exist under this directory.
      checkpointables: A mapping from checkpointable name to checkpointable.

    Returns:
      An awaitable that represents a background save operation.
    """
    save_ops = []
    for name, checkpointable in checkpointables.items():
      handler = registration.resolve_handler_for_save(
          self._handler_registry, checkpointable, name=name
      )
      checkpointable_directory = self._get_checkpointable_directory(
          directory, name
      )
      save_ops.append(handler.save(checkpointable_directory, checkpointable))
    save_awaitables = await asyncio.gather(*save_ops)

    async def _run_background():
      await asyncio.gather(*save_awaitables)

    return _run_background()

  def _existing_items(self, directory: epath.Path) -> List[str]:
    return [p.name for p in directory.iterdir() if p.is_dir()]

  async def load(
      self,
      directory: path_types.Path,
      abstract_checkpointables: dict[str, Any] | None = None,
  ) -> Awaitable[dict[str, Any]]:
    """Loads multiple checkpointables from individual subdirectories.

    Args:
      directory: The directory to load the checkpointables from. The
        checkpointables subdirectories exist under this directory.
      abstract_checkpointables: A mapping from checkpointable name to abstract
        checkpointable. Only the checkpointables listed will be loaded. If None,
        all checkpointables in the directory will be loaded.

    Returns:
      An awaitable that represents a background load operation.
    """
    abstract_checkpointables = abstract_checkpointables or {}
    loadable_checkpointable_names_to_handlers = self._get_loadable_handlers(
        directory, abstract_checkpointables
    )

    load_ops = []
    for (
        checkpointable_name,
        handler,
    ) in loadable_checkpointable_names_to_handlers.items():
      checkpointable_directory = self._get_checkpointable_directory(
          directory, checkpointable_name
      )
      abstract_checkpointable = (
          abstract_checkpointables[checkpointable_name]
          if checkpointable_name in abstract_checkpointables
          else None
      )
      load_ops.append(
          handler.load(
              checkpointable_directory,
              abstract_checkpointable,
          )
      )
    load_awaitables = await asyncio.gather(*load_ops)

    async def _run_background() -> dict[str, Any]:
      loaded_checkpointables = []
      # TODO(b/398249409) Cannot use asyncio.gather because asyncio.run
      # is used in underlying implementation.
      for a in load_awaitables:
        loaded = await a
        loaded_checkpointables.append(loaded)
      return {
          checkpointable_name: loaded
          for checkpointable_name, loaded in zip(
              loadable_checkpointable_names_to_handlers.keys(),
              loaded_checkpointables,
          )
      }

    return _run_background()

  def _get_loadable_handlers(
      self, directory: path_types.Path, abstract_checkpointables: dict[str, Any]
  ):
    """Returns a mapping from checkpointable name to loadable handler."""
    existing_checkpointable_names_to_handler_typestrs = (
        self._get_saved_handler_typestrs(directory)
    )
    abstract_checkpointables = abstract_checkpointables or {
        name: None for name in existing_checkpointable_names_to_handler_typestrs
    }

    loadable_checkpointable_names_to_handlers = {}
    for name, abstract_checkpointable in abstract_checkpointables.items():
      if name not in existing_checkpointable_names_to_handler_typestrs:
        raise KeyError(
            f'Checkpointable "{name}" was not found in the checkpoint.'
            ' Available names:'
            f' {existing_checkpointable_names_to_handler_typestrs.keys()}'
        )
      handler_typestr = existing_checkpointable_names_to_handler_typestrs[name]
      handler = registration.resolve_handler_for_load(
          self._handler_registry,
          abstract_checkpointable,
          name=name,
          handler_typestr=handler_typestr,
      )
      loadable_checkpointable_names_to_handlers[name] = handler

    return loadable_checkpointable_names_to_handlers

  def _get_saved_handler_typestrs(
      self,
      directory: path_types.Path,
  ) -> dict[str, str]:
    """Reads from the checkpoint metadata to get saved handler typestrs."""
    serialized_metadata = self._metadata_store.read(
        checkpoint_metadata.step_metadata_file_path(directory)
    )
    saved_metadata = step_metadata_serialization.deserialize(
        serialized_metadata or {}
    )
    assert isinstance(saved_metadata.item_handlers, dict)
    return saved_metadata.item_handlers
