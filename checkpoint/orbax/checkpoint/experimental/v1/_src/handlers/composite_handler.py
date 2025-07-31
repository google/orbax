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

"""Defines `CompositeHandler`, a helper component for saving and loading."""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable

from absl import logging
from etils import epath
from orbax.checkpoint._src.metadata import checkpoint as checkpoint_metadata
from orbax.checkpoint._src.metadata import step_metadata_serialization
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.path import async_path
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.handlers import registration
from orbax.checkpoint.experimental.v1._src.handlers import types as handler_types
import orbax.checkpoint.experimental.v1._src.handlers.global_registration  # pylint: disable=unused-import
from orbax.checkpoint.experimental.v1._src.path import format_utils
from orbax.checkpoint.experimental.v1._src.path import types as path_types


StepMetadata = checkpoint_metadata.StepMetadata
CompositeItemMetadata = checkpoint_metadata.CompositeItemMetadata

ORBAX_CHECKPOINT_INDICATOR_FILE = 'orbax.checkpoint'


_V0_ERROR_MESSAGE = (
    'If your checkpoint was saved with the Orbax V0 API, please follow the'
    ' instructions at'
    ' https://orbax.readthedocs.io/en/latest/guides/checkpoint/v1/orbax_v0_to_v1_migration.html'
    ' to load it with the Orbax V1 API.'
)


def _existing_checkpointable_names(directory: epath.Path) -> set[str]:
  return {p.name for p in directory.iterdir() if p.is_dir()}


async def _create_orbax_identifier_file(
    directory: path_types.PathAwaitingCreation, primary_host: int | None
):
  """Creates a file called `orbax.checkpoint` for easy identification."""
  directory = await directory.await_creation()
  if multihost.is_primary_host(primary_host):
    await async_path.touch(
        directory / 'orbax.checkpoint', exist_ok=False
    )


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
      handler_registry: registration.CheckpointableHandlerRegistry,
  ):
    self._context = context_lib.get_context()
    # Create a copy to prevent mutation by the caller.
    self._handler_registry = registration.local_registry(
        handler_registry, include_global_registry=False
    )
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

  async def save(
      self,
      directory: path_types.PathAwaitingCreation,
      checkpointables: dict[str, Any],
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
    context = context_lib.get_context()
    handlers_for_save = self.get_handlers_for_save(checkpointables)
    save_ops = []
    for checkpointable_name, checkpointable in checkpointables.items():
      save_ops.append(
          handlers_for_save[checkpointable_name].save(
              directory / checkpointable_name, checkpointable
          )
      )
    save_awaitables = await asyncio.gather(*save_ops)

    async def _run_background():
      await asyncio.gather(
          *save_awaitables,
          _create_orbax_identifier_file(
              directory, context.multiprocessing_options.primary_host
          ),
      )

    return _run_background()

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

    Raises:
      KeyError: If any of the specified checkpointable names are not found in
      the checkpoint.
    """
    abstract_checkpointables = abstract_checkpointables or {}
    handlers_for_load = self.get_handlers_for_load(
        directory, abstract_checkpointables
    )
    existing_checkpointable_names = _existing_checkpointable_names(directory)
    if not abstract_checkpointables:
      abstract_checkpointables = {
          name: None
          for name in handlers_for_load.keys()
          if name not in format_utils.RESERVED_CHECKPOINTABLE_KEYS
          and name in existing_checkpointable_names
      }
    if any(
        name not in existing_checkpointable_names
        for name in abstract_checkpointables.keys()
    ):
      raise KeyError(
          'Requested checkpointables for loading were not found in the'
          ' checkpoint. Available checkpointables:'
          f' {existing_checkpointable_names}'
      )

    load_ops = []
    for (
        checkpointable_name,
        abstract_checkpointable,
    ) in abstract_checkpointables.items():
      handler = handlers_for_load[checkpointable_name]
      load_ops.append(
          handler.load(
              directory / checkpointable_name,
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
              abstract_checkpointables.keys(),
              loaded_checkpointables,
          )
      }

    return _run_background()

  def get_handlers_for_save(
      self, checkpointables: dict[str, Any]
  ) -> dict[str, handler_types.CheckpointableHandler]:
    """Returns a mapping from checkpointable name to handler."""
    return {
        checkpointable_name: registration.resolve_handler_for_save(
            self._handler_registry, checkpointable, name=checkpointable_name
        )
        for checkpointable_name, checkpointable in checkpointables.items()
    }

  def get_handlers_for_load(
      self, directory: path_types.Path, abstract_checkpointables: dict[str, Any]
  ) -> dict[str, handler_types.CheckpointableHandler]:
    """Returns a mapping from checkpointable name to handler."""
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
      self, directory: path_types.Path
  ) -> dict[str, str]:
    """Reads from the checkpoint metadata to get saved handler typestrs."""
    step_metadata_file_path = checkpoint_metadata.step_metadata_file_path(
        directory
    )
    if step_metadata_file_path.exists():
      serialized_metadata = self._metadata_store.read(step_metadata_file_path)
      saved_metadata = step_metadata_serialization.deserialize(
          serialized_metadata or {}
      )
      if isinstance(saved_metadata.item_handlers, dict):
        return saved_metadata.item_handlers  # found step level metadata.
      raise ValueError(
          f'Path at {directory} contains subdirectories:'
          f' {format_utils.subdirs(directory)}, which are expected to match the'
          ' keys given by the _CHECKPOINT_METADATA file:'
          f' {saved_metadata.item_handlers}. If you intended to load a pytree'
          ' checkpoint from the given path, then please consider using'
          ' `loading.load_pytree(..., checkpointable_name=None)` instead.'
          f' {_V0_ERROR_MESSAGE}'
      )

    logging.warning(
        'Given dir does not contain checkpoint metadata file: %s. Trying to get'
        ' saved handlers from checkpoint metadata in each of the checkpointable'
        ' subdirectory.',
        directory,
    )

    saved_handler_typestrs: dict[str, str] = {}
    for checkpointable_path in directory.iterdir():
      serialized_metadata = self._metadata_store.read(
          checkpoint_metadata.step_metadata_file_path(checkpointable_path)
      )
      if serialized_metadata is None:
        continue
      saved_metadata = step_metadata_serialization.deserialize(
          serialized_metadata
      )
      if isinstance(saved_metadata.item_handlers, dict):
        raise ValueError(
            f'Path at {directory} contains subdirectories:'
            f' {format_utils.subdirs(directory)}, which are expected to match'
            ' the keys given by the _CHECKPOINT_METADATA file:'
            f' {saved_metadata.item_handlers}. If you intended to load a pytree'
            ' checkpoint from the given path, then please consider using'
            ' `loading.load_pytree(..., checkpointable_name=None)` instead.'
            f' {_V0_ERROR_MESSAGE}'
        )
      item_handlers = saved_metadata.item_handlers
      if item_handlers is not None:
        checkpointable_name = checkpointable_path.name
        saved_handler_typestrs[checkpointable_name] = item_handlers
    return saved_handler_typestrs
