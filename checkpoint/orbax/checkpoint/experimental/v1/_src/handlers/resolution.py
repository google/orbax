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

"""Logic for resolving handlers for saving and loading."""
from __future__ import annotations

import itertools
from typing import Any

from absl import logging
from orbax.checkpoint._src.metadata import step_metadata_serialization
from orbax.checkpoint._src.path import async_path
from orbax.checkpoint.experimental.v1._src.handlers import registration
from orbax.checkpoint.experimental.v1._src.handlers import types as handler_types
import orbax.checkpoint.experimental.v1._src.handlers.global_registration  # pylint: disable=unused-import
from orbax.checkpoint.experimental.v1._src.metadata import serialization as metadata_serialization
from orbax.checkpoint.experimental.v1._src.path import types as path_types

InternalCheckpointMetadata = (
    step_metadata_serialization.InternalCheckpointMetadata
)


def _subdirs(directory: path_types.Path, *, limit: int = 3) -> list[str]:
  return list(
      itertools.islice(
          (subdir.name for subdir in directory.iterdir() if subdir.is_dir()),
          limit,
      )
  )


_V0_ERROR_MESSAGE = (
    'If your checkpoint was saved with the Orbax V0 API, please follow the'
    ' instructions at'
    ' https://orbax.readthedocs.io/en/latest/guides/checkpoint/v1/orbax_v0_to_v1_migration.html'
    ' to load it with the Orbax V1 API.'
)


async def read_checkpoint_metadata(
    directory: path_types.Path,
) -> InternalCheckpointMetadata:
  """Returns the step metadata for a given path, normalized for V1."""

  serialized_metadata = (
      await metadata_serialization.read(
          metadata_serialization.checkpoint_metadata_file_path(directory)
      )
      or {}
  )

  return InternalCheckpointMetadata.deserialize(serialized_metadata)


def get_handlers_for_save(
    handler_registry: registration.CheckpointableHandlerRegistry,
    checkpointables: dict[str, Any],
) -> dict[str, handler_types.CheckpointableHandler]:
  """Returns a mapping from checkpointable name to handler."""
  return {
      checkpointable_name: registration.resolve_handler_for_save(
          handler_registry, checkpointable, name=checkpointable_name
      )
      for checkpointable_name, checkpointable in checkpointables.items()
  }


async def get_handlers_for_load(
    directory: path_types.Path,
    handler_registry: registration.CheckpointableHandlerRegistry,
    abstract_checkpointables: dict[str, Any],
) -> dict[str, handler_types.CheckpointableHandler]:
  """Returns a mapping from checkpointable name to handler."""
  existing_checkpointable_names_to_handler_typestrs = (
      await _get_saved_handler_typestrs(directory)
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
        handler_registry,
        abstract_checkpointable,
        name=name,
        handler_typestr=handler_typestr,
    )
    loadable_checkpointable_names_to_handlers[name] = handler
  return loadable_checkpointable_names_to_handlers


async def _get_saved_handler_typestrs(
    directory: path_types.Path,
) -> dict[str, str]:
  """Reads from the checkpoint metadata to get saved handler typestrs."""
  checkpoint_metadata_file_path = (
      metadata_serialization.checkpoint_metadata_file_path(directory)
  )
  if await async_path.exists(checkpoint_metadata_file_path):
    checkpoint_metadata = await read_checkpoint_metadata(directory)
    if isinstance(checkpoint_metadata.item_handlers, dict):
      return checkpoint_metadata.item_handlers  # found step level metadata.
    raise ValueError(
        f'Path at {directory} contains subdirectories:'
        f' {_subdirs(directory)}, which are expected to'
        ' match the keys given by the _CHECKPOINT_METADATA file:'
        f' {checkpoint_metadata.item_handlers}. If you intended to load a'
        ' pytree checkpoint from the given path, then please consider using'
        ' `loading.load_pytree(..., checkpointable_name=None)` instead.'
        f' {_V0_ERROR_MESSAGE}'
    )

  logging.warning(
      'Given dir does not contain checkpoint metadata file: %s. Trying to get'
      ' saved handlers from checkpoint metadata in each of the checkpointable'
      ' subdirectory.',
      directory,
  )

  # TODO(b/475265289): Currently, we rely solely on CHECKPOINT_METADATA to
  # find available checkpointables, ignoring valid subdirectories. We
  # should update the composite handler to validate subdirectories to
  # check if any either represents a valid pytree checkpointable or has a
  # name that is registered in the handler registry.
  saved_handler_typestrs: dict[str, str] = {}
  for checkpointable_path in await async_path.iterdir(directory):
    if not await async_path.is_dir(checkpointable_path):
      continue
    checkpoint_metadata = await read_checkpoint_metadata(checkpointable_path)
    if isinstance(checkpoint_metadata.item_handlers, dict):
      raise ValueError(
          f'Path at {directory} contains subdirectories:'
          f' {_subdirs(directory)}, which are expected to'
          ' match the keys given by the _CHECKPOINT_METADATA file:'
          f' {checkpoint_metadata.item_handlers}. If you intended to load a'
          ' pytree checkpoint from the given path, then please consider using'
          ' `loading.load_pytree(..., checkpointable_name=None)` instead.'
          f' {_V0_ERROR_MESSAGE}'
      )
    item_handlers = checkpoint_metadata.item_handlers
    if item_handlers is not None:
      checkpointable_name = checkpointable_path.name
      saved_handler_typestrs[checkpointable_name] = item_handlers
  return saved_handler_typestrs
