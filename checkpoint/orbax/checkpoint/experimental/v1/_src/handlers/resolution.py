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

from typing import Any

from absl import logging
from orbax.checkpoint._src.metadata import step_metadata_serialization
from orbax.checkpoint.experimental.v1._src.handlers import registration
from orbax.checkpoint.experimental.v1._src.handlers import types as handler_types
import orbax.checkpoint.experimental.v1._src.handlers.global_registration  # pylint: disable=unused-import

InternalCheckpointMetadata = (
    step_metadata_serialization.InternalCheckpointMetadata
)


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


def _resolve_single_handler_for_load(
    checkpointable_name: str,
    handler_registry: registration.CheckpointableHandlerRegistry,
    abstract_checkpointable: Any,
    metadata_handler_typestr: str | None,
) -> handler_types.CheckpointableHandler:
  """Logic to resolve a checkpointable's loading handler.

  1. registration.resolve_handler_for_load performs handler discovery based on
  abstract_checkpointable type and handler_typestr.
  2. If this fails or if abstract_checkpointable and handler_typestr are not
  available, we try to resolve using the default pytree handler if registered.

  Args:
    checkpointable_name: The checkpointable name to resolve the handler for.
    handler_registry: The handler registry to use for resolution.
    abstract_checkpointable: The abstract checkpointable to load.
    metadata_handler_typestr: The handler typestr from the checkpoint metadata.

  Returns:
    The handler for the checkpointable.

  Raises:
    registration.NoEntryError: If no handler is resolved and 'pytree' name is
    not registered.
  """
  # 1. Resolve the checkpointable's handler using handler discovery.
  try:
    return registration.resolve_handler_for_load(
        handler_registry,
        abstract_checkpointable,
        name=checkpointable_name,
        handler_typestr=metadata_handler_typestr,
    )
  except registration.NoEntryError as e:
    logging.warning(
        "Failed to resolve handler for checkpointable: '%s'. Attempting to"
        " load using pytree handler. Error: %s",
        checkpointable_name,
        e,
    )

  # 2. If no handler is resolved yet, try to resolve using the default
  # pytree handler.
  pytree_handler = registration.get_registered_handler_by_name(
      handler_registry, "pytree"
  )
  if not pytree_handler:
    raise registration.NoEntryError(
        f"Could not resolve a handler for '{checkpointable_name}' and no"
        f" 'pytree' handler found in {handler_registry})."
        " Please inspect the checkpoint contents via"
        " `loading.checkpointables_metadata`. You may need to provide an"
        " abstract_checkpointable or register a missing handler for this name"
        " or for 'pytree' name which is used as a fallback."
    )
  return pytree_handler


async def get_handlers_for_load(
    handler_registry: registration.CheckpointableHandlerRegistry,
    abstract_checkpointables: dict[str, Any],
    checkpoint_metadata: InternalCheckpointMetadata,
) -> dict[str, handler_types.CheckpointableHandler]:
  """Returns a mapping from checkpointable name to handler.

  Gathers and returns a mapping from checkpointable name to handler by
  checking the following in order:

  1. Check for handler_typestr in checkpoint metadata item_handlers using
  checkpointable_name as key.
  2. Find the handler for each checkpointable using
  _resolve_single_handler_for_load.
  3. If no handler is resolved for a checkpointable, raise a NoEntryError.

  Args:
    handler_registry: The handler registry to use for resolution.
    abstract_checkpointables: The abstract checkpointables to load.
    checkpoint_metadata: InternalCheckpointMetadata to read handler_typestr(s)
      from.

  Returns:
    A mapping from checkpointable name to handler.

  Raises:
    registration.NoEntryError: If no handler is resolved.
  """
  handlers_for_load: dict[str, handler_types.CheckpointableHandler] = {}
  for (
      checkpointable_name,
      abstract_checkpointable,
  ) in abstract_checkpointables.items():
    metadata_handler_typestr = _get_saved_handler_typestr(
        checkpointable_name, checkpoint_metadata
    )
    handlers_for_load[checkpointable_name] = _resolve_single_handler_for_load(
        checkpointable_name,
        handler_registry,
        abstract_checkpointable,
        metadata_handler_typestr,
    )
  return handlers_for_load


async def get_handler_for_load_direct_pytree(
    checkpointable_name: str,
    handler_registry: registration.CheckpointableHandlerRegistry,
    abstract_checkpointable: Any,
    checkpoint_metadata: InternalCheckpointMetadata,
) -> handler_types.CheckpointableHandler:
  """Returns a handler for direct load of a pytree checkpoint.

  1. Check for checkpointable_name in checkpoint metadata item_handlers.
  2. resolve_handler_for_load performs handler discovery based on
  abstract_checkpointable type and handler_typestr.
  2. Find the handler for each checkpointable using
  _resolve_single_handler_for_load.
  3. If no handler is resolved for a checkpointable, raise a NoEntryError.

  Args:
    checkpointable_name: The checkpointable name to resolve the handler for.
    handler_registry: The handler registry to use for resolution.
    abstract_checkpointable: The abstract checkpointable to load.
    checkpoint_metadata: InternalCheckpointMetadata to read handler_typestr
      from.

  Returns:
    The handler for direct load of a pytree checkpoint.
  """
  metadata_handler_typestr = _get_saved_handler_typestr_direct_pytree(
      checkpoint_metadata
  )
  return _resolve_single_handler_for_load(
      checkpointable_name,
      handler_registry,
      abstract_checkpointable,
      metadata_handler_typestr,
  )


def _get_saved_handler_typestr(
    checkpointable_name: str,
    checkpoint_metadata: InternalCheckpointMetadata,
) -> str | None:
  """Reads from the checkpoint metadata to get saved handler typestrs."""
  if isinstance(checkpoint_metadata.item_handlers, dict) and (
      checkpointable_name in checkpoint_metadata.item_handlers
  ):
    return checkpoint_metadata.item_handlers[checkpointable_name]
  return None


def _get_saved_handler_typestr_direct_pytree(
    checkpoint_metadata: InternalCheckpointMetadata,
) -> str | None:
  """Reads from the checkpoint metadata to get saved handler typestrs."""
  if isinstance(checkpoint_metadata.item_handlers, str):
    return checkpoint_metadata.item_handlers
  return None
