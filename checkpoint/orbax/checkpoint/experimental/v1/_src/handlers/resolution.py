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


async def get_handlers_for_load(
    handler_registry: registration.CheckpointableHandlerRegistry,
    abstract_checkpointables: dict[str, Any],
    checkpoint_metadata: InternalCheckpointMetadata,
) -> dict[str, handler_types.CheckpointableHandler | None]:
  """Returns a mapping from checkpointable name to handler.

  Gathers and returns a mapping from checkpointable name to handler by
  checking the following in order:

  1. Check for handler_typestr in checkpoint metadata item_handlers using
  checkpointable_name as key.
  2. resolve_handler_for_load performs handler discovery based on
  abstract_checkpointable type and handler_typestr.
  3. Try to resolve using the default pytree handler if registered.
  4. If no handler is resolved, default to None.

  Args:
    handler_registry: The handler registry to use for resolution.
    abstract_checkpointables: The abstract checkpointables to load.
    checkpoint_metadata: InternalCheckpointMetadata to read handler_typestr(s)
      from.

  Returns:
    A mapping from checkpointable name to handler.
  """
  handlers_for_load: dict[str, handler_types.CheckpointableHandler | None] = {}
  for (
      checkpointable_name,
      abstract_checkpointable,
  ) in abstract_checkpointables.items():
    resolved_handler = None
    # 1. Find handler_typestr in checkpoint metadata item_handlers using
    # checkpointable_name as key.
    metadata_handler_typestr = _get_saved_handler_typestr(
        checkpointable_name, checkpoint_metadata
    )

    # 2. Resolve the handler using handler_typestr and
    # abstract_checkpointable type if either is specified.
    if abstract_checkpointable or metadata_handler_typestr:
      try:
        resolved_handler = registration.resolve_handler_for_load(
            handler_registry,
            abstract_checkpointable,
            name=checkpointable_name,
            handler_typestr=metadata_handler_typestr,
        )
      except registration.NoEntryError as e:
        logging.warning(
            "Failed to resolve handler for checkpointable %s: %s. Attempting to"
            " load using pytree handler, otherwise defaulting to a None"
            " return value.",
            checkpointable_name,
            e,
        )
    else:
      logging.info(
          "No metadata present in checkpoint and no abstract checkpointable"
          " provided for checkpointable: \'%s\'. Attempting to load using"
          " pytree handler, otherwise defaulting to a None return value.",
          checkpointable_name,
      )

    # 4. If no handler is resolved yet, try to resolve using the default
    # pytree handler.
    if not resolved_handler:
      resolved_handler = registration.get_registered_handler_by_name(
          handler_registry, "pytree"
      )

    # 5. If no handler is resolved yet, default to None.
    handlers_for_load[checkpointable_name] = resolved_handler
  return handlers_for_load


async def get_handler_for_load_direct_pytree(
    checkpointable_name: str,
    handler_registry: registration.CheckpointableHandlerRegistry,
    abstract_checkpointable: Any,
    checkpoint_metadata: InternalCheckpointMetadata,
) -> handler_types.CheckpointableHandler | None:
  """Returns a handler for direct load of a pytree checkpoint.

  1. Check for checkpointable_name in checkpoint metadata item_handlers using
  checkpointable_name as key.
  2. resolve_handler_for_load performs handler discovery based on
  abstract_checkpointable type and handler_typestr.
  3. Try to resolve using the default pytree handler if registered.
  4. If no handler is resolved, default to None.

  Args:
    checkpointable_name: The checkpointable name to resolve the handler for.
    handler_registry: The handler registry to use for resolution.
    abstract_checkpointable: The abstract checkpointable to load.
    checkpoint_metadata: InternalCheckpointMetadata to read handler_typestr
      from.

  Returns:
    The handler for direct load of a pytree checkpoint.
  """
  # 1. Check if the handler is explicitly registered.
  resolved_handler = registration.get_registered_handler_by_name(
      handler_registry, checkpointable_name
  )
  if resolved_handler:
    return resolved_handler

  # 2. Find handler_typestr in checkpoint metadata item_handlers if checkpoint
  # metadata is in direct pytree format (str).
  metadata_handler_typestr = None
  if isinstance(checkpoint_metadata.item_handlers, str):
    metadata_handler_typestr = checkpoint_metadata.item_handlers

  # 3. Resolve the handler using handler_typestr and
  # abstract_checkpointable type if either is specified.
  if abstract_checkpointable or metadata_handler_typestr:
    try:
      resolved_handler = registration.resolve_handler_for_load(
          handler_registry,
          abstract_checkpointable,
          name=checkpointable_name,
          handler_typestr=metadata_handler_typestr,
      )
    except registration.NoEntryError as e:
      logging.warning(
          "Failed to resolve handler for checkpointable %s: %s. Attempting to"
          " load using default pytree handler, otherwise defaulting to a"
          " None return value.",
          checkpointable_name,
          e,
      )
  else:
    logging.info(
        "No metadata present in checkpoint and no abstract checkpointable"
        " provided for checkpointable: \'%s\'. Attempting to load using"
        " pytree handler, otherwise defaulting to a None return value.",
        checkpointable_name,
    )

  # 4. If no handler is resolved yet, try to resolve using the default
  # pytree handler.
  if not resolved_handler:
    resolved_handler = registration.get_registered_handler_by_name(
        handler_registry, "pytree"
    )

  # 5. If no handler is resolved yet, default to None.
  return resolved_handler


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

