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

"""Utilities for v0 checkpoint compatibility."""

from typing import Any

from orbax.checkpoint._src.checkpointers import async_checkpointer
from orbax.checkpoint._src.handlers import composite_checkpoint_handler
from orbax.checkpoint._src.handlers import handler_registration as legacy_handler_registration
from orbax.checkpoint._src.serialization import type_handlers
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.handlers import compatibility as handler_compatibility
from orbax.checkpoint.experimental.v1._src.handlers import composite_handler
from orbax.checkpoint.experimental.v1._src.loading import validation
from orbax.checkpoint.experimental.v1._src.metadata import types as metadata_types
from orbax.checkpoint.experimental.v1._src.path import format_utils
from orbax.checkpoint.experimental.v1._src.path import types as path_types
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types


PYTREE_CHECKPOINTABLE_KEY = format_utils.PYTREE_CHECKPOINTABLE_KEY
AbstractPyTree = tree_types.PyTreeOf[tree_types.AbstractLeafType]
CheckpointMetadata = metadata_types.CheckpointMetadata
PLACEHOLDER = type_handlers.PLACEHOLDER


def get_v0_checkpointer_and_args(
    path: path_types.Path,
    abstract_checkpointables: dict[str, Any] | None,
    *,
    context: context_lib.Context,
) -> tuple[
    async_checkpointer.AsyncCheckpointer,
    composite_checkpoint_handler.CompositeArgs,
]:
  """Construct V0 Checkpointer and Args for loading."""
  validation.validate_abstract_checkpointables(abstract_checkpointables)
  abstract_checkpointables = abstract_checkpointables or {}

  # pylint: disable=protected-access
  handlers = composite_handler.CompositeHandler(
      context.checkpointables_options.registry
  ).get_handlers_for_load(path, abstract_checkpointables)
  # pylint: enable=protected-access
  if not abstract_checkpointables:
    abstract_checkpointables = {
        name: None
        for name in handlers.keys()
        if name not in format_utils.RESERVED_CHECKPOINTABLE_KEYS
        and (path / name).exists()
    }

  compatibility_handlers = {
      name: handler_compatibility.get_compatibility_handler(handler)
      for name, handler in handlers.items()
  }
  legacy_handler_registry = (
      legacy_handler_registration.DefaultCheckpointHandlerRegistry()
  )
  for name, handler in compatibility_handlers.items():
    legacy_handler_registry.add(name, handler_compatibility.Args, handler)
  composite_options = composite_checkpoint_handler.CompositeOptions(
      async_options=context.async_options.v0(),
      file_options=context.file_options.v0(),
      multiprocessing_options=context.multiprocessing_options.v0(),
      temporary_path_class=context.file_options.temporary_path_class,
  )
  ckptr = async_checkpointer.AsyncCheckpointer(
      composite_checkpoint_handler.CompositeCheckpointHandler(
          handler_registry=legacy_handler_registry,
          composite_options=composite_options,
      ),
      async_options=context.async_options.v0(),
      multiprocessing_options=context.multiprocessing_options.v0(),
      file_options=context.file_options.v0(),
      temporary_path_class=context.file_options.temporary_path_class,
  )
  args = composite_checkpoint_handler.CompositeArgs(**{
      name: handler_compatibility.Args(checkpointable)
      for name, checkpointable in abstract_checkpointables.items()
  })
  return ckptr, args
