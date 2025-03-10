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

"""Utilities for serializing and deserializing metadata."""

from typing import Any, Sequence

from orbax.checkpoint._src.metadata import checkpoint


CompositeCheckpointHandlerTypeStrs = (
    checkpoint.CompositeCheckpointHandlerTypeStrs
)
CheckpointHandlerTypeStr = checkpoint.CheckpointHandlerTypeStr


def validate_type(obj: Any, field_type: type[Any] | Sequence[type[Any]]):
  if isinstance(field_type, Sequence):
    if not any(isinstance(obj, f_type) for f_type in field_type):
      raise ValueError(
          f'Object must be any one of types {list(field_type)}, got '
          f'{type(obj)}.'
      )
  elif not isinstance(obj, field_type):
    raise ValueError(f'Object must be of type {field_type}, got {type(obj)}.')


def validate_and_process_item_handlers(
    item_handlers: Any,
) -> CompositeCheckpointHandlerTypeStrs | CheckpointHandlerTypeStr | None:
  """Validates and processes item_handlers field."""
  if item_handlers is None:
    return None

  validate_type(item_handlers, [dict, str])
  if isinstance(item_handlers, CompositeCheckpointHandlerTypeStrs):
    for k in item_handlers or {}:
      validate_type(k, str)
    return item_handlers
  elif isinstance(item_handlers, CheckpointHandlerTypeStr):
    return item_handlers


def validate_and_process_custom_metadata(
    custom_metadata: Any,
) -> dict[str, Any]:
  """Validates and processes custom field."""
  if custom_metadata is None:
    return {}

  validate_type(custom_metadata, dict)
  for k in custom_metadata:
    validate_type(k, str)
  return custom_metadata
