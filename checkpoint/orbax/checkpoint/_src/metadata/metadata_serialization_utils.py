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


def validate_type(obj: Any, field_type: type[Any]):
  if not isinstance(obj, field_type):
    raise ValueError(f'Object must be of type {field_type}, got {type(obj)}.')


def validate_field(
    obj: Any,
    field_name: str,
    field_type: type[Any] | Sequence[type[Any]]
):
  """Validates a single field in a dictionary.

  field_type can optionally be a sequence of types, in which case the field
  must be of any one of the types in the sequence.

  Args:
    obj: The object to validate.
    field_name: The name of the field to validate.
    field_type: The type (or sequence of types) of the field to validate.
  """
  if field_name not in obj or obj[field_name] is None:
    return
  field = obj[field_name]
  if isinstance(field_type, Sequence):
    if not any(isinstance(field, f_type) for f_type in field_type):
      raise ValueError(
          f'Metadata field "{field_name}" must be any one of '
          f'types {list(field_type)}, got {type(field)}.'
      )
  elif not isinstance(field, field_type):
    raise ValueError(
        f'Metadata field "{field_name}" must be of type {field_type}, '
        f'got {type(field)}.'
    )


def validate_dict_entry(
    dict_field: dict[Any, Any],
    dict_field_name: str,
    key: Any,
    key_type: type[Any],
    value_type: type[Any] | None = None,
):
  """Validates a single entry in a dictionary field."""
  if not isinstance(key, key_type):
    raise ValueError(
        f'Metadata field "{dict_field_name}" keys must be of type {key_type}, '
        f'got {type(key)}.'
    )
  if value_type is not None:
    dict_field = dict_field[dict_field_name]
    if not isinstance(dict_field[key], value_type):
      raise ValueError(
          f'Metadata field "{dict_field_name}" values must be of '
          f'type {value_type}, got {type(dict_field[key])}.'
      )
