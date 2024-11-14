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

from typing import Any


def validate_field(
    obj: Any,
    field_name: str,
    field_type: type[Any]
):
  if field_name not in obj or obj[field_name] is None:
    return
  field = obj[field_name]
  if not isinstance(field, field_type):
    raise ValueError(
        'StepMetadata {} must be of type {}, got {}.'.format(
            field_name, field_type, type(field)
        )
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
        'StepMetadata {} keys must be of type {}, got {}.'.format(
            dict_field_name, key_type, type(key)
        )
    )
  if value_type is not None:
    dict_field = dict_field[dict_field_name]
    if not isinstance(dict_field[key], value_type):
      raise ValueError(
          'StepMetadata {} values must be of type {}, got {}.'.format(
              dict_field_name, value_type, type(dict_field[key])
          )
      )
