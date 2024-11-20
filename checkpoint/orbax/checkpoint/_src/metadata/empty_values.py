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

"""Handles empty values in the checkpoint PyTree."""

from typing import Any, Mapping
from orbax.checkpoint._src.tree import utils as tree_utils

RESTORE_TYPE_NONE = 'None'
RESTORE_TYPE_DICT = 'Dict'
RESTORE_TYPE_LIST = 'List'
RESTORE_TYPE_TUPLE = 'Tuple'
RESTORE_TYPE_UNKNOWN = 'Unknown'
# TODO: b/365169723 - Handle empty NamedTuple.


# TODO: b/365169723 - Handle empty NamedTuple.
def is_supported_empty_value(value: Any) -> bool:
  """Determines if the *empty* `value` is supported without custom TypeHandler."""
  # Check isinstance first to avoid `not` checks on jax.Arrays (raises error).
  if tree_utils.isinstance_of_namedtuple(value):
    return False
  return (
      isinstance(value, (dict, list, tuple, type(None), Mapping)) and not value
  )


# TODO: b/365169723 - Handle empty NamedTuple.
def get_empty_value_typestr(value: Any) -> str:
  """Returns the typestr constant for the empty value."""
  if not is_supported_empty_value(value):
    raise ValueError(f'{value} is not a supported empty type.')
  if isinstance(value, list):
    return RESTORE_TYPE_LIST
  if isinstance(value, tuple):
    return RESTORE_TYPE_TUPLE
  if isinstance(value, (dict, Mapping)):
    return RESTORE_TYPE_DICT
  if value is None:
    return RESTORE_TYPE_NONE
  raise ValueError(f'Unrecognized empty type: {value}.')


# TODO: b/365169723 - Handle empty NamedTuple.
def is_empty_typestr(typestr: str) -> bool:
  return (
      typestr == RESTORE_TYPE_LIST
      or typestr == RESTORE_TYPE_TUPLE
      or typestr == RESTORE_TYPE_DICT
      or typestr == RESTORE_TYPE_NONE
  )


# TODO: b/365169723 - Handle empty NamedTuple.
def get_empty_value_from_typestr(typestr: str) -> Any:
  if typestr == RESTORE_TYPE_LIST:
    return []
  if typestr == RESTORE_TYPE_TUPLE:
    return tuple()
  if typestr == RESTORE_TYPE_DICT:
    return {}
  if typestr == RESTORE_TYPE_NONE:
    return None
  raise ValueError(f'Unrecognized typestr: {typestr}.')
