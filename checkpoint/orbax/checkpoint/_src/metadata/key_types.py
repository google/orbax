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

"""Defines KeyPythonType enum and helper functions."""

from __future__ import annotations

import enum
from typing import Any

from orbax.checkpoint._src.tree import utils as tree_utils


class KeyType(enum.Enum):
  """Enum representing PyTree key type."""

  SEQUENCE = 1
  DICT = 2

  def to_json(self) -> int:
    return self.value

  @classmethod
  def from_json(cls, value: int) -> KeyType:
    return cls(value)

  @classmethod
  def from_jax_tree_key(cls, key: Any) -> KeyType:
    """Translates the JAX key class into a proto enum."""
    if tree_utils.is_sequence_key(key):
      return cls.SEQUENCE
    elif tree_utils.is_dict_key(key):
      return cls.DICT
    else:
      raise ValueError(f'Unsupported KeyEntry: {type(key)}: "{key}"')


class KeyPythonType(enum.Enum):
  """Enum representing the python type of the key."""

  INT = 1
  STR = 2

  def to_json(self) -> int:
    return self.value

  @classmethod
  def from_json(cls, value: int) -> KeyPythonType:
    return cls(value)

  @classmethod
  def from_jax_python_type(cls, key_python_type: Any) -> KeyPythonType:
    """Translates the JAX key python type into a proto enum."""
    if isinstance(key_python_type, int):
      return cls.INT
    elif isinstance(key_python_type, str):
      return cls.STR
    else:
      raise ValueError(
          f'Unsupported KeyEntry: {type(key_python_type)}: "{key_python_type}"'
      )
