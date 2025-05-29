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

"""Naming utilities."""

import re
from typing import Pattern

# Copied from tensorflow/core/framework/node_def_util.cc
# then tensorflow/python/framework/ops.py
_VALID_OP_NAME_REGEX: Pattern[str] = re.compile(
    r'^[A-Za-z0-9.][A-Za-z0-9_.\\/>-]*$'
)


def is_valid_node_name(sp: str) -> bool:
  return _VALID_OP_NAME_REGEX.match(sp) is not None


def control_input(s: str) -> str:
  return f'^{s}'


def to_slice_notation(s: str, idx: int | None = None) -> str:
  """Converts a string to the slice notation used for Nodef Inputs."""
  if not s:
    raise ValueError('String cannot be empty.')
  if s[0] == '^':
    raise ValueError('This is a control input, can\'t be sliced.')

  s_split = s.split(':')

  if len(s_split) > 1 and s_split[-1].isdigit():
    if int(s_split[-1]) != idx:
      raise ValueError(
          f'This input is already sliced, but at index {s_split[-1]} instead of'
          f' {idx}.'
      )
    return s  # s is already sliced

  idx = idx if idx is not None else 0
  return f'{s}:{idx}'
