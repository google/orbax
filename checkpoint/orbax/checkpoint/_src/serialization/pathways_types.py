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

"""Pathways types.

No dependency on Pathways binaries should be added here, as it should remain
lightweight.
"""

from __future__ import annotations

import enum


class CheckpointingImpl(enum.Enum):
  """The implementation to use for Pathways checkpointing."""

  NO_DISPATCHER = enum.auto()
  COLOCATED_PYTHON = enum.auto()

  @classmethod
  def from_options(
      cls,
      *,
      use_colocated_python: bool = False,
  ) -> CheckpointingImpl:
    """Obtains a CheckpointingImpl from the given options.

    More than one option can be set to True. Resolves in order of priority:
      1. Colocated Python
      4. No Dispatcher

    Args:
      use_colocated_python: Whether to use colocated Python. # BEGIN
      use_remote_python: Whether to use remote Python.
      use_persistence_array_handler: Whether to use the persistence array

    Returns:
      The CheckpointingImpl to use.
    """
    if use_colocated_python:
      return cls.COLOCATED_PYTHON
    else:
      return cls.NO_DISPATCHER
