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

"""Synchronization utilities for futures."""

import enum
import itertools
from orbax.checkpoint._src.multihost import multihost


class HandlerAwaitableSignal(enum.Enum):
  """Defines signals that may be awaited by a `CheckpointHandler`.

  Signals may be passed from `CheckpointManager` or `Checkpointer` layers to
  `CheckpointHandler or below.`

  Attributes:
    STEP_DIRECTORY_CREATION: When recieved, indicates that the step directory
      has been created. The handler should not attempt to write files before the
      directory is created.
    ITEM_DIRECTORY_CREATION: When recieved, indicates that the item directory
      has been created. The handler should not attempt to write files before the
      directory is created.
  """

  STEP_DIRECTORY_CREATION = "step_directory_creation"
  ITEM_DIRECTORY_CREATION = "item_directory_creation"


class HandlerAwaitableSignalBarrierKeyGenerator:
  """A unique barrier key generator for a `HandlerAwaitableSignal`."""

  _operation_id_counter = itertools.count()
  _operation_id = None

  @classmethod
  def next_operation_id(cls) -> int:
    """Increments the operation id counter and returns the new value."""
    cls._operation_id = next(cls._operation_id_counter)
    return cls._operation_id

  @classmethod
  def get_unique_barrier_key(cls, signal: HandlerAwaitableSignal) -> str:
    """Returns a unique barrier key for the signal.

    Args:
      signal: The signal to generate a barrier key for.

    Raises:
      ValueError: If `_operation_id` is not initialized.
    """
    if cls._operation_id is None:
      raise ValueError(
          "_operation_id is not initialized. Please call `next_operation_id()`"
          " first."
      )
    return multihost.unique_barrier_key(
        signal.value, suffix=str(cls._operation_id)
    )
