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

"""Synchronization utilities for futures."""

import enum
import itertools
import threading
from typing import Any, Callable
from orbax.checkpoint._src.multihost import multihost


class HandlerAwaitableSignal(enum.Enum):
  """Defines signals that may be awaited by a `CheckpointHandler`.

  Signals may be passed from `CheckpointManager` or `Checkpointer` layers to
  `CheckpointHandler or below.`

  Attributes:
    AWAITABLE_SIGNALS_CONTRACT: Contract that contains a list of signals that
      may be sent and can be awaited by the handlers.
    STEP_DIRECTORY_CREATION: When recieved, indicates that the step directory
      has been created. The handler should not attempt to write files before the
      directory is created.
    ITEM_DIRECTORY_CREATION: When recieved, indicates that the item directory
      has been created. The handler should not attempt to write files before the
      directory is created.
  """

  AWAITABLE_SIGNALS_CONTRACT = "awaitable_signals_contract"
  STEP_DIRECTORY_CREATION = "step_directory_creation"
  ITEM_DIRECTORY_CREATION = "item_directory_creation"


class OperationIdGenerator:
  """A unique operation id generator."""

  _operation_id_counter = itertools.count()
  _operation_id = next(_operation_id_counter)

  @classmethod
  def next_operation_id(cls) -> int:
    """Increments the operation id counter and returns the new value."""
    cls._operation_id = next(cls._operation_id_counter)
    return cls._operation_id

  @classmethod
  def get_current_operation_id(cls) -> str:
    """Returns the current operation id."""
    return str(cls._operation_id)


class ThreadSaveMultiHostValueHolder:
  """A tracker for tracking host-level op in progress."""

  def __init__(
      self,
      value: Any,
      thread_save_barrier_sync_fn: Callable[[str], None],
      barrier_sync_key_prefix: str | None = None,
  ):
    self._thread_save_barrier_sync_fn = thread_save_barrier_sync_fn
    self._barrier_sync_key_prefix = barrier_sync_key_prefix
    self._value = value
    self._lock = threading.Lock()

  def get_value(self):
    """Gets the value in a thread-safe manner."""
    with self._lock:
      return self._value

  def set_value(self, value: Any):
    """Sets the value across all processes."""
    self._thread_save_barrier_sync_fn(
        multihost.unique_barrier_key(
            "ThreadSaveMultiHostValueHolder:set_value_start",
            prefix=self._barrier_sync_key_prefix,
        ),
    )
    with self._lock:
      self._value = value
      self._thread_save_barrier_sync_fn(
          multihost.unique_barrier_key(
              "ThreadSaveMultiHostValueHolder:set_value_end",
              prefix=self._barrier_sync_key_prefix,
          ),
      )
