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
import logging
from orbax.checkpoint._src.futures import signaling_client
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


class OpTracker:
  """A tracker for tracking host-level op in progress."""

  def __init__(self, tracker_prefix: str, operation_id: str):
    self._operation_id = operation_id
    self._tracker_prefix = tracker_prefix
    logging.info(
        "[process=%s] Created OpTracker for %s with operation id %s",
        multihost.process_index(),
        tracker_prefix,
        operation_id,
    )

  def start(self):
    """Marks the op as in progress for the current process."""
    process_index = multihost.process_index()
    signaling_client.get_signaling_client().key_value_set(
        f"{self._tracker_prefix}_{self._operation_id}/{process_index}",
        str(process_index),
        allow_overwrite=True,
    )

  def complete(self):
    """Marks the op as complete for the current process."""
    process_index = multihost.process_index()
    signaling_client.get_signaling_client().key_value_delete(
        f"{self._tracker_prefix}_{self._operation_id}/{process_index}"
    )

  def get_in_progress_ids(self) -> list[int]:
    """Returns the list of processes in progress for the op."""
    op_in_progress = signaling_client.get_signaling_client().key_value_dir_get(
        f"{self._tracker_prefix}_{self._operation_id}/"
    )
    return [int(process_id) for _, process_id in op_in_progress]


class OpTrackerFactory:
  """A factory for creating `OpTracker` instances."""

  @classmethod
  def create_tracker(cls, tracker_prefix: str) -> OpTracker:
    """Creates a new `OpTracker` instance."""
    return OpTracker(
        tracker_prefix,
        str(OperationIdGenerator.next_operation_id()),
    )
