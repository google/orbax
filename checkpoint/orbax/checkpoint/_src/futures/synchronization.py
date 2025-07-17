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
import time
from typing import Callable, Generic, TypeVar
from absl import logging
from orbax.checkpoint import options as options_lib
from orbax.checkpoint._src.multihost import multihost

_T = TypeVar("_T")


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


class MultihostSynchronizedValue(Generic[_T]):
  """A thread-safe value that is synchronized across all processes."""

  def __init__(
      self,
      value: _T,
      multiprocessing_options: options_lib.MultiprocessingOptions,
      async_options: options_lib.AsyncOptions,
  ):
    """Initializes the thread-safe value."""
    self._multiprocessing_options = multiprocessing_options
    self._async_options = async_options
    self._lock = threading.RLock()
    with self._lock:
      self._value = value

  def _create_thread_safe_barrier_sync_fn(self) -> Callable[[str], None]:
    """Returns a barrier sync function to be called from threads.

    The function accepts a key, but the timeout is already set up using
    `AsyncOptions.timeout_secs` attribute.

    The Jax based barrier sync util, `sync_global_devices`, should not be called
    concurrently. Otherwise, it may cause a deadlock.

    In general, any Jax function with `collectives` should not be called
    concurrently to avoid deadlocks.
    """
    async_options = self._async_options or options_lib.AsyncOptions()
    timeout_secs = async_options.timeout_secs
    barrier_sync_fn = (
        async_options.barrier_sync_fn
        or multihost.get_barrier_sync_fn(
            processes=self._multiprocessing_options.active_processes
        )
    )
    return lambda key: barrier_sync_fn(key=key, timeout_ms=timeout_secs * 1000)

  def get(self) -> _T:
    """Returns the value."""
    with self._lock:
      return self._value

  def set(self, value: _T) -> None:
    """Sets the value across all processes."""
    start_time = time.time()
    thread_safe_barrier_sync_fn = self._create_thread_safe_barrier_sync_fn()
    thread_safe_barrier_sync_fn(
        multihost.unique_barrier_key(
            "ThreadSaveMultiHostValueHolder:set_value_start",
            prefix=self._multiprocessing_options.barrier_sync_key_prefix,
        ),
    )
    with self._lock:
      self._value = value
      thread_safe_barrier_sync_fn(
          multihost.unique_barrier_key(
              "ThreadSaveMultiHostValueHolder:set_value_end",
              prefix=self._multiprocessing_options.barrier_sync_key_prefix,
          ),
      )
    logging.vlog(
        1,
        "[process=%s]MultihostSynchronizedValue set took %s seconds",
        multihost.process_index(),
        time.time() - start_time,
    )
