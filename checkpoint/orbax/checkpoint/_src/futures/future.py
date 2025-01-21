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

"""Futures that can be used for signaling for synchronization."""

import threading
from typing import Any, Coroutine, Optional

from absl import logging
from orbax.checkpoint._src import asyncio_utils
from orbax.checkpoint._src.futures import synchronization
from orbax.checkpoint._src.multihost import multihost
from typing_extensions import Protocol


get_unique_barrier_key = (
    synchronization.HandlerAwaitableSignalBarrierKeyGenerator.get_unique_barrier_key
)
_SIGNAL_ACTION_SUCCESS = 'signal_action_success'


class Future(Protocol):
  """Abstracted Orbax Future class.

  This is used to represent the return value of
  AsyncCheckpointHandler.async_save. This method may return multiple related,
  but potentially distinct, future objects. Common examples may include
  tensorstore.Future or concurrent.futures.Future. Since these types are not
  strictly related to one another, we merely enforce that any returned future
  must have a `result` method which blocks until the future's operation
  completes. Importantly, calling `result` should not *start* execution of the
  future, but merely wait for an ongoing operation to complete.
  """

  def result(self, timeout: Optional[int] = None) -> Any:
    """Waits for the future to complete its operation."""
    ...


class _SignalingThread(threading.Thread):
  """Thread that raises an exception if it encounters an error.

  Waits for signals to be received before proceeding with the target function.
  Then sends signals to indicate that the target function has completed.
  """

  _exception: Optional[Exception] = None

  def __init__(
      self,
      *,
      send_signals: list[synchronization.HandlerAwaitableSignal],
      receive_signals: list[synchronization.HandlerAwaitableSignal],
      timeout_secs: int = 600,
      **kwargs,
  ):
    """Constructor.

    Args:
      send_signals: Signals to send to indicate that the target function has
        completed.
      receive_signals: Signals to wait for before proceeding with the target
        function.
      timeout_secs: Timeout in seconds for waiting for signals.
      **kwargs: Keyword arguments passed to the base class.
    """
    super().__init__(**kwargs)
    self._send_signals = send_signals
    self._receive_signals = receive_signals
    self._timeout_secs = timeout_secs

  def _wait_for_signals(self):
    """Waits for signals to be set."""
    for signal in self._receive_signals:
      logging.vlog(
          1,
          '[process=%d][thread=%s] Waiting for <%s> timeout: %d secs to be set',
          multihost.process_index(),
          threading.current_thread().name,
          signal.value,
          self._timeout_secs,
      )
      barrier_key = get_unique_barrier_key(signal)
      client = multihost.get_jax_distributed_client()
      client.blocking_key_value_get(barrier_key, self._timeout_secs * 1000)

  def _set_signals(self):
    """Sets the barrier keys for the signals using send_signals."""
    for signal in self._send_signals:
      logging.vlog(
          1,
          '[process=%d][thread=%s] Signalling completion of <%s>.',
          multihost.process_index(),
          threading.current_thread().name,
          signal.value,
      )
      barrier_key = get_unique_barrier_key(signal)
      client = multihost.get_jax_distributed_client()
      client.key_value_set(barrier_key, _SIGNAL_ACTION_SUCCESS)

  def run(self):
    """Runs the target function after waiting for signals."""
    try:
      self._wait_for_signals()
      super().run()
      self._set_signals()
    except Exception as e:  # pylint: disable=broad-exception-caught
      self._exception = e

  def join(self, timeout: Optional[float] = None):
    """Waits for the target function to complete."""
    super().join(timeout=timeout)
    if self._exception is not None:
      raise self._exception


class CommitFuture(Future):
  """Represents the result of a background commit.

  May send signals to indicate that the commit has completed. Can also receive
  signals to indicate that the commit should proceed.
  """

  def __init__(
      self,
      coro: Coroutine[Any, Any, None],
      *,
      name: str | None = None,
      send_signals: list[synchronization.HandlerAwaitableSignal] | None = None,
      receive_signals: (
          list[synchronization.HandlerAwaitableSignal] | None
      ) = None,
      timeout_secs: int = 600,
  ):
    """Constructor.

    Args:
      coro: The coroutine to run.
      name: The name of the thread.
      send_signals: Signals to send to indicate that the commit has completed.
      receive_signals: Signals to wait for before proceeding with the commit.
      timeout_secs: Timeout in seconds for waiting for signals.
    """
    super().__init__()
    send_signals = send_signals or []
    receive_signals = receive_signals or []
    self._t = _SignalingThread(
        send_signals=send_signals,
        receive_signals=receive_signals,
        timeout_secs=timeout_secs,
        target=lambda: asyncio_utils.run_sync(coro),
        name=name,
    )
    self._t.start()

  def result(self, timeout: Optional[float] = None) -> Any:
    """Waits for the commit to complete."""
    return self._t.join(timeout=timeout)
