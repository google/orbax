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

"""Utilities for working with threads and asyncio event loops."""

import asyncio
import threading
from typing import Awaitable, Callable, Generic, TypeVar

T = TypeVar('T')


# TODO(b/423708172): Consider using a ThreadPool in the future.
class Thread(threading.Thread, Generic[T]):
  """A Thread that can return a result and raise exceptions."""

  _exception: Exception | None = None
  _result: T

  def __init__(self, target: Callable[[], T], **kwargs):

    def _target_setting_result():
      self._result = target()

    super().__init__(target=_target_setting_result, **kwargs)

  def run(self):
    """Runs the target function after waiting for signals."""
    try:
      super().run()
    except Exception as e:  # pylint: disable=broad-exception-caught
      self._exception = e

  def result(self, timeout: float | None = None) -> T:
    """Waits for the target function to complete."""
    super().join(timeout=timeout)
    if self.is_alive():
      raise TimeoutError(
          f'Thread {self.name} did not complete within {timeout} seconds.'
      )
    if self._exception is not None:
      raise self._exception
    return self._result


# TODO(b/423708172): Eliminate cross-thread event loop sharing.
class BackgroundEventLoopRunner(Generic[T]):
  """A runner for an asyncio event loop.

  This class starts a background thread that runs the event loop with the
  provided `Awaitable`. The event loop is closed when the `Awaitable` completes,
  even without having to call `result()`.

  This class can be useful when certain coroutines need to start in the main
  thread and continue in a background thread. Unless the same event loop is
  used, the coroutines will be cancelled if not finished yet.

  For example, in the blocking part of the save, we can start an operation via
  `asyncio.create_task` that is not necessarily expected to complete before the
  background part starts. (An example would be async directory creation.) The
  task cannot be protected from cancellation unless the blocking part and
  background part share the same event loop, which must be cleaned up when the
  background thread completes.
  """

  def __init__(
      self,
      event_loop: asyncio.AbstractEventLoop,
      target: Awaitable[T],
  ):
    self._event_loop = event_loop
    self._target = target
    self._thread = Thread(target=self._background_task_runner)
    self._thread.start()

  def _background_task_runner(self) -> T:
    result = self._event_loop.run_until_complete(self._target)
    self._event_loop.close()
    return result

  def result(self, timeout: float | None = None) -> T:
    return self._thread.result(timeout=timeout)
