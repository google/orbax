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
from typing import Awaitable, Generic, TypeVar

T = TypeVar('T')


class BackgroundThreadRunner(Generic[T]):
  """A runner for an asyncio event loop in a background thread.

  This class expects an awaitable function that will be run in a background
  thread. It creates an event loop that is passed to the thread. This event loop
  should only be interacted with via asyncio thread-safe APIs, when tasks are
  scheduled from the main thread.
  """

  def __init__(
      self,
      target: Awaitable[T],
  ):
    self._target = target
    self._event_loop = asyncio.new_event_loop()
    self._thread = threading.Thread(
        target=self._event_loop_runner, args=(self._event_loop,), daemon=True
    )
    self._thread.start()
    self._future = asyncio.run_coroutine_threadsafe(
        self._target, self._event_loop
    )

  def _event_loop_runner(self, event_loop: asyncio.AbstractEventLoop):
    event_loop.run_forever()
    event_loop.close()

  def result(self, timeout: float | None = None) -> T:
    r = self._future.result(timeout=timeout)
    if self._thread:
      self._event_loop.call_soon_threadsafe(self._event_loop.stop)
      self._thread.join(timeout=timeout)
      self._thread = None
    return r
