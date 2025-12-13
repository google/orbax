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
from concurrent import futures
import threading
from typing import Awaitable, Generic, TypeVar

T = TypeVar('T')


class BackgroundThreadRunner(Generic[T]):
  """A runner for an asyncio event loop in a background thread.

  This class expects an awaitable function that will be run in a background
  thread. It creates an event loop that is passed to the thread. This event loop
  should only be interacted with via asyncio thread-safe APIs, when tasks are
  scheduled from the main thread.

  TODO(b/407609827): Ensure the event loop can be cleaned up without
  calling `close`.
  """

  def __init__(
      self,
  ):
    self._event_loop = asyncio.new_event_loop()
    self._thread = threading.Thread(
        target=self._event_loop_runner, args=(self._event_loop,), daemon=True
    )
    self._thread.start()
    self._futures: dict[str, futures.Future[T]] = {}

  def _event_loop_runner(self, event_loop: asyncio.AbstractEventLoop):
    event_loop.run_forever()
    event_loop.close()

  def run(self, name: str, target: Awaitable[T]) -> None:
    if self._thread is None:
      raise ValueError('Cannot run after `close` has been called.')
    if name in self._futures:
      raise ValueError(f'A future with name {name} already exists.')
    self._futures[name] = asyncio.run_coroutine_threadsafe(
        target, self._event_loop
    )

  def result(self, name: str, *, timeout: float | None = None) -> T:
    if name not in self._futures:
      raise ValueError(f'No future with name {name} exists.')
    return self._futures[name].result(timeout=timeout)

  def close(self, timeout: float | None = None) -> None:
    if self._thread:
      self._event_loop.call_soon_threadsafe(self._event_loop.stop)
      self._thread.join(timeout=timeout)
      self._thread = None
