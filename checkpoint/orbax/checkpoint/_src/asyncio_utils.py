# Copyright 2026 The Orbax Authors.
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

"""Provides helper async functions."""

import asyncio
from concurrent import futures
import threading
from typing import Any, Coroutine, TypeVar

from absl import logging

try:
  import uvloop  # pylint: disable=g-import-not-at-top
except ImportError:
  uvloop = None

try:
  import nest_asyncio  # pylint: disable=g-import-not-at-top # pytype: disable=import-error
except ImportError:
  nest_asyncio = None

_T = TypeVar('_T')


def _run_event_loop(loop: asyncio.AbstractEventLoop) -> None:
  """Runs the event loop until stop() is called."""
  loop.run_forever()
  loop.close()


def run_sync(coro: Coroutine[Any, Any, _T]) -> _T:
  """Runs a coroutine and returns the result."""
  try:
    # no event loop: ~0.001s, otherwise: ~0.182s
    loop = asyncio.get_running_loop()
  except RuntimeError:
    loop = None

  if loop is None:
    # No event loop is running, so we can safely use asyncio.run.
    return asyncio.run(coro)
  else:
    # An event loop is already running.
    if uvloop is None:
      if nest_asyncio is None:
        raise RuntimeError(
            'nest_asyncio is not installed. Please install it to use run_sync'
            ' with an existing event loop.'
        )
      nest_asyncio.apply()
      return asyncio.run(coro)
    else:
      event_loop = uvloop.new_event_loop()
      thread = threading.Thread(
          target=_run_event_loop, args=(event_loop,), daemon=True
      )
      thread.start()
      try:
        return asyncio.run_coroutine_threadsafe(coro, event_loop).result()
      finally:
        event_loop.call_soon_threadsafe(event_loop.stop)
        thread.join()


class AsyncRunner:
  """Executor that allows running coroutines in a separate thread.

  It creates and manages a new event loop in a dedicated thread, which can then
  be used to run coroutines from other threads using run_coroutine().

  NOTE: This class is currently not intended to be thread-safe. For example,
  calling shutdown() multiple times from multiple threads can result in an
  inconsistent behavior (the subsequent calls will not be guaranteed to wait for
  pending tasks to complete). If the need arises, we will add thread-safety
  in a future iteration.

  NOTE: This class can also be made a context manager, but we have not seen a
  use case for it yet.
  """

  def __init__(self):
    self._loop = asyncio.new_event_loop()
    self._loop_running = threading.Event()
    self._thread = threading.Thread(target=self._run_loop, daemon=True)
    self._thread.start()
    # Wait until the background thread confirms the event loop is running
    self._loop_running.wait()
    self._is_closed = False

  def _run_loop(self) -> None:
    asyncio.set_event_loop(self._loop)
    # Signal that the loop has successfully started
    self._loop.call_soon_threadsafe(self._loop_running.set)
    self._loop.run_forever()
    self._loop.close()

  def run_coroutine(self, coro: Coroutine[Any, Any, _T]) -> futures.Future[_T]:
    """Schedules a coroutine to run in the background thread's event loop.

    Args:
        coro: The coroutine object to run.

    Returns:
        A concurrent.futures.Future object representing the coroutine's result.

    Raises:
        RuntimeError: If the event loop is not running.
    """
    if self._is_closed:
      raise RuntimeError('AsyncRunner has been shut down.')
    return asyncio.run_coroutine_threadsafe(coro, self._loop)

  def shutdown(self, wait: bool = True) -> None:
    """Stops the event loop, waiting for tasks to complete.

    See the note in the class docstring regarding thread-safety.

    Args:
      wait: If True, wait for all tasks to complete before shutting down.
        Otherwise, the shutdown will be non-blocking.
    """
    if self._is_closed:
      return

    self._is_closed = True

    async def _shutdown_tasks():
      current_task = asyncio.current_task(self._loop)
      tasks_to_wait_for = {
          t for t in asyncio.all_tasks(self._loop) if t is not current_task
      }

      if tasks_to_wait_for:
        logging.info(
            'AsyncRunner: Waiting for %d tasks to complete...',
            len(tasks_to_wait_for),
        )
        # return_exceptions=True ensures gather waits for all tasks even if some
        # raise exceptions.
        await asyncio.gather(*tasks_to_wait_for, return_exceptions=True)
        logging.info('AsyncRunner: All tasks finished.')
      else:
        logging.info('AsyncRunner: No active tasks to wait for.')

    logging.info('AsyncRunner: Shutting down (wait=%s)...', wait)
    shutdown_future = asyncio.run_coroutine_threadsafe(
        _shutdown_tasks(), self._loop
    )

    if wait:
      shutdown_future.result()
      # Place the stop command gracefully at the end of the event queue.
      self._loop.call_soon_threadsafe(self._loop.stop)
      # Wait for the thread to exit.
      self._thread.join()
    else:
      # Only signal the event loop to stop, but do not wait for it to exit.
      self._loop.call_soon_threadsafe(self._loop.stop)
    logging.info('AsyncRunner: Stopped.')
