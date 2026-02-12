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
import threading
from typing import Any, Coroutine, TypeVar

import uvloop


_T = TypeVar('_T')


def _run_event_loop(loop: asyncio.AbstractEventLoop) -> None:
  """Runs the event loop until stop() is called."""
  loop.run_forever()
  loop.close()


def run_sync(coro: Coroutine[Any, Any, _T]) -> _T:
  """Runs a coroutine and returns the result."""
  try:
    asyncio.get_running_loop()  # no event loop: ~0.001s, otherwise: ~0.182s
  except RuntimeError:
    # No event loop is running, so we can safely use asyncio.run.
    return asyncio.run(coro)
  else:
    # An event loop is already running.
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
