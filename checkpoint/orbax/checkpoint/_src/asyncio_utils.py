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

"""Provides helper async functions."""

import asyncio
import threading
from typing import Any, Coroutine, TypeVar


_T = TypeVar('_T')


def _run_event_loop(loop: asyncio.AbstractEventLoop):
  """Runs the event loop."""
  loop.run_forever()
  loop.close()


def run_sync(
    coro: Coroutine[Any, Any, _T],
    enable_nest_asyncio: bool = True,  # For testing.
) -> _T:
  """Runs a coroutine and returns the result."""
  try:
    asyncio.get_running_loop()  # no event loop: ~0.001s, otherwise: ~0.182s
    if enable_nest_asyncio:
      event_loop = asyncio.new_event_loop()
      thread = threading.Thread(
          target=_run_event_loop, args=(event_loop,), daemon=True
      )
      thread.start()
      future = asyncio.run_coroutine_threadsafe(coro, event_loop)
      try:
        return future.result()
      finally:
        event_loop.call_soon_threadsafe(event_loop.stop)
        thread.join()
  except RuntimeError:
    pass
  return asyncio.run(coro)
