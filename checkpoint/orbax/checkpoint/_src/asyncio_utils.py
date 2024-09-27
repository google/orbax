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

"""Provides helper async functions."""

from __future__ import annotations

import asyncio
import functools
import threading
from typing import Any, Awaitable, Coroutine, Optional, TypeVar

from absl import logging


_T = TypeVar('_T')


def as_async_function(func):
  """Wraps a function to make it async."""

  @functools.wraps(func)
  async def run(*args, loop=None, executor=None, **kwargs):
    if loop is None:
      loop = asyncio.get_event_loop()
    partial_func = functools.partial(func, *args, **kwargs)
    return await loop.run_in_executor(executor, partial_func)

  return run


class _RunSync:
  """Context manager to run coroutines synchronously.

  Runs the given coroutine in a new event loop running on a new thread, but
  blocks until it is done. It is necessary to avoid error:
  `RuntimeError: asyncio.run() cannot be called from a running event loop`.

  NOTE: Must be called only if an event loop is running.
  """

  def __enter__(self) -> _RunSync:
    self._loop = asyncio.new_event_loop()
    self._runner = threading.Thread(
        target=self._loop.run_forever,
        name=f'_RunSyncEventLoop-{self._loop.time()}',
        daemon=True,
    )
    self._runner.start()
    if logging.vlog_is_on(1):
      logging.vlog(
          1,
          '[event_loop=%s][thread=%s] Started.',
          self._loop,
          self._runner,
      )
    return self

  def __call__(
      self,
      coro: Awaitable[_T],
      timeout: Optional[float] = None,
  ) -> _T:
    if logging.vlog_is_on(1):
      logging.vlog(
          1,
          '[event_loop=%s][thread=%s] Running coroutine=%s',
          self._loop,  # pytype: disable=attribute-error
          self._runner,  # pytype: disable=attribute-error
          coro,
      )
    return asyncio.run_coroutine_threadsafe(
        coro,
        self._loop,  # pytype: disable=attribute-error
    ).result(timeout)

  def __exit__(self, *exc_info: Any) -> None:
    try:
      self._loop.call_soon_threadsafe(self._loop.stop)
      self._runner.join()
    finally:
      if self._loop.is_running():
        self._loop.close()
      if logging.vlog_is_on(1):
        logging.vlog(
            1,
            '[event_loop=%s][thread=%s] Exited.',
            self._loop,
            self._runner,
        )


# Benchmark:
# _run_sync/asyncio.run ~ 2.32x
def _run_sync(coro):
  with _RunSync() as run:
    return run(coro)


def run_sync(coro: Coroutine[Any, Any, _T]) -> _T:
  """Runs a coroutine and returns the result."""
  try:
    asyncio.get_running_loop()  # no event loop: ~0.001s, otherwise: ~0.182s
    # Event loop is running.
    return _run_sync(coro)
  except RuntimeError:
    # No event loop is running.
    return asyncio.run(coro)
