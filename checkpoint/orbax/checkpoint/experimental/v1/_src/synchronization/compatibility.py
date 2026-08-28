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

"""Compatibility utilities for synchronization."""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
import concurrent.futures
import time
from absl import logging
import jax
from orbax.checkpoint._src.futures import future


class FuturesAwaitable:
  """An awaitable that transparently wraps and exposes a list of orbax futures.

  This is a helper for bridging `Awaitable[None]` interfaces with underlying
  orbax `future.Future` lists, permitting cancellation propagation in V0.
  """

  def __init__(
      self,
      commit_futures: Sequence[future.Future],
      coro_fn: Callable[[], Awaitable[None]],
  ):
    self.commit_futures = commit_futures
    self.coro_fn = coro_fn
    self._coro: Awaitable[None] | None = None

  def __await__(self):
    if self._coro is None:
      self._coro = self.coro_fn()
    return self._coro.__await__()


async def async_futures(
    commit_futures: Sequence[future.Future],
    *,
    timeout_secs: float | None = None,
    start_time: float | None = None,
    operation_name: str | None = None,
    re_raise_cancelled: bool = False,
) -> None:
  """Waits for futures and cleanly cancels them on abort."""
  deadline = (
      start_time + timeout_secs
      if timeout_secs is not None and start_time is not None
      else None
  )

  def _wait_with_timeout(f: future.Future):
    if deadline is None:
      return f.result()
    timeout = deadline - time.time()
    if timeout <= 0:
      raise TimeoutError('Overall save timeout exceeded.')
    return f.result(timeout=timeout)

  try:
    await asyncio.gather(
        *[asyncio.to_thread(_wait_with_timeout, f) for f in commit_futures]
    )
  except BaseException as e:
    if isinstance(
        e, (concurrent.futures.CancelledError, asyncio.CancelledError)
    ):
      if operation_name:
        logging.info(
            '[process=%s] %s was safely cancelled.',
            jax.process_index(),
            operation_name,
        )
      else:
        logging.info(
            '[process=%s] async_futures was safely cancelled.',
            jax.process_index(),
        )
      for f in commit_futures:
        if hasattr(f, 'cancel'):
          try:
            f.cancel()
          except Exception as inner_e:  # pylint: disable=broad-exception-caught
            logging.warning(
                'Error cancelling future in async_futures: %s', inner_e
            )
      if re_raise_cancelled:
        raise
      return
    raise
