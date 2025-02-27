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

"""Retry strategies."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar
from absl import logging

_R = TypeVar('_R')


async def retry(
    *,
    awaitable_factory: Callable[[], Awaitable[_R]],
    retry_on_result: Callable[[Any], bool],
    retry_on_exception: Callable[[Exception], bool],
    sleep_for_secs: Callable[[_R | None, Exception | None, int], float],
    max_retries: int,
) -> _R:
  """Retries an awaitable based on result or exception.

  Args:
    awaitable_factory: Creates the awaitable to retry. It is needed because an
      awaitable cannot be awaited more than once.
    retry_on_result: A callable that takes the result of the awaitable and
      returns True if the awaitable should be retried, False otherwise.
    retry_on_exception: A callable that takes an exception and returns True if
      the awaitable should be retried, False otherwise.
    sleep_for_secs: A callable that takes the result of the awaitable, the
      exception raised by the awaitable, and the number of retries remaining,
      and returns the number of seconds to sleep between retries.
    max_retries: The maximum number of times to retry the awaitable.

  Returns:
    The result of the awaitable.

  Raises:
    ValueError: If max_retries is negative.
    Exception: Due to the awaitable raising an exception.
  """
  if max_retries < 0:
    raise ValueError('max_retries must be non-negative.')

  awaitable = awaitable_factory()
  try:
    result = await awaitable_factory()
    if max_retries == 0:
      return result
    if not retry_on_result(result):
      return result
    sleep_secs = sleep_for_secs(result, None, max_retries)
    logging.warning(
        'Will retry after %s seconds due to result=%s, awaitable=%s',
        sleep_secs,
        result,
        awaitable,
    )
    await asyncio.sleep(sleep_secs)
  except Exception as e:  # pylint: disable=broad-except
    if max_retries == 0:
      raise
    if not retry_on_exception(e):
      raise
    sleep_secs = sleep_for_secs(None, e, max_retries)
    logging.warning(
        'Will retry after %s seconds due to exception=%s, awaitable=%s',
        sleep_secs,
        e,
        awaitable,
    )
    await asyncio.sleep(sleep_secs)

  return await retry(
      awaitable_factory=awaitable_factory,
      retry_on_result=retry_on_result,
      retry_on_exception=retry_on_exception,
      sleep_for_secs=sleep_for_secs,
      max_retries=max_retries - 1,
  )
