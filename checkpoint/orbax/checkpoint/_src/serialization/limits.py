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

"""Classes for limiting in-flight bytes during checkpointing."""

import asyncio
import contextlib
from typing import AsyncIterator, Optional, Protocol
from absl import logging
import humanize


class ByteLimiter(Protocol):

  async def wait_for_bytes(self, requested_bytes: int):
    ...

  async def release_bytes(self, requested_bytes: int):
    ...


# Lifted from T5X.
class LimitInFlightBytes(ByteLimiter):
  """Limits in-flight bytes when reading/writing checkpoints per process."""

  def __init__(self, num_bytes: int):
    if num_bytes <= 0:
      raise ValueError(f'Must provide positive `num_bytes`. Found: {num_bytes}')
    self._max_bytes = num_bytes
    self._available_bytes = num_bytes
    self._cv = asyncio.Condition(lock=asyncio.Lock())

  @property
  def max_bytes(self) -> int:
    return self._max_bytes

  async def wait_for_bytes(self, requested_bytes: int):
    """Reserve bytes."""
    if requested_bytes >= self._max_bytes:
      raise ValueError(
          'Requested more bytes than we reserved space for: '
          f'{requested_bytes} > {self._max_bytes}'
      )
    async with self._cv:
      await self._cv.wait_for(lambda: self._available_bytes > requested_bytes)
      self._available_bytes -= requested_bytes
      logging.vlog(
          1,
          'Reserved bytes: %s | Remaining bytes: %s',
          humanize.naturalsize(requested_bytes, binary=True),
          humanize.naturalsize(self._available_bytes, binary=True),
      )
      assert self._available_bytes >= 0

  async def release_bytes(self, requested_bytes: int):
    async with self._cv:
      self._available_bytes += requested_bytes
      logging.vlog(
          1,
          'Releasing bytes: %s | Available bytes: %s',
          humanize.naturalsize(requested_bytes, binary=True),
          humanize.naturalsize(self._available_bytes, binary=True),
      )
      assert self._available_bytes <= self._max_bytes
      self._cv.notify_all()


class UnlimitedInFlightBytes(ByteLimiter):

  async def wait_for_bytes(self, requested_bytes: int):
    del requested_bytes
    return

  async def release_bytes(self, requested_bytes: int):
    del requested_bytes
    return


def get_byte_limiter(concurrent_bytes: Optional[int] = None) -> ByteLimiter:
  if concurrent_bytes is None:
    return UnlimitedInFlightBytes()
  if concurrent_bytes <= 0:
    raise ValueError(
        f'Must provide positive `concurrent_bytes`. Found: {concurrent_bytes}'
    )
  return LimitInFlightBytes(concurrent_bytes)


@contextlib.asynccontextmanager
async def reserved_bytes(
    byte_limiter: ByteLimiter,
    nbytes: int,
) -> AsyncIterator[None]:
  """Reserves some bytes for the duration of the context."""
  await byte_limiter.wait_for_bytes(nbytes)
  try:
    yield
  finally:
    await byte_limiter.release_bytes(nbytes)
