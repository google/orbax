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

"""A signaling client interface and implementations."""

import asyncio
from typing import Protocol, Sequence
from orbax.checkpoint._src.futures import signaling_client


class SignalingClient(Protocol):
  """Client that supports signaling between threads and processes."""

  async def key_value_set(
      self, key: str, value: str, *, allow_overwrite: bool = False
  ):
    """Sets a key-value pair in the client.

    Args:
      key: The key to set.
      value: The value to set.
      allow_overwrite: Whether to allow overwriting an existing value for the
        given key.
    """
    ...

  async def blocking_key_value_get(self, key: str, timeout_secs: int) -> str:
    """Gets the value for a given key in the client.

    Blocks until the key is set or the timeout is reached.

    Args:
      key: The key to get.
      timeout_secs: The timeout in seconds.
    """
    ...

  async def key_value_try_get(self, key: str) -> str | None:
    """Tries to get the value for a given key in the client without blocking.

    Args:
      key: The key to get.
    """
    ...

  async def key_value_delete(self, key: str):
    """Deletes the key-value.

    If the key is a directory, recursively clean up all key-values under the
    directory.

    Args:
      key: The key to delete.
    """
    ...

  async def wait_at_barrier(
      self,
      key: str,
      *,
      timeout_secs: int,
      process_ids: Sequence[int] | None = None,
  ):
    """Waits at a barrier identified by key.

    Args:
      key: The key to wait at.
      timeout_secs: The timeout in seconds.
      process_ids: The participating process ids.
    """
    ...


class _SignalingClient(SignalingClient):
  """An implementation of SignalingClient that wraps V0 implementation."""

  def __init__(self, client: signaling_client.SignalingClient):
    self._client = client

  async def key_value_set(
      self, key: str, value: str, *, allow_overwrite: bool = False
  ):
    return await asyncio.to_thread(
        self._client.key_value_set, key, value, allow_overwrite=allow_overwrite
    )

  async def blocking_key_value_get(self, key: str, timeout_secs: int) -> str:
    return await asyncio.to_thread(
        self._client.blocking_key_value_get, key, timeout_secs
    )

  async def key_value_try_get(self, key: str) -> str | None:
    return await asyncio.to_thread(self._client.key_value_try_get, key)

  async def key_value_delete(self, key: str):
    return await asyncio.to_thread(self._client.key_value_delete, key)

  async def wait_at_barrier(
      self,
      key: str,
      *,
      timeout_secs: int,
      process_ids: Sequence[int] | None = None,
  ):
    return await asyncio.to_thread(
        self._client.wait_at_barrier,
        key,
        timeout_secs=timeout_secs,
        process_ids=process_ids,
    )


def get_signaling_client() -> SignalingClient:
  return _SignalingClient(signaling_client.get_signaling_client())
