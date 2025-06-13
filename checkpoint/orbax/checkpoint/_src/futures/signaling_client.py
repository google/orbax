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

import functools
import threading
import time
from typing import Sequence
from absl import logging
import jax
from orbax.checkpoint._src.multihost import multihost
from typing_extensions import Protocol


class SignalingClient(Protocol):
  """Client that supports signaling between threads and processes."""

  def key_value_set(
      self, key: str, value: str, *, allow_overwrite: bool = False
  ):
    """Sets a key-value pair in the client.

    Args:
      key: The key to set.
      value: The value to set.
      allow_overwrite: Whether to allow overwriting an existing value for the
        given key.
    """

  def blocking_key_value_get(self, key: str, timeout_secs: int) -> str:
    """Gets the value for a given key in the client.

    Blocks until the key is set or the timeout is reached.

    Args:
      key: The key to get.
      timeout_secs: The timeout in seconds.
    """
    ...

  def key_value_try_get(self, key: str) -> str | None:
    """Tries to get the value for a given key in the client without blocking.

    Args:
      key: The key to get.
    """
    ...

  def key_value_delete(self, key: str):
    """Deletes the key-value.

    If the key is a directory, recursively clean up all key-values under the
    directory.

    Args:
      key: The key to delete.
    """
    ...

  def key_value_dir_get(self, key: str) -> Sequence[tuple[str, str]]:
    """Gets all key-value pairs in the directory.

    Args:
      key: The key to get.
    """
    ...

  def wait_at_barrier(
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


class JaxDistributedSignalingClient(SignalingClient):
  """A signaling client that uses a JAX distributed client.

  This class uses a JAX distributed client to implement the signaling client
  interface.
  """

  def __init__(self):
    super().__init__()
    self._client = multihost.get_jax_distributed_client()

  def key_value_set(self, key: str, value: str, allow_overwrite: bool = False):
    """Sets a key-value pair in the client.

    Args:
      key: The key to set.
      value: The value to set.
      allow_overwrite: Whether to allow overwriting an existing value for the
        given key.

    Raises:
      JaxRuntimeError: If key value set fails.
    """
    self._client.key_value_set(key, value, allow_overwrite=allow_overwrite)

  def blocking_key_value_get(self, key: str, timeout_secs: int) -> str:
    """Gets the value for a given key in the client.

    Blocks until the key is set or the timeout is reached.

    Args:
      key: The key to get.
      timeout_secs: The timeout in seconds.

    Returns:
      The value associated with the key.

    Raises:
      JaxRuntimeError: If the key value get fails.
    """
    return str(self._client.blocking_key_value_get(key, timeout_secs * 1000))

  def key_value_try_get(self, key: str) -> str | None:
    """Tries to get the value for a given key in the client without blocking.

    Args:
      key: The key to get.

    Returns:
      The value associated with the key if it exists, otherwise None.
    """
    try:
      return str(self._client.key_value_try_get(key))
    except jax.errors.JaxRuntimeError:
      # Note that JaxRuntimeError may represent issues other than key not found,
      # but we are catching all such issues and returning None.
      logging.vlog(
          1,
          "JaxRuntimeError raised while trying to get key '%s'.",
          key,
          exc_info=True,
      )
      return None

  def key_value_delete(self, key: str):
    """Deletes the key-value.

    If the key is a directory, recursively clean up all key-values under the
    directory.

    Args:
      key: The key to delete.
    """
    self._client.key_value_delete(key)

  def key_value_dir_get(self, key: str) -> Sequence[tuple[str, str]]:
    """Gets all key-value pairs in the directory.

    Args:
      key: The key to get.

    Returns:
      A sequence of (key, value) tuples.
    """
    if not key.endswith("/"):
      raise ValueError(f"Key '{key}' must end with a slash.")
    return self._client.key_value_dir_get(key)

  def wait_at_barrier(
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

    Raises:
      TimeoutError: If the timeout is reached.
    """
    try:
      self._client.wait_at_barrier(
          key, timeout_secs * 1000, process_ids=process_ids
      )
    except jax.errors.JaxRuntimeError as e:
      # Note that JaxRuntimeError may represent issues other than timeouts, but
      # we aim to catch all such issues in caller layers.
      error_msg = f"Timeout waiting at barrier '{key}'."
      if process_ids is not None:
        error_msg += f" Barrier used process ids: {process_ids}."
      error_msg += f" The original error raised was: {str(e)}"
      raise TimeoutError(error_msg) from e


class ThreadSafeKeyValueSignalingClient(SignalingClient):
  """A thread-safe key-value store supporting basic operations and blocking get.

  This should only be used in a single controller setup as it does not support
  interaction with other processes.

  This class uses a dictionary internally to implement signaling client
  interface and protects concurrent access using a threading.Lock. It also uses
  a threading.Condition variable to allow threads to wait efficiently for a key
  to be set.
  """

  def __init__(self):
    super().__init__()
    self._data: dict[str, str] = {}
    self._lock = threading.Lock()
    self._condition = threading.Condition(self._lock)

  def key_value_set(
      self, key: str, value: str, allow_overwrite: bool = False
  ) -> None:
    """Sets the value for a given key in a thread-safe manner.

    If the key already exists and allow_overwrite is True, its value is
    overwritten, else an error is raised.
    Notifies any threads waiting on this key via the condition variable.

    Args:
        key: The key to set. Must be hashable.
        value: The value to associate with the key.
        allow_overwrite: Whether to allow overwriting an existing value for the
          given key.

    Raises:
        KeyError: If the key already exists and allow_overwrite is False.
    """
    with self._lock:
      if key in self._data and not allow_overwrite:
        raise KeyError(f"Key '{key}' already exists.")
      self._data[key] = value
      # Notify potentially waiting threads that the state has changed
      # (specifically, a key has been added or updated).
      self._condition.notify_all()

  def blocking_key_value_get(self, key: str, timeout_secs: int) -> str:
    """Gets the value for a key.

    Blocks until the key exists or a timeout occurs.

    Args:
        key: The key whose value is to be retrieved.
        timeout_secs: The maximum time in seconds to wait for the key to be set.

    Returns:
        The value associated with the key.

    Raises:
        TimeoutError: If the key is not set within the specified timeout.
    """
    end = time.time() + timeout_secs
    with self._condition:  # Acquires the underlying lock
      while key not in self._data:
        notified_in_time = self._condition.wait(timeout=end - time.time())

        if not notified_in_time:
          # Double-check after timeout before raising error
          if key not in self._data:
            raise TimeoutError(f"Timeout waiting for key '{key}'")

      return self._data[key]

  def key_value_try_get(self, key: str) -> str | None:
    """Tries to get the value for a key without blocking.

    Args:
        key: The key whose value is to be retrieved.

    Returns:
        The value associated with the key if it exists, otherwise None.
    """
    with self._lock:
      return self._data.get(key)

  def key_value_delete(self, key: str):
    """Deletes the key-value.

    If the key is a directory - ends with '/', recursively clean up all
    key-values under the directory.

    Args:
        key: The key to delete.
    """
    is_directory_key = key.endswith("/")
    with self._lock:
      if is_directory_key:
        keys_to_delete = [
            full_key
            for full_key in self._data
            if key == full_key or full_key.startswith(key)
        ]
        for k in keys_to_delete:
          del self._data[k]
      else:
        if key in self._data:
          del self._data[key]

  def key_value_dir_get(self, key: str) -> Sequence[tuple[str, str]]:
    """Gets all key-value pairs in the directory.

    Args:
      key: The key to get.

    Returns:
      A sequence of (key, value) tuples.
    """
    if not key.endswith("/"):
      raise ValueError(f"Key '{key}' must end with a slash.")
    keys_to_fetch = [
        full_key
        for full_key in self._data
        if key == full_key or full_key.startswith(key)
    ]
    return [(k, self._data[k]) for k in keys_to_fetch]

  def wait_at_barrier(
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
    raise NotImplementedError(
        "`wait_at_barrier` is not implemented for"
        " `ThreadSafeKeyValueSignalingClient`."
    )


@functools.lru_cache()
def get_signaling_client() -> SignalingClient:
  """Returns the signaling client to use for the current environment."""
  if multihost.is_jax_distributed_client_initialized():
    logging.info("Using JaxDistributedSignalingClient")
    return JaxDistributedSignalingClient()
  else:
    process_count = multihost.process_count()
    if process_count > 1:
      raise RuntimeError(
          "ThreadSafeKeyValueSignalingClient should only be used in a single"
          f" controller setup, process count: {process_count}."
      )
    logging.info("Using ThreadSafeKeyValueSignalingClient")
    return ThreadSafeKeyValueSignalingClient()
