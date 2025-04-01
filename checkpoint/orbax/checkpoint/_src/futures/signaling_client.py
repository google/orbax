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

"""A signaling client interface and implementations."""

import abc
import threading
from typing import Any, Dict, Optional

import jax
from orbax.checkpoint._src.multihost import multihost


class SignalingClient(abc.ABC):
  """An interface for a client that supports signaling between threads and processes."""

  @abc.abstractmethod
  def key_value_set(self, key: str, value: str, allow_overwrite: bool = False):
    """Sets a key-value pair in the client.

    Args:
      key: The key to set.
      value: The value to set.
      allow_overwrite: Whether to allow overwriting an existing value for the
        given key.
    """

  @abc.abstractmethod
  def blocking_key_value_get(self, key: str, timeout_secs: int) -> str:
    """Gets the value for a given key in the client blocking until the key is set or the timeout is reached.

    Args:
      key: The key to get.
      timeout_secs: The timeout in seconds.
    """

  @abc.abstractmethod
  def key_value_try_get(self, key: str) -> Optional[str]:
    """Tries to get the value for a given key in the client without blocking.

    Args:
      key: The key to get.
    """

  @abc.abstractmethod
  def key_value_delete(self, key: str):
    """Deletes the key-value.

    If the key is a directory, recursively clean up all key-values under the
    directory.

    Args:
      key: The key to delete.
    """


class JaxDistributedSignalingClient(SignalingClient):
  """A signaling client that uses a JAX distributed client.

  This class uses a JAX distributed client to implement the signaling client
  interface.
  """

  def __init__(self):
    self._client = multihost.get_jax_distributed_client()

  def key_value_set(self, key: str, value: str, allow_overwrite: bool = False):
    """Sets a key-value pair in the client.

    Args:
      key: The key to set.
      value: The value to set.
      allow_overwrite: Whether to allow overwriting an existing value for the
        given key.

    Raises:
      JaxRuntimeError: If the key already exists and allow_overwrite is False.
    """
    self._client.key_value_set(key, value, allow_overwrite=allow_overwrite)

  def blocking_key_value_get(self, key: str, timeout_secs: int) -> str:
    """Gets the value for a given key in the client blocking until the key is set or the timeout is reached.

    Args:
      key: The key to get.
      timeout_secs: The timeout in seconds.

    Returns:
      The value associated with the key or raises a JaxRuntimeError if the
      timeout is reached.

    Raises:
      JaxRuntimeError: If the timeout is reached.
    """
    return str(self._client.blocking_key_value_get(key, timeout_secs * 1000))

  def key_value_try_get(self, key: str) -> Optional[str]:
    """Tries to get the value for a given key in the client without blocking.

    Args:
      key: The key to get.

    Returns:
      The value associated with the key if it exists, otherwise None.
    """
    try:
      return str(self._client.key_value_try_get(key))
    except jax.errors.JaxRuntimeError:
      return None

  def key_value_delete(self, key: str):
    """Deletes the key-value.

    If the key is a directory, recursively clean up all key-values under the
    directory.

    Args:
      key: The key to delete.
    """
    self._client.key_value_delete(key)


class ThreadSafeKeyValueSignalingClient(SignalingClient):
  """A thread-safe key-value store supporting basic operations and blocking get.

  This class uses a dictionary internally to implement signaling client
  interface and protects concurrent access
  using a threading.Lock. It also uses a threading.Condition variable
  to allow threads to wait efficiently for a key to be set.
  """

  def __init__(self):
    self._data: Dict[str, str] = {}
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
        ValueError: If the key already exists and allow_overwrite is False.
    """
    with self._lock:
      if key in self._data and not allow_overwrite:
        raise ValueError(f"Key '{key}' already exists.")
      self._data[key] = value
      # Notify potentially waiting threads that the state has changed
      # (specifically, a key has been added or updated).
      self._condition.notify_all()

  def blocking_key_value_get(self, key: str, timeout_secs: int) -> Any:
    """Gets the value for a key, blocking until the key exists or a timeout occurs.

    Args:
        key: The key whose value is to be retrieved.
        timeout_secs: The maximum time in seconds to wait for the key to be set.

    Returns:
        The value associated with the key.

    Raises:
        TimeoutError: If the key is not set within the specified timeout.
    """
    with self._condition:  # Acquires the underlying lock
      while key not in self._data:
        notified_in_time = self._condition.wait(timeout=float(timeout_secs))

        if not notified_in_time:
          # Double-check after timeout before raising error
          if key not in self._data:
            raise TimeoutError(f"Timeout waiting for key '{key}'")

      return self._data[key]

  def key_value_try_get(self, key: str) -> Optional[Any]:
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

    If the key is a directory, recursively clean up all key-values under the
    directory.

    Args:
        key: The key to delete.

    Returns:
        True if the key was found and deleted, False otherwise.
    """
    is_directory_key = key.endswith("/")
    with self._lock:
      if is_directory_key:
        for full_key in self._data:
          if key == full_key or full_key.startswith(key):
            del self._data[full_key]
      else:
        if key in self._data:
          del self._data[key]
