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

import threading
import time
from unittest import mock

from absl.testing import absltest
import jax
from orbax.checkpoint._src.futures import signaling_client
from orbax.checkpoint._src.multihost import multihost


class TestThreadSafeKeyValueSignalingClient(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.client = signaling_client.ThreadSafeKeyValueSignalingClient()

  def test_key_value_set_and_get(self):
    self.client.key_value_set("key1", "value1")
    self.assertEqual(self.client.key_value_try_get("key1"), "value1")
    self.assertEqual(self.client.blocking_key_value_get("key1", 1), "value1")

  def test_key_value_try_get_non_existent(self):
    self.assertIsNone(self.client.key_value_try_get("non_existent_key"))

  def test_key_value_set_overwrite_allowed(self):
    self.client.key_value_set("key1", "value1")
    self.client.key_value_set("key1", "value2", allow_overwrite=True)
    self.assertEqual(self.client.key_value_try_get("key1"), "value2")

  def test_key_value_set_overwrite_disallowed(self):
    self.client.key_value_set("key1", "value1")
    with self.assertRaisesRegex(KeyError, "Key 'key1' already exists."):
      self.client.key_value_set("key1", "value2", allow_overwrite=False)
    # Ensure original value is unchanged
    self.assertEqual(self.client.key_value_try_get("key1"), "value1")

  def test_blocking_get_timeout(self):
    with self.assertRaisesRegex(
        TimeoutError, "Timeout waiting for key 'timeout_key'"
    ):
      self.client.blocking_key_value_get("timeout_key", 1)

  def test_blocking_get_waits_and_gets_value(self):
    key = "concurrent_key"
    value = "concurrent_value"
    timeout_secs = 2
    result = []
    error = []

    def target_get():
      try:
        val = self.client.blocking_key_value_get(key, timeout_secs)
        result.append(val)
      except TimeoutError as e:
        error.append(e)

    def target_set():
      time.sleep(timeout_secs / 2)  # Wait less than the timeout
      self.client.key_value_set(key, value)

    getter_thread = threading.Thread(target=target_get)
    setter_thread = threading.Thread(target=target_set)

    getter_thread.start()
    setter_thread.start()

    getter_thread.join(timeout=timeout_secs * 2)
    setter_thread.join(timeout=timeout_secs * 2)

    self.assertEmpty(error, f"Getter thread raised an error: {error}")
    self.assertEqual(result, [value])

  def test_blocking_get_timeout_before_set(self):
    key = "late_key"
    value = "late_value"
    timeout_secs = 1

    def target_get():
      with self.assertRaises(TimeoutError):
        self.client.blocking_key_value_get(key, timeout_secs)

    def target_set():
      time.sleep(timeout_secs * 2)  # Wait longer than the timeout
      self.client.key_value_set(key, value)

    getter_thread = threading.Thread(target=target_get)
    setter_thread = threading.Thread(target=target_set)

    getter_thread.start()
    setter_thread.start()

    getter_thread.join(timeout=timeout_secs * 3)
    setter_thread.join(timeout=timeout_secs * 3)

    # Check the value was eventually set, even though the getter timed out
    self.assertEqual(self.client.key_value_try_get(key), value)

  def test_key_value_delete_single_key(self):
    self.client.key_value_set("key_to_delete", "value_to_delete")
    self.assertIsNotNone(self.client.key_value_try_get("key_to_delete"))
    self.client.key_value_delete("key_to_delete")
    self.assertIsNone(self.client.key_value_try_get("key_to_delete"))

  def test_key_value_delete_non_existent_key(self):
    try:
      self.client.key_value_delete("non_existent_key_to_delete")
    except KeyError as e:
      self.fail(f"Deleting non-existent key raised an exception: {e}")

  def test_key_value_delete_directory_key(self):
    self.client.key_value_set("dir/", "dir_value")
    self.client.key_value_set("dir/key1", "value1")
    self.client.key_value_set("dir/key2", "value2")
    self.client.key_value_set("dir/subdir/key3", "value3")
    self.client.key_value_set("otherdir/key4", "value4")

    # Delete the 'directory'
    self.client.key_value_delete("dir/")

    self.assertIsNone(self.client.key_value_try_get("dir/"))
    self.assertIsNone(self.client.key_value_try_get("dir/key1"))
    self.assertIsNone(self.client.key_value_try_get("dir/key2"))
    self.assertIsNone(self.client.key_value_try_get("dir/subdir/key3"))
    # Ensure key outside the 'directory' remains
    self.assertEqual(self.client.key_value_try_get("otherdir/key4"), "value4")


# Use mocking to avoid requiring a real JAX distributed setup
@mock.patch.object(multihost, "get_jax_distributed_client")
class TestJaxDistributedSignalingClient(absltest.TestCase):

  def test_init(self, mock_get_jax_client):
    signaling_client.JaxDistributedSignalingClient()
    mock_get_jax_client.assert_called_once()

  def test_key_value_set(self, mock_get_jax_client):
    mock_client = mock.MagicMock()
    mock_get_jax_client.return_value = mock_client
    client = signaling_client.JaxDistributedSignalingClient()

    client.key_value_set("jax_key", "jax_value", allow_overwrite=True)

    mock_client.key_value_set.assert_called_once_with(
        "jax_key", "jax_value", allow_overwrite=True
    )

  def test_key_value_set_raises_jax_runtime_error(self, mock_get_jax_client):
    mock_client = mock.MagicMock()
    # Configure the mock to raise JaxRuntimeError
    mock_client.key_value_set.side_effect = jax.errors.JaxRuntimeError(
        "Simulated JAX error"
    )
    mock_get_jax_client.return_value = mock_client
    client = signaling_client.JaxDistributedSignalingClient()

    with self.assertRaises(jax.errors.JaxRuntimeError):
      client.key_value_set(
          "jax_key_exists", "some_value", allow_overwrite=False
      )
    mock_client.key_value_set.assert_called_once_with(
        "jax_key_exists", "some_value", allow_overwrite=False
    )

  def test_blocking_key_value_get(self, mock_get_jax_client):
    mock_client = mock.MagicMock()
    # Simulate JAX client returning bytes
    mock_client.blocking_key_value_get.return_value = "jax_return_value"
    mock_get_jax_client.return_value = mock_client
    client = signaling_client.JaxDistributedSignalingClient()

    value = client.blocking_key_value_get("jax_get_key", 10)

    self.assertEqual(value, "jax_return_value")
    # JAX client expects timeout in milliseconds
    mock_client.blocking_key_value_get.assert_called_once_with(
        "jax_get_key", 10 * 1000
    )

  def test_blocking_key_value_get_raises_jax_runtime_error(
      self, mock_get_jax_client
  ):
    mock_client = mock.MagicMock()
    mock_client.blocking_key_value_get.side_effect = jax.errors.JaxRuntimeError(
        "Simulated JAX timeout"
    )
    mock_get_jax_client.return_value = mock_client
    client = signaling_client.JaxDistributedSignalingClient()

    with self.assertRaises(jax.errors.JaxRuntimeError):
      client.blocking_key_value_get("jax_timeout_key", 5)
    mock_client.blocking_key_value_get.assert_called_once_with(
        "jax_timeout_key", 5 * 1000
    )

  def test_key_value_try_get(self, mock_get_jax_client):
    mock_client = mock.MagicMock()
    mock_client.key_value_try_get.return_value = "jax_try_value"
    mock_get_jax_client.return_value = mock_client
    client = signaling_client.JaxDistributedSignalingClient()

    value = client.key_value_try_get("jax_try_key")

    self.assertEqual(value, "jax_try_value")
    mock_client.key_value_try_get.assert_called_once_with("jax_try_key")

  def test_key_value_try_get_returns_none_on_error(self, mock_get_jax_client):
    mock_client = mock.MagicMock()
    mock_client.key_value_try_get.side_effect = jax.errors.JaxRuntimeError(
        "Simulated JAX key not found"
    )
    mock_get_jax_client.return_value = mock_client
    client = signaling_client.JaxDistributedSignalingClient()

    value = client.key_value_try_get("jax_try_fail_key")
    self.assertIsNone(value)
    mock_client.key_value_try_get.assert_called_once_with("jax_try_fail_key")

  def test_key_value_delete(self, mock_get_jax_client):
    mock_client = mock.MagicMock()
    mock_get_jax_client.return_value = mock_client
    client = signaling_client.JaxDistributedSignalingClient()

    client.key_value_delete("jax_delete_key")
    mock_client.key_value_delete.assert_called_once_with("jax_delete_key")


class TestGetSignalingClient(absltest.TestCase):

  def setUp(self):
    super().setUp()
    signaling_client.get_signaling_client.cache_clear()

  def tearDown(self):
    super().tearDown()
    signaling_client.get_signaling_client.cache_clear()

  @mock.patch.object(multihost, "is_jax_distributed_client_initialized")
  @mock.patch.object(multihost, "get_jax_distributed_client")
  def test_returns_jax_client_when_initialized(
      self, mock_is_init, mock_get_jax_client
  ):
    mock_is_init.return_value = True
    mock_client = mock.MagicMock()
    mock_get_jax_client.return_value = mock_client

    client = signaling_client.get_signaling_client()

    mock_is_init.assert_called_once()
    mock_get_jax_client.assert_called_once()
    self.assertIsInstance(
        client, signaling_client.JaxDistributedSignalingClient
    )

    # Test singleton behavior
    client2 = signaling_client.get_signaling_client()
    self.assertIs(client, client2)

  @mock.patch.object(multihost, "is_jax_distributed_client_initialized")
  def test_returns_thread_safe_client_when_not_initialized(self, mock_is_init):
    mock_is_init.return_value = False

    client = signaling_client.get_signaling_client()

    mock_is_init.assert_called_once()
    self.assertIsInstance(
        client, signaling_client.ThreadSafeKeyValueSignalingClient
    )

    # Test singleton behavior
    client2 = signaling_client.get_signaling_client()
    self.assertIs(client, client2)

  @mock.patch.object(
      multihost, "is_jax_distributed_client_initialized", return_value=False
  )
  @mock.patch.object(multihost, "process_count", return_value=2)
  def test_raises_error_when_multiprocess_and_not_initialized(
      self, mock_is_init, mock_process_count
  ):

    with self.assertRaisesRegex(
        RuntimeError,
        "ThreadSafeKeyValueSignalingClient should only be used in a single"
        " controller setup, process count: 2.",
    ):
      signaling_client.get_signaling_client()
    mock_is_init.assert_called_once()
    mock_process_count.assert_called_once()


if __name__ == "__main__":
  absltest.main()
