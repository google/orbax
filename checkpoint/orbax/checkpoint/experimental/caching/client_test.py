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

"""Unit tests for the storage service client."""

import unittest
from unittest import mock
from absl.testing import absltest
from orbax.checkpoint.experimental.caching import client
import requests


class StorageServiceClientTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_post = self.enter_context(
        mock.patch.object(requests, "post", autospec=True)
    )
    self.client = client.StorageServiceClient(service_url="http://test-url")

  def enter_context(self, context_manager):
    result = context_manager.__enter__()
    self.addCleanup(context_manager.__exit__, None, None, None)
    return result

  def test_resolve(self):
    mock_response = mock.Mock()
    mock_response.json.return_value = {"path": "/path/to/asset"}
    mock_response.raise_for_status.return_value = None
    self.mock_post.return_value = mock_response

    result = self.client.resolve(123, 456)

    self.assertEqual(result, "/path/to/asset")
    self.mock_post.assert_called_once_with(
        "http://test-url/resolve", json={"execution_id": 123, "step": 456}
    )

  def test_finalize(self):
    mock_response = mock.Mock()
    mock_response.raise_for_status.return_value = None
    self.mock_post.return_value = mock_response

    self.client.finalize(123, 456)

    self.mock_post.assert_called_once_with(
        "http://test-url/finalize", json={"execution_id": 123, "step": 456}
    )

  def test_prefetch(self):
    mock_response = mock.Mock()
    mock_response.raise_for_status.return_value = None
    self.mock_post.return_value = mock_response

    self.client.prefetch(123, 456)

    self.mock_post.assert_called_once_with(
        "http://test-url/prefetch", json={"execution_id": 123, "step": 456}
    )

  def test_await_transfer(self):
    mock_response = mock.Mock()
    mock_response.raise_for_status.return_value = None
    self.mock_post.return_value = mock_response

    self.client.await_transfer(123, 456)

    self.mock_post.assert_called_once_with(
        "http://test-url/await_transfer",
        json={"execution_id": 123, "step": 456},
    )


if __name__ == "__main__":
  absltest.main()
