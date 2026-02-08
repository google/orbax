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

"""Unit tests for the storage service API server."""

from unittest import mock
from absl.testing import absltest
from fastapi import testclient
from orbax.checkpoint.experimental.caching import server
from orbax.checkpoint.experimental.caching import service


class ServerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.client = testclient.TestClient(server.uvicorn_app)
    # Mock the storage_service instance in server module
    self.mock_service = mock.Mock(spec=service.StorageService)
    self.enter_context(
        mock.patch.object(server, "storage_service", self.mock_service)
    )
    # Mock Assets object and its to_json for inspect endpoint
    self.mock_service.assets = mock.Mock()
    self.mock_service.storages = mock.Mock()
    self.mock_service.storages.to_json.return_value = {}
    self.mock_service.assets.to_json.return_value = {}

  def test_hello(self):
    response = self.client.get("/")
    self.assertEqual(response.status_code, 200)
    self.assertEqual(response.json(), "Hello from Storage Service!")

  def test_resolve(self):
    self.mock_service.resolve.return_value = "/path/to/asset"
    response = self.client.post(
        "/resolve", json={"execution_id": 123, "step": 100}
    )
    self.assertEqual(response.status_code, 200)
    self.assertEqual(response.json(), {"path": "/path/to/asset"})
    self.mock_service.resolve.assert_called_once_with(service.AssetId(123, 100))

  def test_exists(self):
    self.mock_service.assets.exists.return_value = True
    response = self.client.post(
        "/exists", json={"execution_id": 123, "step": 100}
    )
    self.assertEqual(response.status_code, 200)
    self.assertEqual(response.json(), {"exists": True})
    self.mock_service.assets.exists.assert_called_once_with(
        service.AssetId(123, 100)
    )

  def test_finalize(self):
    response = self.client.post(
        "/finalize", json={"execution_id": 123, "step": 100}
    )
    self.assertEqual(response.status_code, 200)
    self.assertEqual(response.json(), {"status": "ok"})
    self.mock_service.finalize.assert_called_once_with(
        service.AssetId(123, 100)
    )

  def test_prefetch(self):
    response = self.client.post(
        "/prefetch", json={"execution_id": 123, "step": 100}
    )
    self.assertEqual(response.status_code, 200)
    self.assertEqual(response.json(), {"status": "ok"})
    self.mock_service.prefetch.assert_called_once_with(
        service.AssetId(123, 100)
    )

  def test_await_transfer(self):
    response = self.client.post(
        "/await_transfer", json={"execution_id": 123, "step": 100}
    )
    self.assertEqual(response.status_code, 200)
    self.assertEqual(response.json(), {"status": "ok"})
    self.mock_service.await_transfer.assert_called_once_with(
        service.AssetId(123, 100)
    )

  def test_inspect(self):
    response = self.client.get("/inspect")
    self.assertEqual(response.status_code, 200)
    self.assertEqual(response.json(), {"storages": {}, "assets": {}})


if __name__ == "__main__":
  absltest.main()
