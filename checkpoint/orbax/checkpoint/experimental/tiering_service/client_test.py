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

"""Unit tests for the CTS Client library and utility modules."""

import unittest
from unittest import mock
import grpc
from orbax.checkpoint.experimental.tiering_service import client
from orbax.checkpoint.experimental.tiering_service import client_auth
from orbax.checkpoint.experimental.tiering_service import environment
from orbax.checkpoint.experimental.tiering_service.proto import tiering_service_pb2


class EnvironmentTest(unittest.IsolatedAsyncioTestCase):

  @mock.patch("os.environ", {})
  @mock.patch("httpx.AsyncClient.get")
  async def test_get_gcp_zone_metadata_server(self, mock_get):
    mock_response = mock.MagicMock()
    mock_response.status_code = 200
    mock_response.text = "projects/123456/zones/us-east5-a"
    mock_get.return_value = mock_response

    zone = await environment.get_gcp_zone()
    self.assertEqual(zone, "us-east5-a")

  @mock.patch("os.environ", {"GCP_ZONE": "us-west1-b"})
  async def test_get_gcp_zone_env_override(self):
    zone = await environment.get_gcp_zone()
    self.assertEqual(zone, "us-west1-b")

  @mock.patch("os.environ", {})
  @mock.patch("httpx.AsyncClient.get")
  async def test_get_gcp_zone_timeout(self, mock_get):
    mock_get.side_effect = Exception("Connection timeout")
    zone = await environment.get_gcp_zone()
    self.assertIsNone(zone)

  @mock.patch("os.environ", {})
  @mock.patch("httpx.AsyncClient.get")
  async def test_get_gcp_region_metadata_server(self, mock_get):
    mock_response = mock.MagicMock()
    mock_response.status_code = 200
    mock_response.text = "projects/123456/zones/us-east5-a"
    mock_get.return_value = mock_response

    region = await environment.get_gcp_region()
    self.assertEqual(region, "us-east5")

  @mock.patch("os.environ", {"GCP_REGION": "us-west1"})
  async def test_get_gcp_region_env_override(self):
    region = await environment.get_gcp_region()
    self.assertEqual(region, "us-west1")


class ClientAuthTest(unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    super().setUp()
    client_auth._CREDENTIALS = None

  def tearDown(self):
    client_auth._CREDENTIALS = None
    super().tearDown()

  @mock.patch("google.auth.default")
  @mock.patch("google.auth.transport.requests.Request")
  async def test_get_oauth_token_success(self, _, mock_default):
    mock_creds = mock.MagicMock()
    mock_creds.valid = False
    mock_creds.token = "fake-access-token"

    def mock_refresh(_):
      mock_creds.valid = True

    mock_creds.refresh.side_effect = mock_refresh
    mock_default.return_value = (mock_creds, "fake-project")

    # First call: should trigger credentials discovery and refresh
    token = await client_auth.get_oauth_token()
    self.assertEqual(token, "fake-access-token")
    mock_creds.refresh.assert_called_once()

    # Second call: should reuse cached credentials and skip refresh
    mock_creds.refresh.reset_mock()
    token = await client_auth.get_oauth_token()
    self.assertEqual(token, "fake-access-token")
    mock_creds.refresh.assert_not_called()

  @mock.patch("google.auth.default")
  async def test_get_oauth_token_failure(self, mock_default):
    mock_default.side_effect = Exception("No ADC credentials")
    token = await client_auth.get_oauth_token()
    self.assertIsNone(token)


class TieringClientTest(unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    super().setUp()
    self.stub_mock = mock.AsyncMock()
    self.insecure_channel_mock = mock.MagicMock()
    self.channel_close_mock = mock.AsyncMock()
    self.insecure_channel_mock.close = self.channel_close_mock

  @mock.patch("grpc.aio.insecure_channel")
  @mock.patch(
      "orbax.checkpoint.experimental.tiering_service.proto.tiering_service_pb2_grpc.TieringServiceStub"
  )
  async def test_connect_and_close(
      self, mock_stub_class, mock_insecure_channel
  ):
    mock_insecure_channel.return_value = self.insecure_channel_mock
    mock_stub_class.return_value = self.stub_mock

    client_inst = client.TieringClient()
    await client_inst.connect()
    self.assertEqual(client_inst._stub, self.stub_mock)

    await client_inst.close()
    self.assertIsNone(client_inst._channel)
    self.assertIsNone(client_inst._stub)
    self.channel_close_mock.assert_called_once()

  @mock.patch("grpc.aio.secure_channel")
  @mock.patch("grpc.local_channel_credentials")
  @mock.patch(
      "orbax.checkpoint.experimental.tiering_service.proto.tiering_service_pb2_grpc.TieringServiceStub"
  )
  async def test_connect_secure_local(
      self, mock_stub_class, mock_local_creds, mock_secure_channel
  ):
    mock_local_creds.return_value = "local-creds"
    mock_secure_channel.return_value = self.insecure_channel_mock
    mock_stub_class.return_value = self.stub_mock

    # Local loopback addresses should use local channel credentials.
    client_inst = client.TieringClient(
        server_address="localhost:50051", secure=True
    )
    await client_inst.connect()
    mock_local_creds.assert_called_once()
    mock_secure_channel.assert_called_once_with(
        "localhost:50051", "local-creds"
    )

  @mock.patch("grpc.aio.secure_channel")
  @mock.patch("grpc.ssl_channel_credentials")
  @mock.patch("grpc.local_channel_credentials")
  @mock.patch(
      "orbax.checkpoint.experimental.tiering_service.proto.tiering_service_pb2_grpc.TieringServiceStub"
  )
  async def test_connect_secure_remote(
      self,
      mock_stub_class,
      mock_local_creds,
      mock_ssl_creds,
      mock_secure_channel,
  ):
    mock_ssl_creds.return_value = "ssl-creds"
    mock_secure_channel.return_value = self.insecure_channel_mock
    mock_stub_class.return_value = self.stub_mock

    # Remote addresses should use SSL credentials directly.
    client_inst = client.TieringClient(
        server_address="cts-server:50051", secure=True
    )
    await client_inst.connect()
    mock_local_creds.assert_not_called()
    mock_ssl_creds.assert_called_once()
    mock_secure_channel.assert_called_once_with("cts-server:50051", "ssl-creds")

  @mock.patch("grpc.aio.insecure_channel")
  @mock.patch(
      "orbax.checkpoint.experimental.tiering_service.proto.tiering_service_pb2_grpc.TieringServiceStub"
  )
  @mock.patch(
      "orbax.checkpoint.experimental.tiering_service.client_auth.get_oauth_token"
  )
  @mock.patch(
      "orbax.checkpoint.experimental.tiering_service.environment.get_gcp_zone"
  )
  @mock.patch(
      "orbax.checkpoint.experimental.tiering_service.environment.get_gcp_region"
  )
  @mock.patch(
      "orbax.checkpoint.experimental.tiering_service.environment.get_current_user"
  )
  async def test_reserve_success(
      self,
      mock_get_user,
      mock_get_region,
      mock_get_zone,
      mock_get_token,
      mock_stub_class,
      mock_insecure_channel,
  ):
    mock_insecure_channel.return_value = self.insecure_channel_mock
    mock_stub_class.return_value = self.stub_mock
    mock_get_token.return_value = "fake-token"
    mock_get_zone.return_value = "us-east5-a"
    mock_get_region.return_value = "us-east5"
    mock_get_user.return_value = "test-user"

    backend_l0 = tiering_service_pb2.StorageBackend(level=0, prefix="/lustre")
    tp_l0 = tiering_service_pb2.TierPath(
        storage_backend=backend_l0,
        path="/lustre/path1",
        tier_path_uuid="tp-uuid-1",
    )
    asset = tiering_service_pb2.Asset(
        uuid="asset-uuid-1234", tier_paths=[tp_l0]
    )
    reserve_resp = tiering_service_pb2.ReserveResponse(
        asset=asset,
        keep_alive_interval_seconds=60,
        tier_path_uuid="tp-uuid-1",
    )
    self.stub_mock.Reserve.return_value = reserve_resp

    client_inst = client.TieringClient()
    uuid, path = await client_inst.reserve(path="logical/path")

    self.assertEqual(uuid, "asset-uuid-1234")
    self.assertEqual(path, "/lustre/path1")
    self.stub_mock.Reserve.assert_called_once()
    args, kwargs = self.stub_mock.Reserve.call_args
    request = args[0]
    self.assertEqual(request.user, "test-user")
    self.assertEqual(request.zone, "us-east5-a")
    self.assertEqual(request.region, "us-east5")
    self.assertEqual(
        kwargs["metadata"], [("authorization", "Bearer fake-token")]
    )

  @mock.patch("grpc.aio.insecure_channel")
  @mock.patch(
      "orbax.checkpoint.experimental.tiering_service.proto.tiering_service_pb2_grpc.TieringServiceStub"
  )
  @mock.patch(
      "orbax.checkpoint.experimental.tiering_service.client_auth.get_oauth_token"
  )
  @mock.patch(
      "orbax.checkpoint.experimental.tiering_service.environment.get_gcp_zone"
  )
  @mock.patch(
      "orbax.checkpoint.experimental.tiering_service.environment.get_gcp_region"
  )
  @mock.patch(
      "orbax.checkpoint.experimental.tiering_service.environment.get_current_user"
  )
  async def test_reserve_caching_behavior(
      self,
      mock_get_user,
      mock_get_region,
      mock_get_zone,
      mock_get_token,
      mock_stub_class,
      mock_insecure_channel,
  ):
    mock_insecure_channel.return_value = self.insecure_channel_mock
    mock_stub_class.return_value = self.stub_mock
    mock_get_token.return_value = "fake-token"
    mock_get_zone.return_value = "us-east5-a"
    mock_get_region.return_value = "us-east5"
    mock_get_user.return_value = "test-user"

    backend_l0 = tiering_service_pb2.StorageBackend(level=0, prefix="/lustre")
    tp_l0 = tiering_service_pb2.TierPath(
        storage_backend=backend_l0,
        path="/lustre/path1",
        tier_path_uuid="tp-uuid-1",
    )
    asset = tiering_service_pb2.Asset(
        uuid="asset-uuid-1234", tier_paths=[tp_l0]
    )
    reserve_resp = tiering_service_pb2.ReserveResponse(
        asset=asset,
        keep_alive_interval_seconds=60,
        tier_path_uuid="tp-uuid-1",
    )
    self.stub_mock.Reserve.return_value = reserve_resp

    client_inst = client.TieringClient()
    # Call reserve twice
    await client_inst.reserve(path="logical/path")
    await client_inst.reserve(path="logical/path2")

    # Verify environment lookup is cached and called only once
    mock_get_zone.assert_called_once()
    mock_get_region.assert_called_once()

  @mock.patch("grpc.aio.insecure_channel")
  @mock.patch(
      "orbax.checkpoint.experimental.tiering_service.proto.tiering_service_pb2_grpc.TieringServiceStub"
  )
  async def test_reserve_rpc_failure(
      self, mock_stub_class, mock_insecure_channel
  ):
    mock_insecure_channel.return_value = self.insecure_channel_mock
    mock_stub_class.return_value = self.stub_mock

    rpc_error = grpc.aio.AioRpcError(
        code=grpc.StatusCode.INTERNAL,
        initial_metadata=grpc.aio.Metadata(),
        trailing_metadata=grpc.aio.Metadata(),
        details="database error",
    )
    self.stub_mock.Reserve.side_effect = rpc_error

    client_inst = client.TieringClient()
    with self.assertRaises(RuntimeError) as context:
      await client_inst.reserve(path="logical/path")
    self.assertIn("Reserve RPC failed", str(context.exception))


if __name__ == "__main__":
  unittest.main()
