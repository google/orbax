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

import asyncio
import unittest
from unittest import mock

import grpc
from orbax.checkpoint.experimental.tiering_service import client
from orbax.checkpoint.experimental.tiering_service import client_auth
from orbax.checkpoint.experimental.tiering_service import environment
from orbax.checkpoint.experimental.tiering_service.proto import tiering_service_pb2

from google.protobuf import timestamp_pb2


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

    token = await client_auth.get_oauth_token()
    self.assertEqual(token, "fake-access-token")
    mock_creds.refresh.assert_called_once()

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
    client_auth._LOCK = None
    client_auth._CREDENTIALS = None
    self.stub_mock = mock.AsyncMock()
    self.insecure_channel_mock = mock.MagicMock()
    self.channel_close_mock = mock.AsyncMock()
    self.insecure_channel_mock.close = self.channel_close_mock
    self.client = client.TieringClient()
    self.addAsyncCleanup(self.client.close)

  @mock.patch("grpc.aio.insecure_channel")
  @mock.patch(
      "orbax.checkpoint.experimental.tiering_service.proto.tiering_service_pb2_grpc.TieringServiceStub"
  )
  async def test_connect_and_close(
      self, mock_stub_class, mock_insecure_channel
  ):
    mock_insecure_channel.return_value = self.insecure_channel_mock
    mock_stub_class.return_value = self.stub_mock

    client_inst = self.client
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

    client_inst = client.TieringClient(
        server_address="localhost:50051", secure=True
    )
    self.addAsyncCleanup(client_inst.close)
    await client_inst.connect()
    mock_local_creds.assert_called_once()
    mock_secure_channel.assert_called_once_with(
        "localhost:50051", "local-creds"
    )
    await client_inst.close()

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

    client_inst = client.TieringClient(
        server_address="cts-server:50051", secure=True
    )
    self.addAsyncCleanup(client_inst.close)
    await client_inst.connect()
    mock_local_creds.assert_not_called()
    mock_ssl_creds.assert_called_once()
    mock_secure_channel.assert_called_once_with("cts-server:50051", "ssl-creds")
    await client_inst.close()

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

    client_inst = self.client
    await client_inst.connect()
    uuid, path = await client_inst.reserve(path="logical/path", tags=["my-tag"])

    self.assertEqual(uuid, "asset-uuid-1234")
    self.assertEqual(path, "/lustre/path1")
    self.stub_mock.Reserve.assert_called_once()
    args, kwargs = self.stub_mock.Reserve.call_args
    request = args[0]
    self.assertEqual(request.user, "test-user")
    self.assertEqual(request.zone, "us-east5-a")
    self.assertEqual(request.region, "us-east5")
    self.assertEqual(list(request.tags), ["my-tag"])
    self.assertEqual(
        kwargs["metadata"], [("authorization", "Bearer fake-token")]
    )
    await client_inst.close()

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
    self.stub_mock.Finalize.return_value = (
        tiering_service_pb2.FinalizeResponse()
    )

    client_inst = self.client
    await client_inst.connect()
    uuid1, _ = await client_inst.reserve(path="logical/path")
    await client_inst.finalize(uuid1)
    await client_inst.reserve(path="logical/path2")

    mock_get_zone.assert_called_once()
    mock_get_region.assert_called_once()
    await client_inst.close()

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

    client_inst = self.client
    await client_inst.connect()
    with self.assertRaises(RuntimeError) as context:
      await client_inst.reserve(path="logical/path")
    self.assertIn("Reserve RPC failed", str(context.exception))
    await client_inst.close()

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
  async def test_reserve_starts_and_stops_keep_alive(
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

    original_sleep = asyncio.sleep
    with mock.patch(
        "orbax.checkpoint.experimental.tiering_service.client.asyncio.sleep"
    ) as mock_sleep:

      async def mock_sleep_fn(delay):
        if delay == 0:
          await original_sleep(0)
        else:
          await original_sleep(0.01)

      mock_sleep.side_effect = mock_sleep_fn

      backend_l0 = tiering_service_pb2.StorageBackend(level=0, prefix="/lustre")
      tp_l0 = tiering_service_pb2.TierPath(
          storage_backend=backend_l0,
          path="/lustre/path1",
          tier_path_uuid="tp-uuid-1",
      )
      asset = tiering_service_pb2.Asset(uuid="asset-1", tier_paths=[tp_l0])
      self.stub_mock.Reserve.return_value = tiering_service_pb2.ReserveResponse(
          asset=asset,
          keep_alive_interval_seconds=10,
          tier_path_uuid="tp-uuid-1",
      )
      self.stub_mock.ReserveKeepAlive.return_value = (
          tiering_service_pb2.ReserveKeepAliveResponse(
              keep_alive_interval_seconds=10
          )
      )

      client_inst = self.client
      await client_inst.connect()
      uuid, _ = await client_inst.reserve(path="logical/path")

      self.assertEqual(uuid, "asset-1")
      self.assertEqual(client_inst._active_asset_uuid, "asset-1")
      self.assertIsNotNone(client_inst._keep_alive_task)

      self.stub_mock.Finalize.return_value = (
          tiering_service_pb2.FinalizeResponse()
      )
      await client_inst.finalize(uuid="asset-1")
      await asyncio.sleep(0)

      self.assertIsNone(client_inst._active_asset_uuid)
      self.assertIsNone(client_inst._keep_alive_task)
      await client_inst.close()

  @mock.patch("grpc.aio.insecure_channel")
  @mock.patch(
      "orbax.checkpoint.experimental.tiering_service.proto.tiering_service_pb2_grpc.TieringServiceStub"
  )
  async def test_reserve_raises_if_already_active(
      self, mock_stub_class, mock_insecure_channel
  ):
    mock_insecure_channel.return_value = self.insecure_channel_mock
    mock_stub_class.return_value = self.stub_mock

    backend_l0 = tiering_service_pb2.StorageBackend(level=0, prefix="/lustre")
    tp_l0 = tiering_service_pb2.TierPath(
        storage_backend=backend_l0,
        path="/lustre/path1",
        tier_path_uuid="tp-uuid-1",
    )
    asset = tiering_service_pb2.Asset(uuid="asset-1", tier_paths=[tp_l0])
    self.stub_mock.Reserve.return_value = tiering_service_pb2.ReserveResponse(
        asset=asset,
        keep_alive_interval_seconds=60,
        tier_path_uuid="tp-uuid-1",
    )

    client_inst = self.client
    await client_inst.connect()
    await client_inst.reserve(path="logical/path1")

    with self.assertRaises(RuntimeError) as ctx:
      await client_inst.reserve(path="logical/path2")
    self.assertIn("already managing active asset", str(ctx.exception))
    await client_inst.close()

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
  async def test_prefetch_resolves_immediately_if_ready(
      self,
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

    ready_time = timestamp_pb2.Timestamp(seconds=123456)
    backend_l0 = tiering_service_pb2.StorageBackend(level=0, prefix="/lustre")
    tp_l0 = tiering_service_pb2.TierPath(
        storage_backend=backend_l0,
        path="/lustre/path1",
        ready_at=ready_time,
        tier_path_uuid="tp-uuid-2",
    )
    asset = tiering_service_pb2.Asset(uuid="asset-2", tier_paths=[tp_l0])
    self.stub_mock.Prefetch.return_value = tiering_service_pb2.PrefetchResponse(
        asset=asset,
        keep_alive_interval_seconds=10,
        closest_tier_path_uuid="tp-uuid-2",
    )

    client_inst = self.client
    await client_inst.connect()
    path = await client_inst.prefetch(path="logical/path")
    self.assertEqual(path, "/lustre/path1")
    self.assertEqual(client_inst._active_asset_uuid, "asset-2")

    await client_inst.release("asset-2")
    self.assertIsNone(client_inst._active_asset_uuid)
    await client_inst.close()

  @mock.patch("grpc.aio.insecure_channel")
  @mock.patch(
      "orbax.checkpoint.experimental.tiering_service.proto.tiering_service_pb2_grpc.TieringServiceStub"
  )
  async def test_prefetch_raises_if_already_active(
      self, mock_stub_class, mock_insecure_channel
  ):
    mock_insecure_channel.return_value = self.insecure_channel_mock
    mock_stub_class.return_value = self.stub_mock

    ready_time = timestamp_pb2.Timestamp(seconds=123456)
    backend_l0 = tiering_service_pb2.StorageBackend(level=0, prefix="/lustre")
    tp_l0 = tiering_service_pb2.TierPath(
        storage_backend=backend_l0,
        path="/lustre/path1",
        ready_at=ready_time,
        tier_path_uuid="tp-uuid-2",
    )
    asset = tiering_service_pb2.Asset(uuid="asset-2", tier_paths=[tp_l0])
    self.stub_mock.Prefetch.return_value = tiering_service_pb2.PrefetchResponse(
        asset=asset,
        keep_alive_interval_seconds=10,
        closest_tier_path_uuid="tp-uuid-2",
    )

    client_inst = self.client
    await client_inst.connect()
    await client_inst.prefetch(path="logical/path1")

    with self.assertRaises(RuntimeError) as ctx:
      await client_inst.prefetch(path="logical/path2")
    self.assertIn("already managing active asset", str(ctx.exception))
    await client_inst.close()

  @mock.patch("grpc.aio.insecure_channel")
  @mock.patch(
      "orbax.checkpoint.experimental.tiering_service.proto.tiering_service_pb2_grpc.TieringServiceStub"
  )
  async def test_connect_failure_unblocks_waiting_callers(
      self, mock_stub_class, mock_insecure_channel
  ):
    mock_insecure_channel.return_value = self.insecure_channel_mock
    mock_stub_class.return_value = self.stub_mock

    client_inst = client.TieringClient()
    with mock.patch.object(
        client_inst,
        "_async_connect",
        side_effect=Exception("Connection Failed"),
    ):

      async def call_connect():
        try:
          await client_inst.connect()
          return None
        except Exception as e:  # pylint: disable=broad-exception-caught
          return e

      tasks = [asyncio.create_task(call_connect()) for _ in range(3)]
      results = await asyncio.gather(*tasks)
      self.assertEqual(len(results), 3)
      for res in results:
        self.assertIsInstance(res, Exception)

  @mock.patch("grpc.aio.insecure_channel")
  @mock.patch(
      "orbax.checkpoint.experimental.tiering_service.proto.tiering_service_pb2_grpc.TieringServiceStub"
  )
  async def test_close_stops_loop_on_teardown_error(
      self, mock_stub_class, mock_insecure_channel
  ):
    mock_insecure_channel.return_value = self.insecure_channel_mock
    mock_stub_class.return_value = self.stub_mock
    self.channel_close_mock.side_effect = Exception("Channel error")

    client_inst = client.TieringClient()
    await client_inst.connect()

    with self.assertRaises(Exception):
      await client_inst.close()

    self.assertIsNone(client_inst._loop)
    self.assertIsNone(client_inst._thread)

  @mock.patch("grpc.aio.insecure_channel")
  @mock.patch(
      "orbax.checkpoint.experimental.tiering_service.proto.tiering_service_pb2_grpc.TieringServiceStub"
  )
  async def test_delete(self, mock_stub_class, mock_insecure_channel):
    mock_insecure_channel.return_value = self.insecure_channel_mock
    mock_stub_class.return_value = self.stub_mock
    self.stub_mock.Delete.return_value = tiering_service_pb2.DeleteResponse()

    client_inst = self.client
    await client_inst.connect()
    await client_inst.delete(uuid="asset-uuid-delete")
    await asyncio.sleep(0.1)

    self.stub_mock.Delete.assert_called_once()
    args, _ = self.stub_mock.Delete.call_args
    self.assertEqual(args[0].uuid, "asset-uuid-delete")
    await client_inst.close()

  @mock.patch("grpc.aio.insecure_channel")
  @mock.patch(
      "orbax.checkpoint.experimental.tiering_service.proto.tiering_service_pb2_grpc.TieringServiceStub"
  )
  async def test_info(self, mock_stub_class, mock_insecure_channel):
    mock_insecure_channel.return_value = self.insecure_channel_mock
    mock_stub_class.return_value = self.stub_mock

    asset = tiering_service_pb2.Asset(uuid="asset-uuid-info")
    info_resp = tiering_service_pb2.InfoResponse(assets=[asset])
    self.stub_mock.Info.return_value = info_resp

    client_inst = self.client
    await client_inst.connect()
    assets = await client_inst.info(uuid="asset-uuid-info")
    self.assertEqual(len(assets), 1)
    self.assertEqual(assets[0].uuid, "asset-uuid-info")
    self.stub_mock.Info.assert_called_once()
    await client_inst.close()

  @mock.patch("grpc.aio.insecure_channel")
  @mock.patch(
      "orbax.checkpoint.experimental.tiering_service.proto.tiering_service_pb2_grpc.TieringServiceStub"
  )
  async def test_release_path(self, mock_stub_class, mock_insecure_channel):
    mock_insecure_channel.return_value = self.insecure_channel_mock
    mock_stub_class.return_value = self.stub_mock

    ready_time = timestamp_pb2.Timestamp(seconds=123456)
    backend_l0 = tiering_service_pb2.StorageBackend(level=0, prefix="/lustre")
    tp_l0 = tiering_service_pb2.TierPath(
        storage_backend=backend_l0,
        path="/lustre/path1",
        ready_at=ready_time,
        tier_path_uuid="tp-uuid-2",
    )
    asset = tiering_service_pb2.Asset(
        uuid="asset-uuid-release", tier_paths=[tp_l0]
    )
    self.stub_mock.Prefetch.return_value = tiering_service_pb2.PrefetchResponse(
        asset=asset,
        keep_alive_interval_seconds=10,
        closest_tier_path_uuid="tp-uuid-2",
    )

    client_inst = self.client
    await client_inst.connect()
    path = await client_inst.prefetch(uuid="asset-uuid-release")
    self.assertEqual(path, "/lustre/path1")
    self.assertEqual(client_inst._active_asset_uuid, "asset-uuid-release")

    await client_inst.release_path("/lustre/path1")
    self.assertIsNone(client_inst._active_asset_uuid)
    await client_inst.close()


if __name__ == "__main__":
  unittest.main()
