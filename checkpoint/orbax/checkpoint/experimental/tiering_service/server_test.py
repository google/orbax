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

import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import aiosqlite  # pylint: disable=unused-import
import greenlet  # pylint: disable=unused-import
import grpc
from orbax.checkpoint.experimental.tiering_service import db_lib
from orbax.checkpoint.experimental.tiering_service import server
from orbax.checkpoint.experimental.tiering_service import server_config
from orbax.checkpoint.experimental.tiering_service.proto import tiering_service_pb2

from google.protobuf import timestamp_pb2


class TieringServiceTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  def setUp(self):
    super().setUp()
    storage_backends_config = [
        {
            "level": 0,
            "backend_type": "BACKEND_TYPE_LUSTRE",
            "prefix": "/mnt/lustre",
            "zone": "us-central1-a",
        },
        {
            "level": 1,
            "backend_type": "BACKEND_TYPE_GCS",
            "prefix": "gs://my-bucket",
            "region": "us-central1",
        },
    ]
    self.config = self._setup_config(
        {"storage_backends": storage_backends_config}
    )
    self.servicer = server.TieringServiceServicer(self.config)
    self.context = mock.create_autospec(
        grpc.aio.ServicerContext, instance=True, spec_set=True
    )
    # Mock metadata for OAuth token
    self.context.invocation_metadata = mock.AsyncMock(
        return_value=(("authorization", "Bearer valid-mock-token"),)
    )

  async def asyncSetUp(self):
    super().asyncSetUp()
    await server.setup_storage_backends(self.config)
    await self.servicer.initialize()

  async def asyncTearDown(self):
    await self.servicer.close()
    await super().asyncTearDown()

  async def _reserve_asset(self):
    reserve_req = tiering_service_pb2.ReserveRequest(
        path="test/path", user="test-user", zone="us-central1-a"
    )
    reserve_res = await self.servicer.Reserve(reserve_req, self.context)
    return reserve_res.asset.uuid

  def _setup_config(self, config_dict):
    config = server_config.parse_config(config_dict)
    tmp_file = self.create_tempfile()
    config.db_connection_str = f"sqlite+aiosqlite:///{tmp_file.full_path}"
    return config

  async def test_reserve_success(self):
    request = tiering_service_pb2.ReserveRequest(
        path="test/path",
        user="test-user",
        zone="us-central1-a",
        tags=["tag1"],
    )
    response = await self.servicer.Reserve(request, self.context)

    self.assertEqual(response.asset.path, "test/path")
    self.assertEqual(response.asset.user, "test-user")
    self.assertEqual(
        response.asset.state, tiering_service_pb2.ASSET_STATE_ACTIVE_WRITE
    )
    self.assertLen(response.asset.tier_paths, 1)
    self.assertStartsWith(response.asset.tier_paths[0].path, "/mnt/lustre")

  async def test_reserve_twice(self):
    request1 = tiering_service_pb2.ReserveRequest(
        path="test/path1",
        user="test-user",
        zone="us-central1-a",
        tags=["tag1"],
    )
    response1 = await self.servicer.Reserve(request1, self.context)
    self.assertEqual(response1.asset.path, "test/path1")

    request2 = tiering_service_pb2.ReserveRequest(
        path="test/path2",
        user="test-user",
        zone="us-central1-a",
        tags=["tag2"],
    )
    response2 = await self.servicer.Reserve(request2, self.context)
    self.assertEqual(response2.asset.path, "test/path2")

  async def test_reserve_keep_alive_not_found(self):
    request = tiering_service_pb2.ReserveKeepAliveRequest(uuid="invalid-uuid")
    await self.servicer.ReserveKeepAlive(request, self.context)

    self.context.abort.assert_called_once_with(
        grpc.StatusCode.NOT_FOUND, "Asset not found"
    )

  async def test_reserve_invalid_argument(self):
    request = tiering_service_pb2.ReserveRequest(
        path="test/path",
        user="test-user",
    )
    await self.servicer.Reserve(request, self.context)

    self.context.abort.assert_called_once_with(
        grpc.StatusCode.INVALID_ARGUMENT, "No zone or region specified"
    )

  async def test_finalize_success(self):
    asset_uuid = await self._reserve_asset()
    finalize_req = tiering_service_pb2.FinalizeRequest(uuid=asset_uuid)
    finalize_res = await self.servicer.Finalize(finalize_req, self.context)

    self.assertEqual(
        finalize_res.asset.state, tiering_service_pb2.ASSET_STATE_STORED
    )

  async def test_finalize_permission_denied(self):
    asset_uuid = await self._reserve_asset()
    # Remove token from context to simulate missing auth
    self.context.invocation_metadata.return_value = ()

    finalize_req = tiering_service_pb2.FinalizeRequest(uuid=asset_uuid)
    await self.servicer.Finalize(finalize_req, self.context)

    self.context.abort.assert_called_once_with(
        grpc.StatusCode.PERMISSION_DENIED, "Insufficient Lustre permissions"
    )

  async def test_finalize_already_finalized_raises_failed_precondition(self):
    asset_uuid = await self._reserve_asset()
    await self.servicer.Finalize(
        tiering_service_pb2.FinalizeRequest(uuid=asset_uuid), self.context
    )

    # Try to finalize again
    finalize_req = tiering_service_pb2.FinalizeRequest(uuid=asset_uuid)
    await self.servicer.Finalize(finalize_req, self.context)

    self.context.abort.assert_called_once_with(
        grpc.StatusCode.FAILED_PRECONDITION,
        "Asset is in state ASSET_STATE_STORED, expected"
        " ASSET_STATE_ACTIVE_WRITE",
    )

  async def test_delete_unimplemented(self):
    asset_uuid = await self._reserve_asset()
    await self.servicer.Delete(
        tiering_service_pb2.DeleteRequest(uuid=asset_uuid), self.context
    )
    self.context.abort.assert_called_once_with(
        grpc.StatusCode.UNIMPLEMENTED, "Delete Not Implemented"
    )

  async def test_info_success(self):
    asset_uuid = await self._reserve_asset()
    response = await self.servicer.Info(
        tiering_service_pb2.InfoRequest(uuid=asset_uuid), self.context
    )
    self.assertEqual(response.assets[0].uuid, asset_uuid)

  async def test_prefetch_unimplemented(self):
    asset_uuid = await self._reserve_asset()
    # Finalize the asset to STORED state so it is eligible for prefetching
    finalize_req = tiering_service_pb2.FinalizeRequest(uuid=asset_uuid)
    await self.servicer.Finalize(finalize_req, self.context)

    prefetch_req = tiering_service_pb2.PrefetchRequest(
        uuid=asset_uuid, zone="us-central1-a"
    )
    await self.servicer.Prefetch(prefetch_req, self.context)

    self.context.abort.assert_called_once_with(
        grpc.StatusCode.UNIMPLEMENTED, "Prefetch Not Implemented"
    )

  async def test_prefetch_keep_alive_unimplemented(self):
    asset_uuid = await self._reserve_asset()
    req = tiering_service_pb2.PrefetchKeepAliveRequest(uuid=asset_uuid)
    await self.servicer.PrefetchKeepAlive(req, self.context)

    self.context.abort.assert_called_once_with(
        grpc.StatusCode.UNIMPLEMENTED, "PrefetchKeepAlive Not Implemented"
    )

  async def test_prefetch_invalid_argument(self):
    asset_uuid = await self._reserve_asset()
    prefetch_req = tiering_service_pb2.PrefetchRequest(uuid=asset_uuid)
    await self.servicer.Prefetch(prefetch_req, self.context)

    self.context.abort.assert_called_once_with(
        grpc.StatusCode.INVALID_ARGUMENT, "No location specified"
    )

  def test_tier_path_presence(self):
    tp = tiering_service_pb2.TierPath()
    self.assertFalse(tp.HasField("ready_at"))
    self.assertFalse(tp.HasField("expires_at"))

    now = timestamp_pb2.Timestamp()
    now.GetCurrentTime()
    tp.ready_at.CopyFrom(now)
    tp.expires_at.CopyFrom(now)

    self.assertTrue(tp.HasField("ready_at"))
    self.assertTrue(tp.HasField("expires_at"))

  async def test_reserve_permission_denied(self):
    # Remove token from context to simulate missing auth
    self.context.invocation_metadata.return_value = ()

    request = tiering_service_pb2.ReserveRequest(
        path="test/path",
        user="test-user",
        zone="us-central1-a",
    )
    await self.servicer.Reserve(request, self.context)

    self.context.abort.assert_called_once_with(
        grpc.StatusCode.PERMISSION_DENIED, "Insufficient Lustre permissions"
    )

  @parameterized.named_parameters(
      (
          "uninitialized",
          [
              {
                  "level": 0,
                  "backend_type": "BACKEND_TYPE_LUSTRE",
                  "prefix": "/mnt/lustre",
                  "zone": "us-central1-a",
              },
              {
                  "level": 1,
                  "backend_type": "BACKEND_TYPE_GCS",
                  "prefix": "gs://my-bucket",
                  "region": "us-central1",
              },
          ],
          True,
      ),
      (
          "matching",
          [
              {
                  "level": 1,
                  "backend_type": "BACKEND_TYPE_GCS",
                  "prefix": "gs://my-bucket",
                  "region": "us-central1",
              },
          ],
          False,
      ),
  )
  async def test_setup_storage_backends_success(
      self, storage_backends_config, check_uninitialized
  ):
    config = self._setup_config({"storage_backends": storage_backends_config})

    if check_uninitialized:
      await server.setup_storage_backends(config)
      self.assertTrue(await db_lib.async_is_db_initialized(config))
      await db_lib.async_verify_db(config)
    else:
      await server.setup_storage_backends(config)
      await server.setup_storage_backends(config)

  async def test_setup_storage_backends_mismatch(self):
    config_dict = {
        "storage_backends": [
            {
                "level": 1,
                "backend_type": "BACKEND_TYPE_GCS",
                "prefix": "gs://my-bucket",
                "region": "us-central1",
            },
        ]
    }
    config = self._setup_config(config_dict)

    # Initialize DB
    await server.setup_storage_backends(config)

    # Mismatching config
    config_mod_dict = {
        "storage_backends": [
            {
                "level": 1,
                "backend_type": "BACKEND_TYPE_GCS",
                "prefix": "gs://my-bucket",
                "region": "us-east1",
            },
        ]
    }
    config_mod = server_config.parse_config(config_mod_dict)
    config_mod.db_connection_str = config.db_connection_str

    with self.assertRaisesRegex(
        ValueError, "Configuration expects StorageBackend with key"
    ):
      await server.setup_storage_backends(config_mod)


if __name__ == "__main__":
  absltest.main()
