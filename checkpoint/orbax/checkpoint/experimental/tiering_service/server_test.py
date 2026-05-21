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
    self.servicer = server.TieringServiceServicer()
    self.context = mock.create_autospec(
        grpc.ServicerContext, instance=True, spec_set=True
    )
    # Mock metadata for OAuth token
    self.context.invocation_metadata.return_value = (
        ("authorization", "Bearer valid-mock-token"),
    )
    # Clear internal state between tests
    server._assets_by_uuid = {}

  def _reserve_asset(self):
    reserve_req = tiering_service_pb2.ReserveRequest(
        path="test/path", user="test-user", zone="us-central1-a"
    )
    reserve_res = self.servicer.Reserve(reserve_req, self.context)
    return reserve_res.asset.uuid

  def _setup_config(self, config_dict):
    config = server_config.parse_config(config_dict)
    tmp_file = self.create_tempfile()
    config.db_connection_str = f"sqlite+aiosqlite:///{tmp_file.full_path}"
    return config

  def test_reserve_success(self):
    request = tiering_service_pb2.ReserveRequest(
        path="test/path",
        user="test-user",
        zone="us-central1-a",
        tags=["tag1"],
    )
    response = self.servicer.Reserve(request, self.context)

    self.assertEqual(response.asset.path, "test/path")
    self.assertEqual(response.asset.user, "test-user")
    self.assertEqual(
        response.asset.state, tiering_service_pb2.ASSET_STATE_ACTIVE_WRITE
    )
    self.assertLen(response.asset.tier_paths, 1)
    self.assertTrue(response.asset.tier_paths[0].path.startswith("/mnt/lustre"))

  def test_reserve_keep_alive_not_found(self):
    request = tiering_service_pb2.ReserveKeepAliveRequest(uuid="invalid-uuid")
    self.servicer.ReserveKeepAlive(request, self.context)

    self.context.abort.assert_called_once_with(
        grpc.StatusCode.NOT_FOUND, "Asset not found"
    )

  def test_reserve_invalid_argument(self):
    request = tiering_service_pb2.ReserveRequest(
        path="test/path",
        user="test-user",
    )
    self.servicer.Reserve(request, self.context)

    self.context.abort.assert_called_once_with(
        grpc.StatusCode.INVALID_ARGUMENT, "No zone or region specified"
    )

  def test_finalize_success(self):
    asset_uuid = self._reserve_asset()
    finalize_req = tiering_service_pb2.FinalizeRequest(uuid=asset_uuid)
    finalize_res = self.servicer.Finalize(finalize_req, self.context)

    self.assertEqual(
        finalize_res.asset.state, tiering_service_pb2.ASSET_STATE_STORED
    )

  def test_finalize_failed_precondition(self):
    asset_uuid = self._reserve_asset()
    self.servicer.Finalize(
        tiering_service_pb2.FinalizeRequest(uuid=asset_uuid), self.context
    )

    # Try to finalize again
    finalize_req = tiering_service_pb2.FinalizeRequest(uuid=asset_uuid)
    self.servicer.Finalize(finalize_req, self.context)

    self.context.abort.assert_called_once_with(
        grpc.StatusCode.FAILED_PRECONDITION, "Asset not in ACTIVE_WRITE state"
    )

  def test_delete_success(self):
    asset_uuid = self._reserve_asset()
    self.servicer.Delete(
        tiering_service_pb2.DeleteRequest(uuid=asset_uuid), self.context
    )
    self.assertNotIn(asset_uuid, server._assets_by_uuid)

  def test_info_success(self):
    asset_uuid = self._reserve_asset()
    response = self.servicer.Info(
        tiering_service_pb2.InfoRequest(uuid=asset_uuid), self.context
    )
    self.assertEqual(response.asset.uuid, asset_uuid)

  def test_prefetch_success(self):
    asset_uuid = self._reserve_asset()
    prefetch_req = tiering_service_pb2.PrefetchRequest(
        uuid=asset_uuid, zone="us-central1-a"
    )
    response = self.servicer.Prefetch(prefetch_req, self.context)

    self.assertEqual(response.asset.uuid, asset_uuid)

  def test_prefetch_invalid_argument(self):
    asset_uuid = self._reserve_asset()
    prefetch_req = tiering_service_pb2.PrefetchRequest(uuid=asset_uuid)
    self.servicer.Prefetch(prefetch_req, self.context)

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

  def test_reserve_permission_denied(self):
    # Remove token from context to simulate missing auth
    self.context.invocation_metadata.return_value = ()

    request = tiering_service_pb2.ReserveRequest(
        path="test/path",
        user="test-user",
        zone="us-central1-a",
    )
    self.servicer.Reserve(request, self.context)

    self.context.abort.assert_called_once_with(
        grpc.StatusCode.PERMISSION_DENIED, "Insufficient GCS permissions"
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
