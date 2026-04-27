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

from unittest import mock

from absl.testing import absltest
import grpc
from orbax.checkpoint.experimental.tiering_service import server
from orbax.checkpoint.experimental.tiering_service.proto import tiering_service_pb2

from google.protobuf import timestamp_pb2


class TieringServiceTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.servicer = server.TieringServiceServicer()
    self.context = mock.create_autospec(grpc.ServicerContext, instance=True)
    # Mock metadata for OAuth token
    self.context.invocation_metadata.return_value = (
        ("authorization", "Bearer valid-mock-token"),
    )
    # Clear internal state between tests
    server._assets_by_uuid = {}

  def test_reserve_success(self):
    request = tiering_service_pb2.ReserveRequest(
        unique_path="test/path",
        user="test-user",
        zone="us-central1-a",
        tags=["tag1"],
    )
    response = self.servicer.Reserve(request, self.context)

    self.assertEqual(response.asset.unique_path, "test/path")
    self.assertEqual(response.asset.user, "test-user")
    self.assertEqual(
        response.asset.state, tiering_service_pb2.ASSET_STATE_ACTIVE_WRITE
    )
    self.assertLen(response.asset.tier_paths, 1)
    self.assertTrue(response.asset.tier_paths[0].path.startswith("gs://"))

  def test_reserve_keep_alive_not_found(self):
    request = tiering_service_pb2.ReserveKeepAliveRequest(uuid="invalid-uuid")
    self.servicer.ReserveKeepAlive(request, self.context)

    self.context.abort.assert_called_once_with(
        grpc.StatusCode.NOT_FOUND, "Asset not found"
    )

  def test_reserve_invalid_argument(self):
    request = tiering_service_pb2.ReserveRequest(
        unique_path="test/path",
        user="test-user",
    )
    self.servicer.Reserve(request, self.context)

    self.context.abort.assert_called_once_with(
        grpc.StatusCode.INVALID_ARGUMENT, "No zone or region specified"
    )

  def test_finalize_success(self):
    # First reserve
    reserve_req = tiering_service_pb2.ReserveRequest(
        unique_path="test/path", user="test-user", zone="us-central1-a"
    )
    reserve_res = self.servicer.Reserve(reserve_req, self.context)
    asset_uuid = reserve_res.asset.uuid

    # Then finalize
    finalize_req = tiering_service_pb2.FinalizeRequest(uuid=asset_uuid)
    finalize_res = self.servicer.Finalize(finalize_req, self.context)

    self.assertEqual(
        finalize_res.asset.state, tiering_service_pb2.ASSET_STATE_STORED
    )

  def test_finalize_failed_precondition(self):
    # Reserve and then finalize once
    reserve_req = tiering_service_pb2.ReserveRequest(
        unique_path="test/path", user="test-user", zone="us-central1-a"
    )
    reserve_res = self.servicer.Reserve(reserve_req, self.context)
    asset_uuid = reserve_res.asset.uuid
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
    reserve_req = tiering_service_pb2.ReserveRequest(
        unique_path="test/path", user="test-user", zone="us-central1-a"
    )
    reserve_res = self.servicer.Reserve(reserve_req, self.context)
    asset_uuid = reserve_res.asset.uuid

    self.servicer.Delete(
        tiering_service_pb2.DeleteRequest(uuid=asset_uuid), self.context
    )
    self.assertNotIn(asset_uuid, server._assets_by_uuid)

  def test_info_success(self):
    reserve_req = tiering_service_pb2.ReserveRequest(
        unique_path="test/path", user="test-user", zone="us-central1-a"
    )
    reserve_res = self.servicer.Reserve(reserve_req, self.context)
    asset_uuid = reserve_res.asset.uuid

    response = self.servicer.Info(
        tiering_service_pb2.InfoRequest(uuid=asset_uuid), self.context
    )
    self.assertEqual(response.asset.uuid, asset_uuid)

  def test_prefetch_success(self):
    reserve_req = tiering_service_pb2.ReserveRequest(
        unique_path="test/path", user="test-user", zone="us-central1-a"
    )
    reserve_res = self.servicer.Reserve(reserve_req, self.context)
    asset_uuid = reserve_res.asset.uuid

    prefetch_req = tiering_service_pb2.PrefetchRequest(
        uuid=asset_uuid, zone="us-central1-a"
    )
    response = self.servicer.Prefetch(prefetch_req, self.context)

    self.assertEqual(response.asset.uuid, asset_uuid)

  def test_prefetch_invalid_argument(self):
    reserve_req = tiering_service_pb2.ReserveRequest(
        unique_path="test/path", user="test-user", zone="us-central1-a"
    )
    reserve_res = self.servicer.Reserve(reserve_req, self.context)
    asset_uuid = reserve_res.asset.uuid

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
        unique_path="test/path",
        user="test-user",
        zone="us-central1-a",
    )
    self.servicer.Reserve(request, self.context)

    self.context.abort.assert_called_once_with(
        grpc.StatusCode.PERMISSION_DENIED, "Insufficient GCS permissions"
    )


if __name__ == "__main__":
  absltest.main()
