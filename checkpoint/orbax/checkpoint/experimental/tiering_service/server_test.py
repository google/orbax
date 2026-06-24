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
from orbax.checkpoint.experimental.tiering_service import assets
from orbax.checkpoint.experimental.tiering_service import auth
from orbax.checkpoint.experimental.tiering_service import db_lib
from orbax.checkpoint.experimental.tiering_service import db_schema
from orbax.checkpoint.experimental.tiering_service import server
from orbax.checkpoint.experimental.tiering_service import server_config
from orbax.checkpoint.experimental.tiering_service.proto import tiering_service_pb2
from sqlalchemy.future import select

from google.protobuf import timestamp_pb2


async def _setup_prefetch_req(
    servicer: server.TieringServiceServicer,
    asset_uuid: str,
    context: grpc.aio.ServicerContext,
) -> tiering_service_pb2.PrefetchRequest:
  del servicer, context
  return tiering_service_pb2.PrefetchRequest(
      uuid=asset_uuid, zone="us-central1-b"
  )


async def _setup_prefetch_keep_alive_req(
    servicer: server.TieringServiceServicer,
    asset_uuid: str,
    context: grpc.aio.ServicerContext,
) -> tiering_service_pb2.PrefetchKeepAliveRequest:
  """Sets up a prefetch keep-alive request by prefetching to zone B first."""
  prefetch_res = await servicer.Prefetch(
      tiering_service_pb2.PrefetchRequest(
          uuid=asset_uuid, zone="us-central1-b"
      ),
      context,
  )
  tp_b = next(
      tp for tp in prefetch_res.asset.tier_paths if "lustre-b" in tp.path
  )
  return tiering_service_pb2.PrefetchKeepAliveRequest(
      tier_path_uuid=tp_b.tier_path_uuid
  )


async def _setup_delete_invalid_uuid_req(
    servicer: server.TieringServiceServicer,
    asset_uuid: str,
    context: grpc.aio.ServicerContext,
) -> tiering_service_pb2.DeleteRequest:
  del servicer, asset_uuid, context
  return tiering_service_pb2.DeleteRequest(uuid="invalid-uuid")


async def _setup_delete_invalid_path_req(
    servicer: server.TieringServiceServicer,
    asset_uuid: str,
    context: grpc.aio.ServicerContext,
) -> tiering_service_pb2.DeleteRequest:
  del servicer, asset_uuid, context
  return tiering_service_pb2.DeleteRequest(path="non-existent/path")


async def _setup_delete_already_deleted_req(
    servicer: server.TieringServiceServicer,
    asset_uuid: str,
    context: grpc.aio.ServicerContext,
) -> tiering_service_pb2.DeleteRequest:
  await servicer.Delete(
      tiering_service_pb2.DeleteRequest(uuid=asset_uuid), context
  )
  # Simulate job processed (mark state as DELETED).
  async with servicer._session_scope() as session:
    db_assets = await assets.fetch_asset_by_uuid(session, asset_uuid)
    db_assets[0].state = db_schema.AssetState.ASSET_STATE_DELETED
    await session.commit()
  return tiering_service_pb2.DeleteRequest(uuid=asset_uuid)


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

  def _get_multi_lustre_config(self, zone_prefixes):
    """Sets up config with multiple Lustre backends and a default GCS backend."""
    storage_backends_config = []
    for zone, suffix in zone_prefixes:
      storage_backends_config.append({
          "level": 0,
          "backend_type": "BACKEND_TYPE_LUSTRE",
          "prefix": f"/mnt/lustre-{suffix}",
          "zone": zone,
      })
    # Add the default GCS backend.
    storage_backends_config.append({
        "level": 1,
        "backend_type": "BACKEND_TYPE_GCS",
        "prefix": "gs://my-bucket",
        "region": "us-central1",
    })
    return self._setup_config({"storage_backends": storage_backends_config})

  async def _setup_servicer_and_asset(self):
    """Sets up a servicer with 2 Lustre backends and reserves/finalizes an asset."""
    config = self._get_multi_lustre_config([
        ("us-central1-a", "a"),
        ("us-central1-b", "b"),
    ])
    servicer = server.TieringServiceServicer(config)
    await server.setup_storage_backends(config)
    await servicer.initialize()
    self.addAsyncCleanup(servicer.close)

    # Reserve and Finalize on A.
    reserve_res = await servicer.Reserve(
        tiering_service_pb2.ReserveRequest(
            path="test/path", user="test-user", zone="us-central1-a"
        ),
        self.context,
    )
    asset_uuid = reserve_res.asset.uuid
    await servicer.Finalize(
        tiering_service_pb2.FinalizeRequest(uuid=asset_uuid), self.context
    )
    return servicer, asset_uuid

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
    self.assertEqual(
        response.tier_path_uuid, response.asset.tier_paths[0].tier_path_uuid
    )
    self.assertTrue(response.tier_path_uuid)

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
    # Remove token from context to simulate missing auth.
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

    # Try to finalize again.
    finalize_req = tiering_service_pb2.FinalizeRequest(uuid=asset_uuid)
    await self.servicer.Finalize(finalize_req, self.context)

    self.context.abort.assert_called_once_with(
        grpc.StatusCode.FAILED_PRECONDITION,
        "Asset is in state ASSET_STATE_STORED, expected"
        " ASSET_STATE_ACTIVE_WRITE",
    )

  async def test_delete_by_uuid_success(self):
    asset_uuid = await self._reserve_asset()
    request = tiering_service_pb2.DeleteRequest(uuid=asset_uuid)
    response = await self.servicer.Delete(request, self.context)

    with self.subTest("Response is not None"):
      self.assertIsNotNone(response)

    with self.subTest("Database asset state"):
      async with self.servicer._session_scope() as session:
        db_assets = await assets.fetch_asset_by_uuid(session, asset_uuid)
        self.assertLen(db_assets, 1)
        self.assertEqual(
            db_assets[0].state, db_schema.AssetState.ASSET_STATE_ACTIVE_WRITE
        )
        self.assertIsNone(db_assets[0].deleted_at)

    with self.subTest("Delete job queued"):
      async with self.servicer._session_scope() as session:
        stmt = select(db_schema.AssetJob).filter_by(
            asset_uuid=asset_uuid,
            request_type=db_schema.RequestType.REQUEST_TYPE_DELETE_FROM_ALL_TIERS,
        )
        result = await session.execute(stmt)
        jobs = result.scalars().all()
        self.assertLen(jobs, 1)
        self.assertEqual(jobs[0].status, db_schema.JobStatus.JOB_STATUS_QUEUED)

  async def test_delete_by_path_success(self):
    asset_uuid = await self._reserve_asset()
    request = tiering_service_pb2.DeleteRequest(path="test/path")
    response = await self.servicer.Delete(request, self.context)

    self.assertIsNotNone(response)

    # Verify DB state (should not change state immediately).
    async with self.servicer._session_scope() as session:
      db_assets = await assets.fetch_asset_by_uuid(session, asset_uuid)
      self.assertLen(db_assets, 1)
      self.assertEqual(
          db_assets[0].state, db_schema.AssetState.ASSET_STATE_ACTIVE_WRITE
      )

      # Verify AssetJob is queued.
      stmt = select(db_schema.AssetJob).filter_by(
          asset_uuid=asset_uuid,
          request_type=db_schema.RequestType.REQUEST_TYPE_DELETE_FROM_ALL_TIERS,
      )
      result = await session.execute(stmt)
      jobs = result.scalars().all()
      self.assertLen(jobs, 1)
      self.assertEqual(jobs[0].status, db_schema.JobStatus.JOB_STATUS_QUEUED)

  async def test_delete_finalized_success(self):
    asset_uuid = await self._reserve_asset()
    await self.servicer.Finalize(
        tiering_service_pb2.FinalizeRequest(uuid=asset_uuid), self.context
    )

    request = tiering_service_pb2.DeleteRequest(uuid=asset_uuid)
    response = await self.servicer.Delete(request, self.context)
    self.assertIsNotNone(response)

    # Verify state remains STORED.
    async with self.servicer._session_scope() as session:
      db_assets = await assets.fetch_asset_by_uuid(session, asset_uuid)
      self.assertLen(db_assets, 1)
      self.assertEqual(
          db_assets[0].state, db_schema.AssetState.ASSET_STATE_STORED
      )

  async def test_delete_prefetched_success(self):
    servicer, asset_uuid = await self._setup_servicer_and_asset()

    # Prefetch to zone B (creates a second tier path).
    await servicer.Prefetch(
        tiering_service_pb2.PrefetchRequest(
            uuid=asset_uuid, zone="us-central1-b"
        ),
        self.context,
    )

    # Delete the asset.
    response = await servicer.Delete(
        tiering_service_pb2.DeleteRequest(uuid=asset_uuid), self.context
    )

    with self.subTest("Delete response is not None"):
      self.assertIsNotNone(response)

    async with servicer._session_scope() as session:
      with self.subTest("DB state is STORED"):
        db_assets = await assets.fetch_asset_by_uuid(session, asset_uuid)
        self.assertLen(db_assets, 1)
        self.assertEqual(
            db_assets[0].state, db_schema.AssetState.ASSET_STATE_STORED
        )

      with self.subTest("Delete job is queued"):
        stmt = select(db_schema.AssetJob).filter_by(
            asset_uuid=asset_uuid,
            request_type=db_schema.RequestType.REQUEST_TYPE_DELETE_FROM_ALL_TIERS,
        )
        result = await session.execute(stmt)
        jobs = result.scalars().all()
        self.assertLen(jobs, 1)

  async def test_delete_permission_denied(self):
    asset_uuid = await self._reserve_asset()
    # Remove token from context to simulate missing auth.
    self.context.invocation_metadata.return_value = ()

    request = tiering_service_pb2.DeleteRequest(uuid=asset_uuid)
    await self.servicer.Delete(request, self.context)

    self.context.abort.assert_called_once_with(
        grpc.StatusCode.PERMISSION_DENIED, "Insufficient Lustre permissions"
    )

  async def test_delete_prefetched_permission_denied_on_any_backend(self):
    servicer, asset_uuid = await self._setup_servicer_and_asset()

    # Prefetch to zone B.
    await servicer.Prefetch(
        tiering_service_pb2.PrefetchRequest(
            uuid=asset_uuid, zone="us-central1-b"
        ),
        self.context,
    )

    # Mock OAuth verification failing for zone B's prefix but passing for
    # zone A.
    async def mock_has_write_permission(unused_token, *, backend, path):
      del unused_token, backend  # Unused
      if "lustre-b" in path:
        return False
      return True

    with mock.patch.object(
        auth,
        "has_write_permission",
        autospec=True,
        side_effect=mock_has_write_permission,
    ):
      await servicer.Delete(
          tiering_service_pb2.DeleteRequest(uuid=asset_uuid), self.context
      )

    self.context.abort.assert_called_once_with(
        grpc.StatusCode.PERMISSION_DENIED, "Insufficient Lustre permissions"
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="invalid_uuid",
          setup_req_func=_setup_delete_invalid_uuid_req,
      ),
      dict(
          testcase_name="invalid_path",
          setup_req_func=_setup_delete_invalid_path_req,
      ),
      dict(
          testcase_name="already_deleted",
          setup_req_func=_setup_delete_already_deleted_req,
      ),
  )
  async def test_delete_not_found_parameterized(self, setup_req_func):
    servicer, asset_uuid = await self._setup_servicer_and_asset()

    req = await setup_req_func(servicer, asset_uuid, self.context)

    self.context.abort.reset_mock()
    await servicer.Delete(req, self.context)

    self.context.abort.assert_called_once_with(
        grpc.StatusCode.NOT_FOUND, "Asset not found"
    )

  async def test_info_success(self):
    asset_uuid = await self._reserve_asset()
    response = await self.servicer.Info(
        tiering_service_pb2.InfoRequest(uuid=asset_uuid), self.context
    )
    self.assertEqual(response.assets[0].uuid, asset_uuid)

  async def test_prefetch_success_rpc_response(self):
    servicer, asset_uuid = await self._setup_servicer_and_asset()

    prefetch_req = tiering_service_pb2.PrefetchRequest(
        uuid=asset_uuid, zone="us-central1-b"
    )
    prefetch_res = await servicer.Prefetch(prefetch_req, self.context)

    self.assertEqual(prefetch_res.asset.uuid, asset_uuid)
    paths = [tp.path for tp in prefetch_res.asset.tier_paths]
    self.assertCountEqual(
        paths,
        ["/mnt/lustre-a/test/path", "/mnt/lustre-b/test/path"],
    )
    # In the setup, index 1 corresponds to zone B's Lustre backend target
    self.assertEqual(
        prefetch_res.closest_tier_path_uuid,
        prefetch_res.asset.tier_paths[1].tier_path_uuid,
    )
    self.assertTrue(prefetch_res.closest_tier_path_uuid)

  async def test_prefetch_success_db_job_creation(self):
    servicer, asset_uuid = await self._setup_servicer_and_asset()

    prefetch_req = tiering_service_pb2.PrefetchRequest(
        uuid=asset_uuid, zone="us-central1-b"
    )
    await servicer.Prefetch(prefetch_req, self.context)

    async with servicer._session_scope() as session:
      stmt = select(db_schema.AssetJob).filter_by(
          asset_uuid=asset_uuid,
          request_type=db_schema.RequestType.REQUEST_TYPE_COPY,
      )
      result = await session.execute(stmt)
      jobs = result.scalars().all()
      self.assertLen(jobs, 1)
      self.assertEqual(jobs[0].status, db_schema.JobStatus.JOB_STATUS_QUEUED)
      target_tp_id = jobs[0].target_tier_path_id

      stmt_tp = select(db_schema.TierPath).filter_by(
          asset_uuid=asset_uuid, path="/mnt/lustre-b/test/path"
      )
      result_tp = await session.execute(stmt_tp)
      tp_b = result_tp.scalars().first()
      self.assertIsNotNone(tp_b)
      self.assertEqual(target_tp_id, tp_b.id)
      self.assertIsNone(tp_b.ready_at)

  async def test_prefetch_idempotent(self):
    servicer, asset_uuid = await self._setup_servicer_and_asset()

    # 1. Prefetch from B (first time).
    await servicer.Prefetch(
        tiering_service_pb2.PrefetchRequest(
            uuid=asset_uuid, zone="us-central1-b"
        ),
        self.context,
    )

    # 2. Prefetch from B (second time).
    await servicer.Prefetch(
        tiering_service_pb2.PrefetchRequest(
            uuid=asset_uuid, zone="us-central1-b"
        ),
        self.context,
    )

    # Verify only ONE job was created.
    async with servicer._session_scope() as session:
      stmt = select(db_schema.AssetJob).filter_by(
          asset_uuid=asset_uuid,
          request_type=db_schema.RequestType.REQUEST_TYPE_COPY,
      )
      result = await session.execute(stmt)
      jobs = result.scalars().all()
      self.assertLen(jobs, 1)

  async def test_prefetch_already_ready(self):
    # If we prefetch to the same zone where it was reserved and finalized,
    # it should be already ready, so no job should be created.
    asset_uuid = await self._reserve_asset()
    await self.servicer.Finalize(
        tiering_service_pb2.FinalizeRequest(uuid=asset_uuid), self.context
    )

    # Prefetch to the same zone "us-central1-a"
    prefetch_req = tiering_service_pb2.PrefetchRequest(
        uuid=asset_uuid, zone="us-central1-a"
    )
    response = await self.servicer.Prefetch(prefetch_req, self.context)

    # Verify response
    self.assertEqual(response.asset.uuid, asset_uuid)
    self.assertLen(response.asset.tier_paths, 1)
    self.assertIsNotNone(response.asset.tier_paths[0].ready_at.ToDatetime())

    # Verify NO job was created
    async with self.servicer._session_scope() as session:
      stmt = select(db_schema.AssetJob).filter_by(
          asset_uuid=asset_uuid,
          request_type=db_schema.RequestType.REQUEST_TYPE_COPY,
      )
      result = await session.execute(stmt)
      jobs = result.scalars().all()
      self.assertEmpty(jobs)

  async def test_prefetch_permission_denied(self):
    servicer, asset_uuid = await self._setup_servicer_and_asset()

    # Remove token from context to simulate missing auth.
    self.context.invocation_metadata.return_value = ()

    # Prefetch from B (should fail).
    prefetch_req = tiering_service_pb2.PrefetchRequest(
        uuid=asset_uuid, zone="us-central1-b"
    )
    await servicer.Prefetch(prefetch_req, self.context)

    self.context.abort.assert_called_once_with(
        grpc.StatusCode.PERMISSION_DENIED,
        "Insufficient read permissions on source Lustre",
    )

  async def test_prefetch_permission_denied_on_target(self):
    servicer, asset_uuid = await self._setup_servicer_and_asset()

    # Mock has_read_permission to succeed for source (lustre-a) but fail for
    # target (lustre-b).
    async def mock_has_read_permission(unused_token, *, backend, path):
      del unused_token, backend  # Unused.
      if "lustre-b" in path:
        return False
      return True

    with mock.patch.object(
        auth,
        "has_read_permission",
        autospec=True,
        side_effect=mock_has_read_permission,
    ):
      prefetch_req = tiering_service_pb2.PrefetchRequest(
          uuid=asset_uuid, zone="us-central1-b"
      )
      await servicer.Prefetch(prefetch_req, self.context)

    self.context.abort.assert_called_once_with(
        grpc.StatusCode.PERMISSION_DENIED,
        "Insufficient read permissions on target Lustre",
    )

  @parameterized.named_parameters(
      dict(testcase_name="asset_not_finalized", reserve_asset=True),
      dict(testcase_name="asset_does_not_exist", reserve_asset=False),
  )
  async def test_prefetch_not_found(self, reserve_asset):
    asset_uuid = "invalid-uuid"
    if reserve_asset:
      asset_uuid = await self._reserve_asset()

    prefetch_req = tiering_service_pb2.PrefetchRequest(
        uuid=asset_uuid, zone="us-central1-a"
    )
    await self.servicer.Prefetch(prefetch_req, self.context)

    self.context.abort.assert_called_once_with(
        grpc.StatusCode.NOT_FOUND, "Asset not found"
    )

  async def test_prefetch_backend_not_found(self):
    asset_uuid = await self._reserve_asset()
    prefetch_req = tiering_service_pb2.PrefetchRequest(
        uuid=asset_uuid, zone="us-central1-b"
    )
    await self.servicer.Prefetch(prefetch_req, self.context)

    self.context.abort.assert_called_once_with(
        grpc.StatusCode.NOT_FOUND,
        "No level 0 storage backend found for zone:us-central1-b / region:None",
    )

  async def test_prefetch_backend_not_found_region_only(self):
    prefetch_req = tiering_service_pb2.PrefetchRequest(
        uuid="dummy-uuid", region="us-east1"
    )
    await self.servicer.Prefetch(prefetch_req, self.context)

    self.context.abort.assert_called_once_with(
        grpc.StatusCode.NOT_FOUND,
        "No level 0 storage backend found for zone:None / region:us-east1",
    )

  async def test_prefetch_keep_alive_grpc_success(self):
    servicer, asset_uuid = await self._setup_servicer_and_asset()

    # 1. Prefetch on B (triggers job & creates TierPath on B)
    prefetch_res = await servicer.Prefetch(
        tiering_service_pb2.PrefetchRequest(
            uuid=asset_uuid, zone="us-central1-b"
        ),
        self.context,
    )
    self.assertLen(prefetch_res.asset.tier_paths, 2)
    tp_b = next(
        tp for tp in prefetch_res.asset.tier_paths if "lustre-b" in tp.path
    )
    self.assertTrue(tp_b.HasField("expires_at"))
    initial_expires_at = tp_b.expires_at.ToDatetime()

    # 2. Call PrefetchKeepAlive
    keep_alive_req = tiering_service_pb2.PrefetchKeepAliveRequest(
        tier_path_uuid=tp_b.tier_path_uuid
    )
    keep_alive_res = await servicer.PrefetchKeepAlive(
        keep_alive_req, self.context
    )

    # Verify TTL is extended
    tp_b_extended = next(
        tp for tp in keep_alive_res.asset.tier_paths if "lustre-b" in tp.path
    )
    self.assertGreater(
        tp_b_extended.expires_at.ToDatetime(), initial_expires_at
    )

  async def test_prefetch_keep_alive_not_found_fails(self):
    req = tiering_service_pb2.PrefetchKeepAliveRequest(
        tier_path_uuid="non-existent-uuid"
    )
    await self.servicer.PrefetchKeepAlive(req, self.context)
    self.context.abort.assert_called_once_with(
        grpc.StatusCode.NOT_FOUND, "TierPath not found"
    )

  async def test_prefetch_keep_alive_multi_zone_isolation(self):
    config = self._get_multi_lustre_config([
        ("us-central1-a", "a"),
        ("us-central1-b", "b"),
        ("us-central1-c", "c"),
    ])
    servicer = server.TieringServiceServicer(config)
    await server.setup_storage_backends(config)
    await servicer.initialize()
    self.addAsyncCleanup(servicer.close)

    # 1. Reserve and Finalize on C
    reserve_res = await servicer.Reserve(
        tiering_service_pb2.ReserveRequest(
            path="test/path", user="test-user", zone="us-central1-c"
        ),
        self.context,
    )
    asset_uuid = reserve_res.asset.uuid
    await servicer.Finalize(
        tiering_service_pb2.FinalizeRequest(uuid=asset_uuid), self.context
    )

    # 2. Prefetch to A (Zone A)
    prefetch_res_a = await servicer.Prefetch(
        tiering_service_pb2.PrefetchRequest(
            uuid=asset_uuid, zone="us-central1-a"
        ),
        self.context,
    )
    tp_a = next(
        tp for tp in prefetch_res_a.asset.tier_paths if "lustre-a" in tp.path
    )
    expires_at_a_initial = tp_a.expires_at.ToDatetime()

    # 3. Prefetch to B (Zone B)
    prefetch_res_b = await servicer.Prefetch(
        tiering_service_pb2.PrefetchRequest(
            uuid=asset_uuid, zone="us-central1-b"
        ),
        self.context,
    )
    tp_b = next(
        tp for tp in prefetch_res_b.asset.tier_paths if "lustre-b" in tp.path
    )
    expires_at_b_initial = tp_b.expires_at.ToDatetime()

    # 4. Extend Zone A's TTL
    keep_alive_req = tiering_service_pb2.PrefetchKeepAliveRequest(
        tier_path_uuid=tp_a.tier_path_uuid
    )
    keep_alive_res = await servicer.PrefetchKeepAlive(
        keep_alive_req, self.context
    )

    # Verify Zone A's TTL is extended
    tp_a_extended = next(
        tp for tp in keep_alive_res.asset.tier_paths if "lustre-a" in tp.path
    )
    self.assertGreater(
        tp_a_extended.expires_at.ToDatetime(), expires_at_a_initial
    )

    # Verify Zone B's TTL remains strictly unchanged (isolation)
    tp_b_post = next(
        tp for tp in keep_alive_res.asset.tier_paths if "lustre-b" in tp.path
    )
    self.assertEqual(tp_b_post.expires_at.ToDatetime(), expires_at_b_initial)

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
    # Remove token from context to simulate missing auth.
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

  @parameterized.named_parameters(
      dict(
          testcase_name="prefetch",
          rpc_name="Prefetch",
          setup_req_func=_setup_prefetch_req,
      ),
      dict(
          testcase_name="prefetch_keep_alive",
          rpc_name="PrefetchKeepAlive",
          setup_req_func=_setup_prefetch_keep_alive_req,
      ),
  )
  async def test_rpc_fails_if_delete_pending(self, rpc_name, setup_req_func):
    servicer, asset_uuid = await self._setup_servicer_and_asset()

    req = await setup_req_func(servicer, asset_uuid, self.context)

    await servicer.Delete(
        tiering_service_pb2.DeleteRequest(uuid=asset_uuid), self.context
    )

    self.context.abort.reset_mock()
    rpc_method = getattr(servicer, rpc_name)
    await rpc_method(req, self.context)

    self.context.abort.assert_called_once()
    status_code, message = self.context.abort.call_args[0]
    self.assertEqual(status_code, grpc.StatusCode.FAILED_PRECONDITION)
    self.assertIn("marked for deletion", message)


class CtsServerTest(unittest.IsolatedAsyncioTestCase):

  @mock.patch(
      "orbax.checkpoint.experimental.tiering_service."
      "server_config.load_config"
  )
  @mock.patch(
      "orbax.checkpoint.experimental.tiering_service."
      "server.setup_storage_backends"
  )
  @mock.patch("grpc.aio.server")
  @mock.patch(
      "orbax.checkpoint.experimental.tiering_service."
      "server.TieringServiceServicer"
  )
  @mock.patch(
      "orbax.checkpoint.experimental.tiering_service."
      "job_worker.run_tiering_service_worker_loop"
  )
  async def test_serve_default_does_not_start_worker(
      self,
      mock_run_worker,
      mock_servicer_class,
      mock_grpc_server,
      mock_setup_backends,
      mock_load_config,
  ):
    del mock_setup_backends
    mock_load_config.return_value = mock.MagicMock()
    mock_server_instance = mock.MagicMock()
    mock_server_instance.start = mock.AsyncMock()
    mock_server_instance.wait_for_termination = mock.AsyncMock()
    mock_server_instance.stop = mock.AsyncMock()
    mock_grpc_server.return_value = mock_server_instance

    mock_servicer_instance = mock.AsyncMock()
    mock_servicer_class.return_value = mock_servicer_instance

    server_cli = server.CtsServer()
    await server_cli.serve("dummy.yaml")

    mock_run_worker.assert_not_called()

  @mock.patch(
      "orbax.checkpoint.experimental.tiering_service."
      "server_config.load_config"
  )
  @mock.patch(
      "orbax.checkpoint.experimental.tiering_service."
      "server.setup_storage_backends"
  )
  @mock.patch("grpc.aio.server")
  @mock.patch(
      "orbax.checkpoint.experimental.tiering_service."
      "server.TieringServiceServicer"
  )
  @mock.patch(
      "orbax.checkpoint.experimental.tiering_service."
      "job_worker.run_tiering_service_worker_loop"
  )
  async def test_serve_starts_worker_if_requested(
      self,
      mock_run_worker,
      mock_servicer_class,
      mock_grpc_server,
      mock_setup_backends,
      mock_load_config,
  ):
    del mock_setup_backends
    mock_load_config.return_value = mock.MagicMock()
    mock_server_instance = mock.MagicMock()
    mock_server_instance.start = mock.AsyncMock()
    mock_server_instance.wait_for_termination = mock.AsyncMock()
    mock_server_instance.stop = mock.AsyncMock()
    mock_grpc_server.return_value = mock_server_instance

    mock_servicer_instance = mock.AsyncMock()
    mock_servicer_class.return_value = mock_servicer_instance

    mock_worker_instance = mock.AsyncMock()
    mock_run_worker.return_value = mock_worker_instance

    server_cli = server.CtsServer()
    await server_cli.serve("dummy.yaml", start_tiering_service_worker=True)

    mock_run_worker.assert_called_once()
    mock_worker_instance.stop.assert_called_once()


if __name__ == "__main__":
  absltest.main()
