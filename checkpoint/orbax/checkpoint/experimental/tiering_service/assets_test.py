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

import datetime
import unittest

from absl.testing import absltest
from absl.testing import parameterized
import aiosqlite  # pylint: disable=unused-import
import greenlet  # pylint: disable=unused-import
from orbax.checkpoint.experimental.tiering_service import assets
from orbax.checkpoint.experimental.tiering_service import db_schema
from orbax.checkpoint.experimental.tiering_service import storage_backend as storage_backend_lib
from orbax.checkpoint.experimental.tiering_service.proto import tiering_service_pb2
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.future import select
from sqlalchemy.orm import joinedload

from google.protobuf import timestamp_pb2


class AssetsProtoTest(absltest.TestCase):

  def test_proto_from_db_asset_basic(self):
    db_asset = db_schema.Asset(
        asset_uuid="test-uuid",
        path="test/path",
        user="test-user",
        tags=["tag1", "tag2"],
        state=db_schema.AssetState.ASSET_STATE_ACTIVE_WRITE,
    )
    proto_asset = assets.proto_from_db_asset(db_asset)

    self.assertEqual(proto_asset.uuid, "test-uuid")
    self.assertEqual(proto_asset.path, "test/path")
    self.assertEqual(proto_asset.user, "test-user")
    self.assertEqual(list(proto_asset.tags), ["tag1", "tag2"])
    self.assertEqual(
        proto_asset.state, tiering_service_pb2.ASSET_STATE_ACTIVE_WRITE
    )
    self.assertFalse(proto_asset.HasField("created_at"))
    self.assertFalse(proto_asset.HasField("finalized_at"))
    self.assertFalse(proto_asset.HasField("deleted_at"))
    self.assertFalse(proto_asset.HasField("updated_at"))
    self.assertEmpty(proto_asset.tier_paths)

  def test_proto_from_db_asset_timestamps(self):
    dt_created = datetime.datetime(
        2026, 1, 1, 10, 0, 0, tzinfo=datetime.timezone.utc
    )
    expected_ts_created = timestamp_pb2.Timestamp()
    expected_ts_created.FromDatetime(dt_created)

    tz_est = datetime.timezone(datetime.timedelta(hours=-5))
    dt_finalized = datetime.datetime(2026, 1, 2, 11, 0, 0, tzinfo=tz_est)
    expected_ts_finalized = timestamp_pb2.Timestamp()
    expected_ts_finalized.FromDatetime(dt_finalized)

    tz_pst = datetime.timezone(datetime.timedelta(hours=-8))
    dt_deleted = datetime.datetime(2026, 1, 3, 12, 0, 0, tzinfo=tz_pst)
    expected_ts_deleted = timestamp_pb2.Timestamp()
    expected_ts_deleted.FromDatetime(dt_deleted)

    tz_jst = datetime.timezone(datetime.timedelta(hours=9))
    dt_updated = datetime.datetime(2026, 1, 4, 13, 0, 0, tzinfo=tz_jst)
    expected_ts_updated = timestamp_pb2.Timestamp()
    expected_ts_updated.FromDatetime(dt_updated)

    db_asset = db_schema.Asset(
        asset_uuid="test-uuid",
        path="test/path",
        user="test-user",
        state=db_schema.AssetState.ASSET_STATE_ACTIVE_WRITE,
        created_at=dt_created,
        finalized_at=dt_finalized,
        deleted_at=dt_deleted,
        updated_at=dt_updated,
    )
    proto_asset = assets.proto_from_db_asset(db_asset)

    self.assertEqual(proto_asset.created_at, expected_ts_created)
    self.assertEqual(proto_asset.finalized_at, expected_ts_finalized)
    self.assertEqual(proto_asset.deleted_at, expected_ts_deleted)
    self.assertEqual(proto_asset.updated_at, expected_ts_updated)

  def test_proto_from_db_asset_tier_path_and_backend(self):
    tz_ist = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    dt_ready = datetime.datetime(2026, 1, 5, 14, 0, 0, tzinfo=tz_ist)
    expected_ts_ready = timestamp_pb2.Timestamp()
    expected_ts_ready.FromDatetime(dt_ready)

    tz_cest = datetime.timezone(datetime.timedelta(hours=2))
    dt_expires = datetime.datetime(2026, 1, 6, 15, 0, 0, tzinfo=tz_cest)
    expected_ts_expires = timestamp_pb2.Timestamp()
    expected_ts_expires.FromDatetime(dt_expires)

    db_backend = db_schema.StorageBackend(
        id=1,
        level=0,
        backend_type=db_schema.BackendType.BACKEND_TYPE_LUSTRE,
        prefix="/mnt/lustre",
        zone="us-central1-a",
    )
    db_tp = db_schema.TierPath(
        id=10,
        path="/mnt/lustre/test/path",
        ready_at=dt_ready,
        expires_at=dt_expires,
        storage_backend=db_backend,
        state=db_schema.TierPathState.READY,
    )
    db_asset = db_schema.Asset(
        asset_uuid="test-uuid",
        path="test/path",
        user="test-user",
        state=db_schema.AssetState.ASSET_STATE_ACTIVE_WRITE,
        tier_paths=[db_tp],
    )
    proto_asset = assets.proto_from_db_asset(db_asset)

    self.assertLen(proto_asset.tier_paths, 1)
    tp_proto = proto_asset.tier_paths[0]
    self.assertEqual(tp_proto.id, 10)
    self.assertEqual(tp_proto.path, "/mnt/lustre/test/path")
    self.assertEqual(tp_proto.ready_at, expected_ts_ready)
    self.assertEqual(tp_proto.expires_at, expected_ts_expires)
    self.assertEqual(
        tp_proto.state,
        tiering_service_pb2.TierPathState.TIER_PATH_STATE_READY,
    )

    sb_proto = tp_proto.storage_backend
    self.assertEqual(sb_proto.id, 1)
    self.assertEqual(sb_proto.level, 0)
    self.assertEqual(
        sb_proto.backend_type, tiering_service_pb2.BACKEND_TYPE_LUSTRE
    )
    self.assertEqual(sb_proto.prefix, "/mnt/lustre")
    self.assertEqual(sb_proto.zone, "us-central1-a")

  def test_proto_from_db_asset_backend_with_region(self):
    db_backend = db_schema.StorageBackend(
        id=2,
        level=1,
        backend_type=db_schema.BackendType.BACKEND_TYPE_GCS,
        prefix="gs://my-bucket",
        region="us-central1",
    )
    db_tp = db_schema.TierPath(
        id=11,
        path="gs://my-bucket/test/path",
        storage_backend=db_backend,
    )
    db_asset = db_schema.Asset(
        asset_uuid="test-uuid",
        path="test/path",
        user="test-user",
        state=db_schema.AssetState.ASSET_STATE_ACTIVE_WRITE,
        tier_paths=[db_tp],
    )
    proto_asset = assets.proto_from_db_asset(db_asset)

    self.assertLen(proto_asset.tier_paths, 1)
    sb_proto = proto_asset.tier_paths[0].storage_backend
    self.assertEqual(sb_proto.id, 2)
    self.assertEqual(sb_proto.region, "us-central1")
    self.assertFalse(sb_proto.HasField("zone"))
    self.assertFalse(sb_proto.HasField("multi_regions"))

  def test_proto_from_db_asset_backend_with_multi_regions(self):
    db_backend = db_schema.StorageBackend(
        id=3,
        level=1,
        backend_type=db_schema.BackendType.BACKEND_TYPE_GCS,
        prefix="gs://my-multi-bucket",
        multi_regions=["us", "eu"],
    )
    db_tp = db_schema.TierPath(
        id=12,
        path="gs://my-multi-bucket/test/path",
        storage_backend=db_backend,
    )
    db_asset = db_schema.Asset(
        asset_uuid="test-uuid",
        path="test/path",
        user="test-user",
        state=db_schema.AssetState.ASSET_STATE_ACTIVE_WRITE,
        tier_paths=[db_tp],
    )
    proto_asset = assets.proto_from_db_asset(db_asset)

    self.assertLen(proto_asset.tier_paths, 1)
    sb_proto = proto_asset.tier_paths[0].storage_backend
    self.assertEqual(sb_proto.id, 3)
    self.assertEqual(list(sb_proto.multi_regions.regions), ["us", "eu"])
    self.assertFalse(sb_proto.HasField("zone"))
    self.assertFalse(sb_proto.HasField("region"))


class AssetsDbTest(parameterized.TestCase, unittest.IsolatedAsyncioTestCase):

  def _assert_date_time_equal(self, dt1, dt2):
    if dt1 is None or dt2 is None:
      self.assertEqual(dt1, dt2)
      return
    if dt1.tzinfo is None and dt2.tzinfo is not None:
      dt2 = dt2.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    elif dt1.tzinfo is not None and dt2.tzinfo is None:
      dt1 = dt1.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    self.assertEqual(dt1, dt2)

  async def asyncSetUp(self) -> None:
    await super().asyncSetUp()
    tmp_file = self.create_tempfile()
    self.engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_file.full_path}"
    )
    async with self.engine.begin() as conn:
      await conn.run_sync(db_schema.Base.metadata.create_all)
    self.session_maker = async_sessionmaker(self.engine, expire_on_commit=False)

  async def asyncTearDown(self) -> None:
    await self.engine.dispose()
    await super().asyncTearDown()

  async def test_create_or_fetch_asset_and_queries(self):
    async with self.session_maker() as session:
      backend = db_schema.StorageBackend(
          level=0,
          backend_type=db_schema.BackendType.BACKEND_TYPE_LUSTRE,
          prefix="/mnt/lustre",
          zone="us-central1-a",
      )
      session.add(backend)
      await session.commit()

      request = tiering_service_pb2.ReserveRequest(
          path="test/path",
          user="test-user",
          zone="us-central1-a",
          tags=["tag1"],
      )
      config = tiering_service_pb2.ServerConfig(
          client_keep_alive_interval_seconds=600
      )

      tp_uuid = "test-tp-uuid"
      storage_path = "test/storage/path"
      # Create asset.
      asset = await assets.create_or_fetch_asset(
          session,
          request,
          backend,
          config,
          tier_path_uuid=tp_uuid,
          storage_path=storage_path,
      )
      self.assertEqual(asset.path, "test/path")
      self.assertLen(asset.tier_paths, 1)

      # Try creating it again (triggers unique conflict fetch fallback).
      asset2 = await assets.create_or_fetch_asset(
          session,
          request,
          backend,
          config,
          tier_path_uuid=tp_uuid,
          storage_path=storage_path,
      )
      self.assertEqual(asset2.asset_uuid, asset.asset_uuid)

      # Query active asset by path.
      fetched_assets = await assets.fetch_asset_by_path(
          session,
          "test/path",
          inclusive_filter=[
              db_schema.AssetState.ASSET_STATE_ACTIVE_WRITE,
              db_schema.AssetState.ASSET_STATE_STORED,
          ],
      )
      self.assertLen(fetched_assets, 1)
      self.assertEqual(fetched_assets[0].asset_uuid, asset.asset_uuid)

      # Query asset by uuid.
      fetched_uuids = await assets.fetch_asset_by_uuid(
          session, asset.asset_uuid
      )
      self.assertLen(fetched_uuids, 1)
      self.assertEqual(fetched_uuids[0].asset_uuid, asset.asset_uuid)

      # Query asset by identifier (uuid or path).
      fetched_ids = await assets.fetch_asset_by_identifier(
          session, asset.asset_uuid, ""
      )
      self.assertLen(fetched_ids, 1)
      self.assertEqual(fetched_ids[0].asset_uuid, asset.asset_uuid)

  async def test_mutations_keep_alive_finalize(self):
    async with self.session_maker() as session:
      backend = db_schema.StorageBackend(
          level=0,
          backend_type=db_schema.BackendType.BACKEND_TYPE_LUSTRE,
          prefix="/mnt/lustre",
          zone="us-central1-a",
      )
      session.add(backend)
      await session.commit()

      request = tiering_service_pb2.ReserveRequest(
          path="mut/path",
          user="test-user",
          zone="us-central1-a",
      )
      config = tiering_service_pb2.ServerConfig(
          client_keep_alive_interval_seconds=600
      )
      tp_uuid = "test-tp-uuid"
      storage_path = storage_backend_lib.get_storage_path(
          backend, request.path, tp_uuid
      )
      asset = await assets.create_or_fetch_asset(
          session,
          request,
          backend,
          config,
          tier_path_uuid=tp_uuid,
          storage_path=storage_path,
      )

      # Keep alive.
      initial_expires_at = asset.write_expires_at
      updated = await assets.reserve_keep_alive(
          session, asset.asset_uuid, datetime.timedelta(seconds=1200)
      )
      self.assertIsNotNone(updated)
      assert (
          updated is not None
          and updated.write_expires_at is not None
          and initial_expires_at is not None
      )
      self.assertGreater(updated.write_expires_at, initial_expires_at)

      # Verify keep alive persistence.
      async with self.session_maker() as session2:
        fetched = await assets.fetch_asset_by_uuid(session2, asset.asset_uuid)
        self.assertLen(fetched, 1)
        self._assert_date_time_equal(
            fetched[0].write_expires_at, updated.write_expires_at
        )

      # Finalize.
      finalized = await assets.finalize_asset(session, asset)
      self.assertIsNotNone(finalized)
      self.assertEqual(finalized.state, db_schema.AssetState.ASSET_STATE_STORED)
      self.assertLen(finalized.tier_paths, 1)
      self.assertEqual(finalized.tier_paths[0].ready_at, finalized.finalized_at)
      self.assertEqual(
          finalized.tier_paths[0].state, db_schema.TierPathState.READY
      )

      # Verify finalize persistence.
      async with self.session_maker() as session3:
        fetched3 = await assets.fetch_asset_by_uuid(session3, asset.asset_uuid)
        self.assertLen(fetched3, 1)
        self.assertEqual(
            fetched3[0].state, db_schema.AssetState.ASSET_STATE_STORED
        )
        self._assert_date_time_equal(
            fetched3[0].finalized_at, finalized.finalized_at
        )
        self.assertLen(fetched3[0].tier_paths, 1)
        self.assertEqual(
            fetched3[0].tier_paths[0].state, db_schema.TierPathState.READY
        )

  async def test_trigger_l0_to_l1_copy_success(self):
    async with self.session_maker() as session:
      # Setup L0 backend (Lustre)
      b0 = db_schema.StorageBackend(
          level=0,
          backend_type=db_schema.BackendType.BACKEND_TYPE_LUSTRE,
          prefix="/mnt/lustre",
          zone="us-central1-a",
      )
      # Setup L1 backend (GCS)
      b1 = db_schema.StorageBackend(
          level=1,
          backend_type=db_schema.BackendType.BACKEND_TYPE_GCS,
          prefix="gs://my-bucket",
          region="us-central1",
      )
      session.add_all([b0, b1])
      await session.commit()

      request = tiering_service_pb2.ReserveRequest(
          path="test/path/copy_test",
          user="test-user",
          zone="us-central1-a",
      )
      tp_uuid = "test-tp-uuid-b0"
      storage_path = storage_backend_lib.get_storage_path(
          b0, request.path, tp_uuid
      )
      asset = await assets.create_or_fetch_asset(
          session,
          request,
          b0,
          tiering_service_pb2.ServerConfig(
              client_keep_alive_interval_seconds=600
          ),
          tier_path_uuid=tp_uuid,
          storage_path=storage_path,
      )
      # Finalize the asset first
      finalized = await assets.finalize_asset(session, asset)

      # Now trigger copy
      await assets.trigger_l0_to_l1_copy(session, finalized)

      # Verify L1 tier path and copy job were created
      async with self.session_maker() as session2:
        fetched = await assets.fetch_asset_by_uuid(session2, asset.asset_uuid)
        self.assertLen(fetched, 1)
        db_asset = fetched[0]
        # Should have 2 tier paths now: b0 (Lustre) and b1 (GCS)
        self.assertLen(db_asset.tier_paths, 2)

        l1_tp = next(
            tp for tp in db_asset.tier_paths if tp.storage_backend_id == b1.id
        )
        self.assertIsNotNone(l1_tp)
        self.assertEqual(l1_tp.state, db_schema.TierPathState.PENDING)
        self.assertTrue(
            l1_tp.path.startswith("gs://my-bucket/test/path/copy_test/")
        )

        # Verify copy job
        stmt = select(db_schema.AssetJob).filter_by(
            asset_uuid=asset.asset_uuid,
            request_type=db_schema.RequestType.REQUEST_TYPE_COPY,
            status=db_schema.JobStatus.JOB_STATUS_QUEUED,
            target_tier_path_id=l1_tp.id,
        )
        result = await session2.execute(stmt)
        job = result.scalars().first()
        self.assertIsNotNone(job)

  async def test_trigger_l0_to_l1_copy_multiple_l1_backends_raises_error(self):
    async with self.session_maker() as session:
      b0 = db_schema.StorageBackend(
          level=0,
          backend_type=db_schema.BackendType.BACKEND_TYPE_LUSTRE,
          prefix="/mnt/lustre",
          zone="us-central1-a",
      )
      b1_a = db_schema.StorageBackend(
          level=1,
          backend_type=db_schema.BackendType.BACKEND_TYPE_GCS,
          prefix="gs://bucket-a",
          region="us-west1",
      )
      b1_b = db_schema.StorageBackend(
          level=1,
          backend_type=db_schema.BackendType.BACKEND_TYPE_GCS,
          prefix="gs://bucket-b",
          region="us-east1",
      )
      session.add_all([b0, b1_a, b1_b])
      await session.commit()

      request = tiering_service_pb2.ReserveRequest(
          path="test/path/copy_test_fail",
          user="test-user",
          zone="us-central1-a",
      )
      tp_uuid = "test-tp-uuid-b0-fail"
      storage_path = storage_backend_lib.get_storage_path(
          b0, request.path, tp_uuid
      )
      asset = await assets.create_or_fetch_asset(
          session,
          request,
          b0,
          tiering_service_pb2.ServerConfig(
              client_keep_alive_interval_seconds=600
          ),
          tier_path_uuid=tp_uuid,
          storage_path=storage_path,
      )
      finalized = await assets.finalize_asset(session, asset)

      with self.assertRaisesRegex(
          ValueError, "No matching Level 1 backend found"
      ):
        await assets.trigger_l0_to_l1_copy(session, finalized)

  async def test_queries_filtering(self):
    async with self.session_maker() as session:
      backend = db_schema.StorageBackend(
          level=0,
          backend_type=db_schema.BackendType.BACKEND_TYPE_LUSTRE,
          prefix="/mnt/lustre",
          zone="us-central1-a",
      )
      session.add(backend)
      await session.commit()

      config = tiering_service_pb2.ServerConfig(
          client_keep_alive_interval_seconds=600
      )

      # Create Asset A.
      request_a = tiering_service_pb2.ReserveRequest(
          path="path/A",
          user="user-a",
          zone="us-central1-a",
      )
      tp_uuid_a = "test-tp-uuid-a"
      storage_path_a = storage_backend_lib.get_storage_path(
          backend, request_a.path, tp_uuid_a
      )
      asset_a = await assets.create_or_fetch_asset(
          session,
          request_a,
          backend,
          config,
          tier_path_uuid=tp_uuid_a,
          storage_path=storage_path_a,
      )

      # Create Asset B.
      request_b = tiering_service_pb2.ReserveRequest(
          path="path/B",
          user="user-b",
          zone="us-central1-a",
      )
      tp_uuid_b = "test-tp-uuid-b"
      storage_path_b = storage_backend_lib.get_storage_path(
          backend, request_b.path, tp_uuid_b
      )
      asset_b = await assets.create_or_fetch_asset(
          session,
          request_b,
          backend,
          config,
          tier_path_uuid=tp_uuid_b,
          storage_path=storage_path_b,
      )

      # Verify fetch_asset_by_path only returns the matched asset.
      fetched_a_by_path = await assets.fetch_asset_by_path(session, "path/A")
      self.assertLen(fetched_a_by_path, 1)
      self.assertEqual(fetched_a_by_path[0].asset_uuid, asset_a.asset_uuid)

      fetched_b_by_path = await assets.fetch_asset_by_path(session, "path/B")
      self.assertLen(fetched_b_by_path, 1)
      self.assertEqual(fetched_b_by_path[0].asset_uuid, asset_b.asset_uuid)

      # Verify fetch_asset_by_uuid only returns the matched asset.
      fetched_a_by_uuid = await assets.fetch_asset_by_uuid(
          session, asset_a.asset_uuid
      )
      self.assertLen(fetched_a_by_uuid, 1)
      self.assertEqual(fetched_a_by_uuid[0].path, "path/A")

      fetched_b_by_uuid = await assets.fetch_asset_by_uuid(
          session, asset_b.asset_uuid
      )
      self.assertLen(fetched_b_by_uuid, 1)
      self.assertEqual(fetched_b_by_uuid[0].path, "path/B")

  async def _set_a_finalized_asset(
      self, session: AsyncSession
  ) -> tuple[
      db_schema.Asset, db_schema.StorageBackend, db_schema.StorageBackend
  ]:
    """Sets up a finalized asset in the database.

    Creates two storage backends and one asset. The asset is initially reserved
    against one backend and then immediately finalized.

    Args:
      session: The SQLAlchemy AsyncSession to use for database operations.

    Returns:
      A tuple (asset, b1, b2), where asset is the finalized db_schema.Asset,
      b1 is the first db_schema.StorageBackend used for the initial reservation,
      and b2 is the second db_schema.StorageBackend.
    """
    b1 = db_schema.StorageBackend(
        level=0,
        backend_type=db_schema.BackendType.BACKEND_TYPE_LUSTRE,
        prefix="/mnt/lustre-a",
        zone="us-central1-a",
    )
    b2 = db_schema.StorageBackend(
        level=0,
        backend_type=db_schema.BackendType.BACKEND_TYPE_LUSTRE,
        prefix="/mnt/lustre-b",
        zone="us-central1-b",
    )
    session.add_all([b1, b2])  # pyrefly: ignore[missing-attribute]
    await session.commit()

    request = tiering_service_pb2.ReserveRequest(
        path="test/path/finalized_asset",
        user="test-user",
        zone="us-central1-a",
    )
    tp_uuid = "test-tp-uuid-b1"
    storage_path = storage_backend_lib.get_storage_path(
        b1, request.path, tp_uuid
    )
    reserved_asset = await assets.create_or_fetch_asset(
        session,
        request,
        b1,
        tiering_service_pb2.ServerConfig(
            client_keep_alive_interval_seconds=600
        ),
        tier_path_uuid=tp_uuid,
        storage_path=storage_path,
    )
    finalized_asset = await assets.finalize_asset(session, reserved_asset)
    return finalized_asset, b1, b2

  async def test_create_prefetch_job_returns_created_and_updated_asset(self):
    async with self.session_maker() as session:
      asset, _, b2 = await self._set_a_finalized_asset(session)
      tp_uuid = "test-prefetch-tp-uuid-b2"
      storage_path = storage_backend_lib.get_storage_path(
          b2, asset.path, tp_uuid
      )
      result = await assets.create_prefetch_job(
          session,
          asset,
          backend=b2,
          storage_path=storage_path,
          tier_path_uuid=tp_uuid,
          client_keep_alive_interval=datetime.timedelta(seconds=600),
      )
      self.assertTrue(result.created)
      self.assertIsNotNone(result.asset)

  async def test_create_prefetch_job_updates_tier_paths(self):
    async with self.session_maker() as session:
      asset, b1, b2 = await self._set_a_finalized_asset(session)
      tp_uuid = "test-prefetch-tp-uuid-b2"
      storage_path = storage_backend_lib.get_storage_path(
          b2, asset.path, tp_uuid
      )
      result = await assets.create_prefetch_job(
          session,
          asset,
          backend=b2,
          storage_path=storage_path,
          tier_path_uuid=tp_uuid,
          client_keep_alive_interval=datetime.timedelta(seconds=600),
      )
      updated_asset = result.asset
      self.assertIsNotNone(updated_asset)
      paths = [tp.path for tp in updated_asset.tier_paths]
      self.assertCountEqual(
          paths,
          [
              storage_backend_lib.get_storage_path(
                  b1, asset.path, "test-tp-uuid-b1"
              ),
              storage_path,
          ],
      )

  async def test_create_prefetch_job_db_tier_path_not_ready(self):
    async with self.session_maker() as session:
      asset, _, b2 = await self._set_a_finalized_asset(session)
      tp_uuid = "test-prefetch-tp-uuid-b2"
      storage_path = storage_backend_lib.get_storage_path(
          b2, asset.path, tp_uuid
      )
      await assets.create_prefetch_job(
          session,
          asset,
          backend=b2,
          storage_path=storage_path,
          tier_path_uuid=tp_uuid,
          client_keep_alive_interval=datetime.timedelta(seconds=600),
      )

      stmt_tp = select(db_schema.TierPath).filter_by(
          asset_uuid=asset.asset_uuid, storage_backend_id=b2.id
      )
      result_tp = await session.execute(stmt_tp)
      tp_b = result_tp.scalars().first()
      self.assertIsNotNone(tp_b)
      self.assertEqual(tp_b.path, storage_path)
      self.assertIsNone(tp_b.ready_at)

  async def test_create_prefetch_job_db_queues_copy_job(self):
    async with self.session_maker() as session:
      asset, _, b2 = await self._set_a_finalized_asset(session)
      tp_uuid = "test-prefetch-tp-uuid-b2"
      storage_path = storage_backend_lib.get_storage_path(
          b2, asset.path, tp_uuid
      )
      await assets.create_prefetch_job(
          session,
          asset,
          backend=b2,
          storage_path=storage_path,
          tier_path_uuid=tp_uuid,
          client_keep_alive_interval=datetime.timedelta(seconds=600),
      )

      stmt_tp = select(db_schema.TierPath).filter_by(
          asset_uuid=asset.asset_uuid, storage_backend_id=b2.id
      )
      result_tp = await session.execute(stmt_tp)
      tp_b = result_tp.scalars().first()
      self.assertIsNotNone(tp_b)

      stmt_job = select(db_schema.AssetJob).filter_by(
          asset_uuid=asset.asset_uuid,
          target_tier_path_id=tp_b.id,
          request_type=db_schema.RequestType.REQUEST_TYPE_COPY,
      )
      result_job = await session.execute(stmt_job)
      job = result_job.scalars().first()
      self.assertIsNotNone(job)
      self.assertEqual(job.status, db_schema.JobStatus.JOB_STATUS_QUEUED)

  @parameterized.named_parameters(
      dict(
          testcase_name="same_path",
          concurrent_path="/mnt/lustre-b/test/path",
          attempted_path="/mnt/lustre-b/test/path",
      ),
      dict(
          testcase_name="different_path",
          concurrent_path="/mnt/lustre-b/concurrent/path",
          attempted_path="/mnt/lustre-b/test/path",
      ),
  )
  async def test_create_prefetch_job_concurrent_fails_gracefully(
      self, concurrent_path: str, attempted_path: str
  ):
    async with self.session_maker() as session1:
      asset1, _, sb2 = await self._set_a_finalized_asset(session1)
      asset_uuid = asset1.asset_uuid
      b2_id = sb2.id

      async with self.session_maker() as session2:
        tp_b = db_schema.TierPath(
            asset_uuid=asset_uuid,
            storage_backend_id=b2_id,
            path=concurrent_path,
        )
        session2.add(tp_b)
        await session2.flush()

        job_b = db_schema.AssetJob(
            asset_uuid=asset_uuid,
            request_type=db_schema.RequestType.REQUEST_TYPE_COPY,
            status=db_schema.JobStatus.JOB_STATUS_QUEUED,
            target_tier_path_id=tp_b.id,
        )
        session2.add(job_b)
        await session2.commit()

      result = await assets.create_prefetch_job(
          session1,
          asset1,
          backend=sb2,
          storage_path=attempted_path,
          tier_path_uuid="test-prefetch-attempt-uuid",
          client_keep_alive_interval=datetime.timedelta(seconds=600),
      )

      self.assertFalse(result.created)

      # Verify DB has only the concurrent TierPath (no new one created).
      stmt_tp = select(db_schema.TierPath).filter_by(
          asset_uuid=asset_uuid, storage_backend_id=b2_id
      )
      result_tp = await session1.execute(stmt_tp)
      tps = result_tp.scalars().all()
      self.assertLen(tps, 1)
      self.assertEqual(tps[0].path, concurrent_path)

  async def test_create_prefetch_job_sets_expires_at_and_uuid(self):
    async with self.session_maker() as session:
      asset, _, b2 = await self._set_a_finalized_asset(session)
      tp_uuid = "test-prefetch-tp-uuid-b2"
      storage_path = storage_backend_lib.get_storage_path(
          b2, asset.path, tp_uuid
      )
      result = await assets.create_prefetch_job(
          session,
          asset,
          backend=b2,
          storage_path=storage_path,
          tier_path_uuid=tp_uuid,
          client_keep_alive_interval=datetime.timedelta(seconds=600),
      )
      updated_asset = result.asset
      self.assertIsNotNone(updated_asset)

      tp_b = next(
          tp
          for tp in updated_asset.tier_paths
          if tp.storage_backend_id == b2.id
      )
      self.assertEqual(tp_b.tier_path_uuid, tp_uuid)
      self.assertIsNotNone(tp_b.expires_at)

  async def test_prefetch_keep_alive_success(self):
    async with self.session_maker() as session:
      asset, _, b2 = await self._set_a_finalized_asset(session)
      tp_uuid = "test-prefetch-tp-uuid-b2"
      storage_path = storage_backend_lib.get_storage_path(
          b2, asset.path, tp_uuid
      )
      result = await assets.create_prefetch_job(
          session,
          asset,
          backend=b2,
          storage_path=storage_path,
          tier_path_uuid=tp_uuid,
          client_keep_alive_interval=datetime.timedelta(seconds=600),
      )
      updated_asset = result.asset
      self.assertIsNotNone(updated_asset)
      tp_b = next(
          tp
          for tp in updated_asset.tier_paths
          if tp.storage_backend_id == b2.id
      )
      initial_expires_at = tp_b.expires_at

      extended_asset = await assets.prefetch_keep_alive(
          session,
          tier_path_uuid=tp_b.tier_path_uuid,
          interval=datetime.timedelta(seconds=1200),
      )
      self.assertIsNotNone(extended_asset)
      tp_b_extended = next(
          tp
          for tp in extended_asset.tier_paths
          if tp.storage_backend_id == b2.id
      )
      self.assertGreater(tp_b_extended.expires_at, initial_expires_at)

  async def test_prefetch_keep_alive_not_found(self):
    async with self.session_maker() as session:
      result = await assets.prefetch_keep_alive(
          session,
          tier_path_uuid="non-existent-uuid",
          interval=datetime.timedelta(seconds=1200),
      )
      self.assertIsNone(result)

  async def test_prefetch_keep_alive_no_op_when_permanent(self):
    async with self.session_maker() as session:
      asset, _, b2 = await self._set_a_finalized_asset(session)
      tp_uuid = "test-prefetch-tp-uuid-b2"
      storage_path = storage_backend_lib.get_storage_path(
          b2, asset.path, tp_uuid
      )
      result = await assets.create_prefetch_job(
          session,
          asset,
          backend=b2,
          storage_path=storage_path,
          tier_path_uuid=tp_uuid,
          client_keep_alive_interval=datetime.timedelta(seconds=600),
      )
      updated_asset = result.asset
      self.assertIsNotNone(updated_asset)
      tp_b = next(
          tp
          for tp in updated_asset.tier_paths
          if tp.storage_backend_id == b2.id
      )
      tp_b.expires_at = None
      await session.commit()

      extended_asset = await assets.prefetch_keep_alive(
          session,
          tier_path_uuid=tp_b.tier_path_uuid,
          interval=datetime.timedelta(seconds=1200),
      )
      self.assertIsNotNone(extended_asset)
      tp_b_extended = next(
          tp
          for tp in extended_asset.tier_paths
          if tp.storage_backend_id == b2.id
      )
      self.assertIsNone(tp_b_extended.expires_at)

  async def test_prefetch_keep_alive_only_extends(self):
    async with self.session_maker() as session:
      asset, _, b2 = await self._set_a_finalized_asset(session)
      tp_uuid = "test-prefetch-tp-uuid-b2"
      storage_path = storage_backend_lib.get_storage_path(
          b2, asset.path, tp_uuid
      )
      result = await assets.create_prefetch_job(
          session,
          asset,
          backend=b2,
          storage_path=storage_path,
          tier_path_uuid=tp_uuid,
          client_keep_alive_interval=datetime.timedelta(seconds=600),
      )
      updated_asset = result.asset
      self.assertIsNotNone(updated_asset)
      tp_b = next(
          tp
          for tp in updated_asset.tier_paths
          if tp.storage_backend_id == b2.id
      )
      initial_expires_at = tp_b.expires_at

      extended_asset = await assets.prefetch_keep_alive(
          session,
          tier_path_uuid=tp_b.tier_path_uuid,
          interval=datetime.timedelta(seconds=10),
      )
      self.assertIsNotNone(extended_asset)
      tp_b_extended = next(
          tp
          for tp in extended_asset.tier_paths
          if tp.storage_backend_id == b2.id
      )
      self.assertEqual(tp_b_extended.expires_at, initial_expires_at)

  async def test_prefetch_keep_alive_fails_if_deleting(self):
    async with self.session_maker() as session:
      asset, _, b2 = await self._set_a_finalized_asset(session)
      tp_uuid = "test-prefetch-tp-uuid-b2"
      storage_path = storage_backend_lib.get_storage_path(
          b2, asset.path, tp_uuid
      )
      result = await assets.create_prefetch_job(
          session,
          asset,
          backend=b2,
          storage_path=storage_path,
          tier_path_uuid=tp_uuid,
          client_keep_alive_interval=datetime.timedelta(seconds=600),
      )
      updated_asset = result.asset
      self.assertIsNotNone(updated_asset)
      tp_b = next(
          tp
          for tp in updated_asset.tier_paths
          if tp.storage_backend_id == b2.id
      )

      await assets.queue_delete_asset_job(session, asset)

      with self.assertRaisesRegex(
          assets.DeletionPendingError, "marked for deletion"
      ):
        await assets.prefetch_keep_alive(
            session,
            tier_path_uuid=tp_b.tier_path_uuid,
            interval=datetime.timedelta(seconds=1200),
        )

  async def test_prefetch_keep_alive_fails_if_instance_deleting(self):
    async with self.session_maker() as session:
      asset, _, b2 = await self._set_a_finalized_asset(session)
      tp_uuid = "test-prefetch-tp-uuid-b2"
      storage_path = storage_backend_lib.get_storage_path(
          b2, asset.path, tp_uuid
      )
      result = await assets.create_prefetch_job(
          session,
          asset,
          backend=b2,
          storage_path=storage_path,
          tier_path_uuid=tp_uuid,
          client_keep_alive_interval=datetime.timedelta(seconds=600),
      )
      updated_asset = result.asset
      self.assertIsNotNone(updated_asset)
      tp_b = next(
          tp
          for tp in updated_asset.tier_paths
          if tp.storage_backend_id == b2.id
      )

      db_job = db_schema.AssetJob(
          asset_uuid=asset.asset_uuid,
          request_type=db_schema.RequestType.REQUEST_TYPE_DELETE_FROM_INSTANCE,
          status=db_schema.JobStatus.JOB_STATUS_QUEUED,
          target_tier_path_id=tp_b.id,
      )
      session.add(db_job)
      await session.commit()

      with self.assertRaisesRegex(
          assets.DeletionPendingError, "marked for deletion"
      ):
        await assets.prefetch_keep_alive(
            session,
            tier_path_uuid=tp_b.tier_path_uuid,
            interval=datetime.timedelta(seconds=1200),
        )

  async def test_create_prefetch_job_fails_if_deleting(self):
    async with self.session_maker() as session:
      asset, _, b2 = await self._set_a_finalized_asset(session)
      tp_uuid = "test-prefetch-tp-uuid-b2"
      storage_path = storage_backend_lib.get_storage_path(
          b2, asset.path, tp_uuid
      )

      await assets.queue_delete_asset_job(session, asset)

      with self.assertRaisesRegex(
          assets.DeletionPendingError, "marked for deletion"
      ):
        await assets.create_prefetch_job(
            session,
            asset,
            backend=b2,
            storage_path=storage_path,
            tier_path_uuid=tp_uuid,
            client_keep_alive_interval=datetime.timedelta(seconds=600),
        )

  async def test_is_delete_pending_true(self):
    async with self.session_maker() as session:
      asset, _, _ = await self._set_a_finalized_asset(session)

      # Insert a pending delete job.
      db_job = db_schema.AssetJob(
          asset_uuid=asset.asset_uuid,
          request_type=db_schema.RequestType.REQUEST_TYPE_DELETE_FROM_ALL_TIERS,
          status=db_schema.JobStatus.JOB_STATUS_QUEUED,
      )
      session.add(db_job)
      await session.commit()

      self.assertTrue(
          await assets.is_delete_pending(session, asset_uuid=asset.asset_uuid)
      )

  async def test_is_delete_pending_false(self):
    async with self.session_maker() as session:
      asset, _, _ = await self._set_a_finalized_asset(session)
      # No job inserted.
      self.assertFalse(
          await assets.is_delete_pending(session, asset_uuid=asset.asset_uuid)
      )

  async def test_is_tier_path_delete_pending_true(self):
    async with self.session_maker() as session:
      asset, _, b2 = await self._set_a_finalized_asset(session)
      tp_uuid = "test-prefetch-tp-uuid-b2"
      storage_path = storage_backend_lib.get_storage_path(
          b2, asset.path, tp_uuid
      )
      result = await assets.create_prefetch_job(
          session,
          asset,
          backend=b2,
          storage_path=storage_path,
          tier_path_uuid=tp_uuid,
          client_keep_alive_interval=datetime.timedelta(seconds=600),
      )
      updated_asset = result.asset
      self.assertIsNotNone(updated_asset)
      assert updated_asset is not None
      tp_b = next(
          tp
          for tp in updated_asset.tier_paths
          if tp.storage_backend_id == b2.id
      )

      # Insert a pending delete from instance job.
      db_job = db_schema.AssetJob(
          asset_uuid=asset.asset_uuid,
          request_type=db_schema.RequestType.REQUEST_TYPE_DELETE_FROM_INSTANCE,
          status=db_schema.JobStatus.JOB_STATUS_QUEUED,
          target_tier_path_id=tp_b.id,
      )
      session.add(db_job)
      await session.commit()

      self.assertTrue(
          await assets.is_tier_path_delete_pending(
              session, asset_uuid=asset.asset_uuid, tier_path_id=tp_b.id
          )
      )

  async def test_is_tier_path_delete_pending_false(self):
    async with self.session_maker() as session:
      asset, _, b2 = await self._set_a_finalized_asset(session)
      tp_uuid = "test-prefetch-tp-uuid-b2"
      storage_path = storage_backend_lib.get_storage_path(
          b2, asset.path, tp_uuid
      )
      result = await assets.create_prefetch_job(
          session,
          asset,
          backend=b2,
          storage_path=storage_path,
          tier_path_uuid=tp_uuid,
          client_keep_alive_interval=datetime.timedelta(seconds=600),
      )
      updated_asset = result.asset
      self.assertIsNotNone(updated_asset)
      assert updated_asset is not None
      tp_b = next(
          tp
          for tp in updated_asset.tier_paths
          if tp.storage_backend_id == b2.id
      )
      # No job inserted.
      self.assertFalse(
          await assets.is_tier_path_delete_pending(
              session, asset_uuid=asset.asset_uuid, tier_path_id=tp_b.id
          )
      )

  async def test_queue_delete_asset_job_success(self):
    async with self.session_maker() as session:
      asset, _, _ = await self._set_a_finalized_asset(session)

      await assets.queue_delete_asset_job(session, asset)

      stmt = select(db_schema.AssetJob).filter_by(
          asset_uuid=asset.asset_uuid,
          request_type=db_schema.RequestType.REQUEST_TYPE_DELETE_FROM_ALL_TIERS,
      )
      result = await session.execute(stmt)
      job = result.scalars().first()
      self.assertIsNotNone(job)
      self.assertEqual(job.status, db_schema.JobStatus.JOB_STATUS_QUEUED)

  async def test_queue_delete_asset_job_duplicate(self):
    async with self.session_maker() as session:
      asset, _, _ = await self._set_a_finalized_asset(session)

      await assets.queue_delete_asset_job(session, asset)

      with self.assertRaisesRegex(ValueError, "has pending delete"):
        await assets.queue_delete_asset_job(session, asset)

      stmt = select(db_schema.AssetJob).filter_by(
          asset_uuid=asset.asset_uuid,
          request_type=db_schema.RequestType.REQUEST_TYPE_DELETE_FROM_ALL_TIERS,
      )
      result = await session.execute(stmt)
      jobs = result.scalars().all()
      self.assertLen(jobs, 1)

  async def test_queue_delete_asset_job_already_deleted(self):
    async with self.session_maker() as session:
      asset, _, _ = await self._set_a_finalized_asset(session)

      # Simulate another process deleting the asset in the DB.
      async with self.session_maker() as session2:
        db_assets = await assets.fetch_asset_by_uuid(session2, asset.asset_uuid)
        db_asset2 = db_assets[0]
        db_asset2.state = db_schema.AssetState.ASSET_STATE_DELETED
        await session2.commit()

      # Now try to queue delete job using the stale 'asset' object (which is
      # still STORED in memory).
      # But queue_delete_asset_job will re-fetch it with lock and see it is
      # DELETED.
      with self.assertRaisesRegex(ValueError, "is already deleted"):
        await assets.queue_delete_asset_job(session, asset)

      stmt = select(db_schema.AssetJob).filter_by(
          asset_uuid=asset.asset_uuid,
          request_type=db_schema.RequestType.REQUEST_TYPE_DELETE_FROM_ALL_TIERS,
      )
      res = await session.execute(stmt)
      jobs = res.scalars().all()
      self.assertEmpty(jobs)

  async def test_begin_delete_asset_lazy_load_success(self):
    async with self.session_maker() as session:
      asset, _, _ = await self._set_a_finalized_asset(session)
      asset_uuid = asset.asset_uuid

    # Now load asset in a new session without eagerly loading tier_paths.
    async with self.session_maker() as session:
      stmt = select(db_schema.Asset).where(
          db_schema.Asset.asset_uuid == asset_uuid
      )
      res = await session.execute(stmt)
      db_asset = res.scalars().first()
      self.assertIsNotNone(db_asset)

      await assets.begin_delete_asset(session, db_asset)
      await session.commit()

    async with self.session_maker() as session:
      db_assets = await assets.fetch_asset_by_uuid(session, asset_uuid)
      db_asset = db_assets[0]
      self.assertEqual(db_asset.state, db_schema.AssetState.ASSET_STATE_DELETED)
      self.assertNotEmpty(db_asset.tier_paths)
      for tp in db_asset.tier_paths:
        self.assertEqual(tp.state, db_schema.TierPathState.DELETE_IN_PROCESS)

  async def test_finalize_asset_lazy_load_success(self):
    async with self.session_maker() as session:
      b1 = db_schema.StorageBackend(
          level=0,
          backend_type=db_schema.BackendType.BACKEND_TYPE_LUSTRE,
          prefix="/mnt/lustre-a",
          zone="us-central1-a",
      )
      session.add(b1)  # pyrefly: ignore[missing-attribute]
      await session.commit()

      request = tiering_service_pb2.ReserveRequest(
          path="test/path/unfinalized_asset",
          user="test-user",
          zone="us-central1-a",
      )
      tp_uuid = "test-tp-uuid-unfinalized"
      storage_path = storage_backend_lib.get_storage_path(
          b1, request.path, tp_uuid
      )
      reserved_asset = await assets.create_or_fetch_asset(
          session,
          request,
          b1,
          tiering_service_pb2.ServerConfig(
              client_keep_alive_interval_seconds=600
          ),
          tier_path_uuid=tp_uuid,
          storage_path=storage_path,
      )
      asset_uuid = reserved_asset.asset_uuid

    async with self.session_maker() as session:
      stmt = select(db_schema.Asset).where(
          db_schema.Asset.asset_uuid == asset_uuid
      )
      res = await session.execute(stmt)
      db_asset = res.scalars().first()
      self.assertIsNotNone(db_asset)

      await assets.finalize_asset(session, db_asset)
      await session.commit()

    async with self.session_maker() as session:
      db_assets = await assets.fetch_asset_by_uuid(session, asset_uuid)
      db_asset = db_assets[0]
      self.assertEqual(db_asset.state, db_schema.AssetState.ASSET_STATE_STORED)
      self.assertNotEmpty(db_asset.tier_paths)
      for tp in db_asset.tier_paths:
        self.assertIsNotNone(tp.ready_at)
        self.assertEqual(tp.state, db_schema.TierPathState.READY)

  async def test_create_prefetch_job_lazy_load_success(self):
    async with self.session_maker() as session:
      asset, _, b2 = await self._set_a_finalized_asset(session)
      asset_uuid = asset.asset_uuid
      b2_id = b2.id

    # Now load asset in a new session without eagerly loading tier_paths.
    async with self.session_maker() as session:
      stmt = select(db_schema.Asset).where(
          db_schema.Asset.asset_uuid == asset_uuid
      )
      res = await session.execute(stmt)
      db_asset = res.scalars().first()
      self.assertIsNotNone(db_asset)

      res_b2 = await session.execute(
          select(db_schema.StorageBackend).where(
              db_schema.StorageBackend.id == b2_id
          )
      )
      db_backend = res_b2.scalars().first()
      self.assertIsNotNone(db_backend)
      assert db_backend is not None

      tp_uuid = "test-prefetch-tp-uuid-lazy"
      storage_path = storage_backend_lib.get_storage_path(
          db_backend, db_asset.path, tp_uuid
      )

      result = await assets.create_prefetch_job(
          session,
          db_asset,
          backend=db_backend,
          storage_path=storage_path,
          tier_path_uuid=tp_uuid,
          client_keep_alive_interval=datetime.timedelta(seconds=600),
      )
      self.assertTrue(result.created)
      await session.commit()

    async with self.session_maker() as session:
      db_assets = await assets.fetch_asset_by_uuid(session, asset_uuid)
      db_asset = db_assets[0]
      # Verify the new tier path is in the DB.
      self.assertLen(
          db_asset.tier_paths, 2
      )  # L0 reserved (1), L0 prefetched (2)
      tp_prefetched = next(
          tp for tp in db_asset.tier_paths if tp.tier_path_uuid == tp_uuid
      )
      self.assertEqual(tp_prefetched.state, db_schema.TierPathState.PENDING)

  async def test_complete_delete_asset_lazy_load_success(self):
    async with self.session_maker() as session:
      asset, _, _ = await self._set_a_finalized_asset(session)
      asset_uuid = asset.asset_uuid

    # Now load asset in a new session without eagerly loading tier_paths.
    async with self.session_maker() as session:
      stmt = select(db_schema.Asset).where(
          db_schema.Asset.asset_uuid == asset_uuid
      )
      res = await session.execute(stmt)
      db_asset = res.scalars().first()
      self.assertIsNotNone(db_asset)

      await assets.complete_delete_asset(session, db_asset)
      await session.commit()

    async with self.session_maker() as session:
      db_assets = await assets.fetch_asset_by_uuid(session, asset_uuid)
      db_asset = db_assets[0]
      self.assertEqual(db_asset.state, db_schema.AssetState.ASSET_STATE_DELETED)
      self.assertNotEmpty(db_asset.tier_paths)
      for tp in db_asset.tier_paths:
        self.assertEqual(tp.state, db_schema.TierPathState.DELETED)

  async def test_finalize_asset_success_with_multiple_assets(self):
    async with self.session_maker() as session:
      b0 = db_schema.StorageBackend(
          level=0,
          backend_type=db_schema.BackendType.BACKEND_TYPE_LUSTRE,
          prefix="/mnt/lustre-a",
          zone="us-central1-a",
      )
      session.add(b0)
      await session.commit()

      # Create Asset 1
      request1 = tiering_service_pb2.ReserveRequest(
          path="test/path/asset1",
          user="test-user",
          zone="us-central1-a",
      )
      asset1 = await assets.create_or_fetch_asset(
          session,
          request1,
          b0,
          tiering_service_pb2.ServerConfig(
              client_keep_alive_interval_seconds=600
          ),
          tier_path_uuid="tp-uuid-1",
          storage_path="/mnt/lustre-a/asset1",
      )

      # Create Asset 2
      request2 = tiering_service_pb2.ReserveRequest(
          path="test/path/asset2",
          user="test-user",
          zone="us-central1-a",
      )
      asset2 = await assets.create_or_fetch_asset(
          session,
          request2,
          b0,
          tiering_service_pb2.ServerConfig(
              client_keep_alive_interval_seconds=600
          ),
          tier_path_uuid="tp-uuid-2",
          storage_path="/mnt/lustre-a/asset2",
      )

      # Finalize Asset 2
      finalized = await assets.finalize_asset(session, asset2)
      await session.commit()

      # Verify we finalized Asset 2 and NOT Asset 1
      self.assertEqual(finalized.asset_uuid, asset2.asset_uuid)

      async with self.session_maker() as session2:
        fetched1 = await assets.fetch_asset_by_uuid(session2, asset1.asset_uuid)
        self.assertEqual(
            fetched1[0].state,
            db_schema.AssetState.ASSET_STATE_ACTIVE_WRITE,
        )

        fetched2 = await assets.fetch_asset_by_uuid(session2, asset2.asset_uuid)
        self.assertEqual(
            fetched2[0].state, db_schema.AssetState.ASSET_STATE_STORED
        )

  async def test_create_prefetch_job_success_with_multiple_assets(self):
    """Verifies prefetch job creation isolates TierPaths to the correct asset."""
    async with self.session_maker() as session:
      # Setup L0 backend (Lustre) and L1 backend (GCS)
      b0 = db_schema.StorageBackend(
          level=0,
          backend_type=db_schema.BackendType.BACKEND_TYPE_LUSTRE,
          prefix="/mnt/lustre-a",
          zone="us-central1-a",
      )
      b1 = db_schema.StorageBackend(
          level=1,
          backend_type=db_schema.BackendType.BACKEND_TYPE_GCS,
          prefix="gs://my-bucket",
          region="us-central1",
      )
      session.add_all([b0, b1])
      await session.commit()

      # Create and finalize Asset 1
      request1 = tiering_service_pb2.ReserveRequest(
          path="test/path/asset1",
          user="test-user",
          zone="us-central1-a",
      )
      asset1 = await assets.create_or_fetch_asset(
          session,
          request1,
          b0,
          tiering_service_pb2.ServerConfig(
              client_keep_alive_interval_seconds=600
          ),
          tier_path_uuid="tp-uuid-1",
          storage_path="/mnt/lustre-a/asset1",
      )
      await assets.finalize_asset(session, asset1)

      # Create and finalize Asset 2 (the one we will prefetch)
      request2 = tiering_service_pb2.ReserveRequest(
          path="test/path/asset2",
          user="test-user",
          zone="us-central1-a",
      )
      asset2 = await assets.create_or_fetch_asset(
          session,
          request2,
          b0,
          tiering_service_pb2.ServerConfig(
              client_keep_alive_interval_seconds=600
          ),
          tier_path_uuid="tp-uuid-2",
          storage_path="/mnt/lustre-a/asset2",
      )
      await assets.finalize_asset(session, asset2)
      # Trigger L0-L1 copy
      await assets.trigger_l0_to_l1_copy(session, asset2)
      await session.commit()

    # Simulate copy completed and L0 eviction
    async with self.session_maker() as session:
      stmt = (
          select(db_schema.Asset)
          .where(db_schema.Asset.asset_uuid == asset2.asset_uuid)
          .options(joinedload(db_schema.Asset.tier_paths))
      )
      res = await session.execute(stmt)
      db_asset = res.scalars().first()
      self.assertIsNotNone(db_asset)
      assert db_asset is not None
      for tp in db_asset.tier_paths:
        if tp.storage_backend_id == b0.id:
          tp.state = db_schema.TierPathState.DELETED
        elif tp.storage_backend_id == b1.id:
          tp.state = db_schema.TierPathState.READY
          tp.ready_at = datetime.datetime.now(datetime.timezone.utc)
      await session.commit()

    # Now in a new session, prefetch Asset 2
    async with self.session_maker() as session2:
      # Fetch L0 backend again for prefetch target
      res_b0 = await session2.execute(
          select(db_schema.StorageBackend).where(
              db_schema.StorageBackend.id == b0.id
          )
      )
      db_b0 = res_b0.scalars().first()
      self.assertIsNotNone(db_b0)
      assert db_b0 is not None

      # Fetch Asset 2 again to pass to prefetch
      db_assets2 = await assets.fetch_asset_by_uuid(session2, asset2.asset_uuid)
      db_asset2 = db_assets2[0]

      tp_uuid = "prefetch-tp-uuid-2"
      storage_path = storage_backend_lib.get_storage_path(
          db_b0, db_asset2.path, tp_uuid
      )

      # Trigger prefetch on Asset 2
      result = await assets.create_prefetch_job(
          session2,
          db_asset2,
          backend=db_b0,
          storage_path=storage_path,
          tier_path_uuid=tp_uuid,
          client_keep_alive_interval=datetime.timedelta(seconds=600),
      )
      self.assertTrue(result.created)
      asset_res = result.asset
      assert asset_res is not None
      self.assertEqual(asset_res.asset_uuid, asset2.asset_uuid)
      await session2.commit()

      # Verify that Asset 2 got the new tier path, but Asset 1 did not
      async with self.session_maker() as session3:
        fetched1 = await assets.fetch_asset_by_uuid(session3, asset1.asset_uuid)
        self.assertNotIn(
            tp_uuid, [tp.tier_path_uuid for tp in fetched1[0].tier_paths]
        )

        fetched2 = await assets.fetch_asset_by_uuid(session3, asset2.asset_uuid)
        # Asset 2 should have the prefetch path
        self.assertIn(
            tp_uuid, [tp.tier_path_uuid for tp in fetched2[0].tier_paths]
        )

  async def test_begin_delete_tier_path(self):
    async with self.session_maker() as session:
      # Setup asset with a tier path
      b0 = db_schema.StorageBackend(
          level=0,
          backend_type=db_schema.BackendType.BACKEND_TYPE_LUSTRE,
          prefix="/mnt/lustre-a",
          zone="us-central1-a",
      )
      session.add(b0)
      await session.commit()
      request = tiering_service_pb2.ReserveRequest(
          path="test/path/delete_tp",
          user="test-user",
          zone="us-central1-a",
      )
      asset = await assets.create_or_fetch_asset(
          session,
          request,
          b0,
          tiering_service_pb2.ServerConfig(
              client_keep_alive_interval_seconds=600
          ),
          tier_path_uuid="tp-uuid-del",
          storage_path="/mnt/lustre-a/delete_tp",
      )
      tp = asset.tier_paths[0]
      # Set it to ready first
      tp.state = db_schema.TierPathState.READY
      tp.ready_at = datetime.datetime.now(datetime.timezone.utc)
      await session.commit()

      # Now begin delete
      updated_tp = await assets.begin_delete_tier_path(session, tp)
      await session.commit()

      self.assertEqual(
          updated_tp.state, db_schema.TierPathState.DELETE_IN_PROCESS
      )
      self.assertIsNone(updated_tp.ready_at)
      self.assertIsNone(updated_tp.expires_at)

      # Verify persistence
      async with self.session_maker() as session2:
        stmt = select(db_schema.TierPath).where(db_schema.TierPath.id == tp.id)
        res = await session2.execute(stmt)
        db_tp = res.scalars().first()
        self.assertIsNotNone(db_tp)
        assert db_tp is not None
        self.assertEqual(db_tp.state, db_schema.TierPathState.DELETE_IN_PROCESS)

  async def test_complete_delete_tier_path(self):
    async with self.session_maker() as session:
      b0 = db_schema.StorageBackend(
          level=0,
          backend_type=db_schema.BackendType.BACKEND_TYPE_LUSTRE,
          prefix="/mnt/lustre-a",
          zone="us-central1-a",
      )
      session.add(b0)
      await session.commit()
      request = tiering_service_pb2.ReserveRequest(
          path="test/path/complete_del_tp",
          user="test-user",
          zone="us-central1-a",
      )
      asset = await assets.create_or_fetch_asset(
          session,
          request,
          b0,
          tiering_service_pb2.ServerConfig(
              client_keep_alive_interval_seconds=600
          ),
          tier_path_uuid="tp-uuid-comp-del",
          storage_path="/mnt/lustre-a/complete_del_tp",
      )
      tp = asset.tier_paths[0]
      tp.state = db_schema.TierPathState.DELETE_IN_PROCESS
      await session.commit()

      # Now complete delete
      updated_tp = await assets.complete_delete_tier_path(session, tp)
      await session.commit()

      self.assertEqual(updated_tp.state, db_schema.TierPathState.DELETED)
      self.assertIsNone(updated_tp.ready_at)
      self.assertIsNone(updated_tp.expires_at)

      # Verify persistence
      async with self.session_maker() as session2:
        stmt = select(db_schema.TierPath).where(db_schema.TierPath.id == tp.id)
        res = await session2.execute(stmt)
        db_tp = res.scalars().first()
        self.assertIsNotNone(db_tp)
        assert db_tp is not None
        self.assertEqual(db_tp.state, db_schema.TierPathState.DELETED)

  async def test_begin_delete_asset_success_with_multiple_assets(self):
    async with self.session_maker() as session:
      b0 = db_schema.StorageBackend(
          level=0,
          backend_type=db_schema.BackendType.BACKEND_TYPE_LUSTRE,
          prefix="/mnt/lustre-a",
          zone="us-central1-a",
      )
      session.add(b0)
      await session.commit()

      # Create Asset 1
      request1 = tiering_service_pb2.ReserveRequest(
          path="test/path/asset1",
          user="test-user",
          zone="us-central1-a",
      )
      asset1 = await assets.create_or_fetch_asset(
          session,
          request1,
          b0,
          tiering_service_pb2.ServerConfig(
              client_keep_alive_interval_seconds=600
          ),
          tier_path_uuid="tp-uuid-1",
          storage_path="/mnt/lustre-a/asset1",
      )
      await assets.finalize_asset(session, asset1)

      # Create Asset 2 (the one we will delete)
      request2 = tiering_service_pb2.ReserveRequest(
          path="test/path/asset2",
          user="test-user",
          zone="us-central1-a",
      )
      asset2 = await assets.create_or_fetch_asset(
          session,
          request2,
          b0,
          tiering_service_pb2.ServerConfig(
              client_keep_alive_interval_seconds=600
          ),
          tier_path_uuid="tp-uuid-2",
          storage_path="/mnt/lustre-a/asset2",
      )
      await assets.finalize_asset(session, asset2)
      await session.commit()

    # Now call begin_delete_asset on Asset 2
    async with self.session_maker() as session2:
      db_assets2 = await assets.fetch_asset_by_uuid(session2, asset2.asset_uuid)
      db_asset2 = db_assets2[0]

      deleted = await assets.begin_delete_asset(session2, db_asset2)
      await session2.commit()

      self.assertIsNotNone(deleted)
      self.assertEqual(deleted.asset_uuid, asset2.asset_uuid)
      self.assertEqual(deleted.state, db_schema.AssetState.ASSET_STATE_DELETED)

    # Verify that Asset 1 is UNTOUCHED (still STORED)
    async with self.session_maker() as session3:
      fetched1 = await assets.fetch_asset_by_uuid(session3, asset1.asset_uuid)
      self.assertEqual(
          fetched1[0].state, db_schema.AssetState.ASSET_STATE_STORED
      )
      # Also verify Asset 1's tier paths are still READY
      for tp in fetched1[0].tier_paths:
        self.assertEqual(tp.state, db_schema.TierPathState.READY)

      # Verify Asset 2 is DELETED and its tier paths are DELETE_IN_PROCESS
      fetched2 = await assets.fetch_asset_by_uuid(session3, asset2.asset_uuid)
      self.assertEqual(
          fetched2[0].state, db_schema.AssetState.ASSET_STATE_DELETED
      )
      for tp in fetched2[0].tier_paths:
        self.assertEqual(tp.state, db_schema.TierPathState.DELETE_IN_PROCESS)

  async def test_complete_delete_asset_success_with_multiple_assets(self):
    async with self.session_maker() as session:
      b0 = db_schema.StorageBackend(
          level=0,
          backend_type=db_schema.BackendType.BACKEND_TYPE_LUSTRE,
          prefix="/mnt/lustre-a",
          zone="us-central1-a",
      )
      session.add(b0)
      await session.commit()

      # Create Asset 1
      request1 = tiering_service_pb2.ReserveRequest(
          path="test/path/asset1",
          user="test-user",
          zone="us-central1-a",
      )
      asset1 = await assets.create_or_fetch_asset(
          session,
          request1,
          b0,
          tiering_service_pb2.ServerConfig(
              client_keep_alive_interval_seconds=600
          ),
          tier_path_uuid="tp-uuid-1",
          storage_path="/mnt/lustre-a/asset1",
      )
      await assets.finalize_asset(session, asset1)

      # Create Asset 2 (the one we will delete)
      request2 = tiering_service_pb2.ReserveRequest(
          path="test/path/asset2",
          user="test-user",
          zone="us-central1-a",
      )
      asset2 = await assets.create_or_fetch_asset(
          session,
          request2,
          b0,
          tiering_service_pb2.ServerConfig(
              client_keep_alive_interval_seconds=600
          ),
          tier_path_uuid="tp-uuid-2",
          storage_path="/mnt/lustre-a/asset2",
      )
      await assets.finalize_asset(session, asset2)
      await session.commit()

    # Transition Asset 2 to DELETE_IN_PROCESS via begin_delete_asset
    async with self.session_maker() as session2:
      db_assets2 = await assets.fetch_asset_by_uuid(session2, asset2.asset_uuid)
      db_asset2 = db_assets2[0]
      await assets.begin_delete_asset(session2, db_asset2)
      await session2.commit()

    # Now call complete_delete_asset on Asset 2
    async with self.session_maker() as session3:
      db_assets2 = await assets.fetch_asset_by_uuid(session3, asset2.asset_uuid)
      db_asset2 = db_assets2[0]

      completed = await assets.complete_delete_asset(session3, db_asset2)
      await session3.commit()

      self.assertIsNotNone(completed)
      self.assertEqual(completed.asset_uuid, asset2.asset_uuid)

    # Verify that Asset 1 is UNTOUCHED (still STORED and tier paths READY)
    async with self.session_maker() as session4:
      fetched1 = await assets.fetch_asset_by_uuid(session4, asset1.asset_uuid)
      self.assertEqual(
          fetched1[0].state, db_schema.AssetState.ASSET_STATE_STORED
      )
      for tp in fetched1[0].tier_paths:
        self.assertEqual(tp.state, db_schema.TierPathState.READY)

      # Verify Asset 2 is DELETED and its tier paths are DELETED
      fetched2 = await assets.fetch_asset_by_uuid(session4, asset2.asset_uuid)
      self.assertEqual(
          fetched2[0].state, db_schema.AssetState.ASSET_STATE_DELETED
      )
      for tp in fetched2[0].tier_paths:
        self.assertEqual(tp.state, db_schema.TierPathState.DELETED)

  async def test_trigger_l0_to_l1_copy_zone_match(self):
    async with self.session_maker() as session:
      b0 = db_schema.StorageBackend(
          level=0,
          backend_type=db_schema.BackendType.BACKEND_TYPE_LUSTRE,
          prefix="/mnt/lustre",
          zone="us-central1-a",
      )
      b1_a = db_schema.StorageBackend(
          level=1,
          backend_type=db_schema.BackendType.BACKEND_TYPE_GCS,
          prefix="gs://bucket-a",
          zone="us-central1-a",
      )
      b1_b = db_schema.StorageBackend(
          level=1,
          backend_type=db_schema.BackendType.BACKEND_TYPE_GCS,
          prefix="gs://bucket-b",
          zone="us-east1-a",
      )
      session.add_all([b0, b1_a, b1_b])
      await session.commit()

      request = tiering_service_pb2.ReserveRequest(
          path="test/path/copy_zone_match",
          user="test-user",
          zone="us-central1-a",
      )
      tp_uuid = "tp-uuid-b0"
      storage_path = storage_backend_lib.get_storage_path(
          b0, request.path, tp_uuid
      )
      asset = await assets.create_or_fetch_asset(
          session,
          request,
          b0,
          tiering_service_pb2.ServerConfig(
              client_keep_alive_interval_seconds=600
          ),
          tier_path_uuid=tp_uuid,
          storage_path=storage_path,
      )
      finalized = await assets.finalize_asset(session, asset)

      await assets.trigger_l0_to_l1_copy(session, finalized)
      await session.commit()

      # Verify it selected b1_a (same zone) and NOT b1_b
      async with self.session_maker() as session2:
        stmt = (
            select(db_schema.Asset)
            .where(db_schema.Asset.asset_uuid == finalized.asset_uuid)
            .options(joinedload(db_schema.Asset.tier_paths))
        )
        res = await session2.execute(stmt)
        db_asset = res.scalars().first()
        self.assertIsNotNone(db_asset)
        assert db_asset is not None

        # Should have 2 tier paths: L0 (b0) and L1 (b1_a)
        self.assertLen(db_asset.tier_paths, 2)
        backends = [tp.storage_backend_id for tp in db_asset.tier_paths]
        self.assertIn(b0.id, backends)
        self.assertIn(b1_a.id, backends)
        self.assertNotIn(b1_b.id, backends)

  async def test_trigger_l0_to_l1_copy_region_match(self):
    async with self.session_maker() as session:
      b0 = db_schema.StorageBackend(
          level=0,
          backend_type=db_schema.BackendType.BACKEND_TYPE_LUSTRE,
          prefix="/mnt/lustre",
          zone="us-central1-a",
      )
      b1_a = db_schema.StorageBackend(
          level=1,
          backend_type=db_schema.BackendType.BACKEND_TYPE_GCS,
          prefix="gs://bucket-a",
          region="us-central1",
      )
      b1_b = db_schema.StorageBackend(
          level=1,
          backend_type=db_schema.BackendType.BACKEND_TYPE_GCS,
          prefix="gs://bucket-b",
          region="us-east1",
      )
      session.add_all([b0, b1_a, b1_b])
      await session.commit()

      request = tiering_service_pb2.ReserveRequest(
          path="test/path/copy_region_match",
          user="test-user",
          zone="us-central1-a",
      )
      tp_uuid = "tp-uuid-b0-reg"
      storage_path = storage_backend_lib.get_storage_path(
          b0, request.path, tp_uuid
      )
      asset = await assets.create_or_fetch_asset(
          session,
          request,
          b0,
          tiering_service_pb2.ServerConfig(
              client_keep_alive_interval_seconds=600
          ),
          tier_path_uuid=tp_uuid,
          storage_path=storage_path,
      )
      finalized = await assets.finalize_asset(session, asset)

      await assets.trigger_l0_to_l1_copy(session, finalized)
      await session.commit()

      # Verify it selected b1_a (region matches L0 zone's region)
      async with self.session_maker() as session2:
        stmt = (
            select(db_schema.Asset)
            .where(db_schema.Asset.asset_uuid == finalized.asset_uuid)
            .options(joinedload(db_schema.Asset.tier_paths))
        )
        res = await session2.execute(stmt)
        db_asset = res.scalars().first()
        self.assertIsNotNone(db_asset)
        assert db_asset is not None

        self.assertLen(db_asset.tier_paths, 2)
        backends = [tp.storage_backend_id for tp in db_asset.tier_paths]
        self.assertIn(b0.id, backends)
        self.assertIn(b1_a.id, backends)
        self.assertNotIn(b1_b.id, backends)

  async def test_trigger_l0_to_l1_copy_multi_region_match(self):
    async with self.session_maker() as session:
      b0 = db_schema.StorageBackend(
          level=0,
          backend_type=db_schema.BackendType.BACKEND_TYPE_LUSTRE,
          prefix="/mnt/lustre",
          zone="us-central1-a",
      )
      b1_a = db_schema.StorageBackend(
          level=1,
          backend_type=db_schema.BackendType.BACKEND_TYPE_GCS,
          prefix="gs://bucket-a",
          multi_regions=["us-central1", "us-east1"],
      )
      b1_b = db_schema.StorageBackend(
          level=1,
          backend_type=db_schema.BackendType.BACKEND_TYPE_GCS,
          prefix="gs://bucket-b",
          region="us-west1",
      )
      session.add_all([b0, b1_a, b1_b])
      await session.commit()

      request = tiering_service_pb2.ReserveRequest(
          path="test/path/copy_multi_region_match",
          user="test-user",
          zone="us-central1-a",
      )
      tp_uuid = "tp-uuid-b0-mreg"
      storage_path = storage_backend_lib.get_storage_path(
          b0, request.path, tp_uuid
      )
      asset = await assets.create_or_fetch_asset(
          session,
          request,
          b0,
          tiering_service_pb2.ServerConfig(
              client_keep_alive_interval_seconds=600
          ),
          tier_path_uuid=tp_uuid,
          storage_path=storage_path,
      )
      finalized = await assets.finalize_asset(session, asset)

      await assets.trigger_l0_to_l1_copy(session, finalized)
      await session.commit()

      # Verify it selected b1_a (L0 region is in b1_a's multi-regions)
      async with self.session_maker() as session2:
        stmt = (
            select(db_schema.Asset)
            .where(db_schema.Asset.asset_uuid == finalized.asset_uuid)
            .options(joinedload(db_schema.Asset.tier_paths))
        )
        res = await session2.execute(stmt)
        db_asset = res.scalars().first()
        self.assertIsNotNone(db_asset)
        assert db_asset is not None

        self.assertLen(db_asset.tier_paths, 2)
        backends = [tp.storage_backend_id for tp in db_asset.tier_paths]
        self.assertIn(b0.id, backends)
        self.assertIn(b1_a.id, backends)
        self.assertNotIn(b1_b.id, backends)


if __name__ == "__main__":
  absltest.main()
