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

import asyncio
import datetime
import multiprocessing
import unittest

from absl.testing import absltest
from absl.testing import parameterized
import aiosqlite  # pylint: disable=unused-import
import greenlet  # pylint: disable=unused-import
from orbax.checkpoint.experimental.tiering_service import db_schema
import sqlalchemy
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.future import select
from sqlalchemy.orm import sessionmaker


class DbSchemaTest(parameterized.TestCase, unittest.IsolatedAsyncioTestCase):

  async def asyncSetUp(self) -> None:
    await super().asyncSetUp()
    tmp_file = self.create_tempfile()
    self.db_path = tmp_file.full_path

    self.engine = create_async_engine(
        f"sqlite+aiosqlite:///{self.db_path}",
        echo=True,
    )

    async with self.engine.begin() as conn:
      await conn.exec_driver_sql("PRAGMA journal_mode=WAL")
      await conn.run_sync(db_schema.Base.metadata.create_all)

    self.session_maker = sessionmaker(
        self.engine, expire_on_commit=False, class_=AsyncSession
    )

  async def asyncTearDown(self) -> None:
    async with self.engine.begin() as conn:
      await conn.run_sync(db_schema.Base.metadata.drop_all)
    await self.engine.dispose()
    await super().asyncTearDown()

  async def test_create_asset(self) -> None:
    async with self.session_maker() as session:
      asset = db_schema.Asset(
          path="/experiment/step1",
          user="testuser",
          tags=["tag1", "tag2"],
          state=db_schema.AssetState.ASSET_STATE_ACTIVE_WRITE,
          created_at=datetime.datetime(2026, 1, 1, 10, 0, 0),
      )
      session.add(asset)
      await session.commit()

      generated_uuid = asset.asset_uuid
      self.assertIsNotNone(generated_uuid)

      result = await session.execute(
          select(db_schema.Asset).filter_by(asset_uuid=generated_uuid)
      )
      fetched = result.scalars().first()
      self.assertIsNotNone(fetched)
      self.assertEqual(fetched.path, "/experiment/step1")
      self.assertEqual(fetched.tags, ["tag1", "tag2"])
      self.assertEqual(
          fetched.state, db_schema.AssetState.ASSET_STATE_ACTIVE_WRITE
      )

  async def test_update_asset_state(self) -> None:
    async with self.session_maker() as session:
      asset = db_schema.Asset(
          asset_uuid="uuid-456",
          path="/experiment/step2",
          user="testuser",
          state=db_schema.AssetState.ASSET_STATE_ACTIVE_WRITE,
      )
      session.add(asset)
      await session.commit()

      result = await session.execute(
          select(db_schema.Asset).filter_by(asset_uuid="uuid-456")
      )
      asset_fetch = result.scalars().first()
      asset_fetch.state = db_schema.AssetState.ASSET_STATE_STORED
      await session.commit()

      result = await session.execute(
          select(db_schema.Asset).filter_by(asset_uuid="uuid-456")
      )
      fetched = result.scalars().first()
      self.assertEqual(fetched.state, db_schema.AssetState.ASSET_STATE_STORED)

  async def test_add_tier_path(self) -> None:
    async with self.session_maker() as session:
      asset = db_schema.Asset(
          asset_uuid="uuid-789",
          path="/experiment/step3",
          user="testuser",
      )
      backend0 = db_schema.StorageBackend(
          level=0,
          zone="us-east5-a",
          backend_type=db_schema.BackendType.BACKEND_TYPE_LUSTRE,
      )
      backend1 = db_schema.StorageBackend(
          level=1,
          multi_regions=["us-central1", "us-east1"],
          backend_type=db_schema.BackendType.BACKEND_TYPE_GCS,
      )
      tier_path0 = db_schema.TierPath(
          asset_uuid="uuid-789",
          storage_backend=backend0,
          path="/lustre/path/1",
      )
      tier_path1 = db_schema.TierPath(
          asset_uuid="uuid-789",
          storage_backend=backend1,
          path="/gcs/path/2",
      )
      session.add(asset)
      session.add(backend0)
      session.add(backend1)
      session.add(tier_path0)
      session.add(tier_path1)
      await session.commit()

      result = await session.execute(
          select(db_schema.Asset)
          .options(
              sqlalchemy.orm.selectinload(
                  db_schema.Asset.tier_paths
              ).selectinload(db_schema.TierPath.storage_backend)
          )
          .filter_by(asset_uuid="uuid-789")
      )
      fetched = result.scalars().first()
      self.assertLen(fetched.tier_paths, 2)
      tp0 = next(
          tp for tp in fetched.tier_paths if tp.storage_backend.level == 0
      )
      tp1 = next(
          tp for tp in fetched.tier_paths if tp.storage_backend.level == 1
      )
      self.assertEqual(tp0.path, "/lustre/path/1")
      self.assertEqual(tp0.storage_backend.zone, "us-east5-a")
      self.assertEqual(tp1.path, "/gcs/path/2")
      self.assertEqual(
          tp1.storage_backend.multi_regions,
          ["us-central1", "us-east1"],
      )

  async def test_add_tier_path_fails_multiple_locations(self) -> None:
    async with self.session_maker() as session:
      backend = db_schema.StorageBackend(
          level=0,
          zone="us-central1-a",
          backend_type=db_schema.BackendType.BACKEND_TYPE_GCS,
      )
      session.add(backend)
      await session.commit()

      asset = db_schema.Asset(
          asset_uuid="uuid-dup-locations",
          path="/experiment/dup_locations",
          user="testuser",
      )

      tp1 = db_schema.TierPath(
          storage_backend=backend,
          path="/path1",
      )
      asset.tier_paths.append(tp1)
      session.add(asset)

      await session.commit()

      tp2 = db_schema.TierPath(
          storage_backend=backend,
          path="/dup_path",
      )
      asset.tier_paths.append(tp2)
      with self.assertRaisesRegex(
          sqlalchemy.exc.IntegrityError,
          "UNIQUE constraint failed: tier_paths.asset_uuid,"
          " tier_paths.storage_backend_id",
      ):
        await session.commit()

  async def test_storage_backend_fails_multiple_locations_zone(self) -> None:
    async with self.session_maker() as session:
      asset = db_schema.Asset(
          asset_uuid="uuid-distinct-backends-zone",
          path="/experiment/distinct_backends_zone",
          user="testuser",
      )
      b1 = db_schema.StorageBackend(
          level=0,
          zone="us-central1-a",
          backend_type=db_schema.BackendType.BACKEND_TYPE_LUSTRE,
      )
      b2 = db_schema.StorageBackend(
          level=0,
          zone="us-central1-a",
          backend_type=db_schema.BackendType.BACKEND_TYPE_LUSTRE,
      )
      tp1 = db_schema.TierPath(storage_backend=b1, path="/path1")
      tp2 = db_schema.TierPath(storage_backend=b2, path="/path2")
      asset.tier_paths.extend([tp1, tp2])
      session.add(asset)
      with self.assertRaisesRegex(ValueError, "Duplicate zone"):
        await session.commit()

  async def test_storage_backend_fails_multiple_locations_region(self) -> None:
    async with self.session_maker() as session:
      asset = db_schema.Asset(
          asset_uuid="uuid-distinct-backends-region",
          path="/experiment/distinct_backends_region",
          user="testuser",
      )
      b1 = db_schema.StorageBackend(
          level=0,
          region="us-central1",
          backend_type=db_schema.BackendType.BACKEND_TYPE_GCS,
      )
      b2 = db_schema.StorageBackend(
          level=0,
          region="us-central1",
          backend_type=db_schema.BackendType.BACKEND_TYPE_GCS,
      )
      tp1 = db_schema.TierPath(storage_backend=b1, path="/path1")
      tp2 = db_schema.TierPath(storage_backend=b2, path="/path2")
      asset.tier_paths.extend([tp1, tp2])
      session.add(asset)
      with self.assertRaisesRegex(ValueError, "Duplicate region"):
        await session.commit()

  async def test_storage_backend_fails_multiple_locations_multi_regions(
      self,
  ) -> None:
    async with self.session_maker() as session:
      asset = db_schema.Asset(
          asset_uuid="uuid-distinct-backends-mr",
          path="/experiment/distinct_backends_mr",
          user="testuser",
      )
      b1 = db_schema.StorageBackend(
          level=0,
          multi_regions=["us-central1", "us-east1"],
          backend_type=db_schema.BackendType.BACKEND_TYPE_GCS,
      )
      # Order of regions shouldn't matter
      b2 = db_schema.StorageBackend(
          level=0,
          multi_regions=["us-east1", "us-central1"],
          backend_type=db_schema.BackendType.BACKEND_TYPE_GCS,
      )
      tp1 = db_schema.TierPath(storage_backend=b1, path="/path1")
      tp2 = db_schema.TierPath(storage_backend=b2, path="/path2")
      asset.tier_paths.extend([tp1, tp2])
      session.add(asset)
      with self.assertRaisesRegex(ValueError, "Duplicate multi_regions"):
        await session.commit()

  async def test_add_tier_path_fails_no_locations(self) -> None:
    async with self.session_maker() as session:
      with self.assertRaisesRegex(
          sqlalchemy.exc.IntegrityError, "check_mutually_exclusive_locations"
      ):
        invalid_backend_empty = db_schema.StorageBackend(
            level=0,
        )
        session.add(invalid_backend_empty)
        await session.commit()

  async def test_asset_job_queue(self) -> None:
    async with self.session_maker() as session:
      asset = db_schema.Asset(
          asset_uuid="uuid-queue",
          path="/experiment/queue",
          user="testuser",
      )
      backend = db_schema.StorageBackend(
          level=0,
          zone="us-central1-a",
          backend_type=db_schema.BackendType.BACKEND_TYPE_GCS,
      )
      tier_path = db_schema.TierPath(
          asset_uuid="uuid-queue", storage_backend=backend, path="/path1"
      )
      session.add_all([asset, backend, tier_path])
      await session.flush()

      job1 = db_schema.AssetJob(
          asset_uuid="uuid-queue",
          request_type=db_schema.RequestType.REQUEST_TYPE_COPY,
          status=db_schema.JobStatus.JOB_STATUS_QUEUED,
          target_tier_path_id=tier_path.id,
      )
      job2 = db_schema.AssetJob(
          asset_uuid="uuid-queue",
          request_type=db_schema.RequestType.REQUEST_TYPE_DELETE_FROM_INSTANCE,
          status=db_schema.JobStatus.JOB_STATUS_QUEUED,
          target_tier_path_id=tier_path.id,
      )
      session.add_all([job1, job2])
      await session.commit()

      result = await session.execute(
          select(db_schema.AssetJob)
          .filter_by(asset_uuid="uuid-queue")
          .order_by(db_schema.AssetJob.id)
      )
      jobs = result.scalars().all()
      self.assertLen(jobs, 2)
      self.assertEqual(
          jobs[0].request_type, db_schema.RequestType.REQUEST_TYPE_COPY
      )
      self.assertEqual(
          jobs[1].request_type,
          db_schema.RequestType.REQUEST_TYPE_DELETE_FROM_INSTANCE,
      )

      jobs[0].status = db_schema.JobStatus.JOB_STATUS_COMPLETED
      await session.commit()

      result = await session.execute(
          select(db_schema.AssetJob).filter_by(id=job1.id)
      )
      fetched_job = result.scalars().first()
      self.assertEqual(
          fetched_job.status, db_schema.JobStatus.JOB_STATUS_COMPLETED
      )

  async def test_create_asset_duplicates_allowed_for_deleted_incomplete(self):
    # Verify we can have duplicate path for DELETED or INCOMPLETE states
    async with self.session_maker() as session:
      asset1 = db_schema.Asset(
          path="/experiment/dup_allow",
          user="testuser",
          state=db_schema.AssetState.ASSET_STATE_DELETED,
      )
      asset2 = db_schema.Asset(
          path="/experiment/dup_allow",
          user="testuser",
          state=db_schema.AssetState.ASSET_STATE_INCOMPLETE,
      )
      session.add(asset1)
      session.add(asset2)
      await session.commit()

  async def test_create_asset_duplicates_blocked_for_active_stored(
      self,
  ) -> None:
    async with self.session_maker() as session:
      asset3 = db_schema.Asset(
          path="/experiment/dup_block",
          user="testuser",
          state=db_schema.AssetState.ASSET_STATE_ACTIVE_WRITE,
      )
      session.add(asset3)
      await session.commit()

    async with self.session_maker() as session:
      asset4 = db_schema.Asset(
          path="/experiment/dup_block",
          user="testuser",
          state=db_schema.AssetState.ASSET_STATE_STORED,
      )
      session.add(asset4)
      with self.assertRaisesRegex(
          sqlalchemy.exc.IntegrityError,
          "UNIQUE constraint failed: assets.path",
      ):
        await session.commit()

  @parameterized.named_parameters(
      dict(
          testcase_name="same_backend_type",
          backend1=db_schema.StorageBackend(
              level=1,
              zone="us-central1-a",
              backend_type=db_schema.BackendType.BACKEND_TYPE_LUSTRE,
          ),
          backend2=db_schema.StorageBackend(
              level=1,
              zone="us-central1-b",
              backend_type=db_schema.BackendType.BACKEND_TYPE_GCS,
          ),
          expected_exception=ValueError,
          expected_regex="same backend_type",
      ),
      dict(
          testcase_name="duplicate_zone_at_same_level",
          backend1=db_schema.StorageBackend(
              level=1,
              zone="us-central1-a",
              backend_type=db_schema.BackendType.BACKEND_TYPE_LUSTRE,
          ),
          backend2=db_schema.StorageBackend(
              level=1,
              zone="us-central1-a",
              backend_type=db_schema.BackendType.BACKEND_TYPE_LUSTRE,
          ),
          expected_exception=ValueError,
          expected_regex="Duplicate zone",
      ),
      dict(
          testcase_name="duplicate_region_at_same_level",
          backend1=db_schema.StorageBackend(
              level=1,
              region="us-central1",
              backend_type=db_schema.BackendType.BACKEND_TYPE_GCS,
          ),
          backend2=db_schema.StorageBackend(
              level=1,
              region="us-central1",
              backend_type=db_schema.BackendType.BACKEND_TYPE_GCS,
          ),
          expected_exception=ValueError,
          expected_regex="Duplicate region",
      ),
      dict(
          testcase_name="duplicate_multi_regions_at_same_level",
          backend1=db_schema.StorageBackend(
              level=1,
              multi_regions=["us-central1", "us-east1"],
              backend_type=db_schema.BackendType.BACKEND_TYPE_GCS,
          ),
          backend2=db_schema.StorageBackend(
              level=1,
              multi_regions=["us-east1", "us-central1"],
              backend_type=db_schema.BackendType.BACKEND_TYPE_GCS,
          ),
          expected_exception=ValueError,
          expected_regex="Duplicate multi_regions",
      ),
  )
  async def test_storage_backend_validation(
      self,
      backend1,
      backend2,
      expected_exception,
      expected_regex,
  ) -> None:
    async with self.session_maker() as session:
      b1 = backend1
      b2 = backend2
      session.add(b1)
      session.add(b2)
      with self.assertRaisesRegex(expected_exception, expected_regex):
        await session.commit()


def _worker_add_job(
    db_path: str, request_type_val: int, tp_id: int | None
) -> tuple[int, bool]:
  engine = sqlalchemy.create_engine(
      f"sqlite:///{db_path}", connect_args={"timeout": 30}
  )
  db_session = sqlalchemy.orm.sessionmaker(engine)
  with db_session() as session:
    job = db_schema.AssetJob(
        asset_uuid="uuid-queue-multi",
        request_type=db_schema.RequestType(request_type_val),
        status=db_schema.JobStatus.JOB_STATUS_QUEUED,
        target_tier_path_id=tp_id,
    )
    try:
      session.add(job)
      session.commit()
      return (request_type_val, True)
    except sqlalchemy.exc.IntegrityError:
      return (request_type_val, False)


def _worker_create_asset(db_path: str) -> bool:
  engine = sqlalchemy.create_engine(
      f"sqlite:///{db_path}", connect_args={"timeout": 30}
  )
  db_session = sqlalchemy.orm.sessionmaker(engine)
  with db_session() as session:
    asset = db_schema.Asset(
        path="/experiment/race_condition",
        user="testuser",
        state=db_schema.AssetState.ASSET_STATE_ACTIVE_WRITE,
    )
    session.add(asset)
    try:
      session.commit()
      return True
    except sqlalchemy.exc.IntegrityError:
      return False


class DbSchemaMultiprocessTest(
    absltest.TestCase, unittest.IsolatedAsyncioTestCase
):

  async def asyncSetUp(self) -> None:
    await super().asyncSetUp()
    tmp_file = self.create_tempfile()
    self.db_path = tmp_file.full_path

    self.engine = create_async_engine(
        f"sqlite+aiosqlite:///{self.db_path}",
        echo=True,
    )

    async with self.engine.begin() as conn:
      await conn.exec_driver_sql("PRAGMA journal_mode=WAL")
      await conn.run_sync(db_schema.Base.metadata.create_all)

    self.session_maker = sessionmaker(
        self.engine, expire_on_commit=False, class_=AsyncSession
    )

  async def asyncTearDown(self) -> None:
    async with self.engine.begin() as conn:
      await conn.run_sync(db_schema.Base.metadata.drop_all)
    await self.engine.dispose()
    await super().asyncTearDown()

  def test_asset_job_queue_multiprocess(self) -> None:
    async def _setup():
      async with self.session_maker() as session:
        asset = db_schema.Asset(
            asset_uuid="uuid-queue-multi",
            path="/experiment/queue-multi",
            user="testuser",
        )
        sb = db_schema.StorageBackend(level=0, zone="us-central1-a")
        tp = db_schema.TierPath(
            asset_uuid="uuid-queue-multi", storage_backend=sb, path="/path1"
        )
        session.add_all([asset, sb, tp])
        await session.commit()
        return tp.id

    tp_id = asyncio.run(_setup())

    job_types = [
        (int(db_schema.RequestType.REQUEST_TYPE_COPY), tp_id),
        (int(db_schema.RequestType.REQUEST_TYPE_DELETE_FROM_INSTANCE), tp_id),
        (int(db_schema.RequestType.REQUEST_TYPE_DELETE_FROM_ALL_TIERS), None),
    ]

    with multiprocessing.Pool(processes=3) as pool:
      results = pool.starmap(
          _worker_add_job,
          [(self.db_path, jt, target_id) for jt, target_id in job_types],
      )

    for request_type_val, success in results:
      with self.subTest(request_type=request_type_val):
        self.assertTrue(success)

    async def _verify():
      async with self.session_maker() as session:
        result = await session.execute(
            select(db_schema.AssetJob).filter_by(asset_uuid="uuid-queue-multi")
        )
        jobs = result.scalars().all()
        self.assertLen(jobs, 3)

        found_types = [j.request_type for j in jobs]
        expected_types = [
            db_schema.RequestType.REQUEST_TYPE_COPY,
            db_schema.RequestType.REQUEST_TYPE_DELETE_FROM_INSTANCE,
            db_schema.RequestType.REQUEST_TYPE_DELETE_FROM_ALL_TIERS,
        ]
        self.assertCountEqual(found_types, expected_types)

    asyncio.run(_verify())

  def test_create_asset_multiprocess(self) -> None:
    with multiprocessing.Pool(processes=5) as pool:
      results = pool.map(_worker_create_asset, [self.db_path] * 5)

    successes = results.count(True)
    self.assertEqual(successes, 1)


if __name__ == "__main__":
  absltest.main()
