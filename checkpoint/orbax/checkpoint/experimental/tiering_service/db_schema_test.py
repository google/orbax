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
from concurrent import futures
import datetime
import os
import tempfile
import unittest

from absl.testing import absltest
import aiosqlite  # pylint: disable=unused-import
import greenlet  # pylint: disable=unused-import
from orbax.checkpoint.experimental.tiering_service import db_schema
import sqlalchemy
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.future import select
from sqlalchemy.orm import sessionmaker

Asset = db_schema.Asset
TierPath = db_schema.TierPath
AssetJob = db_schema.AssetJob
AssetState = db_schema.AssetState
BackendType = db_schema.BackendType
JobStatus = db_schema.JobStatus

ThreadPoolExecutor = futures.ThreadPoolExecutor


class DbSchemaTest(unittest.IsolatedAsyncioTestCase):

  async def asyncSetUp(self):

    self.db_fd, self.db_path = tempfile.mkstemp()

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

  async def asyncTearDown(self):
    async with self.engine.begin() as conn:
      await conn.run_sync(db_schema.Base.metadata.drop_all)
    await self.engine.dispose()

    os.close(self.db_fd)
    os.unlink(self.db_path)

  async def test_create_asset(self):
    async with self.session_maker() as session:
      asset = Asset(
          unique_path="/experiment/step1",
          user="testuser",
          tags=["tag1", "tag2"],
          state=AssetState.ASSET_STATE_ACTIVE_WRITE,
          created_at=datetime.datetime.now(),
      )
      session.add(asset)
      await session.commit()

      generated_uuid = asset.uuid
      self.assertIsNotNone(generated_uuid)

      result = await session.execute(
          select(Asset).filter_by(uuid=generated_uuid)
      )
      fetched = result.scalars().first()
      self.assertIsNotNone(fetched)
      self.assertEqual(fetched.unique_path, "/experiment/step1")
      self.assertEqual(fetched.tags, ["tag1", "tag2"])
      self.assertEqual(fetched.state, AssetState.ASSET_STATE_ACTIVE_WRITE)

  async def test_update_asset_state(self):
    async with self.session_maker() as session:
      asset = Asset(
          uuid="uuid-456",
          unique_path="/experiment/step2",
          user="testuser",
          state=AssetState.ASSET_STATE_ACTIVE_WRITE,
      )
      session.add(asset)
      await session.commit()

      result = await session.execute(select(Asset).filter_by(uuid="uuid-456"))
      asset_fetch = result.scalars().first()
      asset_fetch.state = AssetState.ASSET_STATE_STORED
      await session.commit()

      result = await session.execute(select(Asset).filter_by(uuid="uuid-456"))
      fetched = result.scalars().first()
      self.assertEqual(fetched.state, AssetState.ASSET_STATE_STORED)

  async def test_add_tier_path(self):
    async with self.session_maker() as session:
      asset = Asset(
          uuid="uuid-789",
          unique_path="/experiment/step3",
          user="testuser",
      )
      tier_path = TierPath(
          asset_uuid="uuid-789",
          level=0,
          multi_region=["us-central1", "us-east1"],
          backend_type=BackendType.BACKEND_TYPE_LUSTRE,
          path="/lustre/path/1",
      )
      session.add(asset)
      session.add(tier_path)
      await session.commit()

      result = await session.execute(
          select(Asset)
          .options(sqlalchemy.orm.selectinload(Asset.tier_paths))
          .filter_by(uuid="uuid-789")
      )
      fetched = result.scalars().first()
      self.assertEqual(len(fetched.tier_paths), 1)
      self.assertEqual(fetched.tier_paths[0].path, "/lustre/path/1")
      self.assertEqual(
          fetched.tier_paths[0].multi_region, ["us-central1", "us-east1"]
      )

    # Verify check constraint fails if both zone and region are set
    async with self.session_maker() as session:
      with self.assertRaises(sqlalchemy.exc.IntegrityError):
        invalid_path = TierPath(
            asset_uuid="uuid-789",
            level=0,
            zone="us-central1-a",
            region="us-central1",
            path="/lustre/path/bad",
        )
        session.add(invalid_path)
        await session.commit()

    # Verify check constraint fails if NONE are set
    async with self.session_maker() as session:
      with self.assertRaises(sqlalchemy.exc.IntegrityError):
        invalid_path_empty = TierPath(
            asset_uuid="uuid-789",
            level=0,
            path="/lustre/path/bad2",
        )
        session.add(invalid_path_empty)
        await session.commit()

  async def test_asset_job_queue(self):
    async with self.session_maker() as session:
      asset = Asset(
          uuid="uuid-queue",
          unique_path="/experiment/queue",
          user="testuser",
      )
      job1 = AssetJob(
          asset_uuid="uuid-queue",
          request_type="PREFETCH",
          status=JobStatus.QUEUED,
          payload={"source": "t1", "destination": "t0"},
      )
      job2 = AssetJob(
          asset_uuid="uuid-queue",
          request_type="DELETE",
          status=JobStatus.QUEUED,
      )
      session.add(asset)
      session.add(job1)
      session.add(job2)
      await session.commit()

      result = await session.execute(
          select(AssetJob)
          .filter_by(asset_uuid="uuid-queue")
          .order_by(AssetJob.id)
      )
      jobs = result.scalars().all()
      self.assertEqual(len(jobs), 2)
      self.assertEqual(jobs[0].request_type, "PREFETCH")
      self.assertEqual(jobs[1].request_type, "DELETE")

      jobs[0].status = JobStatus.COMPLETED
      jobs[0].payload = {"status": "done"}
      await session.commit()

      result = await session.execute(select(AssetJob).filter_by(id=job1.id))
      fetched_job = result.scalars().first()
      self.assertEqual(fetched_job.status, JobStatus.COMPLETED)
      self.assertEqual(fetched_job.payload, {"status": "done"})

  async def test_asset_job_queue_multiprocess(self):
    # Verify that multiple threads can concurrently insert jobs for the same
    # asset. Order doesn't matter, just that all jobs are successfully added.
    # Create parent asset first so foreign key constraints are satisfied
    async with self.session_maker() as session:
      asset = Asset(
          uuid="uuid-queue-multi",
          unique_path="/experiment/queue-multi",
          user="testuser",
      )
      session.add(asset)
      await session.commit()

    def add_job_task(request_type):
      async def _task():

        async with self.session_maker() as session:
          job = AssetJob(
              asset_uuid="uuid-queue-multi",
              request_type=request_type,
              status=JobStatus.QUEUED,
          )
          session.add(job)
          while True:
            try:
              await session.commit()
              return True
            except sqlalchemy.exc.OperationalError as e:
              if "database is locked" in str(e):
                await asyncio.sleep(0.1)
                continue
              raise
          return False

      return asyncio.run(_task())

    loop = asyncio.get_running_loop()
    job_types = ["PREFETCH", "DELETE", "COPY", "FINALIZES"]
    with ThreadPoolExecutor(max_workers=len(job_types)) as pool:
      results = await asyncio.gather(
          *(loop.run_in_executor(pool, add_job_task, jt) for jt in job_types)
      )

    self.assertTrue(all(results))

    # Verify all jobs are present
    async with self.session_maker() as session:
      result = await session.execute(
          select(AssetJob).filter_by(asset_uuid="uuid-queue-multi")
      )
      jobs = result.scalars().all()
      self.assertEqual(len(jobs), 4)

      found_types = [j.request_type for j in jobs]
      self.assertCountEqual(found_types, job_types)

  async def test_create_asset_multiprocess(self):
    # Create tasks trying to add the same unique_path simultaneously
    # Only one task can succeed.
    def create_asset_task():

      async def _task():
        async with self.session_maker() as session:
          asset = Asset(
              unique_path="/experiment/race_condition",
              user="testuser",
              state=AssetState.ASSET_STATE_ACTIVE_WRITE,
          )
          session.add(asset)
          while True:
            try:
              await session.commit()
              return True
            except sqlalchemy.exc.IntegrityError:
              return False
            except sqlalchemy.exc.OperationalError as e:
              if "database is locked" in str(e):
                # This is expected, retry
                await asyncio.sleep(0.1)
                continue
              raise
          return False

      return asyncio.run(_task())

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=5) as pool:
      results = await asyncio.gather(
          loop.run_in_executor(pool, create_asset_task),
          loop.run_in_executor(pool, create_asset_task),
          loop.run_in_executor(pool, create_asset_task),
          loop.run_in_executor(pool, create_asset_task),
          loop.run_in_executor(pool, create_asset_task),
      )

    # Verify exactly one worker succeeded and others failed due to constraint
    successes = sum(results)
    self.assertEqual(successes, 1)

  async def test_create_asset_duplicates_allowed_for_deleted_incomplete(self):
    # Verify we can have duplicate unique_path for DELETED or INCOMPLETE states
    async with self.session_maker() as session:
      asset1 = Asset(
          unique_path="/experiment/dup",
          user="testuser",
          state=AssetState.ASSET_STATE_DELETED,
      )
      asset2 = Asset(
          unique_path="/experiment/dup",
          user="testuser",
          state=AssetState.ASSET_STATE_INCOMPLETE,
      )
      session.add(asset1)
      session.add(asset2)
      await session.commit()  # Should succeed

    # Verify we CANNOT have duplicates for ACTIVE_WRITE or STORED
    async with self.session_maker() as session:
      asset3 = Asset(
          unique_path="/experiment/dup",
          user="testuser",
          state=AssetState.ASSET_STATE_ACTIVE_WRITE,
      )
      session.add(asset3)
      await session.commit()  # Should succeed (first active/stored)

    async with self.session_maker() as session:
      asset4 = Asset(
          unique_path="/experiment/dup",
          user="testuser",
          state=AssetState.ASSET_STATE_STORED,
      )
      session.add(asset4)
      with self.assertRaises(sqlalchemy.exc.IntegrityError):
        await session.commit()

  async def test_validate_tier_paths_same_backend(self):
    async with self.session_maker():
      asset = Asset(
          uuid="uuid-val-backend",
          unique_path="/experiment/val_backend",
          user="testuser",
      )
      asset.tier_paths = [
          TierPath(
              level=1,
              zone="us-central1-a",
              backend_type=BackendType.BACKEND_TYPE_LUSTRE,
              path="/path1",
          ),
          TierPath(
              level=1,
              zone="us-central1-b",
              backend_type=BackendType.BACKEND_TYPE_GCS,
              path="/path2",
          ),
      ]
      with self.assertRaises(ValueError):
        asset.validate_pre_commit()

  async def test_validate_tier_paths_no_duplicates(self):
    async with self.session_maker():
      asset = Asset(
          uuid="uuid-val-dup",
          unique_path="/experiment/val_dup",
          user="testuser",
      )
      asset.tier_paths = [
          TierPath(
              level=1,
              zone="us-central1-a",
              backend_type=BackendType.BACKEND_TYPE_GCS,
              path="/path1",
          ),
          TierPath(
              level=1,
              zone="us-central1-a",
              backend_type=BackendType.BACKEND_TYPE_GCS,
              path="/path2",
          ),
      ]
      with self.assertRaises(ValueError):
        asset.validate_pre_commit()

  async def test_validate_tier_paths_multi_region_disorder(self):
    async with self.session_maker():
      asset = Asset(
          uuid="uuid-val-mr",
          unique_path="/experiment/val_mr",
          user="testuser",
      )
      asset.tier_paths = [
          TierPath(
              level=1,
              multi_region=["us-central1", "us-east1"],
              backend_type=BackendType.BACKEND_TYPE_GCS,
              path="/path1",
          ),
          TierPath(
              level=1,
              multi_region=["us-east1", "us-central1"],
              backend_type=BackendType.BACKEND_TYPE_GCS,
              path="/path2",
          ),
      ]
      with self.assertRaises(ValueError):
        asset.validate_pre_commit()


if __name__ == "__main__":
  absltest.main()
