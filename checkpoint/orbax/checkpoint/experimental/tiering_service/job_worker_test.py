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
import unittest

from absl.testing import absltest
from orbax.checkpoint.experimental.tiering_service import db_lib
from orbax.checkpoint.experimental.tiering_service import db_schema
from orbax.checkpoint.experimental.tiering_service import job_worker
from orbax.checkpoint.experimental.tiering_service import server_config
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import sessionmaker


class TieringServiceWorkerTest(
    absltest.TestCase, unittest.IsolatedAsyncioTestCase
):

  async def asyncSetUp(self):
    await super().asyncSetUp()
    storage_backends_config = [
        {
            "level": 0,
            "backend_type": "BACKEND_TYPE_LUSTRE",
            "prefix": "/mnt/lustre-a",
            "zone": "us-central1-a",
        },
        {
            "level": 0,
            "backend_type": "BACKEND_TYPE_LUSTRE",
            "prefix": "/mnt/lustre-b",
            "zone": "us-central1-b",
        },
        {
            "level": 1,
            "backend_type": "BACKEND_TYPE_GCS",
            "prefix": "gs://my-bucket",
            "region": "us-central1",
        },
    ]
    self.config = server_config.parse_config({
        "storage_backends": storage_backends_config,
        "max_active_jobs_per_backend": 1,
    })
    # Use temp file SQLite for testing
    self.tmp_file = self.create_tempfile()
    self.config.db_connection_str = (
        f"sqlite+aiosqlite:///{self.tmp_file.full_path}"
    )

    await db_lib.async_initialize_db(self.config)
    self.engine = db_lib.get_async_engine(self.config)
    self.session_maker = sessionmaker(
        self.engine, expire_on_commit=False, class_=AsyncSession
    )
    # Short poll interval for fast tests
    self.worker = job_worker.TieringServiceWorker(
        self.session_maker,
        self.config,
        lease_duration_seconds=2,  # Short lease for testing expiration
        poll_interval_seconds=1,
    )

  async def asyncTearDown(self):
    await self.worker.stop()
    await self.engine.dispose()
    await super().asyncTearDown()

  async def _create_asset_and_job(
      self,
      session,
      *,
      asset_uuid,
      path,
      target_backend_id,
      source_backend_id,
      job_status=db_schema.JobStatus.JOB_STATUS_QUEUED,
  ):
    """Helper to create an asset, its tier paths, and a prefetch job."""
    del self
    asset = db_schema.Asset(
        asset_uuid=asset_uuid,
        path=path,
        user="test-user",
        state=db_schema.AssetState.ASSET_STATE_STORED,
    )
    session.add(asset)

    # Source TierPath (ready)
    source_tp = db_schema.TierPath(
        asset_uuid=asset_uuid,
        storage_backend_id=source_backend_id,
        path=f"gs://my-bucket/{path}",
        ready_at=datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc),
    )
    session.add(source_tp)

    # Target TierPath (pending)
    target_tp = db_schema.TierPath(
        asset_uuid=asset_uuid,
        storage_backend_id=target_backend_id,
        path=f"/mnt/lustre/{path}",
    )
    session.add(target_tp)
    await session.flush()

    job = db_schema.AssetJob(
        asset_uuid=asset_uuid,
        request_type=db_schema.RequestType.REQUEST_TYPE_COPY,
        status=job_status,
        target_tier_path_id=target_tp.id,
    )
    session.add(job)
    await session.commit()
    return job, target_tp

  async def test_job_acquisition_success(self):
    async with self.session_maker() as session:
      # Get backend IDs
      result = await session.execute(select(db_schema.StorageBackend))
      backends = result.scalars().all()
      lustre_a = next(b for b in backends if b.zone == "us-central1-a")
      gcs = next(b for b in backends if b.region == "us-central1")

      await self._create_asset_and_job(
          session,
          asset_uuid="asset-1",
          path="path/1",
          target_backend_id=lustre_a.id,
          source_backend_id=gcs.id,
      )

    # Start worker to process the job
    await self.worker.start()

    # Wait for job to complete with timeout (stub completes it immediately)
    for _ in range(5):
      await asyncio.sleep(1)
      async with self.session_maker() as session:
        result = await session.execute(select(db_schema.AssetJob))
        job = result.scalars().first()
        if job.status == db_schema.JobStatus.JOB_STATUS_COMPLETED:
          break

    await self.worker.stop()

    async with self.session_maker() as session:
      # Verify job status transitions to COMPLETED
      result = await session.execute(select(db_schema.AssetJob))
      job = result.scalars().first()
      self.assertEqual(job.status, db_schema.JobStatus.JOB_STATUS_COMPLETED)
      self.assertIsNotNone(job.completed_at)
      self.assertIsNone(job.worker_host)
      self.assertIsNone(job.worker_pid)

      # Verify target TierPath is ready
      result_tp = await session.execute(
          select(db_schema.TierPath).where(
              db_schema.TierPath.asset_uuid == "asset-1"
          )
      )
      tps = result_tp.scalars().all()
      target_tp = next(tp for tp in tps if "lustre" in tp.path)
      self.assertEqual(target_tp.state, db_schema.TierPathState.READY)
      self.assertIsNotNone(target_tp.ready_at)
      self.assertIsNotNone(target_tp.expires_at)


if __name__ == "__main__":
  absltest.main()
