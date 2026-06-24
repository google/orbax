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
from unittest import mock
import uuid

from absl import logging
from absl.testing import absltest
import aiosqlite  # pylint: disable=unused-import
import greenlet  # pylint: disable=unused-import
from orbax.checkpoint.experimental.tiering_service import db_lib
from orbax.checkpoint.experimental.tiering_service import db_schema
from orbax.checkpoint.experimental.tiering_service import gcp_storage_client
from orbax.checkpoint.experimental.tiering_service import job_worker
from orbax.checkpoint.experimental.tiering_service import server_config
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import sessionmaker


class DummyGcpParallelstoreClient(gcp_storage_client.GCPStorageClient):
  """Dummy implementation of GCPStorageClient for testing."""

  def __init__(self):
    """Initializes the dummy client with empty operations list."""
    super().__init__()
    self.operations = {}

  async def trigger_copy(
      self,
      request_id: str,
      source_path: str,
      destination_path: str,
  ) -> str:
    """Triggers copy in progress."""
    self.operations[request_id] = {
        "status": gcp_storage_client.OperationStatus.IN_PROGRESS,
        "progress": 0,
        "type": "copy",
    }
    logging.info(
        "Dummy triggered copy %s -> %s, request_id: %s",
        source_path,
        destination_path,
        request_id,
    )
    return request_id

  async def poll_operation(
      self,
      request_id: str,
  ) -> gcp_storage_client.Result:
    """Polls the status of the specified GCP operation."""
    op = self.operations.get(request_id)
    if not op:
      return gcp_storage_client.Result(
          status=gcp_storage_client.OperationStatus.FAILED,
          detail_info={"error": "Operation not found"},
      )

    if op["status"] == gcp_storage_client.OperationStatus.IN_PROGRESS:
      op["progress"] += 50  # Progress by 50% each poll
      if op["progress"] >= 100:
        op["status"] = gcp_storage_client.OperationStatus.SUCCESS

    return gcp_storage_client.Result(
        status=op["status"],
        detail_info={
            "bytes_copied": op["progress"] * 1000,
            "total_bytes": 100000,
        },
    )


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
    # Use in-memory shared SQLite for testing to avoid file locking conflicts
    db_name = f"testdb_{uuid.uuid4().hex}"
    self.config.db_connection_str = (
        f"sqlite+aiosqlite:///file:{db_name}?mode=memory&cache=shared&uri=true"
    )

    self.engine = db_lib.get_async_engine(self.config)
    # Open and keep a connection alive to prevent SQLite from deleting the
    # in-memory shared database when engines are disposed/recreated.
    self._keep_alive_conn = await self.engine.connect()

    await db_lib.async_initialize_db(self.config)
    self.session_maker = sessionmaker(
        self.engine, expire_on_commit=False, class_=AsyncSession
    )
    self.gcp_client = DummyGcpParallelstoreClient()
    self.determine_client_mock = self.enter_context(
        mock.patch.object(gcp_storage_client, "determine_client", autospec=True)
    )
    self.determine_client_mock.return_value = self.gcp_client

    # Short poll interval for fast tests
    self.worker = job_worker.TieringServiceWorker(
        self.session_maker,
        self.config,
        lease_duration_seconds=2,  # Short lease for testing expiration
        poll_interval_seconds=1,
    )
    self.random_patcher = mock.patch("random.uniform", return_value=0.01)
    self.random_patcher.start()

  async def asyncTearDown(self):
    await self.worker.stop()
    if hasattr(self, "random_patcher"):
      self.random_patcher.stop()
    if hasattr(self, "_keep_alive_conn"):
      await self._keep_alive_conn.close()
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

    # Wait for job to trigger with timeout
    for _ in range(10):
      await asyncio.sleep(0.5)
      async with self.session_maker() as session:
        result = await session.execute(select(db_schema.AssetJob))
        job = result.scalars().first()
        if (
            job.status == db_schema.JobStatus.JOB_STATUS_PROCESSING
            and job.transfer_status is not None
            and "request_id" in job.transfer_status
        ):
          break

    await self.worker.stop()

    async with self.session_maker() as session:
      # Verify job status transitions to PROCESSING and has transfer metadata
      result = await session.execute(select(db_schema.AssetJob))
      job = result.scalars().first()
      self.assertEqual(job.status, db_schema.JobStatus.JOB_STATUS_PROCESSING)
      self.assertIsNone(job.completed_at)
      self.assertEqual(job.worker_host, self.worker._hostname)
      self.assertEqual(job.worker_pid, self.worker._pid)

      self.assertIsNotNone(job.transfer_status)
      self.assertIn("request_id", job.transfer_status)
      self.assertEqual(
          job.transfer_status["client_type"], "DummyGcpParallelstoreClient"
      )

      # Verify target TierPath remains PENDING (since it is not yet completed
      # by polling)
      result_tp = await session.execute(
          select(db_schema.TierPath).where(
              db_schema.TierPath.asset_uuid == "asset-1"
          )
      )
      tps = result_tp.scalars().all()
      target_tp = next(tp for tp in tps if "lustre" in tp.path)
      self.assertEqual(target_tp.state, db_schema.TierPathState.IN_PROGRESS)
      self.assertIsNone(target_tp.ready_at)

  async def test_concurrency_limit_respected(self):
    async with self.session_maker() as session:
      result = await session.execute(select(db_schema.StorageBackend))
      backends = result.scalars().all()
      lustre_a = next(b for b in backends if b.zone == "us-central1-a")
      gcs = next(b for b in backends if b.region == "us-central1")

      # Create 2 jobs targeting the SAME backend (Lustre A)
      await self._create_asset_and_job(
          session,
          asset_uuid="asset-1",
          path="path/1",
          target_backend_id=lustre_a.id,
          source_backend_id=gcs.id,
      )
      await self._create_asset_and_job(
          session,
          asset_uuid="asset-2",
          path="path/2",
          target_backend_id=lustre_a.id,
          source_backend_id=gcs.id,
      )

    # Mock poll_operation to return IN_PROGRESS to prevent job completion
    with mock.patch.object(
        self.gcp_client,
        "poll_operation",
        return_value=gcp_storage_client.Result(
            status=gcp_storage_client.OperationStatus.IN_PROGRESS,
            detail_info={"bytes_copied": 0, "total_bytes": 100000},
        ),
    ):
      await self.worker.start()
      # Wait for at least one job to transition to PROCESSING
      for _ in range(50):
        async with self.session_maker() as session:
          result = await session.execute(select(db_schema.AssetJob))
          jobs_list = result.scalars().all()
          if any(
              j.status == db_schema.JobStatus.JOB_STATUS_PROCESSING
              for j in jobs_list
          ):
            break
        await asyncio.sleep(0.1)
      await self.worker.stop()

    async with self.session_maker() as session:
      result = await session.execute(
          select(db_schema.AssetJob).order_by(db_schema.AssetJob.id)
      )
      jobs_list = result.scalars().all()

      # One should be PROCESSING, the other still QUEUED
      self.assertEqual(
          jobs_list[0].status, db_schema.JobStatus.JOB_STATUS_PROCESSING
      )
      self.assertEqual(
          jobs_list[1].status, db_schema.JobStatus.JOB_STATUS_QUEUED
      )

  async def test_different_backends_concurrency(self):
    async with self.session_maker() as session:
      result = await session.execute(select(db_schema.StorageBackend))
      backends = result.scalars().all()
      lustre_a = next(b for b in backends if b.zone == "us-central1-a")
      lustre_b = next(b for b in backends if b.zone == "us-central1-b")
      gcs = next(b for b in backends if b.region == "us-central1")

      # Create 2 jobs targeting DIFFERENT backends (Lustre A and Lustre B)
      await self._create_asset_and_job(
          session,
          asset_uuid="asset-1",
          path="path/1",
          target_backend_id=lustre_a.id,
          source_backend_id=gcs.id,
      )
      await self._create_asset_and_job(
          session,
          asset_uuid="asset-2",
          path="path/2",
          target_backend_id=lustre_b.id,
          source_backend_id=gcs.id,
      )

    # Mock poll_operation to return IN_PROGRESS to prevent job completion
    with mock.patch.object(
        self.gcp_client,
        "poll_operation",
        return_value=gcp_storage_client.Result(
            status=gcp_storage_client.OperationStatus.IN_PROGRESS,
            detail_info={"bytes_copied": 0, "total_bytes": 100000},
        ),
    ):
      await self.worker.start()
      # Wait for both jobs to transition to PROCESSING
      for _ in range(50):
        async with self.session_maker() as session:
          result = await session.execute(select(db_schema.AssetJob))
          jobs_list = result.scalars().all()
          if all(
              j.status == db_schema.JobStatus.JOB_STATUS_PROCESSING
              for j in jobs_list
          ):
            break
        await asyncio.sleep(0.1)
      await self.worker.stop()

    async with self.session_maker() as session:
      result = await session.execute(select(db_schema.AssetJob))
      jobs_list = result.scalars().all()

      # Both should be PROCESSING because they target different backends
      self.assertEqual(
          jobs_list[0].status, db_schema.JobStatus.JOB_STATUS_PROCESSING
      )
      self.assertEqual(
          jobs_list[1].status, db_schema.JobStatus.JOB_STATUS_PROCESSING
      )

  async def test_job_trigger_failure(self):
    async with self.session_maker() as session:
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

    # Mock trigger_copy to throw an error
    with mock.patch.object(
        self.gcp_client,
        "trigger_copy",
        autospec=True,
        side_effect=RuntimeError("Mocked trigger failure"),
    ):
      await self.worker.start()
      # Wait for job to fail
      for _ in range(5):
        await asyncio.sleep(0.5)
        async with self.session_maker() as session:
          result = await session.execute(select(db_schema.AssetJob))
          job = result.scalars().first()
          if job.status == db_schema.JobStatus.JOB_STATUS_FAILED:
            break
      await self.worker.stop()

    async with self.session_maker() as session:
      result = await session.execute(select(db_schema.AssetJob))
      job = result.scalars().first()
      self.assertEqual(job.status, db_schema.JobStatus.JOB_STATUS_FAILED)
      self.assertIsNotNone(job.completed_at)
      self.assertIn(
          "Failed to trigger transfer: Mocked trigger failure",
          job.transfer_status.get("error", ""),
      )

      # Verify target TierPath is marked FAILED
      result_tp = await session.execute(
          select(db_schema.TierPath).where(
              db_schema.TierPath.asset_uuid == "asset-1"
          )
      )
      tps = result_tp.scalars().all()
      dest_tp = next(tp for tp in tps if "my-bucket" not in tp.path)
      self.assertEqual(dest_tp.state, db_schema.TierPathState.FAILED)


if __name__ == "__main__":
  absltest.main()
