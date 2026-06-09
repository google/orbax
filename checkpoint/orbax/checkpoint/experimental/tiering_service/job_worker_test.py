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

from absl.testing import absltest
from absl.testing import parameterized
import google.auth
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
    self.gcp_client = job_worker.DummyGcpParallelstoreClient()
    # Short poll interval for fast tests
    self.worker = job_worker.TieringServiceWorker(
        self.session_maker,
        self.config,
        gcp_client=self.gcp_client,
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
    """Helper to create an asset, its tier paths, and a prefetch job.

    Args:
      session: The database session.
      asset_uuid: Unique identifier of the asset.
      path: Relative path of the asset.
      target_backend_id: The ID of the target storage backend.
      source_backend_id: The ID of the source storage backend.
      job_status: The initial status of the prefetch job.

    Returns:
      A tuple of (AssetJob, TierPath) created.
    """
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

    # Wait for job to complete with timeout
    for _ in range(10):
      await asyncio.sleep(1)
      async with self.session_maker() as session:
        result = await session.execute(select(db_schema.AssetJob))
        job = result.scalars().first()
        if job.status == db_schema.JobStatus.JOB_STATUS_COMPLETED:
          break

    await self.worker.stop()

    async with self.session_maker() as session:
      # Verify job status transitions to COMPLETED (since dummy client
      # progresses fast)
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
      self.assertIsNotNone(target_tp.ready_at)
      self.assertIsNotNone(target_tp.expires_at)

  async def test_job_failure_clean_up_target_tier_path(self):
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

    # Mock poll_operation to return a failure
    with mock.patch.object(
        self.gcp_client,
        "poll_operation",
        autospec=True,
        return_value=job_worker.Result(
            status=job_worker.Status.COMPLETED,
            detail_info={"error": "Mocked GCP error"},
        ),
    ):
      await self.worker.start()
      # Wait for job to fail
      for _ in range(10):
        await asyncio.sleep(1)
        async with self.session_maker() as session:
          result = await session.execute(select(db_schema.AssetJob))
          job = result.scalars().first()
          if job.status == db_schema.JobStatus.JOB_STATUS_FAILED:
            break
      await self.worker.stop()

    async with self.session_maker() as session:
      # Verify job status is FAILED, completed_at is set, and
      # target_tier_path_id is NULL
      result = await session.execute(select(db_schema.AssetJob))
      job = result.scalars().first()
      self.assertEqual(job.status, db_schema.JobStatus.JOB_STATUS_FAILED)
      self.assertIsNotNone(job.completed_at)
      self.assertIsNone(job.target_tier_path_id)
      self.assertIsNone(job.worker_host)
      self.assertIsNone(job.worker_pid)

      # Verify target TierPath is deleted
      result_tp = await session.execute(
          select(db_schema.TierPath).where(
              db_schema.TierPath.asset_uuid == "asset-1"
          )
      )
      tps = result_tp.scalars().all()
      # Only source (GCS) TierPath should exist
      self.assertLen(tps, 1)
      self.assertTrue(any("my-bucket" in tp.path for tp in tps))

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

    # Dummy GCP client to keep operations RUNNING so we can check concurrency
    with mock.patch.object(
        self.gcp_client,
        "poll_operation",
        autospec=True,
        return_value=job_worker.Result(
            status=job_worker.Status.IN_PROGRESS,
            detail_info={"bytes_copied": 500},
        ),
    ):
      await self.worker.start()
      await asyncio.sleep(2)
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

    with mock.patch.object(
        self.gcp_client,
        "poll_operation",
        autospec=True,
        return_value=job_worker.Result(
            status=job_worker.Status.IN_PROGRESS,
            detail_info={"bytes_copied": 500},
        ),
    ):
      await self.worker.start()
      await asyncio.sleep(2)
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

  async def test_crash_recovery_on_lease_expiration(self):
    async with self.session_maker() as session:
      result = await session.execute(select(db_schema.StorageBackend))
      backends = result.scalars().all()
      lustre_a = next(b for b in backends if b.zone == "us-central1-a")
      gcs = next(b for b in backends if b.region == "us-central1")

      # Create a job that is already in PROCESSING state but has an
      # expired lease
      job, _ = await self._create_asset_and_job(
          session,
          asset_uuid="asset-1",
          path="path/1",
          target_backend_id=lustre_a.id,
          source_backend_id=gcs.id,
          job_status=db_schema.JobStatus.JOB_STATUS_PROCESSING,
      )
      # Set expired lease
      job.expiration_at = datetime.datetime(
          2020, 1, 1, tzinfo=datetime.timezone.utc
      )
      job.worker_host = "dead-host"
      job.worker_pid = 9999

      # Populate dummy operation in client
      op_id = "operations/import-dummy-id"
      self.gcp_client.operations[op_id] = {
          "status": job_worker.Status.IN_PROGRESS,
          "progress": 0,
          "type": "import",
      }
      job.transfer_status = {
          "request_id": op_id,
          "status": job_worker.Status.IN_PROGRESS.value,
      }
      await session.commit()

    # Start worker
    await self.worker.start()

    # Wait for job to complete with timeout
    for _ in range(10):
      await asyncio.sleep(1)
      async with self.session_maker() as session:
        result = await session.execute(select(db_schema.AssetJob))
        recovered_job = result.scalars().first()
        if recovered_job.status == db_schema.JobStatus.JOB_STATUS_COMPLETED:
          break

    await self.worker.stop()

    async with self.session_maker() as session:
      result = await session.execute(select(db_schema.AssetJob))
      recovered_job = result.scalars().first()

      # The worker should have reclaimed it and eventually completed it
      # (via dummy client)
      self.assertEqual(
          recovered_job.status, db_schema.JobStatus.JOB_STATUS_COMPLETED
      )
      self.assertIsNone(recovered_job.worker_host)


class GcpLustreClientTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  def setUp(self):
    super().setUp()
    self.client = job_worker.GcpLustreClient(
        project="test-project",
        location="us-central1-a",
        instance="test-instance",
    )
    # Mock credentials detection
    self._mock_default = self.enter_context(
        mock.patch.object(google.auth, "default", autospec=True)
    )
    self.mock_creds = mock.MagicMock()
    self.mock_creds.valid = True
    self.mock_creds.token = "fake-token"
    self._mock_default.return_value = (self.mock_creds, "test-project")

  @mock.patch("requests.post")
  async def test_trigger_import_success(self, mock_post):
    mock_resp = mock.MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "name": (
            "projects/test-project/locations/us-central1-a/operations/op-123"
        )
    }
    mock_post.return_value = mock_resp

    op_name = await self.client.trigger_import(
        request_id="req-id",
        source_path="gs://my-bucket/path",
        destination_path="/lustre/path",
    )
    self.assertEqual(
        op_name,
        "projects/test-project/locations/us-central1-a/operations/op-123",
    )
    mock_post.assert_called_once_with(
        (
            "https://lustre.googleapis.com/v1/projects/test-project"
            "/locations/us-central1-a/instances/test-instance:importData"
        ),
        json={
            "gcsPath": {"uri": "gs://my-bucket/path"},
            "lustrePath": {"path": "/lustre/path"},
            "requestId": "req-id",
        },
        headers={
            "Authorization": "Bearer fake-token",
            "Content-Type": "application/json",
        },
    )

  @mock.patch("requests.post")
  async def test_trigger_export_success(self, mock_post):
    mock_resp = mock.MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "name": (
            "projects/test-project/locations/us-central1-a/operations/op-456"
        )
    }
    mock_post.return_value = mock_resp

    op_name = await self.client.trigger_export(
        request_id="req-id",
        source_path="/lustre/path",
        destination_path="gs://my-bucket/path",
    )
    self.assertEqual(
        op_name,
        "projects/test-project/locations/us-central1-a/operations/op-456",
    )
    mock_post.assert_called_once_with(
        (
            "https://lustre.googleapis.com/v1/projects/test-project"
            "/locations/us-central1-a/instances/test-instance:exportData"
        ),
        json={
            "lustrePath": {"path": "/lustre/path"},
            "gcsPath": {"uri": "gs://my-bucket/path"},
            "requestId": "req-id",
        },
        headers={
            "Authorization": "Bearer fake-token",
            "Content-Type": "application/json",
        },
    )

  @mock.patch("requests.get")
  async def test_poll_operation_in_progress(self, mock_get):
    mock_resp = mock.MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "name": (
            "projects/test-project/locations/us-central1-a/operations/op-123"
        ),
        "done": False,
        "metadata": {"progress": 50},
    }
    mock_get.return_value = mock_resp

    result = await self.client.poll_operation(
        "projects/test-project/locations/us-central1-a/operations/op-123"
    )
    self.assertEqual(result.status, job_worker.Status.IN_PROGRESS)
    self.assertEqual(result.detail_info, {"progress": 50})

  @mock.patch("requests.get")
  async def test_poll_operation_completed_success(self, mock_get):
    mock_resp = mock.MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "name": (
            "projects/test-project/locations/us-central1-a/operations/op-123"
        ),
        "done": True,
        "response": {"bytesCopied": "1000"},
    }
    mock_get.return_value = mock_resp

    result = await self.client.poll_operation(
        "projects/test-project/locations/us-central1-a/operations/op-123"
    )
    self.assertEqual(result.status, job_worker.Status.COMPLETED)
    self.assertEqual(result.detail_info, {"bytesCopied": "1000"})

  @mock.patch("requests.get")
  async def test_poll_operation_completed_error(self, mock_get):
    mock_resp = mock.MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "name": (
            "projects/test-project/locations/us-central1-a/operations/op-123"
        ),
        "done": True,
        "error": {"code": 3, "message": "Failed to copy"},
    }
    mock_get.return_value = mock_resp

    result = await self.client.poll_operation(
        "projects/test-project/locations/us-central1-a/operations/op-123"
    )
    self.assertEqual(result.status, job_worker.Status.COMPLETED)
    self.assertEqual(
        result.detail_info, {"error": {"code": 3, "message": "Failed to copy"}}
    )


if __name__ == "__main__":
  absltest.main()
