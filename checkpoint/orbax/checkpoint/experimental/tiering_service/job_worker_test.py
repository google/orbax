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

    def mock_get_client(status_dict, project=None, service_account=None):
      del project, service_account  # Unused
      client_type = status_dict.get("client_type")
      if client_type not in {
          "GcsToGcsClient",
          "LustreToGcsClient",
          "GcsToLustreClient",
          "DummyGcpParallelstoreClient",
      }:
        raise ValueError(f"Unknown or missing client type: {client_type}")
      return self.gcp_client

    self.get_client_mock = self.enter_context(
        mock.patch.object(
            gcp_storage_client,
            "get_client_from_status",
            side_effect=mock_get_client,
        )
    )

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

    # Wait for job to complete with timeout
    for _ in range(10):
      await asyncio.sleep(1.0)
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
        return_value=gcp_storage_client.Result(
            status=gcp_storage_client.OperationStatus.FAILED,
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
      # target_tier_path_id is preserved
      result = await session.execute(select(db_schema.AssetJob))
      job = result.scalars().first()
      self.assertEqual(job.status, db_schema.JobStatus.JOB_STATUS_FAILED)
      self.assertIsNotNone(job.completed_at)
      self.assertIsNotNone(job.target_tier_path_id)
      self.assertIsNone(job.worker_host)
      self.assertIsNone(job.worker_pid)

      # Verify target TierPath is preserved but marked FAILED
      result_tp = await session.execute(
          select(db_schema.TierPath).where(
              db_schema.TierPath.asset_uuid == "asset-1"
          )
      )
      tps = result_tp.scalars().all()
      # Both source (GCS) and target (Lustre) TierPaths should exist
      self.assertLen(tps, 2)
      dest_tp = next(tp for tp in tps if "my-bucket" not in tp.path)
      self.assertEqual(dest_tp.state, db_schema.TierPathState.FAILED)

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
          "status": gcp_storage_client.OperationStatus.IN_PROGRESS,
          "progress": 0,
          "type": "import",
      }
      job.transfer_status = {
          "request_id": op_id,
          "status": gcp_storage_client.OperationStatus.IN_PROGRESS.value,
          "client_type": "GcsToLustreClient",
          "zone": "us-central1-a",
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

  async def test_poll_permanent_logic_error_failing_job(self):
    async with self.session_maker() as session:
      result = await session.execute(select(db_schema.StorageBackend))
      backends = result.scalars().all()
      lustre_a = next(b for b in backends if b.zone == "us-central1-a")
      gcs = next(b for b in backends if b.region == "us-central1")

      job, _ = await self._create_asset_and_job(
          session,
          asset_uuid="asset-err-1",
          path="path/err/1",
          target_backend_id=lustre_a.id,
          source_backend_id=gcs.id,
          job_status=db_schema.JobStatus.JOB_STATUS_PROCESSING,
      )

      # Set up ownership for this worker
      job.worker_host = self.worker._hostname
      job.worker_pid = self.worker._pid
      # Set up an invalid client type in transfer_status to induce a ValueError
      job.transfer_status = {
          "request_id": "operations/dummy-op",
          "status": gcp_storage_client.OperationStatus.IN_PROGRESS.value,
          "client_type": "InvalidClientType",
      }
      await session.commit()

    # Trigger polling manually
    await self.worker._poll_active_jobs()

    # Verify that the job is marked as FAILED
    async with self.session_maker() as session:
      result = await session.execute(select(db_schema.AssetJob))
      recovered_job = result.scalars().first()
      self.assertEqual(
          recovered_job.status, db_schema.JobStatus.JOB_STATUS_FAILED
      )
      self.assertIn(
          "Unknown or missing client type: InvalidClientType",
          recovered_job.transfer_status.get("error", ""),
      )

  async def test_update_job_status_after_poll_acquires_lock(self):
    async with self.session_maker() as session:
      result = await session.execute(select(db_schema.StorageBackend))
      backends = result.scalars().all()
      lustre_a = next(b for b in backends if b.zone == "us-central1-a")
      gcs = next(b for b in backends if b.region == "us-central1")

      job, _ = await self._create_asset_and_job(
          session,
          asset_uuid="asset-lock-1",
          path="path/lock/1",
          target_backend_id=lustre_a.id,
          source_backend_id=gcs.id,
          job_status=db_schema.JobStatus.JOB_STATUS_PROCESSING,
      )
      job.worker_host = self.worker._hostname
      job.worker_pid = self.worker._pid
      await session.commit()

    original_get = AsyncSession.get
    get_calls = []
    async def mock_get(self_session, entity, ident, **kwargs):
      get_calls.append((entity, ident, kwargs))
      return await original_get(self_session, entity, ident, **kwargs)

    with mock.patch.object(AsyncSession, "get", mock_get):
      gcp_result = gcp_storage_client.Result(
          status=gcp_storage_client.OperationStatus.IN_PROGRESS,
          detail_info={"bytes_copied": 100, "total_bytes": 1000},
      )
      status_dict = {
          "client_type": "GcsToLustreClient",
          "zone": "us-central1-a",
      }
      now = datetime.datetime.now(datetime.timezone.utc)
      await self.worker._update_job_status_after_poll(
          job, gcp_result, status_dict, now
      )

    self.assertNotEmpty(get_calls)
    entity, ident, kwargs = get_calls[0]
    self.assertEqual(entity, db_schema.AssetJob)
    self.assertEqual(ident, job.id)
    self.assertTrue(kwargs.get("with_for_update"))

  async def test_update_job_status_after_poll_extends_lease(self):
    async with self.session_maker() as session:
      result = await session.execute(select(db_schema.StorageBackend))
      backends = result.scalars().all()
      lustre_a = next(b for b in backends if b.zone == "us-central1-a")
      gcs = next(b for b in backends if b.region == "us-central1")

      job, _ = await self._create_asset_and_job(
          session,
          asset_uuid="asset-lease-direct",
          path="path/lease/direct",
          target_backend_id=lustre_a.id,
          source_backend_id=gcs.id,
          job_status=db_schema.JobStatus.JOB_STATUS_PROCESSING,
      )
      job.worker_host = self.worker._hostname
      job.worker_pid = self.worker._pid
      # Set initial expiration to some past time
      initial_expiration = datetime.datetime.now(
          datetime.timezone.utc
      ) - datetime.timedelta(seconds=10)
      job.expiration_at = initial_expiration
      await session.commit()

    gcp_result = gcp_storage_client.Result(
        status=gcp_storage_client.OperationStatus.IN_PROGRESS,
        detail_info={"bytes_copied": 100, "total_bytes": 1000},
    )
    status_dict = {
        "client_type": "GcsToLustreClient",
        "zone": "us-central1-a",
    }
    now = datetime.datetime.now(datetime.timezone.utc)

    # Call directly
    await self.worker._update_job_status_after_poll(
        job, gcp_result, status_dict, now
    )

    # Verify expiration_at is now updated to now + lease_duration
    async with self.session_maker() as session:
      db_job = await session.get(db_schema.AssetJob, job.id)
      expected_expiration = now + self.worker._lease_duration
      self.assertEqual(
          db_job.expiration_at.replace(tzinfo=None),
          expected_expiration.replace(tzinfo=None),
      )

  async def test_custom_transfer_status_details_preserved(self):
    async with self.session_maker() as session:
      result = await session.execute(select(db_schema.StorageBackend))
      backends = result.scalars().all()
      lustre_a = next(b for b in backends if b.zone == "us-central1-a")
      gcs = next(b for b in backends if b.region == "us-central1")

      job, _ = await self._create_asset_and_job(
          session,
          asset_uuid="asset-custom-1",
          path="path/custom/1",
          target_backend_id=lustre_a.id,
          source_backend_id=gcs.id,
      )

    with mock.patch.object(
        self.gcp_client,
        "poll_operation",
        return_value=gcp_storage_client.Result(
            status=gcp_storage_client.OperationStatus.IN_PROGRESS,
            detail_info={
                "bytes_copied": 500,
                "total_bytes": 1000,
                "custom_metric": "custom_value",
            },
        ),
    ):
      await self.worker.start()

      custom_metric_val = None
      for _ in range(20):
        await asyncio.sleep(0.5)
        async with self.session_maker() as session:
          db_job = await session.get(db_schema.AssetJob, job.id)
          if (
              db_job.transfer_status
              and db_job.transfer_status.get("custom_metric")
          ):
            custom_metric_val = db_job.transfer_status.get("custom_metric")
            break

      await self.worker.stop()
      self.assertEqual(custom_metric_val, "custom_value")


if __name__ == "__main__":
  absltest.main()
