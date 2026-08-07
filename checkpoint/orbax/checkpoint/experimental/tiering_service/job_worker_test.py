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
from orbax.checkpoint.experimental.tiering_service import assets
from orbax.checkpoint.experimental.tiering_service import db_lib
from orbax.checkpoint.experimental.tiering_service import db_schema
from orbax.checkpoint.experimental.tiering_service import gcp_storage_client
from orbax.checkpoint.experimental.tiering_service import job_worker
from orbax.checkpoint.experimental.tiering_service import server_config
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.pool import NullPool


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

  async def delete_path(self, path: str) -> None:
    """Mock delete_path."""
    logging.info("Dummy deleted path: %s", path)


class TieringServiceWorkerTest(
    absltest.TestCase, unittest.IsolatedAsyncioTestCase
):

  async def _wait_for_condition(
      self, check_fn, max_attempts=50, sleep_time=0.1
  ) -> None:
    """Helper to poll a condition with database lock retries."""
    for i in range(max_attempts):
      try:
        if await check_fn():
          return
      except Exception as e:  # pylint: disable=broad-exception-caught
        if "database table is locked" in str(e):
          logging.info(
              "Database locked during poll, retrying... (attempt %d)", i
          )
        else:
          raise
      await asyncio.sleep(sleep_time)
    # Final check without exception catching
    if not await check_fn():
      self.fail("Condition not met after polling timeout")

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

    self.engine = db_lib.get_async_engine(self.config, poolclass=NullPool)
    # Open and keep a connection alive to prevent SQLite from deleting the
    # in-memory shared database when engines are disposed/recreated.
    self._keep_alive_conn = await self.engine.connect()

    await db_lib.async_initialize_db(self.config)
    self.session_maker = async_sessionmaker(self.engine, expire_on_commit=False)
    self.gcp_client = DummyGcpParallelstoreClient()
    self.determine_client_mock = self.enter_context(
        mock.patch.object(gcp_storage_client, "determine_client", autospec=True)
    )
    self.determine_client_mock.return_value = self.gcp_client

    self.determine_delete_client_mock = self.enter_context(
        mock.patch.object(
            gcp_storage_client, "determine_delete_client", autospec=True
        )
    )
    self.determine_delete_client_mock.return_value = self.gcp_client

    def mock_get_client(status_dict, project=None, service_account=None):
      del project, service_account  # Unused
      client_type = status_dict.get("client_type")
      if client_type not in {
          "GcsToGcsClient",
          "GcsToGcsTransferClient",
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
      request_type=db_schema.RequestType.REQUEST_TYPE_COPY,
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

    if request_type == db_schema.RequestType.REQUEST_TYPE_DELETE_FROM_ALL_TIERS:
      target_tp_id = None
    else:
      target_tp_id = target_tp.id

    job = db_schema.AssetJob(
        asset_uuid=asset_uuid,
        request_type=request_type,
        status=job_status,
        target_tier_path_id=target_tp_id,
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

    async def _check():
      async with self.session_maker() as session:
        result = await session.execute(select(db_schema.AssetJob))
        job = result.scalars().first()
        if job is None:
          return False
        return job.status == db_schema.JobStatus.JOB_STATUS_COMPLETED

    await self._wait_for_condition(_check, max_attempts=10, sleep_time=1.0)

    await self.worker.stop()

    async with self.session_maker() as session:
      # Verify job status transitions to COMPLETED
      result = await session.execute(select(db_schema.AssetJob))
      job = result.scalars().first()
      assert job is not None
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

      async def _check():
        async with self.session_maker() as session:
          result = await session.execute(select(db_schema.AssetJob))
          jobs_list = result.scalars().all()
          return any(
              j.status == db_schema.JobStatus.JOB_STATUS_PROCESSING
              for j in jobs_list
          )

      await self._wait_for_condition(_check, max_attempts=50, sleep_time=0.1)
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

      async def _check():
        async with self.session_maker() as session:
          result = await session.execute(select(db_schema.AssetJob))
          jobs_list = result.scalars().all()
          return all(
              j.status == db_schema.JobStatus.JOB_STATUS_PROCESSING
              for j in jobs_list
          )

      await self._wait_for_condition(_check, max_attempts=50, sleep_time=0.1)
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

      async def _check():
        async with self.session_maker() as session:
          result = await session.execute(select(db_schema.AssetJob))
          job = result.scalars().first()
          if job is None:
            return False
          return job.status == db_schema.JobStatus.JOB_STATUS_FAILED

      await self._wait_for_condition(_check, max_attempts=5, sleep_time=0.5)
      await self.worker.stop()

    async with self.session_maker() as session:
      result = await session.execute(select(db_schema.AssetJob))
      job = result.scalars().first()
      assert job is not None
      self.assertEqual(job.status, db_schema.JobStatus.JOB_STATUS_FAILED)
      self.assertIsNotNone(job.completed_at)
      assert job.transfer_status is not None
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

      async def _check():
        async with self.session_maker() as session:
          result = await session.execute(select(db_schema.AssetJob))
          job = result.scalars().first()
          if job is None:
            return False
          return job.status == db_schema.JobStatus.JOB_STATUS_FAILED

      await self._wait_for_condition(_check, max_attempts=10, sleep_time=1)
      await self.worker.stop()

    async with self.session_maker() as session:
      # Verify job status is FAILED, completed_at is set, and
      # target_tier_path_id is preserved
      result = await session.execute(select(db_schema.AssetJob))
      job = result.scalars().first()
      assert job is not None
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

    async def _check():
      async with self.session_maker() as session:
        result = await session.execute(select(db_schema.AssetJob))
        recovered_job = result.scalars().first()
        return recovered_job.status == db_schema.JobStatus.JOB_STATUS_COMPLETED  # pyrefly: ignore[missing-attribute]

    await self._wait_for_condition(_check, max_attempts=10, sleep_time=1)

    await self.worker.stop()

    async with self.session_maker() as session:
      result = await session.execute(select(db_schema.AssetJob))
      recovered_job = result.scalars().first()

      # The worker should have reclaimed it and eventually completed it
      # (via dummy client)
      self.assertEqual(
          recovered_job.status, db_schema.JobStatus.JOB_STATUS_COMPLETED  # pyrefly: ignore[missing-attribute]
      )
      self.assertIsNone(recovered_job.worker_host)  # pyrefly: ignore[missing-attribute]

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
          recovered_job.status, db_schema.JobStatus.JOB_STATUS_FAILED  # pyrefly: ignore[missing-attribute]
      )
      self.assertIn(
          "Unknown or missing client type: InvalidClientType",
          recovered_job.transfer_status.get("error", ""),  # pyrefly: ignore[missing-attribute]
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
          db_job.expiration_at.replace(tzinfo=None),  # pyrefly: ignore[missing-attribute]
          expected_expiration.replace(tzinfo=None),
      )

  async def test_poll_delete_job_extends_lease(self):
    async with self.session_maker() as session:
      result = await session.execute(select(db_schema.StorageBackend))
      backends = result.scalars().all()
      lustre_a = next(b for b in backends if b.zone == "us-central1-a")
      gcs = next(b for b in backends if b.region == "us-central1")

      job, _ = await self._create_asset_and_job(
          session,
          asset_uuid="asset-delete-lease",
          path="path/delete/lease",
          target_backend_id=lustre_a.id,
          source_backend_id=gcs.id,
          job_status=db_schema.JobStatus.JOB_STATUS_PROCESSING,
          request_type=db_schema.RequestType.REQUEST_TYPE_DELETE_FROM_ALL_TIERS,
      )
      job.worker_host = self.worker._hostname
      job.worker_pid = self.worker._pid
      initial_expiration = datetime.datetime.now(
          datetime.timezone.utc
      ) - datetime.timedelta(seconds=10)
      job.expiration_at = initial_expiration
      await session.commit()

    now = datetime.datetime.now(datetime.timezone.utc)
    # Trigger polling manually
    await self.worker._poll_single_job(job, now)

    # Verify expiration_at is now updated to now + lease_duration
    async with self.session_maker() as session:
      db_job = await session.get(db_schema.AssetJob, job.id)
      expected_expiration = now + self.worker._lease_duration
      self.assertEqual(
          db_job.expiration_at.replace(tzinfo=None),  # pyrefly: ignore[missing-attribute]
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

      async def _check():
        nonlocal custom_metric_val
        async with self.session_maker() as session:
          db_job = await session.get(db_schema.AssetJob, job.id)
          assert db_job is not None
          if db_job.transfer_status and db_job.transfer_status.get(
              "custom_metric"
          ):
            custom_metric_val = db_job.transfer_status.get("custom_metric")
            return True
          return False

      await self._wait_for_condition(_check, max_attempts=20, sleep_time=0.5)

      await self.worker.stop()
      self.assertEqual(custom_metric_val, "custom_value")

  async def test_delete_from_instance_job_lifecycle(self):
    async with self.session_maker() as session:
      result = await session.execute(select(db_schema.StorageBackend))
      backends = result.scalars().all()
      gcs = next(b for b in backends if b.region == "us-central1")

      # Create Asset and ready TierPath
      asset = db_schema.Asset(
          asset_uuid="asset-del-inst-1",
          path="path/del/inst/1",
          user="test-user",
          state=db_schema.AssetState.ASSET_STATE_STORED,
      )
      session.add(asset)
      tp = db_schema.TierPath(
          asset_uuid="asset-del-inst-1",
          storage_backend_id=gcs.id,
          path="gs://my-bucket/path/del/inst/1",
          ready_at=datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc),
          state=db_schema.TierPathState.READY,
      )
      session.add(tp)
      await session.commit()

      # Queue DELETE_FROM_INSTANCE job
      db_job = db_schema.AssetJob(
          asset_uuid=asset.asset_uuid,
          target_tier_path_id=tp.id,
          request_type=db_schema.RequestType.REQUEST_TYPE_DELETE_FROM_INSTANCE,
          status=db_schema.JobStatus.JOB_STATUS_QUEUED,
      )
      session.add(db_job)
      await session.commit()

    # Start worker - it should pick up the job, transition state to
    # DELETE_IN_PROCESS, and complete the deletion.
    await self.worker.start()

    # Verify that the tier path has transitioned to DELETE_IN_PROCESS
    # almost immediately (or upon completion).
    async def _check():
      async with self.session_maker() as session:
        stmt = (
            select(db_schema.TierPath)
            .where(db_schema.TierPath.id == tp.id)
            .execution_options(populate_existing=True)
        )
        res = await session.execute(stmt)
        db_tp = res.scalars().first()
        return db_tp and db_tp.state == db_schema.TierPathState.DELETED

    await self._wait_for_condition(_check, max_attempts=20, sleep_time=0.2)

    await self.worker.stop()

    # Final DB assertion checks
    async with self.session_maker() as session:
      async with session.begin():
        stmt_tp = (
            select(db_schema.TierPath)
            .where(db_schema.TierPath.id == tp.id)
            .execution_options(populate_existing=True)
        )
        res_tp = await session.execute(stmt_tp)
        db_tp = res_tp.scalars().first()
        self.assertEqual(db_tp.state, db_schema.TierPathState.DELETED)  # pyrefly: ignore[missing-attribute]
        self.assertIsNone(db_tp.ready_at)  # pyrefly: ignore[missing-attribute]

        stmt_job = (
            select(db_schema.AssetJob)
            .where(db_schema.AssetJob.id == db_job.id)
            .execution_options(populate_existing=True)
        )
        res_job = await session.execute(stmt_job)
        db_job_res = res_job.scalars().first()
        self.assertEqual(
            db_job_res.status, db_schema.JobStatus.JOB_STATUS_COMPLETED  # pyrefly: ignore[missing-attribute]
        )

  async def test_delete_from_all_tiers_job_lifecycle(self):
    async with self.session_maker() as session:
      result = await session.execute(select(db_schema.StorageBackend))
      backends = result.scalars().all()
      gcs = next(b for b in backends if b.region == "us-central1")
      lustre = next(b for b in backends if b.zone == "us-central1-a")

      # Create Asset and 2 ready TierPaths (Lustre + GCS)
      asset = db_schema.Asset(
          asset_uuid="asset-del-all-1",
          path="path/del/all/1",
          user="test-user",
          state=db_schema.AssetState.ASSET_STATE_STORED,
      )
      session.add(asset)
      tp_gcs = db_schema.TierPath(
          asset_uuid="asset-del-all-1",
          storage_backend_id=gcs.id,
          path="gs://my-bucket/path/del/all/1",
          ready_at=datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc),
          state=db_schema.TierPathState.READY,
      )
      tp_lustre = db_schema.TierPath(
          asset_uuid="asset-del-all-1",
          storage_backend_id=lustre.id,
          path="/mnt/lustre/path/del/all/1",
          ready_at=datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc),
          state=db_schema.TierPathState.READY,
      )
      session.add(tp_gcs)
      session.add(tp_lustre)
      await session.commit()

      # Queue DELETE_FROM_ALL_TIERS job
      await assets.queue_delete_asset_job(session, asset)

      # Find queued job
      stmt = select(db_schema.AssetJob).where(
          db_schema.AssetJob.asset_uuid == asset.asset_uuid,
          db_schema.AssetJob.request_type
          == db_schema.RequestType.REQUEST_TYPE_DELETE_FROM_ALL_TIERS,
      )
      res_job = await session.execute(stmt)
      db_job = res_job.scalars().first()

    # Start worker
    await self.worker.start()

    # Verify both paths deleted
    async def _check():
      async with self.session_maker() as session:
        stmt = (
            select(db_schema.TierPath)
            .where(db_schema.TierPath.id == tp_gcs.id)
            .execution_options(populate_existing=True)
        )
        res = await session.execute(stmt)
        db_tp = res.scalars().first()
        return db_tp and db_tp.state == db_schema.TierPathState.DELETED

    await self._wait_for_condition(_check, max_attempts=20, sleep_time=0.2)

    await self.worker.stop()

    # Assertions
    async with self.session_maker() as session:
      async with session.begin():
        stmt_tp_g = (
            select(db_schema.TierPath)
            .where(db_schema.TierPath.id == tp_gcs.id)
            .execution_options(populate_existing=True)
        )
        res_tp_g = await session.execute(stmt_tp_g)
        db_tp_g = res_tp_g.scalars().first()
        self.assertEqual(db_tp_g.state, db_schema.TierPathState.DELETED)  # pyrefly: ignore[missing-attribute]

        stmt_tp_l = (
            select(db_schema.TierPath)
            .where(db_schema.TierPath.id == tp_lustre.id)
            .execution_options(populate_existing=True)
        )
        res_tp_l = await session.execute(stmt_tp_l)
        db_tp_l = res_tp_l.scalars().first()
        self.assertEqual(db_tp_l.state, db_schema.TierPathState.DELETED)  # pyrefly: ignore[missing-attribute]

        stmt_asset = (
            select(db_schema.Asset)
            .where(db_schema.Asset.asset_uuid == asset.asset_uuid)
            .execution_options(populate_existing=True)
        )
        res_asset = await session.execute(stmt_asset)
        db_asset = res_asset.scalars().first()
        self.assertEqual(
            db_asset.state, db_schema.AssetState.ASSET_STATE_DELETED  # pyrefly: ignore[missing-attribute]
        )

        stmt_job = (
            select(db_schema.AssetJob)
            .where(db_schema.AssetJob.id == db_job.id)  # pyrefly: ignore[missing-attribute]
            .execution_options(populate_existing=True)
        )
        res_job = await session.execute(stmt_job)
        db_job_res = res_job.scalars().first()
        self.assertEqual(
            db_job_res.status, db_schema.JobStatus.JOB_STATUS_COMPLETED  # pyrefly: ignore[missing-attribute]
        )


if __name__ == "__main__":
  absltest.main()
