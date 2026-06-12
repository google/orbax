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

"""Checkpoint Tiering Service (CTS) worker to handle AssetJobs.

Consumes queued AssetJobs and manages the asynchronous data movement (eg. Lustre
- GCS import/export).
"""

import asyncio
import datetime
import os
import socket
from typing import Any
from absl import logging
from orbax.checkpoint.experimental.tiering_service import assets
from orbax.checkpoint.experimental.tiering_service import db_schema
from orbax.checkpoint.experimental.tiering_service.gcp_storage_client import GCPStorageClient
from orbax.checkpoint.experimental.tiering_service.gcp_storage_client import GcsToGcsClient
from orbax.checkpoint.experimental.tiering_service.gcp_storage_client import GcsToLustreClient
from orbax.checkpoint.experimental.tiering_service.gcp_storage_client import LustreToGcsClient
from orbax.checkpoint.experimental.tiering_service.gcp_storage_client import OperationStatus
from orbax.checkpoint.experimental.tiering_service.gcp_storage_client import TransferContext
from orbax.checkpoint.experimental.tiering_service.proto import tiering_service_pb2
import sqlalchemy
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
import sqlalchemy.orm


class TieringServiceWorker:
  """Background worker that processes AssetJobs."""

  def __init__(
      self,
      session_maker: Any,
      config: tiering_service_pb2.ServerConfig,
      gcp_client: GCPStorageClient | None = None,
      *,
      lease_duration_seconds: int = 60,
      poll_interval_seconds: int = 5,
  ):
    """Initializes the background worker.

    Args:
      session_maker: A callable that returns a database session.
      config: The server configuration.
      gcp_client: Client to interact with GCP Parallelstore.
      lease_duration_seconds: Duration of the lease acquired for jobs.
      poll_interval_seconds: Polling interval for checking job status.
    """
    self._session_maker = session_maker
    self._config = config
    self._gcp_client = gcp_client
    self._lease_duration = datetime.timedelta(seconds=lease_duration_seconds)
    self._poll_interval = poll_interval_seconds
    self._hostname = socket.gethostname()
    self._pid = os.getpid()
    self._tasks = []
    self._shutdown_event = asyncio.Event()

  def _get_client_for_backends(
      self,
      source_backend: db_schema.StorageBackend,
      target_backend: db_schema.StorageBackend,
  ) -> GCPStorageClient:
    """Returns the appropriate GCP client for the given transfer backends."""
    if self._gcp_client is not None:
      return self._gcp_client

    project = self._config.gcp_project or None
    service_account = self._config.service_account or None

    if (
        source_backend.backend_type == db_schema.BackendType.BACKEND_TYPE_GCS
        and target_backend.backend_type
        == db_schema.BackendType.BACKEND_TYPE_GCS
    ):
      return GcsToGcsClient(project=project, service_account=service_account)

    if (
        source_backend.backend_type == db_schema.BackendType.BACKEND_TYPE_LUSTRE
        and target_backend.backend_type
        == db_schema.BackendType.BACKEND_TYPE_GCS
    ):
      location = source_backend.zone
      instance = f"lustre-{location}"
      return LustreToGcsClient(
          instance=instance,
          location=location,
          project=project,
          service_account=service_account,
      )

    if (
        source_backend.backend_type == db_schema.BackendType.BACKEND_TYPE_GCS
        and target_backend.backend_type
        == db_schema.BackendType.BACKEND_TYPE_LUSTRE
    ):
      location = target_backend.zone
      instance = f"lustre-{location}"
      return GcsToLustreClient(
          instance=instance,
          location=location,
          project=project,
          service_account=service_account,
      )

    raise ValueError(
        f"Unsupported backend pair: {source_backend.backend_type} and"
        f" {target_backend.backend_type}"
    )

  def _get_client_for_job(self, job: db_schema.AssetJob) -> GCPStorageClient:
    """Returns the client for the given job."""
    if self._gcp_client is not None:
      return self._gcp_client

    target_tp = job.target_tier_path
    asset = job.asset
    source_tp = None
    for tp in asset.tier_paths:
      if tp.id != target_tp.id:
        source_tp = tp
        break

    if not source_tp or not target_tp:
      raise ValueError(
          f"Could not resolve source and target backends for job {job.id}"
      )

    return self._get_client_for_backends(
        source_tp.storage_backend, target_tp.storage_backend
    )

  async def start(self):
    """Starts the background worker loops."""
    logging.info(
        "Starting TieringServiceWorker on %s:%d", self._hostname, self._pid
    )
    self._shutdown_event.clear()
    self._tasks.append(asyncio.create_task(self._run_acquisition_loop()))
    self._tasks.append(asyncio.create_task(self._run_polling_loop()))

  async def stop(self):
    """Stops the background worker loops gracefully."""
    logging.info("Stopping TieringServiceWorker...")
    self._shutdown_event.set()
    if self._tasks:
      # Wait for tasks to cancel and finish
      for task in self._tasks:
        task.cancel()
      await asyncio.gather(*self._tasks, return_exceptions=True)
      self._tasks.clear()
    logging.info("TieringServiceWorker stopped.")

  async def _run_acquisition_loop(self):
    """Periodically acquires and triggers queued jobs."""
    while not self._shutdown_event.is_set():
      try:
        async with self._session_maker() as session:
          async with session.begin():
            job = await self._acquire_next_job(session)
            if job:
              logging.info("Acquired job %d", job.id)
              await self._process_job(session, job)
      except Exception:  # pylint: disable=broad-except
        logging.exception("Error in job acquisition loop.")
      await asyncio.sleep(self._poll_interval)

  async def _run_polling_loop(self):
    """Periodically polls status of active jobs owned by this worker."""
    while not self._shutdown_event.is_set():
      try:
        async with self._session_maker() as session:
          async with session.begin():
            await self._poll_active_jobs(session)
      except Exception:  # pylint: disable=broad-except
        logging.exception("Error in job polling loop.")
      await asyncio.sleep(self._poll_interval)

  async def _acquire_next_job(
      self, session: AsyncSession
  ) -> db_schema.AssetJob | None:
    """Queries the database for the next eligible job and claims it."""
    now = datetime.datetime.now(datetime.timezone.utc)
    max_active = self._config.max_active_jobs_per_backend

    # 1. Identify assets currently executing active transfers (PROCESSING and
    # not expired)
    active_assets_subquery = (
        select(db_schema.AssetJob.asset_uuid)
        .where(
            db_schema.AssetJob.status
            == db_schema.JobStatus.JOB_STATUS_PROCESSING,
            db_schema.AssetJob.expiration_at >= now,
        )
        .scalar_subquery()
    )

    # 2. Identify storage backends that have reached their maximum active
    # job limit (N)
    busy_backends_subquery = (
        select(db_schema.TierPath.storage_backend_id)
        .join(
            db_schema.AssetJob,
            db_schema.AssetJob.target_tier_path_id == db_schema.TierPath.id,
        )
        .where(
            db_schema.AssetJob.status
            == db_schema.JobStatus.JOB_STATUS_PROCESSING,
            db_schema.AssetJob.expiration_at >= now,
        )
        .group_by(db_schema.TierPath.storage_backend_id)
        .having(sqlalchemy.func.count(db_schema.AssetJob.id) >= max_active)
        .scalar_subquery()
    )

    # 3. Fetch the oldest eligible job (QUEUED or expired PROCESSING)
    stmt = (
        select(db_schema.AssetJob)
        .options(
            sqlalchemy.orm.selectinload(
                db_schema.AssetJob.target_tier_path
            ).selectinload(db_schema.TierPath.storage_backend),
            sqlalchemy.orm.selectinload(db_schema.AssetJob.asset)
            .selectinload(db_schema.Asset.tier_paths)
            .selectinload(db_schema.TierPath.storage_backend),
        )
        .join(
            db_schema.TierPath,
            db_schema.AssetJob.target_tier_path_id == db_schema.TierPath.id,
            isouter=True,
        )
        .where(
            sqlalchemy.or_(
                db_schema.AssetJob.status
                == db_schema.JobStatus.JOB_STATUS_QUEUED,
                sqlalchemy.and_(
                    db_schema.AssetJob.status
                    == db_schema.JobStatus.JOB_STATUS_PROCESSING,
                    db_schema.AssetJob.expiration_at < now,
                ),
            ),
            ~db_schema.AssetJob.asset_uuid.in_(active_assets_subquery),
            sqlalchemy.or_(
                db_schema.AssetJob.target_tier_path_id.is_(None),
                ~db_schema.TierPath.storage_backend_id.in_(
                    busy_backends_subquery
                ),
            ),
        )
        .order_by(db_schema.AssetJob.created_at.asc())
        .limit(1)
        .with_for_update(skip_locked=True)
    )

    result = await session.execute(stmt)
    job = result.scalars().first()

    if job:
      # Atomically claim the job
      job.status = db_schema.JobStatus.JOB_STATUS_PROCESSING
      job.expiration_at = now + self._lease_duration
      job.worker_host = self._hostname
      job.worker_pid = self._pid
      job.last_updated_at = now
      session.add(job)
      if (
          job.request_type == db_schema.RequestType.REQUEST_TYPE_COPY
          and job.target_tier_path
      ):
        job.target_tier_path.state = (
            db_schema.TierPathState.IN_PROGRESS
        )
        session.add(job.target_tier_path)

    return job

  async def _process_job(self, session: AsyncSession, job: db_schema.AssetJob):
    """Triggers the GCP transfer for the acquired job."""
    # We need to fetch the source and destination paths.
    # For COPY (Prefetch): Source is GCS (Tier 1), Destination is Lustre
    # (Tier 0)
    # For DELETE_FROM_INSTANCE: We just delete from Lustre.
    # For DELETE_FROM_ALL_TIERS: We delete from all.
    #
    # Actually, jobs.py only handles data movement (COPY/Export).
    # Deletion jobs might be handled differently, but let's see.
    # "manages the asynchronous data movement (Lustre import/export)"
    #
    # If request_type is COPY:
    #   We need to find the GCS path (source) and Lustre path (target).
    #   The job has `target_tier_path_id` which is the Lustre path we are
    #   copying to.
    #   We need to find the GCS path for the same asset.

    if job.request_type == db_schema.RequestType.REQUEST_TYPE_COPY:
      await self._process_copy_job(session, job)
    else:
      # For now, we only handle COPY in the worker. Other request types
      # (e.g. DELETE) are not implemented.
      logging.warning("Unsupported job request type: %s", job.request_type)
      # Mark as failed for now if not COPY
      await self._fail_job(
          session,
          job,
          f"Unsupported request type: {job.request_type}",
          {"status": "FAILED"},
      )

  async def _process_copy_job(
      self, session: AsyncSession, job: db_schema.AssetJob
  ):
    """Processes a COPY job (GCS -> Lustre or Lustre -> GCS)."""
    # Find the target TierPath
    target_tp = job.target_tier_path
    if not target_tp:
      await self._fail_job(
          session,
          job,
          "Target TierPath not found",
          {"status": "FAILED"},
      )
      return

    asset = job.asset
    # Find the source TierPath (the other tier path that is ready)
    # TODO(dnlng): just find a source that is closest to the target.
    source_tp = None
    for tp in asset.tier_paths:
      if tp.id != target_tp.id and tp.ready_at is not None:
        source_tp = tp
        break

    if not source_tp:
      await self._fail_job(
          session,
          job,
          "Source TierPath not found or not ready",
          {"status": "FAILED"},
      )
      return

    # Determine if import or export
    # If source is GCS (level 1) and target is Lustre (level 0) -> Import
    # If source is Lustre (level 0) and target is GCS (level 1) -> Export
    try:
      async with self._get_client_for_backends(
          source_tp.storage_backend, target_tp.storage_backend
      ) as client:
        operation_name = await client.trigger_copy(
            job.request_id, source_tp.path, target_tp.path
        )

      job.transfer_status = {
          "request_id": operation_name,
          "status": OperationStatus.IN_PROGRESS.value,
          "bytes_copied": 0,
          "total_bytes": 0,
      }
      session.add(job)
      logging.info(
          "Triggered transfer for job %d, operation_name: %s",
          job.id,
          operation_name,
      )

    except Exception as e:  # pylint: disable=broad-except
      logging.exception("Failed to trigger transfer for job %d", job.id)
      await self._fail_job(
          session,
          job,
          f"Failed to trigger transfer: {e}",
          {"status": "FAILED"},
      )

  async def _poll_active_jobs(self, session: AsyncSession):
    """Polls status of active jobs owned by this worker."""
    now = datetime.datetime.now(datetime.timezone.utc)
    stmt = (
        select(db_schema.AssetJob)
        .options(
            sqlalchemy.orm.selectinload(
                db_schema.AssetJob.target_tier_path
            ).selectinload(db_schema.TierPath.storage_backend),
            sqlalchemy.orm.selectinload(db_schema.AssetJob.asset)
            .selectinload(db_schema.Asset.tier_paths)
            .selectinload(db_schema.TierPath.storage_backend),
        )
        .where(
            db_schema.AssetJob.status
            == db_schema.JobStatus.JOB_STATUS_PROCESSING,
            db_schema.AssetJob.worker_host == self._hostname,
            db_schema.AssetJob.worker_pid == self._pid,
        )
    )
    result = await session.execute(stmt)
    active_jobs = result.scalars().all()

    for job in active_jobs:
      logging.info("Polling job %d", job.id)
      await self._extend_lease(session, job, now)

      status_dict = job.transfer_status
      if not status_dict or "request_id" not in status_dict:
        logging.warning("Job %d in PROCESSING but has no request_id", job.id)
        continue

      req_id = status_dict["request_id"]
      try:
        async with self._get_client_for_job(job) as client:
          target_tp = job.target_tier_path
          source_tp = None
          for tp in job.asset.tier_paths:
            if tp.id != target_tp.id and tp.ready_at is not None:
              source_tp = tp
              break
          if not source_tp:
            raise ValueError(f"Source TierPath not found for job {job.id}")

          context = TransferContext(
              job_request_id=job.request_id,
              source_path=source_tp.path,
              destination_path=target_tp.path,
              transfer_status=status_dict,
          )
          gcp_result = await client.poll_operation(req_id, context=context)
          logging.info(
              "Job %d GCP status: %s, detail_info: %s",
              job.id,
              gcp_result.status,
              gcp_result.detail_info,
          )

        # Update transfer_status JSON
        new_status = {
            **status_dict,
            "status": gcp_result.status.value,
            **gcp_result.detail_info,
        }
        job.transfer_status = new_status
        job.last_updated_at = now

        if gcp_result.status == OperationStatus.SUCCESS:
          await self._complete_job(session, job, now)
        elif gcp_result.status == OperationStatus.FAILED:
          error_msg = gcp_result.detail_info.get("error", "Unknown GCP error")
          await self._fail_job(session, job, error_msg, new_status, now)
        else:
          session.add(job)

      except Exception:  # pylint: disable=broad-except
        logging.exception("Error polling job %d", job.id)
        # Do not fail immediately on poll error, lease will expire if worker
        # dies.
        # But we can update last_updated_at to show we tried.
        job.last_updated_at = now
        session.add(job)

  async def _extend_lease(
      self,
      session: AsyncSession,
      job: db_schema.AssetJob,
      now: datetime.datetime,
  ):
    """Extends the lease of the job (heartbeat)."""
    job.expiration_at = now + self._lease_duration
    session.add(job)
    logging.info("Extended lease for job %d to %s", job.id, job.expiration_at)

  async def _complete_job(
      self,
      session: AsyncSession,
      job: db_schema.AssetJob,
      now: datetime.datetime,
  ):
    """Marks the job as completed and updates target TierPath."""
    job.status = db_schema.JobStatus.JOB_STATUS_COMPLETED
    job.completed_at = now
    job.worker_host = None
    job.worker_pid = None
    job.expiration_at = None
    session.add(job)

    # Mark target TierPath as ready
    target_tp = job.target_tier_path
    if target_tp:
      target_tp.state = db_schema.TierPathState.READY
      target_tp.ready_at = now
      # Calculate expiration for TierPath if it is Level 0 (Lustre)
      # "the checkpoint could be removed from this location after it expires."
      # GCS (Level 1) paths usually don't expire.
      if (
          target_tp.storage_backend.backend_type
          == db_schema.BackendType.BACKEND_TYPE_LUSTRE
      ):
        # Set expires_at. We can use a default TTL or parse it from config.
        # For now, let's use a default of 1 hour, or use keep_alive_interval.
        # Actually, server has client_keep_alive_interval_seconds
        # (default 1800).
        # Let's use that as a base.
        ttl = datetime.timedelta(
            seconds=self._config.client_keep_alive_interval_seconds
        )
        target_tp.expires_at = assets.calculate_expires_at(ttl)
      session.add(target_tp)

    logging.info(
        "Completed job %d, target TierPath %s marked ready",
        job.id,
        target_tp.path if target_tp else "None",
    )

  async def _fail_job(
      self,
      session: AsyncSession,
      job: db_schema.AssetJob,
      error_msg: str,
      transfer_status: dict[str, Any],
      now: datetime.datetime | None = None,
  ):
    """Marks the job as failed and cleans up target TierPath."""
    if not now:
      now = datetime.datetime.now(datetime.timezone.utc)
    job.status = db_schema.JobStatus.JOB_STATUS_FAILED
    job.completed_at = now
    job.worker_host = None
    job.worker_pid = None
    job.expiration_at = None
    transfer_status["error"] = error_msg
    job.transfer_status = transfer_status
    session.add(job)

    # Clean up target TierPath on failure (set state to FAILED)
    target_tp = job.target_tier_path
    if target_tp:
      target_tp.state = db_schema.TierPathState.FAILED
      session.add(target_tp)

    logging.error("Failed job %d: %s", job.id, error_msg)


async def run_tiering_service_worker_loop(
    session_maker: Any,
    config: tiering_service_pb2.ServerConfig,
    gcp_client: GCPStorageClient | None = None,
    *,
    lease_duration_seconds: int = 60,
    poll_interval_seconds: int = 5,
) -> TieringServiceWorker:
  """Runs the worker loop."""
  worker = TieringServiceWorker(
      session_maker,
      config,
      gcp_client,
      lease_duration_seconds=lease_duration_seconds,
      poll_interval_seconds=poll_interval_seconds,
  )
  await worker.start()
  return worker
