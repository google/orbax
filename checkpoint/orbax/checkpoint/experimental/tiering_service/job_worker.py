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
import random
import socket
from typing import Any
from absl import logging
from orbax.checkpoint.experimental.tiering_service import assets
from orbax.checkpoint.experimental.tiering_service import db_lib
from orbax.checkpoint.experimental.tiering_service import db_schema
from orbax.checkpoint.experimental.tiering_service import gcp_storage_client
from orbax.checkpoint.experimental.tiering_service import storage_backend
from orbax.checkpoint.experimental.tiering_service.proto import tiering_service_pb2
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import joinedload
from sqlalchemy.orm import sessionmaker


class TieringServiceWorker:
  """Background worker that processes AssetJobs."""

  def __init__(
      self,
      session_maker: sessionmaker | None,
      config: tiering_service_pb2.ServerConfig,
      *,
      lease_duration_seconds: int = 60,
      poll_interval_seconds: int = 10,
  ):
    """Initializes the background worker.

    Args:
      session_maker: A callable that returns a database session.
      config: The server configuration.
      lease_duration_seconds: Duration of the lease acquired for jobs.
      poll_interval_seconds: Polling interval for checking job status.
    """
    self._session_maker = session_maker
    self._config = config
    self._lease_duration = datetime.timedelta(seconds=lease_duration_seconds)
    self._poll_interval = poll_interval_seconds
    self._hostname = socket.gethostname()
    self._pid = os.getpid()
    self._tasks = []
    self._shutdown_event = asyncio.Event()
    self._backends = None
    self._backends_to_try = []

  async def start(self):
    """Starts the background worker loops."""
    logging.info(
        "Starting TieringServiceWorker on %s:%d", self._hostname, self._pid
    )
    self._shutdown_event.clear()

    # fetch and cache all storage backends if not already cached.
    async with self._session_maker() as session:
      self._backends = await storage_backend.find_backends_by_level(
          session, level=None
      )

    # Shuffle backends to distribute jobs evenly across workers.
    self._backends_to_try = [b.id for b in self._backends]
    # Try no-backend jobs too (e.g. DELETE_FROM_ALL_TIERS)
    self._backends_to_try.append(None)
    random.shuffle(self._backends_to_try)

    self._tasks.append(asyncio.create_task(self._run_acquisition_loop()))
    self._tasks.append(asyncio.create_task(self._run_polling_loop()))

  async def stop(self):
    """Stops the background worker loops gracefully."""
    logging.info("Stopping TieringServiceWorker...")
    self._shutdown_event.set()
    if self._tasks:
      # Wait for tasks to finish
      await asyncio.gather(*self._tasks, return_exceptions=True)
      self._tasks.clear()
    logging.info("TieringServiceWorker stopped.")

  async def _run_acquisition_loop(self):
    """Periodically acquires and triggers queued jobs."""
    while not self._shutdown_event.is_set():
      try:
        for backend_id in self._backends_to_try:
          job = await db_lib.acquire_next_job(
              session_maker=self._session_maker,
              backend_id=backend_id,
              lease_duration=self._lease_duration,
              hostname=self._hostname,
              pid=self._pid,
              max_active=self._config.max_active_jobs_per_backend,
          )

          if job:
            logging.info(
                "Acquired job %d for backend %s", job.id, str(backend_id)
            )
            await self._process_job(job)

          # Add some random sleep between checks to stagger concurrent workers.
          await asyncio.sleep(random.uniform(1.0, 5.0))

      except Exception:  # pylint: disable=broad-except
        logging.exception("Error in job acquisition loop.")
      await asyncio.sleep(self._poll_interval)

  async def _run_polling_loop(self):
    """Periodically polls status of active jobs owned by this worker."""
    while not self._shutdown_event.is_set():
      try:
        await self._poll_active_jobs()
      except Exception:  # pylint: disable=broad-except
        logging.exception("Error in job polling loop.")
      await asyncio.sleep(self._poll_interval)

  async def _poll_active_jobs(self):
    """Polls status of active jobs owned by this worker."""
    now = datetime.datetime.now(datetime.timezone.utc)
    async with self._session_maker() as session:
      active_jobs = await db_lib.get_active_jobs(
          session, self._hostname, self._pid
      )

    for job in active_jobs:
      await self._poll_single_job(job, now)

  async def _poll_single_job(
      self, job: db_schema.AssetJob, now: datetime.datetime
  ):
    """Polls a single active copy job and updates its status."""
    logging.info("Polling job %d", job.id)

    status_dict = job.transfer_status
    if not status_dict or "request_id" not in status_dict:
      logging.warning("Job %d in PROCESSING but has no request_id", job.id)
      return

    req_id = status_dict["request_id"]
    try:
      client = gcp_storage_client.get_client_from_status(
          status_dict,
          project=self._config.gcp_project or None,
          service_account=self._config.service_account or None,
      )
      try:
        # The network call to client.poll_operation happens OUTSIDE the DB
        # transaction. GCS/Lustre clients don't use context.
        gcp_result = await client.poll_operation(req_id)
        logging.info(
            "Job %d GCP status: %s, detail_info: %s",
            job.id,
            gcp_result.status,
            gcp_result.detail_info,
        )
      finally:
        await client.close()

      await self._update_job_status_after_poll(
          job, gcp_result, status_dict, now
      )

    except ValueError as e:
      await self._handle_polling_error(job, e, status_dict, now)
    except Exception:  # pylint: disable=broad-except
      logging.exception("Error polling job %d", job.id)
      # Note: lease is not extended if we hit transient error.
      async with self._session_maker() as session:
        async with session.begin():
          merged_job = await session.get(
              db_schema.AssetJob, job.id, with_for_update=True
          )
          if merged_job and (
              merged_job.status == db_schema.JobStatus.JOB_STATUS_PROCESSING
              and merged_job.worker_host == self._hostname
              and merged_job.worker_pid == self._pid
          ):
            merged_job.last_updated_at = now
            session.add(merged_job)

  async def _update_job_status_after_poll(
      self,
      job: db_schema.AssetJob,
      gcp_result: gcp_storage_client.Result,
      status_dict: dict[str, Any],
      now: datetime.datetime,
  ):
    """Updates the job status in the database after a poll iteration."""
    # Update transfer_status JSON and heartbeat/lease in a short transaction
    async with self._session_maker() as session:
      async with session.begin():
        merged_job = await session.get(
            db_schema.AssetJob, job.id, with_for_update=True
        )
        if not merged_job:
          logging.warning("Job %d was deleted", job.id)
          return
        if (
            merged_job.status != db_schema.JobStatus.JOB_STATUS_PROCESSING
            or merged_job.worker_host != self._hostname
            or merged_job.worker_pid != self._pid
        ):
          logging.warning(
              "Job %d is no longer owned by this worker (%s:%d). "
              "Skipping status update.",
              job.id,
              self._hostname,
              self._pid,
          )
          return

        # Extend lease only after successful poll
        await self._extend_lease(merged_job, now)

        status_pb = gcp_storage_client.deserialize_transfer_status(status_dict)
        status_pb.status = gcp_result.status.value
        if "bytes_copied" in gcp_result.detail_info:
          status_pb.bytes_copied = gcp_result.detail_info["bytes_copied"]
        if "total_bytes" in gcp_result.detail_info:
          status_pb.total_bytes = gcp_result.detail_info["total_bytes"]

        extra_keys = {"bytes_copied", "total_bytes", "error"}
        custom_details = {
            k: v
            for k, v in gcp_result.detail_info.items()
            if k not in extra_keys
        }
        status_pb.detail_info.update(custom_details)

        new_status = gcp_storage_client.serialize_transfer_status(status_pb)
        merged_job.transfer_status = new_status
        merged_job.last_updated_at = now

        if gcp_result.status == gcp_storage_client.OperationStatus.SUCCESS:
          await self._complete_job(session, merged_job, now)
        elif gcp_result.status == gcp_storage_client.OperationStatus.FAILED:
          error_msg = gcp_result.detail_info.get("error", "Unknown GCP error")
          await self._fail_job(session, merged_job, error_msg, new_status, now)
        else:
          session.add(merged_job)

  async def _handle_polling_error(
      self,
      job: db_schema.AssetJob,
      error: ValueError,
      status_dict: dict[str, Any],
      now: datetime.datetime,
  ):
    """Handles permanent logic errors encountered during job polling."""
    logging.exception("Permanent logic error polling job %d", job.id)
    async with self._session_maker() as session:
      async with session.begin():
        merged_job = await session.get(
            db_schema.AssetJob, job.id, with_for_update=True
        )
        if not merged_job:
          logging.warning("Job %d was deleted", job.id)
          return
        if (
            merged_job.status != db_schema.JobStatus.JOB_STATUS_PROCESSING
            or merged_job.worker_host != self._hostname
            or merged_job.worker_pid != self._pid
        ):
          logging.warning(
              "Job %d is no longer owned by this worker (%s:%d). "
              "Skipping error update.",
              job.id,
              self._hostname,
              self._pid,
          )
          return
        new_status = {
            **status_dict,
            "status": gcp_storage_client.OperationStatus.FAILED.value,
        }
        await self._fail_job(session, merged_job, str(error), new_status, now)

  async def _process_job(self, job: db_schema.AssetJob):
    """Triggers the GCP transfer for the acquired job."""
    # Check if this is a reclaimed job that is already triggered.
    if job.transfer_status and "request_id" in job.transfer_status:
      logging.info(
          "Job %d was reclaimed and already has active transfer (%s). "
          "Skipping trigger.",
          job.id,
          job.transfer_status["request_id"],
      )
      return

    if job.request_type == db_schema.RequestType.REQUEST_TYPE_COPY:
      await self._process_copy_job(job)
    else:
      # TODO: b/503445463 - Support DELETE jobs.
      logging.warning("Unsupported job request type: %s", job.request_type)
      # Mark as failed for now if not COPY
      async with self._session_maker() as session:
        async with session.begin():
          merged_job = await session.get(
              db_schema.AssetJob, job.id, with_for_update=True
          )
          if not merged_job:
            logging.warning("Job %d was deleted", job.id)
            return
          if (
              merged_job.status != db_schema.JobStatus.JOB_STATUS_PROCESSING
              or merged_job.worker_host != self._hostname
              or merged_job.worker_pid != self._pid
          ):
            logging.warning(
                "Job %d is no longer owned by this worker (%s:%d). "
                "Skipping update.",
                job.id,
                self._hostname,
                self._pid,
            )
            return
          await self._fail_job(
              session,
              merged_job,
              f"Unsupported request type: {job.request_type}",
              {"status": "FAILED"},
          )

  def _find_source_tier_path(
      self, asset: db_schema.Asset, target_tp: db_schema.TierPath
  ) -> db_schema.TierPath | None:
    """Finds a ready source TierPath that is not the target TierPath."""
    for tp in asset.tier_paths:
      if tp.id != target_tp.id and tp.ready_at is not None:
        return tp
    return None

  async def _fail_job_by_id(self, job_id: int, error_msg: str):
    """Fails a job by ID, verifying hostname/PID ownership first."""
    async with self._session_maker() as session:
      async with session.begin():
        merged_job = await session.get(
            db_schema.AssetJob, job_id, with_for_update=True
        )
        if (
            merged_job
            and merged_job.status == db_schema.JobStatus.JOB_STATUS_PROCESSING
            and merged_job.worker_host == self._hostname
            and merged_job.worker_pid == self._pid
        ):
          await self._fail_job(
              session,
              merged_job,
              error_msg,
              {"status": "FAILED"},
          )

  async def _save_triggered_transfer_status(
      self,
      job_id: int,
      error_msg: str | None,
      operation_name: str | None,
      client_type: str,
      zone: str | None,
  ):
    """Saves the details of a successfully triggered transfer to the job."""
    async with self._session_maker() as session:
      async with session.begin():
        merged_job = await session.get(
            db_schema.AssetJob, job_id, with_for_update=True
        )
        if not merged_job:
          logging.warning("Job %d was deleted", job_id)
          return
        if (
            merged_job.status != db_schema.JobStatus.JOB_STATUS_PROCESSING
            or merged_job.worker_host != self._hostname
            or merged_job.worker_pid != self._pid
        ):
          logging.warning(
              "Job %d is no longer owned by this worker (%s:%d). "
              "Skipping post-trigger update.",
              job_id,
              self._hostname,
              self._pid,
          )
          return

        if error_msg is not None:
          await self._fail_job(
              session,
              merged_job,
              error_msg,
              {"status": "FAILED"},
          )
        else:
          status_pb = tiering_service_pb2.TransferStatus(
              request_id=operation_name,
              status=gcp_storage_client.OperationStatus.IN_PROGRESS.value,
              client_type=client_type,
          )
          if zone:
            status_pb.zone = zone
          merged_job.transfer_status = (
              gcp_storage_client.serialize_transfer_status(status_pb)
          )
          session.add(merged_job)
          logging.info(
              "Triggered transfer for job %d, op: %s",
              merged_job.id,
              operation_name,
          )

  async def _process_copy_job(self, job: db_schema.AssetJob):
    """Processes a COPY job (eg. GCS -> Lustre or Lustre -> GCS)."""
    target_tp = job.target_tier_path
    if not target_tp:
      await self._fail_job_by_id(job.id, "Target TierPath not found")
      return

    source_tp = self._find_source_tier_path(job.asset, target_tp)
    if not source_tp:
      await self._fail_job_by_id(
          job.id, "Source TierPath not found or not ready"
      )
      return

    try:
      determined_client = gcp_storage_client.determine_client(
          source_tp,
          target_tp,
          project=self._config.gcp_project or None,
          service_account=self._config.service_account or None,
      )
    except ValueError as e:
      await self._fail_job_by_id(job.id, str(e))
      return

    error_msg = None
    operation_name = None
    try:
      operation_name = await determined_client.trigger_copy(
          job.request_id, source_tp.path, target_tp.path
      )
    except Exception as e:  # pylint: disable=broad-except
      logging.exception("Failed to trigger transfer for job %d", job.id)
      error_msg = f"Failed to trigger transfer: {e}"
    finally:
      await determined_client.close()

    await self._save_triggered_transfer_status(
        job_id=job.id,
        error_msg=error_msg,
        operation_name=operation_name,
        client_type=determined_client.__class__.__name__,
        zone=getattr(determined_client, "location", None),
    )

  async def _get_target_tier_path(
      self,
      session: AsyncSession,
      job: db_schema.AssetJob,
      load_backend: bool = False,
  ) -> db_schema.TierPath | None:
    """Retrieves target TierPath by ID, optionally eager-loading the backend."""
    if job.target_tier_path_id is None:
      return None

    if load_backend:
      stmt = (
          select(db_schema.TierPath)
          .options(joinedload(db_schema.TierPath.storage_backend))
          .where(db_schema.TierPath.id == job.target_tier_path_id)
      )
      result = await session.execute(stmt)
      return result.scalars().first()
    else:
      return await session.get(db_schema.TierPath, job.target_tier_path_id)

  async def _fail_job(
      self,
      session: AsyncSession,
      job: db_schema.AssetJob,
      error_msg: str,
      transfer_status: dict[str, Any] | None = None,
      now: datetime.datetime | None = None,
  ):
    """Marks the job as failed."""
    if not now:
      now = datetime.datetime.now(datetime.timezone.utc)
    job.status = db_schema.JobStatus.JOB_STATUS_FAILED
    job.completed_at = now
    job.worker_host = None
    job.worker_pid = None
    job.expiration_at = None
    if transfer_status is None:
      transfer_status = {}
    else:
      transfer_status = dict(transfer_status)
    transfer_status["error"] = error_msg
    job.transfer_status = transfer_status
    session.add(job)

    # Clean up target TierPath on failure (set state to FAILED)
    target_tp = await self._get_target_tier_path(
        session, job, load_backend=False
    )
    if target_tp:
      target_tp.state = db_schema.TierPathState.FAILED
      session.add(target_tp)

    logging.error("Failed job %d: %s", job.id, error_msg)

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
    target_tp = await self._get_target_tier_path(
        session, job, load_backend=True
    )
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
        ttl = datetime.timedelta(seconds=60 * 60)
        target_tp.expires_at = assets.calculate_expires_at(ttl)
      session.add(target_tp)

    logging.info(
        "Completed job %d, target TierPath %s marked ready",
        job.id,
        target_tp.path if target_tp else "None",
    )

  async def _extend_lease(
      self,
      job: db_schema.AssetJob,
      now: datetime.datetime,
  ):
    """Extends the lease of the job (heartbeat)."""
    job.expiration_at = now + self._lease_duration


async def run_tiering_service_worker_loop(
    session_maker: sessionmaker | None,
    config: tiering_service_pb2.ServerConfig,
    *,
    lease_duration_seconds: int = 60,
    poll_interval_seconds: int = 5,
) -> TieringServiceWorker:
  """Runs the worker loop."""
  worker = TieringServiceWorker(
      session_maker,
      config,
      lease_duration_seconds=lease_duration_seconds,
      poll_interval_seconds=poll_interval_seconds,
  )
  await worker.start()
  return worker
