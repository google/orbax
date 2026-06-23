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
from sqlalchemy.orm import sessionmaker


class TieringServiceWorker:
  """Background worker that processes AssetJobs."""

  def __init__(
      self,
      session_maker: sessionmaker | None,
      config: tiering_service_pb2.ServerConfig,
      gcp_client: gcp_storage_client.GCPStorageClient | None = None,
      *,
      lease_duration_seconds: int = 60,
      poll_interval_seconds: int = 10,
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
    self._backends = None
    self._backends_to_try = []

  def _get_client_for_backends(
      self,
      source_backend: db_schema.StorageBackend,
      target_backend: db_schema.StorageBackend,
  ) -> gcp_storage_client.GCPStorageClient:
    """Returns the appropriate GCP client for the given transfer backends."""
    if self._gcp_client is not None:
      return self._gcp_client

    project = self._config.gcp_project or None
    service_account = self._config.service_account or None

    # Determine if Lustre import/export or GCS-to-GCS copy.
    # If source is GCS and target is Lustre -> Import
    # If source is Lustre and target is GCS -> Export
    # If both source  and target are GCS -> GCS-to-GCS copy
    #
    # TODO(dnlng): Support Lustre-to-Lustre copy.

    if (
        source_backend.backend_type == db_schema.BackendType.BACKEND_TYPE_GCS
        and target_backend.backend_type
        == db_schema.BackendType.BACKEND_TYPE_GCS
    ):
      return gcp_storage_client.GcsToGcsClient(
          project=project, service_account=service_account
      )

    if (
        source_backend.backend_type == db_schema.BackendType.BACKEND_TYPE_LUSTRE
        and target_backend.backend_type
        == db_schema.BackendType.BACKEND_TYPE_GCS
    ):
      location = source_backend.zone
      instance = f"lustre-{location}"
      return gcp_storage_client.LustreToGcsClient(
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
      return gcp_storage_client.GcsToLustreClient(
          instance=instance,
          location=location,
          project=project,
          service_account=service_account,
      )

    raise ValueError(
        f"Unsupported backend pair: {source_backend.backend_type} and"
        f" {target_backend.backend_type}"
    )

  def _get_client_for_job(
      self, job: db_schema.AssetJob
  ) -> gcp_storage_client.GCPStorageClient:
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

  async def _process_job(self, job: db_schema.AssetJob):
    """Triggers the GCP transfer for the acquired job."""
    if job.request_type == db_schema.RequestType.REQUEST_TYPE_COPY:
      await self._process_copy_job(job)
    else:
      # TODO: b/503445463 - Support DELETE jobs.
      logging.warning("Unsupported job request type: %s", job.request_type)
      # Mark as failed for now if not COPY
      async with self._session_maker() as session:
        async with session.begin():
          # Re-associate the detached job instance with the new session.
          merged_job = await session.merge(job)
          await self._fail_job(
              session,
              merged_job,
              f"Unsupported request type: {job.request_type}",
              {"status": "FAILED"},
          )

  async def _process_copy_job(self, job: db_schema.AssetJob):
    """Processes a COPY job (eg. GCS -> Lustre or Lustre -> GCS)."""
    error_msg = None
    operation_name = None

    try:
      # Find the target TierPath
      target_tp = job.target_tier_path
      if not target_tp:
        raise ValueError("Target TierPath not found")

      asset = job.asset
      # Find the source TierPath (the other tier path that is ready)
      # For now, we just pick the first available.
      # TODO: b/503445463 - use heuristics to find closest to the target.
      source_tp = None
      for tp in asset.tier_paths:
        if tp.id != target_tp.id and tp.ready_at is not None:
          source_tp = tp
          break

      if not source_tp:
        raise ValueError("Source TierPath not found or not ready")

      client = self._get_client_for_backends(
          source_tp.storage_backend, target_tp.storage_backend
      )
      try:
        # The network call occurs OUTSIDE database transactions.
        operation_name = await client.trigger_copy(
            job.request_id, source_tp.path, target_tp.path
        )
      finally:
        if client is not self._gcp_client:
          await client.close()

    except ValueError as e:
      error_msg = str(e)
    except Exception as e:  # pylint: disable=broad-except
      logging.exception("Failed to trigger transfer for job %d", job.id)
      error_msg = f"Failed to trigger transfer: {e}"

    # Update database inside a single session/transaction block at the end.
    async with self._session_maker() as session:
      async with session.begin():
        # Re-associate the detached job instance with the new session.
        merged_job = await session.merge(job)
        if error_msg is not None:
          await self._fail_job(
              session,
              merged_job,
              error_msg,
              {"status": "FAILED"},
          )
        else:
          merged_job.transfer_status = {
              "request_id": operation_name,
              "status": gcp_storage_client.OperationStatus.IN_PROGRESS.value,
              "bytes_copied": 0,
              "total_bytes": 0,
          }
          session.add(merged_job)
          logging.info(
              "Triggered transfer for job %d, operation_name: %s",
              merged_job.id,
              operation_name,
          )

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
    target_tp = job.target_tier_path
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
        ttl = datetime.timedelta(seconds=60 * 60)
        target_tp.expires_at = assets.calculate_expires_at(ttl)
      session.add(target_tp)

    logging.info(
        "Completed job %d, target TierPath %s marked ready",
        job.id,
        target_tp.path if target_tp else "None",
    )


async def run_tiering_service_worker_loop(
    session_maker: sessionmaker | None,
    config: tiering_service_pb2.ServerConfig,
    gcp_client: gcp_storage_client.GCPStorageClient | None = None,
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
