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
import dataclasses
import datetime
import enum
import os
import socket
from typing import Any
from absl import logging
import google.auth
from google.auth.transport.requests import Request
from orbax.checkpoint.experimental.tiering_service import assets
from orbax.checkpoint.experimental.tiering_service import db_schema
from orbax.checkpoint.experimental.tiering_service.proto import tiering_service_pb2
import requests
import sqlalchemy
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
import sqlalchemy.orm


class Status(enum.Enum):
  """The status of a job or operation."""
  IN_PROGRESS = "IN_PROGRESS"
  SUCCESS = "SUCCESS"
  COMPLETED = "COMPLETED"


@dataclasses.dataclass
class Result:
  """The result of an operation run by a client.

  Attributes:
    status: The completion status of the operation.
    detail_info: Detailed metadata or error message from the operation.
  """
  status: Status
  detail_info: dict[str, Any]


class GcpParallelstoreClient:
  """Client interface to interact with GCP Parallelstore."""

  async def trigger_import(
      self,
      request_id: str,
      source_path: str,
      destination_path: str,
  ) -> str:
    """Triggers import and returns request ID."""
    raise NotImplementedError

  async def trigger_export(
      self,
      request_id: str,
      source_path: str,
      destination_path: str,
  ) -> str:
    """Triggers export and returns request ID."""
    raise NotImplementedError

  async def poll_operation(self, request_id: str) -> Result:
    """Polls operation status and returns a Result object."""
    raise NotImplementedError


class DummyGcpParallelstoreClient(GcpParallelstoreClient):
  """Dummy implementation of GcpParallelstoreClient for testing."""

  def __init__(self):
    """Initializes the dummy client with empty operations list."""
    self.operations = {}

  async def trigger_import(
      self,
      request_id: str,
      source_path: str,
      destination_path: str,
  ) -> str:
    """Triggers import in progress."""
    self.operations[request_id] = {
        "status": Status.IN_PROGRESS,
        "progress": 0,
        "type": "import",
    }
    logging.info(
        "Dummy triggered import %s -> %s, request_id: %s",
        source_path,
        destination_path,
        request_id,
    )
    return request_id

  async def trigger_export(
      self,
      request_id: str,
      source_path: str,
      destination_path: str,
  ) -> str:
    """Triggers export in progress."""
    self.operations[request_id] = {
        "status": Status.IN_PROGRESS,
        "progress": 0,
        "type": "export",
    }
    logging.info(
        "Dummy triggered export %s -> %s, request_id: %s",
        source_path,
        destination_path,
        request_id,
    )
    return request_id

  async def poll_operation(self, request_id: str) -> Result:
    """Polls the status of the specified GCP operation."""
    op = self.operations.get(request_id)
    if not op:
      return Result(
          status=Status.COMPLETED,
          detail_info={"error": "Operation not found"},
      )

    if op["status"] == Status.IN_PROGRESS:
      op["progress"] += 50  # Progress by 50% each poll
      if op["progress"] >= 100:
        op["status"] = Status.COMPLETED

    return Result(
        status=op["status"],
        detail_info={
            "bytes_copied": op["progress"] * 1000,
            "total_bytes": 100000,
        },
    )


class GcpLustreClient(GcpParallelstoreClient):
  """Client interface to interact with GCP Managed Lustre API via REST."""

  def __init__(
      self,
      project: str | None = None,
      location: str | None = None,
      instance: str | None = None,
  ):
    """Initializes the client.

    Args:
      project: The GCP project ID. If not set, it is auto-detected.
      location: The zone of the Managed Lustre instance.
      instance: The Managed Lustre instance ID.
    """
    self.project = project
    self.location = location or os.environ.get("CTS_LUSTRE_LOCATION")
    self.instance = instance or os.environ.get("CTS_LUSTRE_INSTANCE")
    self._credentials = None

  async def _get_token_and_project(self) -> tuple[str, str]:
    """Gets Gcp credentials token and project ID."""
    if not self._credentials:
      self._credentials, detected_project = google.auth.default(
          scopes=["https://www.googleapis.com/auth/cloud-platform"]
      )
      if not self.project:
        self.project = detected_project

    if not self._credentials.valid:
      await asyncio.to_thread(self._credentials.refresh, Request())

    if not self.project:
      raise ValueError("GCP Project ID must be specified or auto-detected.")
    if not self.location:
      raise ValueError(
          "Managed Lustre Location must be specified or set via"
          " CTS_LUSTRE_LOCATION environment variable."
      )
    if not self.instance:
      raise ValueError(
          "Managed Lustre Instance ID must be specified or set via"
          " CTS_LUSTRE_INSTANCE environment variable."
      )

    return self._credentials.token, self.project

  async def trigger_import(
      self,
      request_id: str,
      source_path: str,
      destination_path: str,
  ) -> str:
    """Triggers import from GCS to Lustre and returns the Operation name.

    Args:
      request_id: The unique request ID for the operation.
      source_path: The GCS source bucket/path, e.g. "gs://bucket/path".
      destination_path: The destination path on Lustre.

    Returns:
      The resource name of the triggered LRO Operation.

    Raises:
      RuntimeError: If the import request fails.
    """
    token, project = await self._get_token_and_project()
    url = (
        f"https://lustre.googleapis.com/v1/projects/{project}"
        f"/locations/{self.location}/instances/{self.instance}:importData"
    )
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "gcsPath": {"uri": source_path},
        "lustrePath": {"path": destination_path},
        "requestId": request_id,
    }
    response = await asyncio.to_thread(
        requests.post, url, json=payload, headers=headers
    )
    if response.status_code != 200:
      raise RuntimeError(
          f"Failed to trigger import: {response.status_code} - {response.text}"
      )
    return response.json()["name"]

  async def trigger_export(
      self,
      request_id: str,
      source_path: str,
      destination_path: str,
  ) -> str:
    """Triggers export from Lustre to GCS and returns the Operation name.

    Args:
      request_id: The unique request ID for the operation.
      source_path: The source path on Lustre.
      destination_path: The GCS destination bucket/path, e.g.
        "gs://bucket/path".

    Returns:
      The resource name of the triggered LRO Operation.

    Raises:
      RuntimeError: If the export request fails.
    """
    token, project = await self._get_token_and_project()
    url = (
        f"https://lustre.googleapis.com/v1/projects/{project}"
        f"/locations/{self.location}/instances/{self.instance}:exportData"
    )
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "lustrePath": {"path": source_path},
        "gcsPath": {"uri": destination_path},
        "requestId": request_id,
    }
    response = await asyncio.to_thread(
        requests.post, url, json=payload, headers=headers
    )
    if response.status_code != 200:
      raise RuntimeError(
          f"Failed to trigger export: {response.status_code} - {response.text}"
      )
    return response.json()["name"]

  async def poll_operation(self, request_id: str) -> Result:
    """Polls operation status and returns a Result object.

    Args:
      request_id: The resource name of the Operation, e.g.
        "projects/.../operations/...".

    Returns:
      A Result object representing the operation status.

    Raises:
      RuntimeError: If the poll request fails.
    """
    token, _ = await self._get_token_and_project()
    url = f"https://lustre.googleapis.com/v1/{request_id}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    response = await asyncio.to_thread(requests.get, url, headers=headers)
    if response.status_code != 200:
      raise RuntimeError(
          f"Failed to poll operation: {response.status_code} - {response.text}"
      )
    data = response.json()
    done = data.get("done", False)
    if done:
      if "error" in data:
        return Result(
            status=Status.COMPLETED,
            detail_info={"error": data["error"]},
        )
      else:
        return Result(
            status=Status.COMPLETED,
            detail_info=data.get("response", {}),
        )
    else:
      return Result(
          status=Status.IN_PROGRESS,
          detail_info=data.get("metadata", {}),
      )


class TieringServiceWorker:
  """Background worker that processes AssetJobs."""

  def __init__(
      self,
      session_maker: Any,
      config: tiering_service_pb2.ServerConfig,
      gcp_client: GcpParallelstoreClient | None = None,
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
    self._gcp_client = gcp_client or DummyGcpParallelstoreClient()
    self._lease_duration = datetime.timedelta(seconds=lease_duration_seconds)
    self._poll_interval = poll_interval_seconds
    self._hostname = socket.gethostname()
    self._pid = os.getpid()
    self._tasks = []
    self._shutdown_event = asyncio.Event()

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
      if (
          source_tp.storage_backend.backend_type
          == db_schema.BackendType.BACKEND_TYPE_GCS
          and target_tp.storage_backend.backend_type
          == db_schema.BackendType.BACKEND_TYPE_LUSTRE
      ):
        # Import
        request_id = await self._gcp_client.trigger_import(
            job.request_id, source_tp.path, target_tp.path
        )
      elif (
          source_tp.storage_backend.backend_type
          == db_schema.BackendType.BACKEND_TYPE_LUSTRE
          and target_tp.storage_backend.backend_type
          == db_schema.BackendType.BACKEND_TYPE_GCS
      ):
        # Export
        request_id = await self._gcp_client.trigger_export(
            job.request_id, source_tp.path, target_tp.path
        )
      else:
        raise ValueError(
            "Unsupported transfer from"
            f" {source_tp.storage_backend.backend_type} to"
            f" {target_tp.storage_backend.backend_type}"
        )

      job.transfer_status = {
          "request_id": request_id,
          "status": Status.IN_PROGRESS.value,
          "bytes_copied": 0,
          "total_bytes": 0,
      }
      session.add(job)
      logging.info(
          "Triggered transfer for job %d, request_id: %s", job.id, request_id
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
            ).selectinload(db_schema.TierPath.storage_backend)
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
        gcp_result = await self._gcp_client.poll_operation(req_id)
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

        if gcp_result.status in (Status.SUCCESS, Status.COMPLETED):
          if "error" in gcp_result.detail_info:
            error_msg = gcp_result.detail_info.get("error", "Unknown GCP error")
            await self._fail_job(session, job, error_msg, new_status, now)
          else:
            await self._complete_job(session, job, now)
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

    # Clean up target TierPath on failure
    target_tp = job.target_tier_path
    if target_tp:
      await session.delete(target_tp)

    logging.error("Failed job %d: %s", job.id, error_msg)


async def run_tiering_service_worker_loop(
    session_maker: Any,
    config: tiering_service_pb2.ServerConfig,
    gcp_client: GcpParallelstoreClient | None = None,
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
