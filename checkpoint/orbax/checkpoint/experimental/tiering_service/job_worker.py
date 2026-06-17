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
from absl import logging
from orbax.checkpoint.experimental.tiering_service import assets
from orbax.checkpoint.experimental.tiering_service import db_lib
from orbax.checkpoint.experimental.tiering_service import db_schema
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
    # TODO: b/503445463 - Implement actual job processing logic.
    logging.info("Processing job %d", job.id)
    now = datetime.datetime.now(datetime.timezone.utc)
    async with self._session_maker() as session:
      async with session.begin():
        merged_job = await session.merge(job)
        await self._complete_job(session, merged_job, now)

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
