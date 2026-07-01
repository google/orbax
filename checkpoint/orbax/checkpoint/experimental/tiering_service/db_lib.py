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

"""Database initialization utilities for Tiering Service."""

import contextlib
import datetime
import sqlite3

from orbax.checkpoint.experimental.tiering_service import db_schema
from orbax.checkpoint.experimental.tiering_service.proto import tiering_service_pb2
import sqlalchemy
from sqlalchemy import event
from sqlalchemy.dialects.sqlite.aiosqlite import AsyncAdapt_aiosqlite_connection
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.future import select
import sqlalchemy.orm
from sqlalchemy.orm import sessionmaker


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
  """Enables foreign key constraints on SQLite database connections.

  This is SQLite-specific because other databases (like PostgreSQL) enforce
  foreign keys by default and do not support SQLite's PRAGMA syntax.
  We perform an isinstance check against the standard sqlite3.Connection
  and SQLAlchemy's aiosqlite adapter wrapper to verify if this is an SQLite
  connection.

  Args:
    dbapi_connection: The database connection to configure.
    connection_record: Metadata about the connection.
  """
  del connection_record
  connection_types = (sqlite3.Connection, AsyncAdapt_aiosqlite_connection)

  if isinstance(dbapi_connection, connection_types):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def get_async_engine(config: tiering_service_pb2.ServerConfig) -> AsyncEngine:
  """Returns an AsyncEngine configured from ServerConfig."""
  input_url = config.db_connection_str

  # Make sure we are using the async version of the driver
  if input_url.startswith("psql://"):
    url = input_url.replace("psql://", "postgresql+asyncpg://", 1)
  elif input_url.startswith("sqlite://"):
    url = input_url.replace("sqlite://", "sqlite+aiosqlite://", 1)
  else:
    url = input_url
  return create_async_engine(url)


@contextlib.asynccontextmanager
async def _engine_context(config: tiering_service_pb2.ServerConfig):
  """Yields an AsyncEngine and ensures it is disposed upon exit."""
  engine = get_async_engine(config)
  try:
    yield engine
  finally:
    await engine.dispose()


def _get_backend_key(
    level: int,
    zone: str | None,
    region: str | None,
    multi_regions: list[str] | None,
) -> tuple[int, str | None, str | None, tuple[str, ...] | None]:
  """Generates a unique key for a StorageBackend based on level and location."""
  return (
      level,
      zone,
      region,
      tuple(sorted(multi_regions)) if multi_regions else None,
  )


async def async_initialize_db(config: tiering_service_pb2.ServerConfig) -> None:
  """Initializes the database with the schema and initial data.

  If the database is uninitialized, this function connects to the database,
  creates all necessary tables based on the `db_schema.Base` metadata, and
  populates the `StorageBackend` table with data from the provided server
  configuration if it's empty.

  Args:
    config: The server configuration containing tier information and DB URL.
  """
  async with _engine_context(config) as engine:
    async with engine.begin() as conn:
      await conn.run_sync(db_schema.Base.metadata.create_all)

    session_maker = sessionmaker(
        engine, expire_on_commit=False, class_=AsyncSession
    )
    async with session_maker() as session:
      result = await session.execute(select(db_schema.StorageBackend))
      existing = result.scalars().first()

      if existing is not None:
        return

      for instance in config.storage_backends:
        if instance.backend_type == tiering_service_pb2.BACKEND_TYPE_LUSTRE:
          backend_type = db_schema.BackendType.BACKEND_TYPE_LUSTRE
        elif instance.backend_type == tiering_service_pb2.BACKEND_TYPE_GCS:
          backend_type = db_schema.BackendType.BACKEND_TYPE_GCS
        else:
          raise ValueError(
              f"Unknown storage backend type: {instance.backend_type!r}"
          )

        backend = db_schema.StorageBackend(
            level=instance.level,
            backend_type=backend_type,
            prefix=instance.prefix,
        )
        if instance.HasField("zone"):
          backend.zone = instance.zone
        elif instance.HasField("region"):
          backend.region = instance.region
        elif instance.HasField("multi_regions"):
          backend.multi_regions = list(instance.multi_regions.regions)
        session.add(backend)
      if session.new:
        await session.commit()


async def async_is_db_initialized(
    config: tiering_service_pb2.ServerConfig,
) -> bool:
  """Returns whether the database is already initialized with StorageBackend entries."""
  async with _engine_context(config) as engine:
    session_maker = sessionmaker(
        engine, expire_on_commit=False, class_=AsyncSession
    )
    try:
      async with session_maker() as session:
        result = await session.execute(select(db_schema.StorageBackend))
        return result.scalars().first() is not None
    except OperationalError:  # If tables do not exist yet.
      return False


async def async_verify_db(config: tiering_service_pb2.ServerConfig) -> None:
  """Verifies that the database StorageBackend table matches ServerConfig.

  Args:
    config: The server configuration containing tier information and DB URL.

  Raises:
    ValueError: If there is any mismatch between configuration and database.
  """
  async with _engine_context(config) as engine:
    session_maker = sessionmaker(
        engine, expire_on_commit=False, class_=AsyncSession
    )
    async with session_maker() as session:
      result = await session.execute(select(db_schema.StorageBackend))
      db_backends = result.scalars().all()

  expected_count = len(config.storage_backends)
  if len(db_backends) != expected_count:
    raise ValueError(
        f"Mismatch in total StorageBackend count: DB has {len(db_backends)},"
        f" config expects {expected_count}"
    )

  backend_by_key = {
      _get_backend_key(b.level, b.zone, b.region, b.multi_regions): b
      for b in db_backends
  }

  for instance in config.storage_backends:
    level = instance.level
    if instance.backend_type == tiering_service_pb2.BACKEND_TYPE_LUSTRE:
      expected_type = db_schema.BackendType.BACKEND_TYPE_LUSTRE
    elif instance.backend_type == tiering_service_pb2.BACKEND_TYPE_GCS:
      expected_type = db_schema.BackendType.BACKEND_TYPE_GCS
    else:
      raise ValueError(
          f"Unknown storage backend type: {instance.backend_type!r}"
      )

    zone = instance.zone if instance.HasField("zone") else None
    region = instance.region if instance.HasField("region") else None
    multi_regions = (
        list(instance.multi_regions.regions)
        if instance.HasField("multi_regions")
        else None
    )

    instance_key = _get_backend_key(level, zone, region, multi_regions)
    db_backend = backend_by_key.get(instance_key)

    if db_backend is None:
      raise ValueError(
          f"Configuration expects StorageBackend with key {instance_key!r}"
          f" (prefix={instance.prefix!r}) but not found in Database."
      )

    if db_backend.backend_type != expected_type:
      raise ValueError(
          f"Backend with key {instance_key!r} mismatch"
          f" backend_type: DB has {db_backend.backend_type.name}, config"
          f" expects {expected_type.name}"
      )
    if db_backend.prefix != instance.prefix:
      raise ValueError(
          f"Backend with key {instance_key!r} mismatch"
          f" prefix: DB has {db_backend.prefix!r}, config expects"
          f" {instance.prefix!r}"
      )


async def get_active_jobs(
    session: AsyncSession, hostname: str, pid: int
) -> list[db_schema.AssetJob]:
  """Returns all active PROCESSING jobs owned by this worker."""
  stmt = (
      select(db_schema.AssetJob)
      .where(
          db_schema.AssetJob.status
          == db_schema.JobStatus.JOB_STATUS_PROCESSING,
          db_schema.AssetJob.worker_host == hostname,
          db_schema.AssetJob.worker_pid == pid,
      )
  )
  result = await session.execute(stmt)
  return list(result.scalars().all())


async def _has_eligible_jobs(
    session: AsyncSession,
    backend_id: int | None,
    now: datetime.datetime,
) -> bool:
  """Checks if there are any eligible jobs for the backend without locking."""
  active_assets_subquery = (
      select(db_schema.AssetJob.asset_uuid)
      .where(
          db_schema.AssetJob.status
          == db_schema.JobStatus.JOB_STATUS_PROCESSING,
          db_schema.AssetJob.expiration_at >= now,
      )
      .scalar_subquery()
  )

  if backend_id is None:
    backend_cond = db_schema.AssetJob.target_tier_path_id.is_(None)
  else:
    backend_cond = db_schema.TierPath.storage_backend_id == backend_id

  stmt = (
      select(db_schema.AssetJob.id)
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
          backend_cond,
      )
      .limit(1)
  )
  result = await session.execute(stmt)
  return result.scalar() is not None


async def _try_lock_backend(
    session: AsyncSession,
    backend_id: int,
    max_active: int,
    now: datetime.datetime,
) -> bool:
  """Locks the backend and returns True if it has capacity for more jobs."""
  # Lock only this specific backend to avoid race conditions.
  backend_stmt = (
      select(db_schema.StorageBackend)
      .where(db_schema.StorageBackend.id == backend_id)
      .with_for_update()
  )
  await session.execute(backend_stmt)

  # Re-evaluate active capacity under the backend lock.
  active_count_stmt = (
      select(sqlalchemy.func.count(db_schema.AssetJob.id))
      .join(
          db_schema.TierPath,
          db_schema.AssetJob.target_tier_path_id == db_schema.TierPath.id,
      )
      .where(
          db_schema.AssetJob.status
          == db_schema.JobStatus.JOB_STATUS_PROCESSING,
          db_schema.AssetJob.expiration_at >= now,
          db_schema.TierPath.storage_backend_id == backend_id,
      )
  )
  active_count_result = await session.execute(active_count_stmt)
  active_count = active_count_result.scalar()

  return active_count < max_active


async def _claim_eligible_job(
    session: AsyncSession,
    backend_id: int | None,
    lease_duration: datetime.timedelta,
    hostname: str,
    pid: int,
    now: datetime.datetime,
) -> db_schema.AssetJob | None:
  """Fetches and claims the next eligible job, if any."""
  active_assets_subquery = (
      select(db_schema.AssetJob.asset_uuid)
      .where(
          db_schema.AssetJob.status
          == db_schema.JobStatus.JOB_STATUS_PROCESSING,
          db_schema.AssetJob.expiration_at >= now,
      )
      .scalar_subquery()
  )

  if backend_id is None:
    backend_cond = db_schema.AssetJob.target_tier_path_id.is_(None)
  else:
    backend_cond = db_schema.TierPath.storage_backend_id == backend_id

  # Fetch the next eligible job, filtering for:
  # 1. Jobs that are queued or whose execution lease has expired (stale jobs).
  # 2. Jobs targeting assets that aren't already actively being processed by
  #    another job, preventing concurrency conflicts on the same asset.
  # 3. Jobs belonging to the requested storage backend.
  # We select the oldest job (FIFO order) and use SKIP LOCKED concurrency
  # control to prevent multiple workers from matching or blocking on the same
  # job.
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
          backend_cond,
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
    job.expiration_at = now + lease_duration
    job.worker_host = hostname
    job.worker_pid = pid
    job.last_updated_at = now
    session.add(job)  # pyrefly: ignore[missing-attribute]
    if (
        job.request_type == db_schema.RequestType.REQUEST_TYPE_COPY
        and job.target_tier_path
    ):
      job.target_tier_path.state = db_schema.TierPathState.IN_PROGRESS
      session.add(job.target_tier_path)  # pyrefly: ignore[missing-attribute]

  return job


async def acquire_next_job(
    session_maker: sessionmaker,
    backend_id: int | None,
    lease_duration: datetime.timedelta,
    hostname: str,
    pid: int,
    max_active: int,
) -> db_schema.AssetJob | None:
  """Queries the database for the next eligible job on the given backend and claims it.

  Args:
    session_maker: A session maker or session factory. MUST be configured with
      `expire_on_commit=False` to prevent returned Job relationships from being
      expired upon transaction commit.
    backend_id: The ID of the storage backend.
    lease_duration: Lease duration for the claimed job.
    hostname: Hostname of the claiming worker.
    pid: PID of the claiming worker.
    max_active: Maximum active jobs allowed on this backend.

  Returns:
    The claimed AssetJob instance, or None if no eligible jobs are available or
    if capacity is full.
  """
  now = datetime.datetime.now(datetime.timezone.utc)

  async with session_maker() as session:
    await session.begin()
    try:
      # Check if there are any jobs at all before acquiring locks
      if not await _has_eligible_jobs(session, backend_id, now):
        await session.rollback()
        return None

      if backend_id is not None:
        if not await _try_lock_backend(session, backend_id, max_active, now):
          # no jobs available on this backend, release lock and return None
          await session.rollback()
          return None

      job = await _claim_eligible_job(
          session, backend_id, lease_duration, hostname, pid, now
      )
      if job is None:
        await session.rollback()
        return None

      await session.commit()
      return job
    except Exception:
      await session.rollback()
      raise
