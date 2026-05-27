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
from orbax.checkpoint.experimental.tiering_service import db_schema
from orbax.checkpoint.experimental.tiering_service.proto import tiering_service_pb2
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.future import select
from sqlalchemy.orm import sessionmaker


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
