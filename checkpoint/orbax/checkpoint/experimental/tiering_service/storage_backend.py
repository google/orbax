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

"""Storage backend library for Tiering Service.

This module provides helper functions for querying storage backends from the
database, locating the closest backend based on client location, and formatting
backend paths.
"""

from collections.abc import Sequence
from orbax.checkpoint.experimental.tiering_service import db_schema
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select


async def find_backends_by_level(
    session: AsyncSession,
    level: int = 0,
) -> Sequence[db_schema.StorageBackend]:
  """Queries backends with the given level from the database.

  Args:
    session: The database session.
    level: The tiering level to filter by (default is 0).

  Returns:
    A sequence of StorageBackend objects matching the level.
  """
  result = await session.execute(
      select(db_schema.StorageBackend).filter_by(level=level)
  )
  return result.scalars().all()


def locate_closest_backend(
    backends: Sequence[db_schema.StorageBackend],
    zone: str | None,
    region: str | None,
) -> db_schema.StorageBackend | None:
  """Selects the closest storage backend matching input location metrics.

  Attempts to match the client's zone or region with the available backends.
  If zone is provided, it tries to match the exact zone, or falls back to
  matching the zone prefix with backend region/multi-regions if no region is
  specified. If region is provided, it tries to match the region.

  Args:
    backends: A sequence of StorageBackend objects to choose from.
    zone: The client's zone, or None.
    region: The client's region, or None.

  Returns:
    The closest matching StorageBackend, or None if no match is found.
  """
  if zone is not None:
    # Match the exact zone.
    for backend in backends:
      if backend.zone == zone:
        return backend

    # When no region specified, match the zone prefix with the region.
    if region is None:
      for backend in backends:
        if backend.region is not None and zone.startswith(backend.region):
          return backend
        if backend.multi_regions is not None and zone.startswith(
            tuple(backend.multi_regions)
        ):
          return backend

  if region is not None:
    for backend in backends:
      if backend.region == region:
        return backend
      if backend.multi_regions is not None and region in backend.multi_regions:
        return backend

  return None


def get_storage_path(
    backend: db_schema.StorageBackend,
    relative_path: str,
) -> str:
  """Builds the absolute storage path for the given backend and relative path.

  Combines the backend's prefix with the relative path, ensuring proper
  formatting.

  Args:
    backend: The StorageBackend target.
    relative_path: The relative path of the asset.

  Returns:
    The absolute storage path.
  """
  return f"{backend.prefix.rstrip('/')}/{relative_path.lstrip('/')}"


def get_backend_name(backend: db_schema.StorageBackend) -> str:
  """Returns a user-friendly name for the backend type.

  Args:
    backend: The StorageBackend target.

  Returns:
    A string representation of the backend type (e.g., "GCS", "Lustre",
    "unknown").
  """
  if backend.backend_type == db_schema.BackendType.BACKEND_TYPE_GCS:
    return "GCS"
  elif backend.backend_type == db_schema.BackendType.BACKEND_TYPE_LUSTRE:
    return "Lustre"
  else:
    return "unknown"
