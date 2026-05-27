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

"""Asset management utilities for Tiering Service.

This module handles database operations for creating, fetching, updating,
and finalizing assets. It also provides conversion functions between
database models and protobuf messages.
"""

from collections.abc import Collection, Sequence
import datetime

from absl import logging
from orbax.checkpoint.experimental.tiering_service import db_schema
from orbax.checkpoint.experimental.tiering_service import storage_backend as storage_backend_lib
from orbax.checkpoint.experimental.tiering_service.proto import tiering_service_pb2
import sqlalchemy
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
import sqlalchemy.orm

from google.protobuf import timestamp_pb2


def _proto_from_db_tier_path(
    tier_path: db_schema.TierPath,
) -> tiering_service_pb2.TierPath:
  """Converts a db_schema.TierPath to a tiering_service_pb2.TierPath.

  Extracts storage backend details and timestamps from the database model
  and constructs the corresponding protobuf message.

  Args:
    tier_path: The database TierPath model instance.

  Returns:
    The constructed protobuf TierPath message.
  """
  storage_backend = tier_path.storage_backend

  def _get_location_kwargs(sb: db_schema.StorageBackend):
    if sb.zone is not None:
      return {"zone": sb.zone}
    if sb.region is not None:
      return {"region": sb.region}
    if sb.multi_regions is not None:
      return {
          "multi_regions": tiering_service_pb2.MultipleRegions(
              regions=sb.multi_regions
          )
      }
    return {}

  storage_backend_kwargs = {
      "id": storage_backend.id,
      "level": storage_backend.level,
      "backend_type": storage_backend.backend_type.value,
      "prefix": storage_backend.prefix,
      **_get_location_kwargs(storage_backend),
  }

  proto_storage_backend = tiering_service_pb2.StorageBackend(
      **storage_backend_kwargs
  )

  ready_at_pb = None
  if tier_path.ready_at is not None:
    ready_at_pb = timestamp_pb2.Timestamp()
    ready_at_pb.FromDatetime(tier_path.ready_at)

  expires_at_pb = None
  if tier_path.expires_at is not None:
    expires_at_pb = timestamp_pb2.Timestamp()
    expires_at_pb.FromDatetime(tier_path.expires_at)

  return tiering_service_pb2.TierPath(
      id=tier_path.id,
      path=tier_path.path,
      storage_backend=proto_storage_backend,
      ready_at=ready_at_pb,
      expires_at=expires_at_pb,
  )


def proto_from_db_asset(db_asset: db_schema.Asset) -> tiering_service_pb2.Asset:
  """Converts a db_schema.Asset to a tiering_service_pb2.Asset.

  Maps database fields, including relationships (tier paths) and timestamps,
  to the protobuf Asset representation.

  Args:
    db_asset: The database Asset model instance.

  Returns:
    The constructed protobuf Asset message.
  """
  proto_asset = tiering_service_pb2.Asset(
      uuid=db_asset.asset_uuid,
      path=db_asset.path,
      user=db_asset.user,
      tags=db_asset.tags if db_asset.tags else [],
      state=db_asset.state.value,
      tier_paths=(
          _proto_from_db_tier_path(tier_path)
          for tier_path in db_asset.tier_paths
      ),
  )

  if db_asset.created_at:
    proto_asset.created_at.FromDatetime(db_asset.created_at)
  if db_asset.finalized_at:
    proto_asset.finalized_at.FromDatetime(db_asset.finalized_at)
  if db_asset.deleted_at:
    proto_asset.deleted_at.FromDatetime(db_asset.deleted_at)
  if db_asset.updated_at:
    proto_asset.updated_at.FromDatetime(db_asset.updated_at)

  return proto_asset


async def fetch_asset_by_identifier(
    session: AsyncSession,
    asset_uuid: str | None = None,
    path: str | None = None,
    inclusive_filter: Collection[db_schema.AssetState] | None = None,
) -> Sequence[db_schema.Asset]:
  """Fetches assets using optional asset_uuid or path identifiers with state filtering.

  Queries the database for assets. You must provide either asset_uuid or path.
  If both are provided, asset_uuid takes precedence.

  Args:
    session: The database session.
    asset_uuid: Optional UUID to filter by.
    path: Optional path to filter by.
    inclusive_filter: Optional collection of states to filter by. If provided,
      only assets in these states will be returned.

  Returns:
    A sequence of matching Asset objects.
  """
  if asset_uuid is None and path is None:
    logging.warning("No uuid or path specified")
    return []

  clauses = []
  if asset_uuid is not None:
    clauses.append(db_schema.Asset.asset_uuid == asset_uuid)
  elif path is not None:
    clauses.append(db_schema.Asset.path == path)

  if inclusive_filter is not None:
    clauses.append(db_schema.Asset.state.in_(inclusive_filter))

  stmt_select = (
      select(db_schema.Asset)
      .options(
          sqlalchemy.orm.selectinload(db_schema.Asset.tier_paths).selectinload(
              db_schema.TierPath.storage_backend
          )
      )
      .where(*clauses)
  )

  stmt = (
      stmt_select.order_by(db_schema.Asset.created_at.desc())
      if path is not None
      else stmt_select
  )

  result = await session.execute(stmt)
  return result.scalars().all()


async def fetch_asset_by_path(
    session: AsyncSession,
    path: str,
    inclusive_filter: Collection[db_schema.AssetState] | None = None,
) -> Sequence[db_schema.Asset]:
  """Fetches assets by path with optional state eligibility constraints.

  Args:
    session: The database session.
    path: The asset path to filter by.
    inclusive_filter: Optional collection of states to filter by.

  Returns:
    A sequence of matching Asset objects.
  """
  return await fetch_asset_by_identifier(
      session, path=path, inclusive_filter=inclusive_filter
  )


async def fetch_asset_by_uuid(
    session: AsyncSession,
    asset_uuid: str,
    inclusive_filter: Collection[db_schema.AssetState] | None = None,
) -> Sequence[db_schema.Asset]:
  """Fetches assets by UUID with optional state eligibility constraints.

  Args:
    session: The database session.
    asset_uuid: The asset UUID to filter by.
    inclusive_filter: Optional collection of states to filter by.

  Returns:
    A sequence of matching Asset objects.
  """
  return await fetch_asset_by_identifier(
      session, asset_uuid=asset_uuid, inclusive_filter=inclusive_filter
  )


def calculate_expires_at(
    interval: datetime.timedelta,
    grace_ratio: float = 0.2,
) -> datetime.datetime:
  """Calculates a new expiration timestamp with a grace period buffer.

  The grace period acts as a buffer to account for communication delay.

  Args:
    interval: The base timeout interval.
    grace_ratio: Optional ratio of the interval to use as a grace buffer
      (default is 0.2).

  Returns:
    The calculated expiration datetime in UTC.
  """
  grace_buffer = interval * grace_ratio
  total_interval = interval + grace_buffer
  return datetime.datetime.now(datetime.timezone.utc) + total_interval


async def create_or_fetch_asset(
    session: AsyncSession,
    request: tiering_service_pb2.ReserveRequest,
    backend: db_schema.StorageBackend,
    config: tiering_service_pb2.ServerConfig,
) -> db_schema.Asset:
  """Creates a new asset or fetches an existing one on unique constraint conflict.

  Attempts to insert a new Asset record with ACTIVE_WRITE state and associates
  it with a new TierPath on the specified backend. If an asset with the same
  path already exists and is active/stored, the insert will be as no-op due to
  database constraints, and this function will then return the existing asset.

  Args:
    session: The database session.
    request: The ReserveRequest containing path, user, and tags.
    backend: The StorageBackend to associate the asset with.
    config: The ServerConfig to get the keep-alive interval.

  Returns:
    The created or fetched Asset object.

  Raises:
    ValueError: If creation fails and the existing asset cannot be retrieved.
  """
  db_asset = db_schema.Asset(
      path=request.path,
      user=request.user,
      tags=list(request.tags),
      state=db_schema.AssetState.ASSET_STATE_ACTIVE_WRITE,
      write_expires_at=calculate_expires_at(
          datetime.timedelta(
              seconds=config.client_keep_alive_interval_seconds
          )
      ),
  )
  storage_path = storage_backend_lib.get_storage_path(backend, request.path)
  tier_path = db_schema.TierPath(
      storage_backend=backend,
      path=storage_path,
  )
  db_asset.tier_paths.append(tier_path)

  try:
    session.add(db_asset)
    await session.commit()
    # Refresh the asset to load DB updated fields such as updated_at.
    await session.refresh(
        db_asset, attribute_names=["created_at", "updated_at"]
    )
    return db_asset
  except IntegrityError:
    await session.rollback()

  logging.info(
      "Reserve: Asset path already exists, fetching existing record: %s",
      request.path,
  )
  active_assets = await fetch_asset_by_path(
      session,
      request.path,
      inclusive_filter=[
          db_schema.AssetState.ASSET_STATE_ACTIVE_WRITE,
          db_schema.AssetState.ASSET_STATE_STORED,
      ],
  )
  if not active_assets:
    # This scenario is unlikely unless the asset was deleted after the
    # insert attempt.
    raise ValueError("Failed to retrieve reserved asset.")
  return active_assets[0]


async def reserve_keep_alive(
    session: AsyncSession,
    uuid_val: str,
    interval: datetime.timedelta,
) -> db_schema.Asset | None:
  """Extends the client writing keep alive expiration timestamp for an asset.

  Args:
    session: The database session.
    uuid_val: The UUID of the asset to update.
    interval: The new timeout interval.

  Returns:
    The updated Asset object, or None if the asset was not found.
  """
  db_assets = await fetch_asset_by_uuid(session, uuid_val)
  db_asset = db_assets[0] if db_assets else None
  if not db_asset:
    return None

  db_asset.write_expires_at = calculate_expires_at(interval)
  await session.commit()
  return db_asset


async def finalize_asset(
    session: AsyncSession,
    db_asset: db_schema.Asset,
) -> db_schema.Asset:
  """Finalizes asset status, transitions state to STORED inside a transaction.

  Updates the asset state, sets the finalized timestamp, and marks the
  associated tier path as ready.

  Args:
    session: The database session.
    db_asset: The Asset model instance to finalize.

  Returns:
    The finalized Asset object.

  Raises:
    ValueError: If the asset is not in ACTIVE_WRITE state.
  """
  if db_asset.state != db_schema.AssetState.ASSET_STATE_ACTIVE_WRITE:
    raise ValueError(
        f"Asset {db_asset.asset_uuid} is in state {db_asset.state.name}, but"
        " must be in ASSET_STATE_ACTIVE_WRITE to be finalized."
    )

  now = datetime.datetime.now(datetime.timezone.utc)
  db_asset.state = db_schema.AssetState.ASSET_STATE_STORED
  db_asset.finalized_at = now
  db_asset.write_expires_at = None

  for tier_path in db_asset.tier_paths:
    tier_path.ready_at = now
    # TODO: b/503445463 - Set expires_at when policy is supported.

  await session.commit()
  await session.refresh(db_asset, attribute_names=["updated_at"])
  return db_asset
