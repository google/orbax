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
import dataclasses
import datetime
import uuid

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


class DeletionPendingError(ValueError):
  """Raised when an operation is attempted on an asset/TierPath marked for deletion."""


class PrefetchFailedError(ValueError):
  """Raised when a prefetch operation has failed."""


@dataclasses.dataclass
class CreatePrefetchJobResult:
  """Result of creating a prefetch job.

  Attributes:
    asset: The updated asset, or None if not found.
    created: True if a new job was successfully created, False if it failed due
      to a concurrent insert or there is already an existing job queued.
  """

  asset: db_schema.Asset | None
  created: bool


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

  state_val = tier_path.state or db_schema.TierPathState.UNSPECIFIED
  proto_state_name = f"TIER_PATH_STATE_{state_val.name}"
  state_pb = tiering_service_pb2.TierPathState.Value(proto_state_name)

  return tiering_service_pb2.TierPath(
      id=tier_path.id,
      path=tier_path.path,
      storage_backend=proto_storage_backend,
      ready_at=ready_at_pb,
      expires_at=expires_at_pb,
      tier_path_uuid=tier_path.tier_path_uuid,
      state=state_pb,
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
    *,
    tier_path_uuid: str,
    storage_path: str,
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
    tier_path_uuid: The pre-generated UUID for the new TierPath.
    storage_path: The pre-generated physical storage path.

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
  backend = await session.merge(backend)
  tier_path = db_schema.TierPath(
      storage_backend=backend,
      path=storage_path,
      tier_path_uuid=tier_path_uuid,
  )
  db_asset.tier_paths.append(tier_path)

  try:
    session.add(db_asset)  # pyrefly: ignore[missing-attribute]
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
  stmt = (
      select(db_schema.Asset)
      .where(db_schema.Asset.asset_uuid == db_asset.asset_uuid)
      .options(sqlalchemy.orm.joinedload(db_schema.Asset.tier_paths))
  )
  res = await session.execute(stmt)
  db_asset = res.scalars().first()

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
    tier_path.state = db_schema.TierPathState.READY
    # TODO: b/503445463 - Set expires_at when policy is supported.

  await session.commit()
  await session.refresh(db_asset, attribute_names=["updated_at"])
  return db_asset


def find_matching_backend(
    backends: Sequence[db_schema.StorageBackend],
    source_backend: db_schema.StorageBackend,
) -> db_schema.StorageBackend | None:
  """Finds a storage backend in `backends` that matches `source_backend`.

  The matching logic prioritizes:
  1. Exact zone match (if both source and target have zone).
  2. Region match (if they share the same region, or region extracted from
     zone).
  3. Multi-regions match (if target has multi_regions containing source's
     region).

  Args:
    backends: A list of candidate StorageBackend objects.
    source_backend: The source StorageBackend object.

  Returns:
    The matching StorageBackend object, or None if no match is found.
  """

  def _get_region_from_zone(zone: str) -> str:
    return zone.rsplit("-", 1)[0]

  # Get source region
  source_region = None
  if source_backend.zone:
    source_region = _get_region_from_zone(source_backend.zone)
  elif source_backend.region:
    source_region = source_backend.region

  # 1. Match by zone first (if both have zone)
  if source_backend.zone:
    for b in backends:
      if b.zone and b.zone == source_backend.zone:
        return b

  # 2. Match by region
  if source_region:
    # Try exact region match first
    for b in backends:
      if b.region and b.region == source_region:
        return b
      if b.zone and _get_region_from_zone(b.zone) == source_region:
        return b

    # Try multi-regions match
    for b in backends:
      if b.multi_regions and source_region in b.multi_regions:
        return b

  # 3. Handle source having multi_regions (if L0 is multi-regional)
  if source_backend.multi_regions:
    # Check if target has region that is in source's multi_regions
    for b in backends:
      if b.region and b.region in source_backend.multi_regions:
        return b
      if (
          b.zone
          and _get_region_from_zone(b.zone) in source_backend.multi_regions
      ):
        return b
      if b.multi_regions:
        # Check overlap
        overlap = set(source_backend.multi_regions) & set(b.multi_regions)
        if overlap:
          return b

  return None


async def trigger_l0_to_l1_copy(
    session: AsyncSession,
    db_asset: db_schema.Asset,
) -> None:
  """Triggers copying the asset from Level 0 to Level 1 storage.

  We assume always preserve policy, so trigger L0-L1 immediately.
  Generically calls them L0 and L1 as they are not always Lustre and GCS.

  Args:
    session: The database session.
    db_asset: The Asset model instance to copy.
  """
  # Find the Level 0 backend from the asset's existing tier paths
  l0_backend = None
  for tp in db_asset.tier_paths:
    if tp.storage_backend.level == 0:
      l0_backend = tp.storage_backend
      break

  if l0_backend is None:
    raise ValueError(f"Asset {db_asset.asset_uuid} has no Level 0 backend.")

  # Look up Level 1 storage backend to queue the copy job
  stmt = select(db_schema.StorageBackend).where(
      db_schema.StorageBackend.level == 1
  )
  res = await session.execute(stmt)
  level_1_backends = res.scalars().all()

  l1_backend = find_matching_backend(level_1_backends, l0_backend)
  if level_1_backends and l1_backend is None:
    raise ValueError(
        f"No matching Level 1 backend found for Level 0 backend: {l0_backend}"
    )

  if l1_backend is not None:
    l1_tp_uuid = str(uuid.uuid4())
    l1_path = storage_backend_lib.get_storage_path(
        l1_backend, db_asset.path, l1_tp_uuid
    )
    new_l1_tp = db_schema.TierPath(
        storage_backend=l1_backend,
        path=l1_path,
        tier_path_uuid=l1_tp_uuid,
        state=db_schema.TierPathState.PENDING,
    )
    db_asset.tier_paths.append(new_l1_tp)

    db_job = db_schema.AssetJob(
        asset_uuid=db_asset.asset_uuid,
        request_type=db_schema.RequestType.REQUEST_TYPE_COPY,
        status=db_schema.JobStatus.JOB_STATUS_QUEUED,
        target_tier_path=new_l1_tp,
    )
    session.add(db_job)  # pyrefly: ignore[missing-attribute]
    logging.info(
        "Finalize: Queued copy job to Level 1 for asset %s, target path: %s",
        db_asset.asset_uuid,
        l1_path,
    )
    await session.commit()


async def is_delete_pending(session: AsyncSession, *, asset_uuid: str) -> bool:
  """Checks if there is a pending delete job for the asset."""
  stmt = (
      select(db_schema.AssetJob)
      .filter_by(
          asset_uuid=asset_uuid,
          request_type=db_schema.RequestType.REQUEST_TYPE_DELETE_FROM_ALL_TIERS,
      )
      .where(
          db_schema.AssetJob.status.in_([
              db_schema.JobStatus.JOB_STATUS_QUEUED,
              db_schema.JobStatus.JOB_STATUS_PROCESSING,
          ])
      )
  )
  result = await session.execute(stmt)
  return bool(result.scalars().first())


async def is_tier_path_delete_pending(
    session: AsyncSession,
    *,
    asset_uuid: str,
    tier_path_id: int,
) -> bool:
  """Checks if there is a pending delete job for the specific TierPath."""
  stmt = (
      select(db_schema.AssetJob)
      .filter_by(
          asset_uuid=asset_uuid,
          target_tier_path_id=tier_path_id,
          request_type=db_schema.RequestType.REQUEST_TYPE_DELETE_FROM_INSTANCE,
      )
      .where(
          db_schema.AssetJob.status.in_([
              db_schema.JobStatus.JOB_STATUS_QUEUED,
              db_schema.JobStatus.JOB_STATUS_PROCESSING,
          ])
      )
  )
  result = await session.execute(stmt)
  return bool(result.scalars().first())


async def create_prefetch_job(
    session: AsyncSession,
    db_asset: db_schema.Asset,
    *,
    backend: db_schema.StorageBackend,
    storage_path: str,
    tier_path_uuid: str,
    client_keep_alive_interval: datetime.timedelta,
) -> CreatePrefetchJobResult:
  """Queues a prefetch job for the given asset to the target backend.

  This function executes atomically in a single transaction. It creates both
  the `TierPath` and the corresponding `AssetJob` together, ensuring that we
  never commit a "dangling" `TierPath` without an associated prefetch job.
  Specifically, if an existing `TierPath` already exists and is active or in
  progress, the operation behaves as a no-op.

  Args:
    session: The database session (active transaction).
    db_asset: The asset to prefetch.
    backend: The target storage backend (level 0).
    storage_path: The storage path to use for the new TierPath.
    tier_path_uuid: The pre-generated unique identifier for the TierPath.
    client_keep_alive_interval: The interval to set for the initial expires_at
      of the TierPath.

  Returns:
    A CreatePrefetchJobResult containing the updated asset and a boolean
    indicating whether a new job was created.

  Raises:
    DeletionPendingError: If the asset is already marked for deletion.
  """

  stmt = (
      select(db_schema.Asset)
      .where(db_schema.Asset.asset_uuid == db_asset.asset_uuid)
      .options(sqlalchemy.orm.joinedload(db_schema.Asset.tier_paths))
  )
  res = await session.execute(stmt)
  db_asset = res.scalars().first()

  if db_asset is None:
    return CreatePrefetchJobResult(asset=None, created=False)

  # Check if there is already a preceding delete job
  if await is_delete_pending(session, asset_uuid=db_asset.asset_uuid):
    raise DeletionPendingError(
        f"Cannot prefetch asset {db_asset.asset_uuid} because it is marked for"
        " deletion."
    )
    # TODO: b/503444041 - potentially allow prefetch for assets that deletion
    # hasn't started yet.

  logging.info(
      "Prefetch: Creating new pending TierPath and job for asset %s and"
      " backend %s",
      db_asset.asset_uuid,
      backend.id,
  )
  backend = await session.merge(backend)
  new_tp = db_schema.TierPath(
      storage_backend=backend,
      path=storage_path,
      tier_path_uuid=tier_path_uuid,
      expires_at=calculate_expires_at(client_keep_alive_interval),
  )
  db_asset.tier_paths.append(new_tp)

  db_job = db_schema.AssetJob(
      asset_uuid=db_asset.asset_uuid,
      request_type=db_schema.RequestType.REQUEST_TYPE_COPY,
      status=db_schema.JobStatus.JOB_STATUS_QUEUED,
      target_tier_path=new_tp,
  )
  session.add(db_job)  # pyrefly: ignore[missing-attribute]

  asset_uuid = db_asset.asset_uuid
  backend_id = backend.id
  try:
    await session.commit()
  except IntegrityError:
    await session.rollback()
    logging.debug(
        "Prefetch: Concurrent insert detected for asset %s and backend %s,"
        " rolling back",
        asset_uuid,
        backend_id,
    )
    db_assets = await fetch_asset_by_uuid(session, asset_uuid)
    return CreatePrefetchJobResult(
        asset=(db_assets[0] if db_assets else None), created=False
    )

  await session.refresh(db_asset, attribute_names=["updated_at"])
  return CreatePrefetchJobResult(asset=db_asset, created=True)


async def prefetch_keep_alive(
    session: AsyncSession,
    *,
    tier_path_uuid: str,
    interval: datetime.timedelta,
) -> db_schema.Asset | None:
  """Extend the TierPath's expiration timestamp.

  Args:
    session: The database session.
    tier_path_uuid: The UUID of the TierPath to update.
    interval: The new timeout interval.

  Returns:
    The updated Asset object, or None if the TierPath was not found.

  Raises:
    DeletionPendingError: If the asset associated with the TierPath is marked
      for deletion, or if the specific TierPath instance is marked for
      deletion.
    PrefetchFailedError: If the prefetch operation on the TierPath failed.
  """
  stmt = select(db_schema.TierPath).filter_by(tier_path_uuid=tier_path_uuid)
  result = await session.execute(stmt)
  tp = result.scalars().first()
  if tp is None:
    return None

  if tp.state == db_schema.TierPathState.FAILED:
    raise PrefetchFailedError(f"Prefetch failed for TierPath {tier_path_uuid}.")

  if await is_delete_pending(session, asset_uuid=tp.asset_uuid):
    raise DeletionPendingError(f"Asset {tp.asset_uuid} is marked for deletion.")

  if await is_tier_path_delete_pending(
      session, asset_uuid=tp.asset_uuid, tier_path_id=tp.id
  ):
    raise DeletionPendingError(
        f"TierPath {tier_path_uuid} is marked for deletion."
    )
    # TODO: b/503444041 - potentially allow prefetch for assets that deletion
    # hasn't started yet.

  if tp.expires_at is None:
    logging.debug(
        "TierPath %s has no expires_at (still copying or permanent), no-op",
        tier_path_uuid,
    )
    db_assets = await fetch_asset_by_uuid(session, tp.asset_uuid)
    return db_assets[0] if db_assets else None

  new_expires_at = calculate_expires_at(interval)
  existing_expires_at = tp.expires_at
  compared_expires_at = (
      existing_expires_at.replace(tzinfo=datetime.timezone.utc)
      if existing_expires_at.tzinfo is None
      else existing_expires_at
  )
  if new_expires_at > compared_expires_at:
    logging.debug(
        "Extending TierPath %s expires_at from %s to %s",
        tier_path_uuid,
        existing_expires_at,
        new_expires_at,
    )
    tp.expires_at = new_expires_at
    await session.commit()
  else:
    logging.debug(
        "New expires_at %s is not longer than existing %s for TierPath %s,"
        " no-op",
        new_expires_at,
        existing_expires_at,
        tier_path_uuid,
    )

  db_assets = await fetch_asset_by_uuid(session, tp.asset_uuid)
  return db_assets[0] if db_assets else None


async def queue_delete_asset_job(
    session: AsyncSession,
    db_asset: db_schema.Asset,
) -> None:
  """Queues a delete job for the asset.

  Args:
    session: The database session.
    db_asset: The asset to delete.

  Raises:
    ValueError: If the asset is not found, is already deleted, or already has a
      pending delete job.
  """
  stmt = (
      select(db_schema.Asset)
      .where(db_schema.Asset.asset_uuid == db_asset.asset_uuid)
      .with_for_update()  # Locks the row until session.commit().
      .execution_options(populate_existing=True)
  )
  res = await session.execute(stmt)
  locked_asset = res.scalars().first()
  if locked_asset is None:
    raise ValueError(f"Asset {db_asset.asset_uuid} not found for locking")

  if locked_asset.state == db_schema.AssetState.ASSET_STATE_DELETED:
    raise ValueError(f"Asset {db_asset.asset_uuid} is already deleted.")

  if await is_delete_pending(session, asset_uuid=locked_asset.asset_uuid):
    raise ValueError(f"Asset {db_asset.asset_uuid} has pending delete.")

  db_job = db_schema.AssetJob(
      asset_uuid=locked_asset.asset_uuid,
      request_type=db_schema.RequestType.REQUEST_TYPE_DELETE_FROM_ALL_TIERS,
      status=db_schema.JobStatus.JOB_STATUS_QUEUED,
  )
  session.add(db_job)  # pyrefly: ignore[missing-attribute]

  await session.commit()


async def begin_delete_tier_path(
    session: AsyncSession,
    tier_path: db_schema.TierPath,
) -> db_schema.TierPath:
  """Transitions a tier path state to DELETE_IN_PROCESS and clears read access."""
  tier_path.state = db_schema.TierPathState.DELETE_IN_PROCESS
  tier_path.ready_at = None
  tier_path.expires_at = None
  session.add(tier_path)  # pyrefly: ignore[missing-attribute]
  return tier_path


async def begin_delete_asset(
    session: AsyncSession,
    db_asset: db_schema.Asset,
) -> db_schema.Asset | None:
  """Transitions asset state to DELETED and all its tier paths to DELETE_IN_PROCESS."""
  stmt = (
      select(db_schema.Asset)
      .where(db_schema.Asset.asset_uuid == db_asset.asset_uuid)
      .options(sqlalchemy.orm.joinedload(db_schema.Asset.tier_paths))
  )
  res = await session.execute(stmt)
  db_asset = res.scalars().first()

  if db_asset is None:
    return None

  db_asset.state = db_schema.AssetState.ASSET_STATE_DELETED
  db_asset.deleted_at = datetime.datetime.now(datetime.timezone.utc)

  for tp in db_asset.tier_paths:
    tp.state = db_schema.TierPathState.DELETE_IN_PROCESS
    tp.ready_at = None
    tp.expires_at = None
  return db_asset


async def complete_delete_asset(
    session: AsyncSession,
    db_asset: db_schema.Asset,
) -> db_schema.Asset | None:
  """Transitions asset state to DELETED and marks all tier paths as deleted.

  Args:
    session: The database session.
    db_asset: The Asset model instance to delete.

  Returns:
    The updated Asset object, or None if it was concurrently deleted.
  """
  stmt = (
      select(db_schema.Asset)
      .where(db_schema.Asset.asset_uuid == db_asset.asset_uuid)
      .options(sqlalchemy.orm.joinedload(db_schema.Asset.tier_paths))
  )
  res = await session.execute(stmt)
  db_asset = res.scalars().first()

  if db_asset is None:
    return None

  now = datetime.datetime.now(datetime.timezone.utc)
  db_asset.state = db_schema.AssetState.ASSET_STATE_DELETED
  db_asset.deleted_at = now

  for tp in db_asset.tier_paths:
    tp.state = db_schema.TierPathState.DELETED
    tp.ready_at = None
    tp.expires_at = None

  return db_asset


async def complete_delete_tier_path(
    session: AsyncSession,
    tier_path: db_schema.TierPath,
) -> db_schema.TierPath:
  """Transitions a tier path state to DELETED.

  Args:
    session: The database session.
    tier_path: The TierPath model instance to update.

  Returns:
    The updated TierPath object.
  """
  tier_path.state = db_schema.TierPathState.DELETED
  tier_path.ready_at = None
  tier_path.expires_at = None
  session.add(tier_path)  # pyrefly: ignore[missing-attribute]
  return tier_path
