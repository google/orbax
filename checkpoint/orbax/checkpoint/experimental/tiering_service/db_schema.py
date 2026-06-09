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

"""Checkpoint Tiering Service (CTS) database schema definition.

Provides SQLAlchemy models for tracking assets, tier paths, and job queues.
"""

import enum
import itertools
import uuid

import sqlalchemy.orm

Base = sqlalchemy.orm.declarative_base()


class AssetState(enum.IntEnum):
  """The lifecycle state of an asset tracked by CTS."""

  ASSET_STATE_UNSPECIFIED = 0
  ASSET_STATE_ACTIVE_WRITE = 1
  ASSET_STATE_STORED = 2
  ASSET_STATE_DELETED = 3
  ASSET_STATE_INCOMPLETE = 4


class BackendType(enum.IntEnum):
  """The storage backend type for a tier path."""

  BACKEND_TYPE_UNSPECIFIED = 0
  BACKEND_TYPE_LUSTRE = 1
  BACKEND_TYPE_GCS = 2


class JobStatus(enum.IntEnum):
  """The execution status of an asset job."""

  JOB_STATUS_UNSPECIFIED = 0
  JOB_STATUS_QUEUED = 1
  JOB_STATUS_PROCESSING = 2
  JOB_STATUS_COMPLETED = 3
  JOB_STATUS_FAILED = 4


class RequestType(enum.IntEnum):
  """The operation type requested for an asset job."""

  REQUEST_TYPE_UNSPECIFIED = 0
  REQUEST_TYPE_COPY = 1
  REQUEST_TYPE_DELETE_FROM_INSTANCE = 2
  REQUEST_TYPE_DELETE_FROM_ALL_TIERS = 3


class Asset(Base):
  """A CTS asset representing a complete checkpoint.

  Acts as the primary entity holding assets' metadata and latest storage state.
  Unique asset paths are expected to be unique within the active/stored states.
  Duplicates are allowed for deleted or incomplete states.

  Attributes:
    asset_uuid: A unique identifier for the asset (Primary Key).
    path: The user-defined path identifying the asset.
    user: The user who owns or created the asset.
    tags: Optional JSON field for storing arbitrary tags.
    state: The current lifecycle state of the asset, an AssetState enum.
    created_at: Timestamp when the asset record was created.
    finalized_at: Timestamp when the asset was marked as finalized.
    deleted_at: Timestamp when the asset was marked as deleted.
    updated_at: Timestamp of the last update to the asset record.
    tier_paths: A relationship to the TierPath objects associated with this
      asset.
    jobs: A relationship to the AssetJob objects associated with this asset.
  """

  __tablename__ = "assets"

  asset_uuid = sqlalchemy.Column(
      sqlalchemy.String,
      primary_key=True,
      default=lambda: str(uuid.uuid4()),
  )
  path = sqlalchemy.Column(sqlalchemy.String, index=True, nullable=False)
  user = sqlalchemy.Column(sqlalchemy.String, nullable=False)
  tags = sqlalchemy.Column(sqlalchemy.JSON, nullable=True)
  state = sqlalchemy.Column(
      sqlalchemy.Enum(AssetState), default=AssetState.ASSET_STATE_UNSPECIFIED
  )
  created_at = sqlalchemy.Column(
      sqlalchemy.DateTime(timezone=True),
      server_default=sqlalchemy.sql.func.now(),
      nullable=False,
  )
  finalized_at = sqlalchemy.Column(
      sqlalchemy.DateTime(timezone=True), nullable=True
  )
  deleted_at = sqlalchemy.Column(
      sqlalchemy.DateTime(timezone=True), nullable=True
  )
  write_expires_at = sqlalchemy.Column(
      sqlalchemy.DateTime(timezone=True), nullable=True
  )
  updated_at = sqlalchemy.Column(
      sqlalchemy.DateTime(timezone=True),
      server_default=sqlalchemy.sql.func.now(),
      onupdate=sqlalchemy.sql.func.now(),
      nullable=False,
  )

  tier_paths = sqlalchemy.orm.relationship(
      "TierPath", back_populates="asset", cascade="all, delete-orphan"
  )
  jobs = sqlalchemy.orm.relationship(
      "AssetJob", back_populates="asset", cascade="all, delete-orphan"
  )

  __table_args__ = (
      # Enforce path only for live assets (ACTIVE_WRITE, STORED).
      # Duplicates are allowed for DELETED or INCOMPLETE states.
      sqlalchemy.Index(
          "idx_assets_path_active_stored",
          "path",
          unique=True,
          sqlite_where=sqlalchemy.column("state").in_([
              AssetState.ASSET_STATE_ACTIVE_WRITE.name,
              AssetState.ASSET_STATE_STORED.name,
          ]),
          postgresql_where=sqlalchemy.column("state").in_([
              AssetState.ASSET_STATE_ACTIVE_WRITE.name,
              AssetState.ASSET_STATE_STORED.name,
          ]),
      ),
  )

  def __repr__(self):
    return (
        f"Asset(asset_uuid={self.asset_uuid!r},"
        f" path={self.path!r}, state={self.state.name!r},"
        f" user={self.user!r})"
    )


class StorageBackend(Base):
  """A system-wide available storage instance.

  The table should be populated once by a single CTS server during
  initialization in one transaction.  Afterwards, the content is only validated
  against the server configuration.

  Attributes:
    id: Primary key for the storage backend entry.
    level: An integer representing the tiering level.
    zone: The zone where the storage backend resides.
    region: The region where the storage backend resides.
    multi_regions: A list of regions forming a multi-region deployment.
    backend_type: The type of storage (e.g., Lustre, GCS).
    prefix: Storage backend prefix (e.g., gs://bucket-name, /mnt/lustre/).
    tier_paths: Relationship to the TierPath objects utilizing this backend.
  """

  __tablename__ = "storage_backends"

  id = sqlalchemy.Column(
      sqlalchemy.Integer, primary_key=True, autoincrement=True
  )
  level = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)
  zone = sqlalchemy.Column(sqlalchemy.String, nullable=True)
  region = sqlalchemy.Column(sqlalchemy.String, nullable=True)
  multi_regions = sqlalchemy.Column(sqlalchemy.JSON, nullable=True)
  backend_type = sqlalchemy.Column(
      sqlalchemy.Enum(BackendType), default=BackendType.BACKEND_TYPE_UNSPECIFIED
  )
  prefix = sqlalchemy.Column(sqlalchemy.String, nullable=False)

  tier_paths = sqlalchemy.orm.relationship(
      "TierPath", back_populates="storage_backend", cascade="all, delete-orphan"
  )

  __table_args__ = (
      # Enforce that only one of zone, region, or multi_regions is set.
      sqlalchemy.CheckConstraint(
          "(CASE WHEN zone IS NOT NULL THEN 1 ELSE 0 END + "
          "CASE WHEN region IS NOT NULL THEN 1 ELSE 0 END + "
          "CASE WHEN multi_regions IS NOT NULL THEN 1 ELSE 0 END) = 1",
          name="check_mutually_exclusive_locations",
      ),
  )

  def __repr__(self):
    if self.zone:
      location = f"zone={self.zone!r}"
    elif self.region:
      location = f"region={self.region!r}"
    elif self.multi_regions:
      location = f"multi_regions={self.multi_regions!r}"
    else:
      location = "None"
    return (
        f"StorageBackend(id={self.id}, level={self.level},"
        f" backend_type={self.backend_type.name!r}, prefix={self.prefix!r},"
        f" {location})"
    )

  def validate_pre_commit(self) -> None:
    """Validates StorageBackend constraints before a commit.

    This validates for:
    1.  All StorageBackend entries at the same `level` must share the same
        `backend_type`.
    2.  Within the same `level`, each location identifier (`zone`, `region`, or
        `multi_regions`) must be unique across all StorageBackend entries.

    The validation is performed against other StorageBackend objects currently
    loaded or newly added within the same SQLAlchemy session.

    Raises:
      ValueError: If any of the validation constraints are violated.
    """
    session = sqlalchemy.orm.object_session(self)
    if session is None:
      # No session, so no need to validate.
      raise ValueError("No session found")

    session_backends = [
        obj
        for obj in set(
            itertools.chain(session.new, session.identity_map.values())
        )
        if isinstance(obj, StorageBackend) and obj.level == self.level
    ]
    types = {b.backend_type for b in session_backends}
    if len(types) > 1:
      raise ValueError(
          f"StorageBackend at level {self.level} must have the same"
          f" backend_type, but found conflicting types: {types}"
      )

    seen_zones = set()
    seen_regions = set()
    seen_multis = set()
    for b in session_backends:
      if b.zone:
        if b.zone in seen_zones:
          raise ValueError(f"Duplicate zone[{b.zone}]")
        seen_zones.add(b.zone)
      if b.region:
        if b.region in seen_regions:
          raise ValueError(f"Duplicate region[{b.region}]")
        seen_regions.add(b.region)
      if b.multi_regions:
        sorted_list = (
            sorted(b.multi_regions)
            if isinstance(b.multi_regions, list)
            else b.multi_regions
        )
        mr_val = (
            tuple(sorted_list) if isinstance(sorted_list, list) else sorted_list
        )
        if mr_val in seen_multis:
          raise ValueError(f"Duplicate multi_regions[{mr_val}]")
        seen_multis.add(mr_val)


@sqlalchemy.event.listens_for(StorageBackend, "before_insert")
@sqlalchemy.event.listens_for(StorageBackend, "before_update")
def _validate_storage_backend_before_flush(
    mapper: sqlalchemy.orm.Mapper,
    connection: sqlalchemy.engine.Connection,
    target: StorageBackend,
) -> None:
  del mapper, connection
  target.validate_pre_commit()


class TierPath(Base):
  """A storage location for an asset.

  Asset can be stored in multiple locations across different zones and regions,
  and different storage tiers.

  Attributes:
    id: Primary key for the tier path.
    asset_uuid: Foreign key linking to the `Asset`.
    storage_backend_id: Foreign key linking to the `StorageBackend`.
    path: The concrete storage path (e.g., GCS URI, Lustre path).
    ready_at: Timestamp when the asset became available at this tier path.
    expires_at: Timestamp when the asset is scheduled to expire from this tier
      path.
    tier_path_uuid: A unique identifier for this tier path.
    asset: SQLAlchemy relationship to the `Asset` object.
    storage_backend: SQLAlchemy relationship to the `StorageBackend` object.
  """

  __tablename__ = "tier_paths"

  id = sqlalchemy.Column(
      sqlalchemy.Integer, primary_key=True, autoincrement=True
  )
  asset_uuid = sqlalchemy.Column(
      sqlalchemy.String,
      sqlalchemy.ForeignKey("assets.asset_uuid", ondelete="CASCADE"),
      nullable=False,
  )
  storage_backend_id = sqlalchemy.Column(
      sqlalchemy.Integer,
      sqlalchemy.ForeignKey("storage_backends.id", ondelete="CASCADE"),
      nullable=False,
  )
  path = sqlalchemy.Column(sqlalchemy.String, nullable=False)
  ready_at = sqlalchemy.Column(
      sqlalchemy.DateTime(timezone=True), nullable=True
  )
  expires_at = sqlalchemy.Column(
      sqlalchemy.DateTime(timezone=True), nullable=True
  )
  tier_path_uuid = sqlalchemy.Column(
      sqlalchemy.String,
      unique=True,
      nullable=False,
      default=lambda: str(uuid.uuid4()),
  )

  asset = sqlalchemy.orm.relationship("Asset", back_populates="tier_paths")
  storage_backend = sqlalchemy.orm.relationship(
      "StorageBackend", back_populates="tier_paths"
  )

  __table_args__ = (
      # An asset can have at most one TierPath for a given storage backend.
      sqlalchemy.UniqueConstraint(
          "asset_uuid",
          "storage_backend_id",
          name="uq_tier_path_asset_backend",
      ),
  )

  def __repr__(self):
    return (
        f"TierPath(id={self.id}, asset_uuid='{self.asset_uuid}',"
        f" storage_backend_id={self.storage_backend_id}, path='{self.path}',"
        f" ready_at={self.ready_at}, expires_at={self.expires_at})"
    )


class AssetJob(Base):
  """A queued operation for an Asset.

  This table ensures that multiple servers can try to queue and update job
  status without race conditions.

  Attributes:
    id: Primary key for the job.
    asset_uuid: Foreign key to the target Asset.
    request_type: The requested operation type, an instance of RequestType.
    status: Current execution status of the job, an instance of JobStatus.
    target_tier_path_id: Foreign key to the targeted TierPath for operations
      such as COPY or DELETE_FROM_INSTANCE.
    created_at: Timestamp when the job was created.
    completed_at: Timestamp when the job was completed.
    asset: Relationship to the associated Asset.
    target_tier_path: Relationship to the targeted TierPath.
  """

  __tablename__ = "asset_jobs"

  id = sqlalchemy.Column(
      sqlalchemy.Integer, primary_key=True, autoincrement=True
  )
  asset_uuid = sqlalchemy.Column(
      sqlalchemy.String,
      sqlalchemy.ForeignKey("assets.asset_uuid", ondelete="CASCADE"),
      nullable=False,
  )
  request_type = sqlalchemy.Column(
      sqlalchemy.Enum(RequestType),
      default=RequestType.REQUEST_TYPE_UNSPECIFIED,
      nullable=False,
  )
  status = sqlalchemy.Column(
      sqlalchemy.Enum(JobStatus),
      default=JobStatus.JOB_STATUS_QUEUED,
      index=True,
  )
  # Target tier path for COPY and DELETE_FROM_INSTANCE requests
  target_tier_path_id = sqlalchemy.Column(
      sqlalchemy.Integer,
      sqlalchemy.ForeignKey("tier_paths.id", ondelete="SET NULL"),
      nullable=True,
  )
  request_id = sqlalchemy.Column(
      sqlalchemy.String,
      nullable=False,
      unique=True,
      default=lambda: str(uuid.uuid4()),
  )
  transfer_status = sqlalchemy.Column(sqlalchemy.JSON, nullable=True)
  expiration_at = sqlalchemy.Column(
      sqlalchemy.DateTime(timezone=True), nullable=True
  )
  last_updated_at = sqlalchemy.Column(
      sqlalchemy.DateTime(timezone=True), nullable=True
  )
  worker_host = sqlalchemy.Column(sqlalchemy.String, nullable=True)
  worker_pid = sqlalchemy.Column(sqlalchemy.Integer, nullable=True)

  created_at = sqlalchemy.Column(
      sqlalchemy.DateTime(timezone=True),
      server_default=sqlalchemy.sql.func.now(),
      nullable=False,
  )
  completed_at = sqlalchemy.Column(
      sqlalchemy.DateTime(timezone=True), nullable=True
  )

  asset = sqlalchemy.orm.relationship("Asset", back_populates="jobs")
  target_tier_path = sqlalchemy.orm.relationship("TierPath")

  __table_args__ = (
      # target_tier_path is required in COPY and DELETE_FROM_INSTANCE requests,
      # except when the job has failed and its target tier path was cleaned up.
      sqlalchemy.CheckConstraint(
          """
          (request_type IN ('REQUEST_TYPE_COPY', 'REQUEST_TYPE_DELETE_FROM_INSTANCE')
           AND (status = 'JOB_STATUS_FAILED' OR target_tier_path_id IS NOT NULL))
          OR
          (request_type IN ('REQUEST_TYPE_DELETE_FROM_ALL_TIERS', 'REQUEST_TYPE_UNSPECIFIED')
           AND target_tier_path_id IS NULL)
          """,
          name="check_asset_job_valid_payload",
      ),
  )

  def __repr__(self):
    return (
        f"AssetJob(id={self.id}, asset_uuid='{self.asset_uuid}',"
        f" request_type='{self.request_type.name}',"
        f" status='{self.status.name}',"
        f" target_tier_path_id={self.target_tier_path_id},"
        f" request_id='{self.request_id}',"
        f" created_at={self.created_at}, completed_at={self.completed_at})"
    )
