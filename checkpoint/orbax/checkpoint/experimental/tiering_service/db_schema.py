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

import collections
import enum
import json
from typing import Any, Dict, Optional
import uuid

import sqlalchemy.orm

Base = sqlalchemy.orm.declarative_base()


class AssetState(enum.IntEnum):
  """Represents the lifecycle state of an asset tracked by CTS."""

  ASSET_STATE_UNSPECIFIED = 0
  ASSET_STATE_ACTIVE_WRITE = 1
  ASSET_STATE_STORED = 2
  ASSET_STATE_DELETED = 3
  ASSET_STATE_INCOMPLETE = 4


class BackendType(enum.IntEnum):
  """Identifies the storage backend type for a tier path."""

  BACKEND_TYPE_UNSPECIFIED = 0
  BACKEND_TYPE_LUSTRE = 1
  BACKEND_TYPE_GCS = 2


class JobStatus(enum.Enum):
  """Represents the execution status of an asset job."""

  QUEUED = "QUEUED"
  PROCESSING = "PROCESSING"
  COMPLETED = "COMPLETED"
  FAILED = "FAILED"


class Asset(Base):
  """Database model representing a distinct CTS asset.

  Acts as the primary entity holding assets' metadata and latest storage state.
  Unique asset paths are expected to be unique within the active/stored states.
  Duplicates are allowed for deleted or incomplete states.
  """

  __tablename__ = "assets"

  uuid = sqlalchemy.Column(
      sqlalchemy.String,
      primary_key=True,
      default=lambda: str(uuid.uuid4()),
      index=True,
  )
  unique_path = sqlalchemy.Column(sqlalchemy.String, index=True, nullable=False)
  user = sqlalchemy.Column(sqlalchemy.String, nullable=False)
  tags = sqlalchemy.Column(sqlalchemy.JSON, nullable=True)
  state = sqlalchemy.Column(
      sqlalchemy.Enum(AssetState), default=AssetState.ASSET_STATE_UNSPECIFIED
  )
  created_at = sqlalchemy.Column(sqlalchemy.DateTime, nullable=True)
  finalized_at = sqlalchemy.Column(sqlalchemy.DateTime, nullable=True)
  deleted_at = sqlalchemy.Column(sqlalchemy.DateTime, nullable=True)
  updated_at = sqlalchemy.Column(sqlalchemy.DateTime, nullable=True)

  tier_paths = sqlalchemy.orm.relationship(
      "TierPath", back_populates="asset", cascade="all, delete-orphan"
  )
  jobs = sqlalchemy.orm.relationship(
      "AssetJob", back_populates="asset", cascade="all, delete-orphan"
  )

  def validate_pre_commit(self):
    # Group by level

    levels = collections.defaultdict(list)
    for tp in self.tier_paths:
      levels[tp.level].append(tp)

    for lvl, paths in levels.items():
      # 1. Check backend consistency
      backends = {tp.backend_type for tp in paths}
      if len(backends) > 1:
        raise ValueError(
            f"All tier_paths at the level {lvl} must have the same"
            " backend_type."
        )

      # 2. Duplicate zones/regions/multi_region
      seen_zones = set()
      seen_regions = set()
      seen_multis = set()
      for tp in paths:
        if tp.zone:
          if tp.zone in seen_zones:
            raise ValueError(f"Duplicate zone[{tp.zone}]")
          seen_zones.add(tp.zone)
        if tp.region:
          if tp.region in seen_regions:
            raise ValueError(f"Duplicate region[{tp.region}]")
          seen_regions.add(tp.region)
        if tp.multi_region:
          # Sort elements to make comparison order-agnostic
          sorted_list = (
              sorted(tp.multi_region)
              if isinstance(tp.multi_region, list)
              else tp.multi_region
          )
          mr_val = (
              tuple(sorted_list)
              if isinstance(sorted_list, list)
              else sorted_list
          )
          if mr_val in seen_multis:
            raise ValueError(f"Duplicate multi_region[{mr_val}]")
          seen_multis.add(mr_val)

  __table_args__ = (
      # Enforce unique_path only for live assets (ACTIVE_WRITE, STORED).
      # Duplicates are allowed for DELETED or INCOMPLETE states.
      sqlalchemy.Index(
          "idx_assets_unique_path_active_stored",
          "unique_path",
          unique=True,
          sqlite_where=sqlalchemy.text(
              "state IN ('ASSET_STATE_ACTIVE_WRITE', 'ASSET_STATE_STORED')"
          ),
      ),
  )


class TierPath(Base):
  """Representing a storage location for an asset.

  Asset can be stored in multiple locations across different zones and regions,
  and different storage tiers.
  """

  __tablename__ = "tier_paths"

  id = sqlalchemy.Column(
      sqlalchemy.Integer, primary_key=True, autoincrement=True
  )
  asset_uuid = sqlalchemy.Column(
      sqlalchemy.String,
      sqlalchemy.ForeignKey("assets.uuid", ondelete="CASCADE"),
      nullable=False,
  )
  level = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)
  zone = sqlalchemy.Column(sqlalchemy.String, nullable=True)
  region = sqlalchemy.Column(sqlalchemy.String, nullable=True)
  multi_region = sqlalchemy.Column(sqlalchemy.JSON, nullable=True)
  backend_type = sqlalchemy.Column(
      sqlalchemy.Enum(BackendType), default=BackendType.BACKEND_TYPE_UNSPECIFIED
  )
  path = sqlalchemy.Column(sqlalchemy.String, nullable=False)
  ready_at = sqlalchemy.Column(sqlalchemy.DateTime, nullable=True)
  expires_at = sqlalchemy.Column(sqlalchemy.DateTime, nullable=True)

  asset = sqlalchemy.orm.relationship("Asset", back_populates="tier_paths")

  __table_args__ = (
      # Enforce that only one of zone, region, or multi_region is set.
      sqlalchemy.CheckConstraint(
          "(zone IS NOT NULL) + (region IS NOT NULL) + (multi_region IS NOT"
          " NULL) = 1",
          name="check_mutually_exclusive_locations",
      ),
  )


class AssetJob(Base):
  """Database model representing an ACID transactional job within a queue.

  Validates global serial execution for each asset to eliminate race conditions.
  """

  __tablename__ = "asset_jobs"

  id = sqlalchemy.Column(
      sqlalchemy.Integer, primary_key=True, autoincrement=True
  )
  asset_uuid = sqlalchemy.Column(
      sqlalchemy.String,
      sqlalchemy.ForeignKey("assets.uuid", ondelete="CASCADE"),
      nullable=False,
  )
  request_type = sqlalchemy.Column(sqlalchemy.String, nullable=False)
  status = sqlalchemy.Column(
      sqlalchemy.Enum(JobStatus), default=JobStatus.QUEUED, index=True
  )
  payload_json = sqlalchemy.Column(sqlalchemy.Text, nullable=True)
  created_at = sqlalchemy.Column(sqlalchemy.DateTime, nullable=True)
  completed_at = sqlalchemy.Column(sqlalchemy.DateTime, nullable=True)

  asset = sqlalchemy.orm.relationship("Asset", back_populates="jobs")

  @property
  def payload(self) -> Dict[str, Any]:
    """Gets the deserialized job payload metadata."""
    if not self.payload_json:
      return {}
    return json.loads(self.payload_json)

  @payload.setter
  def payload(self, value: Optional[Dict[str, Any]]):
    """Sets the job payload metadata, serialized as JSON."""
    if value is None:
      self.payload_json = None
    else:
      self.payload_json = json.dumps(value)
