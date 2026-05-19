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

"""Server library utilities for Tiering Service."""

from orbax.checkpoint.experimental.tiering_service import db_schema
from orbax.checkpoint.experimental.tiering_service.proto import tiering_service_pb2


def db_asset_to_proto(db_asset: db_schema.Asset) -> tiering_service_pb2.Asset:
  """Converts a db_schema.Asset to a tiering_service_pb2.Asset."""
  proto_asset = tiering_service_pb2.Asset()
  proto_asset.uuid = db_asset.asset_uuid
  proto_asset.path = db_asset.path
  proto_asset.user = db_asset.user
  if db_asset.tags:
    proto_asset.tags.extend(db_asset.tags)
  proto_asset.state = db_asset.state.value

  if db_asset.created_at:
    proto_asset.created_at.FromDatetime(db_asset.created_at)
  if db_asset.finalized_at:
    proto_asset.finalized_at.FromDatetime(db_asset.finalized_at)
  if db_asset.deleted_at:
    proto_asset.deleted_at.FromDatetime(db_asset.deleted_at)
  if db_asset.updated_at:
    proto_asset.updated_at.FromDatetime(db_asset.updated_at)

  for tp in db_asset.tier_paths:
    proto_tp = proto_asset.tier_paths.add()
    proto_tp.id = tp.id
    proto_tp.path = tp.path
    if tp.ready_at:
      proto_tp.ready_at.FromDatetime(tp.ready_at)
    if tp.expires_at:
      proto_tp.expires_at.FromDatetime(tp.expires_at)

    sb = tp.storage_backend
    proto_sb = proto_tp.storage_backend
    proto_sb.id = sb.id
    proto_sb.level = sb.level
    proto_sb.backend_type = sb.backend_type.value
    proto_sb.prefix = sb.prefix
    if sb.zone:
      proto_sb.zone = sb.zone
    elif sb.region:
      proto_sb.region = sb.region
    elif sb.multi_regions:
      proto_sb.multi_regions.regions.extend(sb.multi_regions)

  return proto_asset
