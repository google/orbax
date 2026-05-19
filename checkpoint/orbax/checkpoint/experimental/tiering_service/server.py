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

"""Checkpoint Tiering Service (CTS) Server implementation."""

import asyncio
from collections.abc import Sequence
from concurrent import futures
import datetime
import os
import sys

from absl import logging
import fire
import grpc
from orbax.checkpoint.experimental.tiering_service import db_lib
from orbax.checkpoint.experimental.tiering_service import db_schema
from orbax.checkpoint.experimental.tiering_service import server_config
from orbax.checkpoint.experimental.tiering_service import server_lib
from orbax.checkpoint.experimental.tiering_service.proto import tiering_service_pb2
from orbax.checkpoint.experimental.tiering_service.proto import tiering_service_pb2_grpc
import sqlalchemy
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import sessionmaker
import uvloop

_BEARER_PREFIX = "Bearer "


async def _get_oauth_token(context: grpc.aio.ServicerContext) -> str | None:
  """Extracts OAuth token from gRPC metadata."""
  logging.debug("Extracting OAuth token from metadata")
  metadata = dict(context.invocation_metadata())
  # Standard header for OAuth tokens in gRPC is 'authorization'
  auth_header = metadata.get("authorization")

  if auth_header is None:
    logging.warning("No authorization header found")
    return None

  if not auth_header.startswith(_BEARER_PREFIX):
    logging.warning("Malfomed authorization header: %s", auth_header)
    return None

  logging.debug("Found authorization header: %s", auth_header)
  return auth_header[len(_BEARER_PREFIX) :]


async def _verify_gcs_permissions(
    token: str | None, path: str, permissions: Sequence[str]
) -> bool:
  """Verifies if the caller has necessary permissions on GCS."""
  logging.info("Verifying GCS permissions for path: %s", path)
  logging.debug("Requested permissions: %s", permissions)
  # TODO: b/503445654 - Implement actual IAM permission verification
  # For now, return True if a token is provided, and False otherwise,
  # to allow testing PERMISSION_DENIED errors.
  logging.debug("Permission check result: %s", token is not None)
  return token is not None


class TieringServiceServicer(tiering_service_pb2_grpc.TieringServiceServicer):
  """Servicer for the TieringService."""

  def __init__(self, config: server_config.ServerConfig):
    super().__init__()
    self._config = config
    self._engine = db_lib.get_async_engine(self._config)
    self._session_maker = sessionmaker(
        self._engine,
        # Prevent sync lazy-loading issues (MissingGreenlet) after commit.
        expire_on_commit=False,
        class_=AsyncSession,
    )
    self._level0_backends = None

  async def _get_level0_backends(self, session: AsyncSession):
    if self._level0_backends is None:
      result = await session.execute(
          select(db_schema.StorageBackend).filter_by(level=0)
      )
      self._level0_backends = result.scalars().all()
    return self._level0_backends

  async def _run_async_db(self, coro_fn):
    async with self._session_maker() as session:
      return await coro_fn(session)

  async def Reserve(
      self,
      request: tiering_service_pb2.ReserveRequest,
      context: grpc.aio.ServicerContext,
  ) -> tiering_service_pb2.ReserveResponse:
    """Reserves a new asset or looks up an existing one."""
    logging.info("Reserve requested for path: %s", request.path)
    token = await _get_oauth_token(context)

    # Verify write permission to the target GCS path/managed folder
    if not await _verify_gcs_permissions(
        token, request.path, ["storage.objects.create"]
    ):
      logging.warning("Permission denied for Reserve on path: %s", request.path)
      await context.abort(
          grpc.StatusCode.PERMISSION_DENIED, "Insufficient GCS permissions"
      )
      return tiering_service_pb2.ReserveResponse()

    if not request.HasField("zone") and not request.HasField("region"):
      logging.error(
          "Reserve: No location specified for path: %s, user: %s",
          request.path,
          request.user,
      )
      await context.abort(
          grpc.StatusCode.INVALID_ARGUMENT, "No zone or region specified"
      )
      return tiering_service_pb2.ReserveResponse()

    async def _coro(
        session: AsyncSession,
    ) -> tiering_service_pb2.ReserveResponse:
      backends = await self._get_level0_backends(session)
      backend = None

      if request.HasField("zone"):
        # Try match the exact zone
        for b in backends:
          if b.zone == request.zone:
            backend = b
            break

        # If no exact zone match, try to region to requestor zone
        if not backend and not request.HasField("region"):
          for b in backends:
            if b.region and request.zone.startswith(b.region):
              backend = b
              break
            if b.multi_regions and any(
                request.zone.startswith(mr) for mr in b.multi_regions
            ):
              backend = b
              break

      # Can't locate by zone, try to locate by region
      if not backend and request.HasField("region"):
        for b in backends:
          if b.region == request.region:
            backend = b
            break
          if b.multi_regions and request.region in b.multi_regions:
            backend = b
            break

      if not backend:
        await context.abort(
            grpc.StatusCode.INTERNAL,
            f"No level 0 storage backend found for zone:{request.zone} /"
            f" region:{request.region}",
        )
        return tiering_service_pb2.ReserveResponse()

      db_asset = db_schema.Asset(
          path=request.path,
          user=request.user,
          tags=list(request.tags) if request.tags else [],
          state=db_schema.AssetState.ASSET_STATE_ACTIVE_WRITE,
          write_expires_at=datetime.datetime.now(datetime.timezone.utc).replace(
              tzinfo=None
          )
          + self._config.client_keep_alive_interval,
      )
      storage_path = f"{backend.prefix.rstrip('/')}/{request.path.lstrip('/')}"
      tp = db_schema.TierPath(
          storage_backend=backend,
          path=storage_path,
      )
      db_asset.tier_paths.append(tp)
      try:
        session.add(db_asset)
        await session.commit()

        # TODO(b/503445463): Trigger async directory creation
      except IntegrityError:
        await session.rollback()
        logging.info(
            "Reserve: Asset path already exists, fetching existing record: %s",
            request.path,
        )

      stmt_fetch = (
          select(db_schema.Asset)
          .options(
              sqlalchemy.orm.selectinload(
                  db_schema.Asset.tier_paths
              ).selectinload(db_schema.TierPath.storage_backend)
          )
          .filter_by(path=request.path)
          .filter(
              db_schema.Asset.state.in_([
                  db_schema.AssetState.ASSET_STATE_ACTIVE_WRITE,
                  db_schema.AssetState.ASSET_STATE_STORED,
              ])
          )
      )
      result_fetch = await session.execute(stmt_fetch)
      db_asset = result_fetch.scalars().first()
      if not db_asset:
        # this should rarely happen, we might need to add retry insertion
        await context.abort(
            grpc.StatusCode.INTERNAL, "Failed to retrieve reserved asset"
        )
        return tiering_service_pb2.ReserveResponse()

      return tiering_service_pb2.ReserveResponse(
          asset=server_lib.db_asset_to_proto(db_asset),
          keep_alive_interval_seconds=int(
              self._config.client_keep_alive_interval.total_seconds()
          ),
      )

    return await self._run_async_db(_coro)

  async def ReserveKeepAlive(
      self,
      request: tiering_service_pb2.ReserveKeepAliveRequest,
      context: grpc.aio.ServicerContext,
  ) -> tiering_service_pb2.ReserveKeepAliveResponse:
    """Extends the writing timeout for an asset."""
    logging.info("ReserveKeepAlive requested for UUID: %s", request.uuid)

    async def _coro(
        session: AsyncSession,
    ) -> tiering_service_pb2.ReserveKeepAliveResponse:
      stmt = (
          select(db_schema.Asset)
          .options(
              sqlalchemy.orm.selectinload(
                  db_schema.Asset.tier_paths
              ).selectinload(db_schema.TierPath.storage_backend)
          )
          .filter_by(asset_uuid=request.uuid)
          .filter(
              db_schema.Asset.state != db_schema.AssetState.ASSET_STATE_DELETED
          )
      )
      result = await session.execute(stmt)
      db_asset = result.scalars().first()
      if not db_asset:
        logging.warning("ReserveKeepAlive: Asset not found: %s", request.uuid)
        await context.abort(grpc.StatusCode.NOT_FOUND, "Asset not found")
        return tiering_service_pb2.ReserveKeepAliveResponse()

      db_asset.write_expires_at = (
          datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
          + self._config.client_keep_alive_interval
      )
      await session.commit()

      return tiering_service_pb2.ReserveKeepAliveResponse(
          keep_alive_interval_seconds=int(
              self._config.client_keep_alive_interval.total_seconds()
          ),
      )

    return await self._run_async_db(_coro)

  async def Finalize(
      self,
      request: tiering_service_pb2.FinalizeRequest,
      context: grpc.aio.ServicerContext,
  ) -> tiering_service_pb2.FinalizeResponse:
    """Finalizes an asset, moving it to STORED state."""
    logging.info("Finalize requested for UUID: %s", request.uuid)
    token = await _get_oauth_token(context)

    async def _coro(
        session: AsyncSession,
    ) -> tiering_service_pb2.FinalizeResponse:
      stmt = (
          select(db_schema.Asset)
          .options(
              sqlalchemy.orm.selectinload(
                  db_schema.Asset.tier_paths
              ).selectinload(db_schema.TierPath.storage_backend)
          )
          .filter_by(asset_uuid=request.uuid)
          .filter(
              db_schema.Asset.state != db_schema.AssetState.ASSET_STATE_DELETED
          )
      )
      result = await session.execute(stmt)
      db_asset = result.scalars().first()
      if not db_asset:
        logging.warning("Finalize: Asset not found: %s", request.uuid)
        await context.abort(grpc.StatusCode.NOT_FOUND, "Asset not found")
        return tiering_service_pb2.FinalizeResponse()

      # Verify write permission before finalizing
      if not await _verify_gcs_permissions(
          token, db_asset.path, ["storage.objects.create"]
      ):
        logging.warning(
            "Permission denied for Finalize on path: %s", db_asset.path
        )
        await context.abort(
            grpc.StatusCode.PERMISSION_DENIED, "Insufficient GCS permissions"
        )
        return tiering_service_pb2.FinalizeResponse()

      if db_asset.state != db_schema.AssetState.ASSET_STATE_ACTIVE_WRITE:
        logging.warning(
            "Finalize: Asset %s not in ACTIVE_WRITE state", request.uuid
        )
        await context.abort(
            grpc.StatusCode.FAILED_PRECONDITION,
            "Asset not in ACTIVE_WRITE state",
        )
        return tiering_service_pb2.FinalizeResponse()

      db_asset.state = db_schema.AssetState.ASSET_STATE_STORED
      db_asset.finalized_at = datetime.datetime.now(
          datetime.timezone.utc
      ).replace(tzinfo=None)
      db_asset.write_expires_at = None
      proto_asset = server_lib.db_asset_to_proto(db_asset)
      await session.commit()

      return tiering_service_pb2.FinalizeResponse(asset=proto_asset)

    return await self._run_async_db(_coro)

  async def Prefetch(
      self,
      request: tiering_service_pb2.PrefetchRequest,
      context: grpc.aio.ServicerContext,
  ) -> tiering_service_pb2.PrefetchResponse:
    """Signals CTS to copy an asset to Tier 0 storage."""
    identifier = request.uuid if request.HasField("uuid") else request.path
    logging.info("Prefetch requested for identifier: %s", identifier)

    if not request.HasField("zone") and not request.HasField("region"):
      logging.error(
          "Prefetch: No location specified for identifier: %s", identifier
      )
      await context.abort(
          grpc.StatusCode.INVALID_ARGUMENT, "No location specified"
      )
      return tiering_service_pb2.PrefetchResponse()

    async def _coro(
        session: AsyncSession,
    ) -> tiering_service_pb2.PrefetchResponse:
      stmt = (
          select(db_schema.Asset)
          .options(
              sqlalchemy.orm.selectinload(
                  db_schema.Asset.tier_paths
              ).selectinload(db_schema.TierPath.storage_backend)
          )
          .filter(
              db_schema.Asset.state != db_schema.AssetState.ASSET_STATE_DELETED
          )
      )
      if request.HasField("uuid"):
        stmt = stmt.filter_by(asset_uuid=request.uuid)
      elif request.HasField("path"):
        stmt = stmt.filter_by(path=request.path).order_by(
            db_schema.Asset.created_at.desc()
        )
      else:
        await context.abort(
            grpc.StatusCode.INVALID_ARGUMENT, "No identifier specified"
        )
        return tiering_service_pb2.PrefetchResponse()

      result = await session.execute(stmt)
      db_asset = result.scalars().first()
      if not db_asset:
        logging.warning("Prefetch: Asset not found: %s", identifier)
        await context.abort(grpc.StatusCode.NOT_FOUND, "Asset not found")
        return tiering_service_pb2.PrefetchResponse()

      # TODO: b/503445654 - Trigger async copy to closest storage tier to user.

      logging.info("Prefetch: UUID: %s", db_asset.asset_uuid)
      return tiering_service_pb2.PrefetchResponse(
          asset=server_lib.db_asset_to_proto(db_asset)
      )

    return await self._run_async_db(_coro)

  async def PrefetchKeepAlive(
      self,
      request: tiering_service_pb2.PrefetchKeepAliveRequest,
      context: grpc.aio.ServicerContext,
  ) -> tiering_service_pb2.PrefetchKeepAliveResponse:
    """Signals that the client is still reading/waiting for promotion."""
    logging.info("PrefetchKeepAlive requested for UUID: %s", request.uuid)

    async def _coro(
        session: AsyncSession,
    ) -> tiering_service_pb2.PrefetchKeepAliveResponse:
      stmt = (
          select(db_schema.Asset)
          .options(
              sqlalchemy.orm.selectinload(
                  db_schema.Asset.tier_paths
              ).selectinload(db_schema.TierPath.storage_backend)
          )
          .filter_by(asset_uuid=request.uuid)
          .filter(
              db_schema.Asset.state != db_schema.AssetState.ASSET_STATE_DELETED
          )
      )
      result = await session.execute(stmt)
      db_asset = result.scalars().first()
      if not db_asset:
        logging.warning("PrefetchKeepAlive: Asset not found: %s", request.uuid)
        await context.abort(grpc.StatusCode.NOT_FOUND, "Asset not found")
        return tiering_service_pb2.PrefetchKeepAliveResponse()

      logging.info("PrefetchKeepAlive: Handled for UUID: %s", request.uuid)
      return tiering_service_pb2.PrefetchKeepAliveResponse(
          asset=server_lib.db_asset_to_proto(db_asset),
          keep_alive_interval_seconds=int(
              self._config.client_keep_alive_interval.total_seconds()
          ),
      )

    return await self._run_async_db(_coro)

  async def Delete(
      self,
      request: tiering_service_pb2.DeleteRequest,
      context: grpc.aio.ServicerContext,
  ) -> tiering_service_pb2.DeleteResponse:
    """Deletes an asset from CTS tracking."""
    identifier = request.uuid if request.HasField("uuid") else request.path
    logging.info("Delete requested for identifier: %s", identifier)

    async def _coro(
        session: AsyncSession,
    ) -> tiering_service_pb2.DeleteResponse:
      stmt = (
          select(db_schema.Asset)
          .options(
              sqlalchemy.orm.selectinload(
                  db_schema.Asset.tier_paths
              ).selectinload(db_schema.TierPath.storage_backend)
          )
          .filter(
              db_schema.Asset.state != db_schema.AssetState.ASSET_STATE_DELETED
          )
      )
      if request.HasField("uuid"):
        stmt = stmt.filter_by(asset_uuid=request.uuid)
      elif request.HasField("path"):
        stmt = stmt.filter_by(path=request.path).order_by(
            db_schema.Asset.created_at.desc()
        )
      else:
        await context.abort(
            grpc.StatusCode.INVALID_ARGUMENT, "No identifier specified"
        )
        return tiering_service_pb2.DeleteResponse()

      result = await session.execute(stmt)
      db_asset = result.scalars().first()
      if db_asset:
        db_asset.state = db_schema.AssetState.ASSET_STATE_DELETED
        db_asset.deleted_at = datetime.datetime.now(
            datetime.timezone.utc
        ).replace(tzinfo=None)
        await session.commit()
        logging.info("Deleted asset with UUID: %s", db_asset.asset_uuid)
      else:
        logging.warning("Delete: Asset not found: %s", identifier)
      return tiering_service_pb2.DeleteResponse()

    return await self._run_async_db(_coro)

  async def Info(
      self,
      request: tiering_service_pb2.InfoRequest,
      context: grpc.aio.ServicerContext,
  ) -> tiering_service_pb2.InfoResponse:
    """Returns metadata about an asset."""
    identifier = request.uuid if request.HasField("uuid") else request.path
    logging.info("Info requested for identifier: %s", identifier)

    async def _coro(session: AsyncSession) -> tiering_service_pb2.InfoResponse:
      stmt = (
          select(db_schema.Asset)
          .options(
              sqlalchemy.orm.selectinload(
                  db_schema.Asset.tier_paths
              ).selectinload(db_schema.TierPath.storage_backend)
          )
          .filter(
              db_schema.Asset.state != db_schema.AssetState.ASSET_STATE_DELETED
          )
      )
      if request.HasField("uuid"):
        stmt = stmt.filter_by(asset_uuid=request.uuid)
      elif request.HasField("path"):
        stmt = stmt.filter_by(path=request.path).order_by(
            db_schema.Asset.created_at.desc()
        )
      else:
        await context.abort(
            grpc.StatusCode.INVALID_ARGUMENT, "No identifier specified"
        )
        return tiering_service_pb2.InfoResponse()

      result = await session.execute(stmt)
      db_asset = result.scalars().first()
      if not db_asset:
        logging.warning("Info: Asset not found: %s", identifier)
        await context.abort(grpc.StatusCode.NOT_FOUND, "Asset not found")
        return tiering_service_pb2.InfoResponse()

      logging.debug("Returning info for asset: %s", db_asset)
      return tiering_service_pb2.InfoResponse(
          asset=server_lib.db_asset_to_proto(db_asset)
      )

    return await self._run_async_db(_coro)


async def setup_storage_backends(
    config: tiering_service_pb2.ServerConfig,
) -> None:
  """Initializes the database if uninitialized, otherwise verifies it matches configuration."""
  if not await db_lib.async_is_db_initialized(config):
    await db_lib.async_initialize_db(config)
  else:
    await db_lib.async_verify_db(config)
  # TODO(b/503445463): Start background garbage collection task to handle
  # expired assets.


class CtsServer:
  """Checkpoint Tiering Service (CTS) Server CLI."""

  async def serve(self, yaml_path: str) -> None:
    """Starts the gRPC server.

    Args:
      yaml_path: Path to the YAML configuration file.
    """
    config = server_config.load_config(yaml_path)
    await setup_storage_backends(config)

    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    tiering_service_pb2_grpc.add_TieringServiceServicer_to_server(
        TieringServiceServicer(), server
    )

    server_creds = os.environ.get("SERVER_CREDS")  # pylint: disable=unused-variable

    server.add_secure_port("[::]:50051", server_creds)
    await server.start()
    await server.wait_for_termination()


def main(argv: Sequence[str] | None = None) -> None:
  """Main entry point for CTS server."""
  if argv is None:
    argv = sys.argv
  uvloop.install()
  try:
    asyncio.get_event_loop()
  except RuntimeError:
    # Create the high-performance uvloop instead
    loop = uvloop.new_event_loop()
    asyncio.set_event_loop(loop)
  fire.Fire(CtsServer, command=argv[1:])


if __name__ == "__main__":
  main()
