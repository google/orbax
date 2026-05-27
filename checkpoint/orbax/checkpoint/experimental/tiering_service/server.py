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
from collections.abc import AsyncIterator, Sequence
from concurrent import futures
import contextlib
import datetime
import os
import pprint
import sys

from absl import logging
import fire
import grpc
from orbax.checkpoint.experimental.tiering_service import assets
from orbax.checkpoint.experimental.tiering_service import auth
from orbax.checkpoint.experimental.tiering_service import db_lib
from orbax.checkpoint.experimental.tiering_service import db_schema
from orbax.checkpoint.experimental.tiering_service import server_config
from orbax.checkpoint.experimental.tiering_service import storage_backend
from orbax.checkpoint.experimental.tiering_service.proto import tiering_service_pb2
from orbax.checkpoint.experimental.tiering_service.proto import tiering_service_pb2_grpc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
import uvloop


def _has_location(
    request: (
        tiering_service_pb2.ReserveRequest | tiering_service_pb2.PrefetchRequest
    ),
) -> bool:
  """Checks whether request specifies a location (zone or region)."""
  return bool(request.zone or request.region)


class TieringServiceServicer(tiering_service_pb2_grpc.TieringServiceServicer):
  """Servicer for the TieringService."""

  def __init__(self, config: tiering_service_pb2.ServerConfig):
    """Initializes the TieringServiceServicer.

    Args:
      config: The server configuration.
    """
    super().__init__()
    self._config = config
    self._engine = db_lib.get_async_engine(self._config)
    self._session_maker = sessionmaker(
        self._engine,
        # Required for async session usage
        expire_on_commit=False,
        class_=AsyncSession,
    )
    self._level0_backends: Sequence[db_schema.StorageBackend] | None = None

  async def initialize(self) -> None:
    """Initializes the servicer, loading static data."""
    async with self._session_maker() as session:
      self._level0_backends = await storage_backend.find_backends_by_level(
          session, level=0
      )

  async def close(self) -> None:
    """Closes stateful resources, disposing of the database engine."""
    await self._engine.dispose()

  @contextlib.asynccontextmanager
  async def _session_scope(self) -> AsyncIterator[AsyncSession]:
    """Provides a transactional scope around a series of database operations."""
    async with self._session_maker() as session:
      try:
        yield session
      finally:
        session.expunge_all()

  async def Reserve(
      self,
      request: tiering_service_pb2.ReserveRequest,
      context: grpc.aio.ServicerContext,
  ) -> tiering_service_pb2.ReserveResponse:
    """Reserves a new asset or looks up an existing one."""
    logging.info("Reserve requested for path: %s", request.path)

    if not _has_location(request):
      logging.error(
          "Reserve: No location specified for path: %s, user: %s",
          request.path,
          request.user,
      )
      await context.abort(
          grpc.StatusCode.INVALID_ARGUMENT, "No zone or region specified"
      )
      return tiering_service_pb2.ReserveResponse()

    # Find the closest level 0 backend
    if self._level0_backends is None:
      await context.abort(
          grpc.StatusCode.FAILED_PRECONDITION, "Servicer not initialized"
      )
      return tiering_service_pb2.ReserveResponse()
    backend = storage_backend.locate_closest_backend(
        self._level0_backends, request.zone, request.region
    )
    if not backend:
      zone_val = request.zone
      region_val = request.region
      await context.abort(
          grpc.StatusCode.INTERNAL,
          f"No level 0 storage backend found for zone:{zone_val} /"
          f" region:{region_val}",
      )
      return tiering_service_pb2.ReserveResponse()

    # Calculate resolved path and check GCS permission.
    storage_path = storage_backend.get_storage_path(backend, request.path)
    token = await auth.get_oauth_token(context)
    if not await auth.has_write_permission(token, backend, storage_path):
      logging.warning(
          "Permission denied for Reserve on storage path: %s", storage_path
      )
      backend_name = storage_backend.get_backend_name(backend)
      await context.abort(
          grpc.StatusCode.PERMISSION_DENIED,
          f"Insufficient {backend_name} permissions",
      )
      return tiering_service_pb2.ReserveResponse()

    async with self._session_scope() as session:
      try:
        db_asset = await assets.create_or_fetch_asset(
            session, request, backend, self._config
        )
      except ValueError as e:
        await context.abort(
            grpc.StatusCode.INTERNAL, f"Failed to reserve asset: {e}"
        )
        return tiering_service_pb2.ReserveResponse()

      return tiering_service_pb2.ReserveResponse(
          asset=assets.proto_from_db_asset(db_asset),
          keep_alive_interval_seconds=self._config.client_keep_alive_interval_seconds,
      )

  async def ReserveKeepAlive(
      self,
      request: tiering_service_pb2.ReserveKeepAliveRequest,
      context: grpc.aio.ServicerContext,
  ) -> tiering_service_pb2.ReserveKeepAliveResponse:
    """Extends the writing timeout for an asset."""
    logging.info("ReserveKeepAlive requested for UUID: %s", request.uuid)

    async with self._session_scope() as session:
      db_asset = await assets.reserve_keep_alive(
          session,
          request.uuid,
          datetime.timedelta(
              seconds=self._config.client_keep_alive_interval_seconds
          ),
      )
      if not db_asset:
        logging.warning("ReserveKeepAlive: Asset not found: %s", request.uuid)
        await context.abort(grpc.StatusCode.NOT_FOUND, "Asset not found")
        return tiering_service_pb2.ReserveKeepAliveResponse()

      return tiering_service_pb2.ReserveKeepAliveResponse(
          keep_alive_interval_seconds=self._config.client_keep_alive_interval_seconds,
      )

  async def Finalize(
      self,
      request: tiering_service_pb2.FinalizeRequest,
      context: grpc.aio.ServicerContext,
  ) -> tiering_service_pb2.FinalizeResponse:
    """Finalizes an asset, moving it to STORED state."""
    logging.info("Finalize requested for UUID: %s", request.uuid)
    token = await auth.get_oauth_token(context)

    async with self._session_scope() as session:
      db_assets = await assets.fetch_asset_by_uuid(session, request.uuid)
      db_asset = db_assets[0] if db_assets else None
      if not db_asset:
        logging.warning("Finalize: Asset not found: %s", request.uuid)
        await context.abort(grpc.StatusCode.NOT_FOUND, "Asset not found")
        return tiering_service_pb2.FinalizeResponse()

      if db_asset.state != db_schema.AssetState.ASSET_STATE_ACTIVE_WRITE:
        logging.warning(
            "Finalize: Asset %s is not in ACTIVE_WRITE state (current state:"
            " %s)",
            request.uuid,
            db_asset.state,
        )
        await context.abort(
            grpc.StatusCode.FAILED_PRECONDITION,
            f"Asset is in state {db_asset.state.name}, expected"
            " ASSET_STATE_ACTIVE_WRITE",
        )
        return tiering_service_pb2.FinalizeResponse()

      # Verify write permission before finalizing.
      tier_path = next(iter(db_asset.tier_paths), None)
      if tier_path is None:
        logging.warning(
            "Finalize: Asset %s has no tier paths despite being in"
            " ACTIVE_WRITE state",
            request.uuid,
        )
        await context.abort(
            grpc.StatusCode.FAILED_PRECONDITION,
            "Asset has no associated storage paths",
        )
        return tiering_service_pb2.FinalizeResponse()

      if not await auth.has_write_permission(
          token, tier_path.storage_backend, tier_path.path
      ):
        logging.warning(
            "Permission denied for Finalize on path: %s",
            tier_path.path,
        )
        backend_name = storage_backend.get_backend_name(
            tier_path.storage_backend
        )
        await context.abort(
            grpc.StatusCode.PERMISSION_DENIED,
            f"Insufficient {backend_name} permissions",
        )
        return tiering_service_pb2.FinalizeResponse()

      try:
        db_asset = await assets.finalize_asset(session, db_asset)
        if not db_asset:
          # This is unlikely to happen since we just finalized the asset
          raise ValueError("Asset not found after finalize")
      except ValueError as e:
        logging.exception("Finalize failed for UUID: %s", request.uuid)
        await context.abort(
            grpc.StatusCode.FAILED_PRECONDITION,
            f"Failed to finalize asset: {e}",
        )
        return tiering_service_pb2.FinalizeResponse()

      return tiering_service_pb2.FinalizeResponse(
          asset=assets.proto_from_db_asset(db_asset)
      )

  async def Prefetch(
      self,
      request: tiering_service_pb2.PrefetchRequest,
      context: grpc.aio.ServicerContext,
  ) -> tiering_service_pb2.PrefetchResponse:
    """Signals CTS to copy an asset to Tier 0 storage."""
    if not _has_location(request):
      await context.abort(
          grpc.StatusCode.INVALID_ARGUMENT, "No location specified"
      )
      return tiering_service_pb2.PrefetchResponse()
    # TODO: b/503445654 - Trigger async copy to closest storage tier to user.

    await context.abort(
        grpc.StatusCode.UNIMPLEMENTED, "Prefetch Not Implemented"
    )
    return tiering_service_pb2.PrefetchResponse()

  async def PrefetchKeepAlive(
      self,
      request: tiering_service_pb2.PrefetchKeepAliveRequest,
      context: grpc.aio.ServicerContext,
  ) -> tiering_service_pb2.PrefetchKeepAliveResponse:
    """Signals that the client is still reading/waiting for promotion."""
    logging.info("PrefetchKeepAlive requested for UUID: %s", request.uuid)

    await context.abort(
        grpc.StatusCode.UNIMPLEMENTED, "PrefetchKeepAlive Not Implemented"
    )
    return tiering_service_pb2.PrefetchKeepAliveResponse()

  async def Delete(
      self,
      request: tiering_service_pb2.DeleteRequest,
      context: grpc.aio.ServicerContext,
  ) -> tiering_service_pb2.DeleteResponse:
    """Deletes an asset from CTS tracking."""
    logging.info("Delete requested")

    await context.abort(grpc.StatusCode.UNIMPLEMENTED, "Delete Not Implemented")
    return tiering_service_pb2.DeleteResponse()

  async def Info(
      self,
      request: tiering_service_pb2.InfoRequest,
      context: grpc.aio.ServicerContext,
  ) -> tiering_service_pb2.InfoResponse:
    """Returns metadata about an asset."""
    identifier = request.uuid if request.HasField("uuid") else request.path
    logging.info("Info requested for identifier: %s", identifier)

    async with self._session_scope() as session:
      db_assets = await assets.fetch_asset_by_identifier(
          session,
          asset_uuid=request.uuid if request.HasField("uuid") else None,
          path=request.path if request.HasField("path") else None,
          inclusive_filter=[
              db_schema.AssetState.ASSET_STATE_ACTIVE_WRITE,
              db_schema.AssetState.ASSET_STATE_STORED,
          ],
      )
      if not db_assets:
        logging.warning("Info: Asset not found: %s", identifier)
        await context.abort(grpc.StatusCode.NOT_FOUND, "Asset not found")
        return tiering_service_pb2.InfoResponse()

      logging.debug("Returning info for assets: %s", pprint.pformat(db_assets))
      return tiering_service_pb2.InfoResponse(
          assets=(assets.proto_from_db_asset(asset) for asset in db_assets)
      )


async def setup_storage_backends(
    config: tiering_service_pb2.ServerConfig,
) -> None:
  """Initializes the database if uninitialized, otherwise verifies it matches configuration."""
  if not await db_lib.async_is_db_initialized(config):
    await db_lib.async_initialize_db(config)
  else:
    await db_lib.async_verify_db(config)


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
    servicer = TieringServiceServicer(config)
    try:
      await servicer.initialize()
      tiering_service_pb2_grpc.add_TieringServiceServicer_to_server(
          servicer, server
      )

      server_creds = os.environ.get("SERVER_CREDS")  # pylint: disable=unused-variable

      server.add_secure_port("[::]:50051", server_creds)
      await server.start()

      # TODO: b/503445463 - Start background garbage collection task to handle
      # expired assets.

      await server.wait_for_termination()
    finally:
      await servicer.close()


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
