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
import os
import sys
import uuid

from absl import logging
import fire
import grpc
from orbax.checkpoint.experimental.tiering_service import db_lib
from orbax.checkpoint.experimental.tiering_service import server_config
from orbax.checkpoint.experimental.tiering_service.proto import tiering_service_pb2
from orbax.checkpoint.experimental.tiering_service.proto import tiering_service_pb2_grpc
import uvloop

from google.protobuf import timestamp_pb2

# TODO: b/503445654 - Internal dummy state for assets.
# (to be replaced by SQL db)
_assets_by_uuid: dict[str, tiering_service_pb2.Asset] = {}
_BEARER_PREFIX = "Bearer "


def _get_oauth_token(context: grpc.ServicerContext) -> str | None:
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


def _get_asset(
    request: (
        tiering_service_pb2.PrefetchRequest
        | tiering_service_pb2.DeleteRequest
        | tiering_service_pb2.InfoRequest
    ),
) -> tiering_service_pb2.Asset | None:
  """Retrieves an asset. Return None if it's not found.

  Args:
    request: A request containing either a UUID or a path.

  Returns:
    The matching Asset if found, or None.
  """
  if request.HasField("uuid"):
    return _assets_by_uuid.get(request.uuid)
  elif request.HasField("path"):
    for asset in _assets_by_uuid.values():
      if asset.path == request.path:
        return asset
  return None


def _verify_gcs_permissions(
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

  def Reserve(
      self,
      request: tiering_service_pb2.ReserveRequest,
      context: grpc.ServicerContext,
  ) -> tiering_service_pb2.ReserveResponse:
    """Reserves a new asset or looks up an existing one."""
    logging.info("Reserve requested for path: %s", request.path)
    token = _get_oauth_token(context)

    # Verify write permission to the target GCS path/managed folder
    if not _verify_gcs_permissions(
        token, request.path, ["storage.objects.create"]
    ):
      logging.warning("Permission denied for Reserve on path: %s", request.path)
      context.abort(
          grpc.StatusCode.PERMISSION_DENIED, "Insufficient GCS permissions"
      )
      # context.abort raises an exception.

    # TODO: b/503445654 - Ensure path is unique.

    # TODO: b/503445654 - Fake a TierPath for now
    asset_uuid = str(uuid.uuid4())
    storage_path = f"/mnt/lustre/{request.path}"

    if not request.HasField("zone") and not request.HasField("region"):
      logging.error(
          "Reserve: No location specified for path: %s, user: %s",
          request.path,
          request.user,
      )
      context.abort(
          grpc.StatusCode.INVALID_ARGUMENT, "No zone or region specified"
      )
      # context.abort raises an exception.

    # TODO: b/503445654 - Tier 0 closest to the user.
    now = timestamp_pb2.Timestamp()
    now.GetCurrentTime()

    asset = tiering_service_pb2.Asset(
        uuid=asset_uuid,
        path=request.path,
        user=request.user,
        tags=request.tags,
        state=tiering_service_pb2.ASSET_STATE_ACTIVE_WRITE,
        created_at=now,
        updated_at=now,
    )

    # TODO: b/503445654 - Look up nearest backend to reserve path for requestor.
    storage_backend_kwargs = {
        "level": 0,
        "backend_type": tiering_service_pb2.BACKEND_TYPE_LUSTRE,
        # database/config.
        "prefix": "/mnt/lustre",
    }
    if request.HasField("zone"):
      storage_backend_kwargs["zone"] = request.zone
    elif request.HasField("region"):
      storage_backend_kwargs["region"] = request.region

    asset.tier_paths.add(
        path=storage_path,
        storage_backend=tiering_service_pb2.StorageBackend(
            **storage_backend_kwargs
        ),
    )

    _assets_by_uuid[asset_uuid] = asset
    logging.info("Reserved asset with UUID: %s", asset_uuid)
    logging.debug("Created asset: %s", asset)
    return tiering_service_pb2.ReserveResponse(
        asset=asset, keep_alive_interval_seconds=300
    )

  def ReserveKeepAlive(
      self,
      request: tiering_service_pb2.ReserveKeepAliveRequest,
      context: grpc.ServicerContext,
  ) -> tiering_service_pb2.ReserveKeepAliveResponse:
    """Extends the writing timeout for an asset."""
    logging.info("ReserveKeepAlive requested for UUID: %s", request.uuid)
    if request.uuid not in _assets_by_uuid:
      logging.warning("ReserveKeepAlive: Asset not found: %s", request.uuid)
      context.abort(grpc.StatusCode.NOT_FOUND, "Asset not found")
      # context.abort raises an exception.

    # TODO: b/503445654 - Update the expiration timer

    return tiering_service_pb2.ReserveKeepAliveResponse(
        keep_alive_interval_seconds=300
    )

  def Finalize(
      self,
      request: tiering_service_pb2.FinalizeRequest,
      context: grpc.ServicerContext,
  ) -> tiering_service_pb2.FinalizeResponse:
    """Finalizes an asset, moving it to STORED state."""
    logging.info("Finalize requested for UUID: %s", request.uuid)
    token = _get_oauth_token(context)

    if request.uuid not in _assets_by_uuid:
      logging.warning("Finalize: Asset not found: %s", request.uuid)
      context.abort(grpc.StatusCode.NOT_FOUND, "Asset not found")
      # context.abort raises an exception.

    asset = _assets_by_uuid[request.uuid]

    # Verify write permission before finalizing
    if not _verify_gcs_permissions(
        token, asset.path, ["storage.objects.create"]
    ):
      logging.warning("Permission denied for Finalize on path: %s", asset.path)
      context.abort(
          grpc.StatusCode.PERMISSION_DENIED, "Insufficient GCS permissions"
      )
      # context.abort raises an exception.

    if asset.state != tiering_service_pb2.ASSET_STATE_ACTIVE_WRITE:
      logging.warning(
          "Finalize: Asset %s not in ACTIVE_WRITE state", request.uuid
      )
      context.abort(
          grpc.StatusCode.FAILED_PRECONDITION, "Asset not in ACTIVE_WRITE state"
      )
      # context.abort raises an exception.

    asset.state = tiering_service_pb2.ASSET_STATE_STORED
    now = timestamp_pb2.Timestamp()
    now.GetCurrentTime()
    asset.updated_at.CopyFrom(now)

    logging.info("Finalized asset with UUID: %s", request.uuid)
    logging.debug("Finalized asset: %s", asset)
    return tiering_service_pb2.FinalizeResponse(asset=asset)

  def Prefetch(
      self,
      request: tiering_service_pb2.PrefetchRequest,
      context: grpc.ServicerContext,
  ) -> tiering_service_pb2.PrefetchResponse:
    """Signals CTS to copy an asset to Tier 0 storage."""
    identifier = request.uuid if request.HasField("uuid") else request.path
    logging.info("Prefetch requested for identifier: %s", identifier)

    asset = _get_asset(request)
    if not asset:
      logging.warning("Prefetch: Asset not found: %s", identifier)
      context.abort(grpc.StatusCode.NOT_FOUND, "Asset not found")
      # context.abort raises an exception.

    if not request.HasField("zone") and not request.HasField("region"):
      logging.error(
          "Prefetch: No location specified for identifier: %s", identifier
      )
      context.abort(grpc.StatusCode.INVALID_ARGUMENT, "No location specified")
      # context.abort raises an exception.
    # TODO: b/503445654 - Trigger async copy to closest storage tier to user.

    logging.info("Prefetch:  UUID: %s", request.uuid)
    return tiering_service_pb2.PrefetchResponse(asset=asset)

  def PrefetchKeepAlive(
      self,
      request: tiering_service_pb2.PrefetchKeepAliveRequest,
      context: grpc.ServicerContext,
  ) -> tiering_service_pb2.PrefetchKeepAliveResponse:
    """Signals that the client is still reading/waiting for promotion."""
    logging.info("PrefetchKeepAlive requested for UUID: %s", request.uuid)
    if request.uuid not in _assets_by_uuid:
      logging.warning("PrefetchKeepAlive: Asset not found: %s", request.uuid)
      context.abort(grpc.StatusCode.NOT_FOUND, "Asset not found")
      # context.abort raises an exception.

    logging.info("PrefetchKeepAlive: Handled for UUID: %s", request.uuid)
    return tiering_service_pb2.PrefetchKeepAliveResponse(
        keep_alive_interval_seconds=300
    )

  def Delete(
      self,
      request: tiering_service_pb2.DeleteRequest,
      context: grpc.ServicerContext,
  ) -> tiering_service_pb2.DeleteResponse:
    """Deletes an asset from CTS tracking."""
    identifier = request.uuid if request.HasField("uuid") else request.path
    logging.info("Delete requested for identifier: %s", identifier)

    asset = _get_asset(request)
    if asset:
      del _assets_by_uuid[asset.uuid]
      logging.info("Deleted asset with UUID: %s", asset.uuid)
    else:
      logging.warning("Delete: Asset not found: %s", identifier)
    return tiering_service_pb2.DeleteResponse()

  def Info(
      self,
      request: tiering_service_pb2.InfoRequest,
      context: grpc.ServicerContext,
  ) -> tiering_service_pb2.InfoResponse:
    """Returns metadata about an asset."""
    identifier = request.uuid if request.HasField("uuid") else request.path
    logging.info("Info requested for identifier: %s", identifier)

    asset = _get_asset(request)
    if not asset:
      logging.warning("Info: Asset not found: %s", identifier)
      context.abort(grpc.StatusCode.NOT_FOUND, "Asset not found")
      # context.abort raises an exception.

    logging.debug("Returning info for asset: %s", asset)
    return tiering_service_pb2.InfoResponse(asset=asset)


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
