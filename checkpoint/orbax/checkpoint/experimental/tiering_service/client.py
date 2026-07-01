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

"""Checkpoint Tiering Service (CTS) client library implementation."""

import asyncio
from collections.abc import Sequence
from typing import Any
import grpc
from orbax.checkpoint.experimental.tiering_service import client_auth
from orbax.checkpoint.experimental.tiering_service import environment
from orbax.checkpoint.experimental.tiering_service.proto import tiering_service_pb2
from orbax.checkpoint.experimental.tiering_service.proto import tiering_service_pb2_grpc


class TieringClient:
  """Client library to communicate with the Checkpoint Tiering Service (CTS)."""

  def __init__(
      self, server_address: str = "localhost:50051", secure: bool = False
  ):
    """Initializes the TieringClient.

    Args:
      server_address: Address of the gRPC server.
      secure: If True, establishes a secure gRPC channel.
    """
    self._server_address = server_address
    self._secure = secure
    self._channel = None
    self._stub = None
    self._zone = None
    self._region = None
    self._env_queried = False

  async def __aenter__(self) -> "TieringClient":
    await self.connect()
    return self

  async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
    await self.close()

  async def connect(self) -> None:
    """Establishes an async gRPC channel with the server."""
    if self._channel is not None:
      return

    if self._secure:
      is_local = (
          "localhost" in self._server_address
          or "127.0.0.1" in self._server_address
      )
      if is_local:
        try:
          # Secure channel setup. Fall back to SSL if local creds not supported.
          creds = grpc.local_channel_credentials()
        except AttributeError:
          creds = grpc.ssl_channel_credentials()
      else:
        creds = grpc.ssl_channel_credentials()
      self._channel = grpc.aio.secure_channel(self._server_address, creds)
    else:
      self._channel = grpc.aio.insecure_channel(self._server_address)

    self._stub = tiering_service_pb2_grpc.TieringServiceStub(self._channel)

  async def close(self) -> None:
    """Closes the gRPC channel."""
    if self._channel is not None:
      await self._channel.close()
      self._channel = None
      self._stub = None

  async def _get_gcp_zone_and_region(self) -> tuple[str | None, str | None]:
    """Retrieves and caches GCP zone and region."""
    if not self._env_queried:
      self._zone = await environment.get_gcp_zone()
      self._region = await environment.get_gcp_region()
      self._env_queried = True
    return self._zone, self._region

  async def _get_auth_metadata(self) -> list[tuple[str, str]]:
    """Retrieves GCP OAuth token and formats it as gRPC metadata."""
    token = await client_auth.get_oauth_token()
    if token:
      return [("authorization", f"Bearer {token}")]
    return []

  async def reserve(
      self,
      path: str,
      tags: Sequence[str] | None = None,
      user: str | None = None,
  ) -> tuple[str, str]:
    """Reserves an asset path on Tier 0 storage.

    Args:
      path: Unique checkpoint logical path.
      tags: Optional list of tags.
      user: Optional owner user. If not specified, auto-discovers.

    Returns:
      A tuple of (asset_uuid, tier0_path).

    Raises:
      RuntimeError: If gRPC call fails or no Tier 0 path is returned.
    """
    if not self._stub:
      await self.connect()

    stub = self._stub
    if stub is None:
      raise RuntimeError("Stub is not initialized after connect.")

    if user is None:
      user = environment.get_current_user()

    zone, region = await self._get_gcp_zone_and_region()

    request = tiering_service_pb2.ReserveRequest(
        path=path,
        tags=tags or [],
        user=user,
    )
    if zone is not None:
      request.zone = zone
    if region is not None:
      request.region = region

    metadata = await self._get_auth_metadata()
    try:
      response = await stub.Reserve(request, metadata=metadata)
    except grpc.aio.AioRpcError as e:
      raise RuntimeError(
          f"Reserve RPC failed: {e.details()} ({e.code()})"
      ) from e

    asset = response.asset
    if not response.tier_path_uuid:
      raise RuntimeError(
          "Reserve succeeded but returned no tier_path_uuid for asset"
          f" {asset.uuid}"
      )

    for tp in asset.tier_paths:
      if tp.tier_path_uuid == response.tier_path_uuid:
        return asset.uuid, tp.path

    raise RuntimeError(
        "Reserve succeeded but returned tier_path_uuid"
        f" {response.tier_path_uuid} which is not found in asset tier paths"
        f" for asset {asset.uuid}"
    )

  async def finalize(self, uuid: str) -> None:
    """Finalizes the asset, marking it stored and immutable.

    Args:
      uuid: Asset UUID to finalize.

    Raises:
      RuntimeError: If gRPC call fails.
    """
    if not self._stub:
      await self.connect()

    stub = self._stub
    if stub is None:
      raise RuntimeError("Stub is not initialized after connect.")

    request = tiering_service_pb2.FinalizeRequest(uuid=uuid)
    metadata = await self._get_auth_metadata()
    try:
      await stub.Finalize(request, metadata=metadata)
    except grpc.aio.AioRpcError as e:
      raise RuntimeError(
          f"Finalize RPC failed: {e.details()} ({e.code()})"
      ) from e

  async def prefetch(
      self,
      path: str | None = None,
      uuid: str | None = None,
  ) -> asyncio.Future[str]:
    """Prefetches the asset to the closest Tier 0 storage.

    Args:
      path: Logical path of the asset.
      uuid: Asset UUID.

    Returns:
      A Future that will resolve to the Tier 0 path when ready.

    Raises:
      ValueError: If neither or both path and uuid are specified.
      RuntimeError: If gRPC call fails.
    """
    if path is None and uuid is None:
      raise ValueError("Either path or uuid must be specified.")
    if path is not None and uuid is not None:
      raise ValueError("Only one of path or uuid can be specified.")

    if not self._stub:
      await self.connect()

    stub = self._stub
    if stub is None:
      raise RuntimeError("Stub is not initialized after connect.")

    zone, region = await self._get_gcp_zone_and_region()

    request = tiering_service_pb2.PrefetchRequest()
    if uuid is not None:
      request.uuid = uuid
    else:
      request.path = path

    if zone is not None:
      request.zone = zone
    if region is not None:
      request.region = region

    metadata = await self._get_auth_metadata()
    try:
      response = await stub.Prefetch(request, metadata=metadata)
    except grpc.aio.AioRpcError as e:
      raise RuntimeError(
          f"Prefetch RPC failed: {e.details()} ({e.code()})"
      ) from e

    future = asyncio.get_running_loop().create_future()
    asset = response.asset
    if response.closest_tier_path_uuid:
      for tp in asset.tier_paths:
        if tp.tier_path_uuid == response.closest_tier_path_uuid:
          if tp.HasField("ready_at"):
            future.set_result(tp.path)
            return future
          break
    else:
      raise RuntimeError(
          "Prefetch succeeded but returned no closest_tier_path_uuid for asset"
          f" {asset.uuid}"
      )

    # TODO: b/503445837 - to poll for prefetch completion.
    return future

  async def release(self, uuid: str) -> None:
    """Client-side release of prefetch keep-alive loop.

    Args:
      uuid: Asset UUID to release.
    """
    # TODO: b/503445837 - implemente release.
    pass

  async def delete(
      self,
      path: str | None = None,
      uuid: str | None = None,
  ) -> None:
    """Queues a delete job for the asset.

    Args:
      path: Logical path of the asset.
      uuid: Asset UUID to delete.

    Raises:
      ValueError: If neither or both path and uuid are specified.
      RuntimeError: If gRPC call fails.
    """
    if path is None and uuid is None:
      raise ValueError("Either path or uuid must be specified.")
    if path is not None and uuid is not None:
      raise ValueError("Only one of path or uuid can be specified.")

    if not self._stub:
      await self.connect()

    stub = self._stub
    if stub is None:
      raise RuntimeError("Stub is not initialized after connect.")

    if uuid is not None:
      request = tiering_service_pb2.DeleteRequest(uuid=uuid)
    else:
      request = tiering_service_pb2.DeleteRequest(path=path)

    metadata = await self._get_auth_metadata()
    try:
      await stub.Delete(request, metadata=metadata)
    except grpc.aio.AioRpcError as e:
      raise RuntimeError(
          f"Delete RPC failed: {e.details()} ({e.code()})"
      ) from e

  async def info(
      self,
      path: str | None = None,
      uuid: str | None = None,
  ) -> list[tiering_service_pb2.Asset]:
    """Retrieves info/metadata for an asset.

    Args:
      path: Logical path of the asset.
      uuid: Asset UUID.

    Returns:
      A list of matching Asset configurations.

    Raises:
      ValueError: If neither or both path and uuid are specified.
      RuntimeError: If gRPC call fails.
    """
    if path is None and uuid is None:
      raise ValueError("Either path or uuid must be specified.")
    if path is not None and uuid is not None:
      raise ValueError("Only one of path or uuid can be specified.")

    if not self._stub:
      await self.connect()

    stub = self._stub
    if stub is None:
      raise RuntimeError("Stub is not initialized after connect.")

    if uuid is not None:
      request = tiering_service_pb2.InfoRequest(uuid=uuid)
    else:
      request = tiering_service_pb2.InfoRequest(path=path)

    metadata = await self._get_auth_metadata()
    try:
      response = await stub.Info(request, metadata=metadata)
      return list(response.assets)
    except grpc.aio.AioRpcError as e:
      raise RuntimeError(f"Info RPC failed: {e.details()} ({e.code()})") from e
