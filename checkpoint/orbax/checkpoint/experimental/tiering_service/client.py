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
import enum
import threading
from typing import Any
from absl import logging
import grpc
from orbax.checkpoint.experimental.tiering_service import client_auth
from orbax.checkpoint.experimental.tiering_service import environment
from orbax.checkpoint.experimental.tiering_service.proto import tiering_service_pb2
from orbax.checkpoint.experimental.tiering_service.proto import tiering_service_pb2_grpc


class JobType(enum.Enum):
  """Job types managed by TieringClient."""

  WRITE = "write"
  PREFETCH = "prefetch"


class TieringClient:
  """Client library to communicate with the Checkpoint Tiering Service (CTS).

  Supports single active checkpoint operation per TieringClient instance.
  """

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
    self._loop: asyncio.AbstractEventLoop | None = None
    self._thread: threading.Thread | None = None

    self._channel = None
    self._stub = None

    self._zone = None
    self._region = None
    self._env_queried = False
    self._env_lock = None

    # Single Active Asset State
    self._active_asset_uuid: str | None = None
    self._active_job_type: JobType | None = None
    self._active_tier_path_uuid: str | None = None
    self._active_path: str | None = None
    self._keep_alive_task: asyncio.Task[None] | None = None
    self._prefetch_future: asyncio.Future[str] | None = None

    self._connect_lock = threading.Lock()
    self._connected_event = threading.Event()
    self._connect_error: Exception | None = None
    self._closed = False
    self._connecting = False

  async def __aenter__(self) -> "TieringClient":
    await self.connect()
    return self

  async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
    await self.close()

  def _get_loop(self) -> asyncio.AbstractEventLoop:
    """Gets the running event loop or raises a RuntimeError if not set."""
    if self._loop is None:
      raise RuntimeError(
          "TieringClient is not connected. Call connect() first."
      )
    return self._loop

  async def _ensure_connected(self) -> None:
    if self._closed:
      raise RuntimeError("TieringClient has been closed")
    if self._loop is None:
      await self.connect()

  def _get_or_create_stub(self) -> tiering_service_pb2_grpc.TieringServiceStub:
    """Gets or creates the gRPC stub."""
    if self._stub is None:
      if self._secure:
        is_local = (
            "localhost" in self._server_address
            or "127.0.0.1" in self._server_address
        )
        if is_local:
          try:
            creds = grpc.local_channel_credentials()
          except AttributeError:
            creds = grpc.ssl_channel_credentials()
        else:
          creds = grpc.ssl_channel_credentials()
        self._channel = grpc.aio.secure_channel(self._server_address, creds)
      else:
        self._channel = grpc.aio.insecure_channel(self._server_address)

      self._stub = tiering_service_pb2_grpc.TieringServiceStub(self._channel)

    return self._stub

  async def _async_connect(self) -> None:
    self._get_or_create_stub()

  async def connect(self) -> None:
    """Establishes a gRPC channel with the server and starts background loop."""
    should_start = False
    with self._connect_lock:
      if self._closed:
        raise RuntimeError("TieringClient has been closed")
      if self._connected_event.is_set() and self._loop is not None:
        return
      if not self._connecting:
        self._connecting = True
        should_start = True

    if should_start:
      self._connect_error = None
      try:
        loop_started = self._start_background_loop()
        await asyncio.to_thread(loop_started.wait)

        fut = asyncio.run_coroutine_threadsafe(
            self._async_connect(), self._loop
        )
        await asyncio.wrap_future(fut)
        self._connected_event.set()
      except Exception as e:
        self._connect_error = e
        self._connected_event.set()  # Unblock waiting callers
        await self._cleanup_connection_failure()
        raise
    else:
      await asyncio.to_thread(self._connected_event.wait)
      if self._connect_error is not None:
        raise RuntimeError(
            f"Connection to TieringClient failed: {self._connect_error}"
        ) from self._connect_error
      if self._loop is None:
        raise RuntimeError("Connection to TieringClient failed.")

  def _start_background_loop(self) -> threading.Event:
    """Creates a new event loop and runs it in a background thread."""
    self._loop = asyncio.new_event_loop()
    loop_started = threading.Event()

    def run_loop():
      if not self._loop:
        raise RuntimeError("Event loop not set.")
      asyncio.set_event_loop(self._loop)
      self._env_lock = asyncio.Lock()
      loop_started.set()
      self._loop.run_forever()

    self._thread = threading.Thread(target=run_loop, daemon=True)
    self._thread.start()
    return loop_started

  async def _cleanup_connection_failure(self) -> None:
    """Cleans up background thread and loop on connection failure."""
    if self._loop is not None:
      self._loop.call_soon_threadsafe(self._loop.stop)
    if self._thread is not None and threading.current_thread() != self._thread:
      await asyncio.to_thread(self._thread.join)
    with self._connect_lock:
      self._connecting = False
      self._loop = None
      self._thread = None

  async def _async_close(self) -> None:
    """Closes all active background tasks, futures, and gRPC channels."""
    if self._keep_alive_task is not None:
      self._keep_alive_task.cancel()
      try:
        await self._keep_alive_task
      except asyncio.CancelledError:
        pass
      self._keep_alive_task = None

    if self._prefetch_future is not None and not self._prefetch_future.done():
      self._prefetch_future.cancel()
      self._prefetch_future = None

    self._reset_active_state()

    chan = self._channel
    if chan is not None:
      await chan.close()
      self._channel = None
      self._stub = None

  async def close(self) -> None:
    """Closes the gRPC channel and stops background loop."""
    self._closed = True
    if self._loop is None:
      return

    try:
      fut = asyncio.run_coroutine_threadsafe(self._async_close(), self._loop)
      await asyncio.wrap_future(fut)
    finally:
      if self._loop and self._loop.is_running():
        self._loop.call_soon_threadsafe(self._loop.stop)
      th = self._thread
      if th is not None and threading.current_thread() != th:
        await asyncio.to_thread(th.join)
        self._thread = None
      self._loop = None
      self._connected_event.clear()
      with self._connect_lock:
        self._connecting = False

  def _reset_active_state(self) -> None:
    self._active_asset_uuid = None
    self._active_job_type = None
    self._active_tier_path_uuid = None
    self._active_path = None

  def _check_active_operation(self) -> None:
    if self._active_asset_uuid is not None:
      job_name = (
          self._active_job_type.value if self._active_job_type else "operation"
      )
      raise RuntimeError(
          f"TieringClient is already managing active asset "
          f"'{self._active_asset_uuid}' ({job_name}). Call finalize() or "
          "release() first, or use a separate TieringClient instance."
      )

  async def _get_gcp_zone_and_region(self) -> tuple[str | None, str | None]:
    """Retrieves and caches GCP zone and region."""
    lock = self._env_lock
    if lock is None:
      lock = asyncio.Lock()
      self._env_lock = lock
    async with lock:
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

  async def _write_keep_alive_loop(
      self, asset_uuid: str, interval: float
  ) -> None:
    """Runs write keep-alive heartbeats."""
    stub = self._get_or_create_stub()
    current_interval = interval
    while True:
      try:
        await asyncio.sleep(current_interval)
        request = tiering_service_pb2.ReserveKeepAliveRequest(uuid=asset_uuid)
        metadata = await self._get_auth_metadata()
        response = await stub.ReserveKeepAlive(
            request, metadata=metadata, timeout=30.0
        )
        current_interval = max(
            1.0, float(response.keep_alive_interval_seconds) * 0.8
        )
      except asyncio.CancelledError:
        break
      except grpc.aio.AioRpcError as e:
        if e.code() == grpc.StatusCode.NOT_FOUND:
          logging.info(
              "ReserveKeepAlive for %s returned NOT_FOUND; exiting keep-alive.",
              asset_uuid,
          )
          break
        logging.warning(
            "Write keep alive RPC failed for %s: %s", asset_uuid, e.details()
        )
        current_interval = min(5.0, current_interval)
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.warning("Write keep alive failed for %s: %s", asset_uuid, e)
        current_interval = min(5.0, current_interval)

  async def _prefetch_keep_alive_loop(
      self, asset_uuid: str, tier_path_uuid: str, interval: float
  ) -> None:
    """Runs prefetch keep-alive heartbeats and resolves future when ready."""
    stub = self._get_or_create_stub()
    current_interval = interval
    while True:
      try:
        await asyncio.sleep(current_interval)
        request = tiering_service_pb2.PrefetchKeepAliveRequest(
            tier_path_uuid=tier_path_uuid
        )
        metadata = await self._get_auth_metadata()
        response = await stub.PrefetchKeepAlive(
            request, metadata=metadata, timeout=30.0
        )
        current_interval = max(
            1.0, float(response.keep_alive_interval_seconds) * 0.8
        )

        for tp in response.asset.tier_paths:
          if tp.tier_path_uuid == tier_path_uuid and tp.HasField("ready_at"):
            if self._prefetch_future and not self._prefetch_future.done():
              self._prefetch_future.set_result(tp.path)
            break
      except asyncio.CancelledError:
        break
      except grpc.aio.AioRpcError as e:
        if e.code() in (
            grpc.StatusCode.NOT_FOUND,
            grpc.StatusCode.FAILED_PRECONDITION,
            grpc.StatusCode.ABORTED,
        ):
          logging.info(
              "PrefetchKeepAlive for %s returned %s; exiting keep-alive.",
              asset_uuid,
              e.code(),
          )
          if self._prefetch_future and not self._prefetch_future.done():
            self._prefetch_future.set_exception(
                RuntimeError(f"Prefetch failed: {e.details()}")
            )
          break
        logging.warning(
            "Prefetch keep alive RPC failed for %s: %s",
            asset_uuid,
            e.details(),
        )
        current_interval = min(5.0, current_interval)
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.warning("Prefetch keep alive failed for %s: %s", asset_uuid, e)
        current_interval = min(5.0, current_interval)

  async def _prepare_reserve_request(
      self,
      path: str,
      tags: Sequence[str] | None,
      user: str | None,
  ) -> tiering_service_pb2.ReserveRequest:
    """Prepares the ReserveRequest protobuf message."""
    if user is None:
      user = environment.get_current_user()
    zone, region = await self._get_gcp_zone_and_region()
    request = tiering_service_pb2.ReserveRequest(
        path=path, tags=tags or [], user=user
    )
    if zone is not None:
      request.zone = zone
    if region is not None:
      request.region = region
    return request

  def _find_reserved_path(
      self, asset: tiering_service_pb2.Asset, tier_path_uuid: str
  ) -> str | None:
    for tp in asset.tier_paths:
      if tp.tier_path_uuid == tier_path_uuid:
        return tp.path
    return None

  async def _async_reserve(
      self,
      path: str,
      tags: Sequence[str] | None = None,
      user: str | None = None,
  ) -> tuple[str, str]:
    """Asynchronous implementation of reserve."""
    self._check_active_operation()
    stub = self._get_or_create_stub()
    request = await self._prepare_reserve_request(path, tags, user)
    metadata = await self._get_auth_metadata()
    try:
      response = await stub.Reserve(request, metadata=metadata, timeout=30.0)
    except grpc.aio.AioRpcError as e:
      raise RuntimeError(
          f"Reserve RPC failed: {e.details()} ({e.code()})"
      ) from e

    asset = response.asset
    asset_uuid = asset.uuid
    if not response.tier_path_uuid:
      raise RuntimeError(
          "Reserve succeeded but returned no tier_path_uuid for asset"
          f" {asset_uuid}"
      )

    reserved_path = self._find_reserved_path(asset, response.tier_path_uuid)
    if reserved_path is None:
      raise RuntimeError(
          f"Reserve succeeded but tier_path_uuid {response.tier_path_uuid}"
          " not found"
      )

    interval = max(1.0, float(response.keep_alive_interval_seconds) * 0.8)
    self._active_asset_uuid = asset_uuid
    self._active_job_type = JobType.WRITE
    self._active_path = reserved_path
    self._active_tier_path_uuid = response.tier_path_uuid
    self._keep_alive_task = self._get_loop().create_task(
        self._write_keep_alive_loop(asset_uuid, interval)
    )

    return asset_uuid, reserved_path

  async def reserve(
      self,
      path: str,
      tags: Sequence[str] | None = None,
      user: str | None = None,
  ) -> tuple[str, str]:
    """Reserves an asset path on Tier 0 storage."""
    await self._ensure_connected()
    lp = self._get_loop()
    fut = asyncio.run_coroutine_threadsafe(
        self._async_reserve(path, tags, user), lp
    )
    return await asyncio.wrap_future(fut)

  async def _async_finalize(self, uuid: str) -> None:
    """Asynchronous implementation of finalize."""
    stub = self._get_or_create_stub()
    request = tiering_service_pb2.FinalizeRequest(uuid=uuid)
    metadata = await self._get_auth_metadata()
    try:
      await stub.Finalize(request, metadata=metadata, timeout=30.0)
    except grpc.aio.AioRpcError as e:
      raise RuntimeError(
          f"Finalize RPC failed: {e.details()} ({e.code()})"
      ) from e
    finally:
      if self._keep_alive_task is not None:
        self._keep_alive_task.cancel()
        self._keep_alive_task = None
      self._reset_active_state()

  async def finalize(self, uuid: str) -> None:
    """Finalizes the asset, marking it stored and immutable."""
    await self._ensure_connected()
    lp = self._get_loop()
    fut = asyncio.run_coroutine_threadsafe(self._async_finalize(uuid), lp)
    await asyncio.wrap_future(fut)

  def _prepare_prefetch_request(
      self,
      path: str | None,
      uuid: str | None,
      zone: str | None,
      region: str | None,
  ) -> tiering_service_pb2.PrefetchRequest:
    """Prepares the PrefetchRequest protobuf message."""
    request = tiering_service_pb2.PrefetchRequest()
    if uuid is not None:
      request.uuid = uuid
    else:
      request.path = path  # pytype: disable=wrong-arg-types
    if zone is not None:
      request.zone = zone
    if region is not None:
      request.region = region
    return request

  def _find_closest_tier_path(
      self, asset: tiering_service_pb2.Asset, closest_tier_path_uuid: str
  ) -> tiering_service_pb2.TierPath | None:
    for tp in asset.tier_paths:
      if tp.tier_path_uuid == closest_tier_path_uuid:
        return tp
    return None

  async def _async_prefetch(
      self,
      path: str | None = None,
      uuid: str | None = None,
  ) -> tuple[str, str]:
    """Asynchronous implementation of prefetch."""
    if path is None and uuid is None:
      raise ValueError("Either path or uuid must be specified.")
    if path is not None and uuid is not None:
      raise ValueError("Only one of path or uuid can be specified.")

    self._check_active_operation()
    stub = self._get_or_create_stub()
    zone, region = await self._get_gcp_zone_and_region()
    request = self._prepare_prefetch_request(path, uuid, zone, region)
    metadata = await self._get_auth_metadata()
    try:
      response = await stub.Prefetch(request, metadata=metadata, timeout=30.0)
    except grpc.aio.AioRpcError as e:
      raise RuntimeError(
          f"Prefetch RPC failed: {e.details()} ({e.code()})"
      ) from e

    asset = response.asset
    asset_uuid = asset.uuid
    if not response.closest_tier_path_uuid:
      raise RuntimeError(
          "Prefetch succeeded but returned no closest_tier_path_uuid for asset"
          f" {asset.uuid}"
      )

    closest_tp = self._find_closest_tier_path(
        asset, response.closest_tier_path_uuid
    )
    if closest_tp is None:
      raise RuntimeError(
          "Prefetch response did not contain closest TierPath for asset"
          f" {asset_uuid}"
      )

    interval = max(1.0, float(response.keep_alive_interval_seconds) * 0.8)
    lp = self._get_loop()
    self._active_asset_uuid = asset_uuid
    self._active_job_type = JobType.PREFETCH
    self._active_path = closest_tp.path
    self._active_tier_path_uuid = closest_tp.tier_path_uuid
    self._prefetch_future = lp.create_future()

    if closest_tp.HasField("ready_at"):
      self._prefetch_future.set_result(closest_tp.path)

    self._keep_alive_task = lp.create_task(
        self._prefetch_keep_alive_loop(
            asset_uuid, closest_tp.tier_path_uuid, interval
        )
    )

    try:
      path_result = await self._prefetch_future
      return asset_uuid, path_result
    except asyncio.CancelledError:
      if self._keep_alive_task is not None:
        self._keep_alive_task.cancel()
        self._keep_alive_task = None
      self._reset_active_state()
      raise

  async def prefetch(
      self,
      path: str | None = None,
      uuid: str | None = None,
  ) -> str:
    """Prefetches the asset to the closest Tier 0 storage."""
    await self._ensure_connected()
    lp = self._get_loop()
    fut = asyncio.run_coroutine_threadsafe(self._async_prefetch(path, uuid), lp)
    _, resolved_path = await asyncio.wrap_future(fut)
    return resolved_path

  async def _async_release(self, uuid: str) -> None:
    del uuid  # Unused.
    if self._keep_alive_task is not None:
      self._keep_alive_task.cancel()
      self._keep_alive_task = None
    if self._prefetch_future is not None and not self._prefetch_future.done():
      self._prefetch_future.cancel()
      self._prefetch_future = None
    self._reset_active_state()

  async def release(self, uuid: str) -> None:
    """Client-side release of prefetch reservation."""
    await self._ensure_connected()
    lp = self._get_loop()
    fut = asyncio.run_coroutine_threadsafe(self._async_release(uuid), lp)
    await asyncio.wrap_future(fut)

  async def release_path(self, path: str) -> None:
    """Releases a prefetch reservation by path string directly."""
    await self._ensure_connected()
    if self._active_path == path or self._active_asset_uuid is not None:
      await self.release(self._active_asset_uuid or "")

  async def _async_delete(
      self,
      path: str | None = None,
      uuid: str | None = None,
  ) -> None:
    """Asynchronous implementation of delete."""
    if path is None and uuid is None:
      raise ValueError("Either path or uuid must be specified.")
    if path is not None and uuid is not None:
      raise ValueError("Only one of path or uuid can be specified.")

    stub = self._get_or_create_stub()
    if uuid is not None:
      request = tiering_service_pb2.DeleteRequest(uuid=uuid)
    else:
      request = tiering_service_pb2.DeleteRequest(path=path)

    metadata = await self._get_auth_metadata()
    try:
      await stub.Delete(request, metadata=metadata, timeout=30.0)
    except grpc.aio.AioRpcError as e:
      raise RuntimeError(
          f"Delete RPC failed: {e.details()} ({e.code()})"
      ) from e

  def _log_fire_and_forget_error(self, fut: Any, error_msg: str) -> None:
    try:
      fut.result()
    except Exception:  # pylint: disable=broad-exception-caught
      logging.exception(error_msg)

  async def delete(
      self,
      path: str | None = None,
      uuid: str | None = None,
  ) -> None:
    """Queues a delete job for the asset."""
    await self._ensure_connected()
    lp = self._get_loop()
    fut = asyncio.run_coroutine_threadsafe(self._async_delete(path, uuid), lp)
    error_msg = f"Delete task failed for path={path}, uuid={uuid}"
    fut.add_done_callback(
        lambda f: self._log_fire_and_forget_error(f, error_msg)
    )

  async def _async_info(
      self,
      path: str | None = None,
      uuid: str | None = None,
  ) -> list[tiering_service_pb2.Asset]:
    """Asynchronous implementation of info."""
    if path is None and uuid is None:
      raise ValueError("Either path or uuid must be specified.")
    if path is not None and uuid is not None:
      raise ValueError("Only one of path or uuid can be specified.")

    stub = self._get_or_create_stub()
    if uuid is not None:
      request = tiering_service_pb2.InfoRequest(uuid=uuid)
    else:
      request = tiering_service_pb2.InfoRequest(path=path)

    metadata = await self._get_auth_metadata()
    try:
      response = await stub.Info(request, metadata=metadata, timeout=30.0)
      return list(response.assets)
    except grpc.aio.AioRpcError as e:
      raise RuntimeError(f"Info RPC failed: {e.details()} ({e.code()})") from e

  async def info(
      self,
      path: str | None = None,
      uuid: str | None = None,
  ) -> list[tiering_service_pb2.Asset]:
    """Retrieves info/metadata for an asset."""
    await self._ensure_connected()
    lp = self._get_loop()
    fut = asyncio.run_coroutine_threadsafe(self._async_info(path, uuid), lp)
    return await asyncio.wrap_future(fut)
