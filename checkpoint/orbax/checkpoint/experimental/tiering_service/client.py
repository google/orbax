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
from typing import Any
from absl import logging
import grpc
from orbax.checkpoint.experimental.tiering_service import client_auth
from orbax.checkpoint.experimental.tiering_service import environment
from orbax.checkpoint.experimental.tiering_service.proto import tiering_service_pb2
from orbax.checkpoint.experimental.tiering_service.proto import tiering_service_pb2_grpc


class JobType(enum.Enum):
  """Job types managed by the centralized keep-alive manager."""

  WRITE = "write"
  PREFETCH = "prefetch"


class _KeepAliveJob:
  """Represents an active keep-alive job managed by the centralized manager."""

  def __init__(
      self,
      asset_uuid: str,
      job_type: JobType,
      interval: float,
      tier_path_uuid: str | None = None,
  ):
    self.asset_uuid = asset_uuid
    self.job_type = job_type
    self.interval = interval
    self.tier_path_uuid = tier_path_uuid
    self.loop = asyncio.get_running_loop()
    self.next_run = self.loop.time() + interval


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
    self._channels: dict[asyncio.AbstractEventLoop, grpc.aio.Channel] = {}
    self._stubs: dict[
        asyncio.AbstractEventLoop, tiering_service_pb2_grpc.TieringServiceStub
    ] = {}
    self._zone = None
    self._region = None
    self._env_queried = False
    self._env_lock = None
    self._keep_alives: dict[tuple[str, JobType], _KeepAliveJob] = {}
    self._keep_alive_manager_tasks: dict[
        asyncio.AbstractEventLoop, asyncio.Task[None]
    ] = {}
    self._keep_alive_events: dict[asyncio.AbstractEventLoop, asyncio.Event] = {}
    self._prefetch_futures: dict[
        asyncio.AbstractEventLoop, dict[str, asyncio.Future[str]]
    ] = {}
    self._path_to_uuid: dict[str, str] = {}

  async def __aenter__(self) -> "TieringClient":
    await self.connect()
    return self

  async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
    await self.close()

  def _get_or_create_stub(self) -> tiering_service_pb2_grpc.TieringServiceStub:
    """Gets or creates the gRPC stub for the current event loop."""
    loop = asyncio.get_running_loop()
    if loop not in self._stubs:
      if self._secure:
        is_local = (
            "localhost" in self._server_address
            or "127.0.0.1" in self._server_address
        )
        if is_local:
          try:
            # Secure channel setup. Fall back to SSL if local creds not
            # supported.
            creds = grpc.local_channel_credentials()
          except AttributeError:
            creds = grpc.ssl_channel_credentials()
        else:
          creds = grpc.ssl_channel_credentials()
        channel = grpc.aio.secure_channel(self._server_address, creds)
      else:
        channel = grpc.aio.insecure_channel(self._server_address)

      self._channels[loop] = channel
      self._stubs[loop] = tiering_service_pb2_grpc.TieringServiceStub(channel)

    return self._stubs[loop]

  async def connect(self) -> None:
    """Establishes an async gRPC channel with the server."""
    self._get_or_create_stub()

  async def close(self) -> None:
    """Closes the gRPC channel."""
    try:
      current_loop = asyncio.get_running_loop()
    except RuntimeError:
      current_loop = None

    # Cancel manager task for current loop
    if current_loop and current_loop in self._keep_alive_manager_tasks:
      task = self._keep_alive_manager_tasks[current_loop]
      task.cancel()
      try:
        await task
      except asyncio.CancelledError:
        pass
      del self._keep_alive_manager_tasks[current_loop]

    # Cancel manager tasks for other loops (or all if no current_loop)
    for loop_val, task in list(self._keep_alive_manager_tasks.items()):
      task.cancel()
      del self._keep_alive_manager_tasks[loop_val]

    # Clean up jobs belonging to current loop (or all if current_loop is None)
    if current_loop:
      for key, job in list(self._keep_alives.items()):
        if job.loop == current_loop:
          del self._keep_alives[key]
      if current_loop in self._keep_alive_events:
        del self._keep_alive_events[current_loop]
    else:
      self._keep_alives.clear()
      self._keep_alive_events.clear()

    # Release pending prefetches and cancel futures belonging to current loop
    if current_loop and current_loop in self._prefetch_futures:
      for asset_uuid, fut in list(self._prefetch_futures[current_loop].items()):
        if not fut.done():
          self._stop_prefetch_keep_alive(asset_uuid)
          fut.cancel()
      del self._prefetch_futures[current_loop]

    # Clean up remaining loops' futures (cancel without awaiting)
    for loop_val, fut_dict in list(self._prefetch_futures.items()):
      for asset_uuid, fut in list(fut_dict.items()):
        if not fut.done():
          fut.cancel()
      del self._prefetch_futures[loop_val]

    for loop_val, channel in list(self._channels.items()):
      if current_loop and loop_val == current_loop:
        await channel.close()
        del self._channels[loop_val]
      elif not current_loop:
        try:
          asyncio.run(channel.close())
        except Exception:  # pylint: disable=broad-except
          pass
        del self._channels[loop_val]

    if not self._channels:
      self._stubs.clear()

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

  def _ensure_manager_running(self) -> None:
    loop = asyncio.get_running_loop()
    if loop not in self._keep_alive_events:
      self._keep_alive_events[loop] = asyncio.Event()
    if (
        loop not in self._keep_alive_manager_tasks
        or self._keep_alive_manager_tasks[loop].done()
    ):
      self._keep_alive_manager_tasks[loop] = asyncio.create_task(
          self._keep_alive_manager_loop()
      )

  def _start_write_keep_alive(self, asset_uuid: str, interval: int) -> None:
    """Starts the write keep-alive background task."""
    job = _KeepAliveJob(
        asset_uuid=asset_uuid,
        job_type=JobType.WRITE,
        interval=max(1.0, float(interval) * 0.8),
    )
    self._keep_alives[(asset_uuid, JobType.WRITE)] = job
    self._ensure_manager_running()
    self._keep_alive_events[job.loop].set()

  def _stop_write_keep_alive(self, asset_uuid: str) -> None:
    """Stops the write keep-alive background task."""
    job = self._keep_alives.pop((asset_uuid, JobType.WRITE), None)
    if job:
      self._keep_alive_events[job.loop].set()

  def _start_prefetch_keep_alive(
      self, asset_uuid: str, tier_path_uuid: str, interval: int
  ) -> None:
    """Starts the prefetch keep-alive background task."""
    job = _KeepAliveJob(
        asset_uuid=asset_uuid,
        job_type=JobType.PREFETCH,
        interval=max(1.0, float(interval) * 0.8),
        tier_path_uuid=tier_path_uuid,
    )
    self._keep_alives[(asset_uuid, JobType.PREFETCH)] = job
    self._ensure_manager_running()
    self._keep_alive_events[job.loop].set()

  def _stop_prefetch_keep_alive(self, asset_uuid: str) -> None:
    """Stops the prefetch keep-alive background task."""
    job = self._keep_alives.pop((asset_uuid, JobType.PREFETCH), None)
    loop = None
    if job:
      self._keep_alive_events[job.loop].set()
      loop = job.loop
    else:
      try:
        loop = asyncio.get_running_loop()
      except RuntimeError:
        pass

    if loop and loop in self._prefetch_futures:
      fut = self._prefetch_futures[loop].pop(asset_uuid, None)
      if fut and not fut.done():
        fut.cancel()

  def _get_earliest_job(
      self, loop: asyncio.AbstractEventLoop
  ) -> tuple[_KeepAliveJob | None, float | None]:
    """Finds the earliest job to run and its scheduled time."""
    earliest_job = None
    earliest_time = None
    for job in self._keep_alives.values():
      if job.loop == loop:
        if earliest_time is None or job.next_run < earliest_time:
          earliest_time = job.next_run
          earliest_job = job
    return earliest_job, earliest_time

  async def _wait_for_next_job(
      self, loop: asyncio.AbstractEventLoop, timeout: float
  ) -> bool:
    """Waits for next job or early wakeup. Returns True if woken up early."""
    event_task = loop.create_task(self._keep_alive_events[loop].wait())
    sleep_task = loop.create_task(asyncio.sleep(timeout))
    try:
      done, _ = await asyncio.wait(
          [event_task, sleep_task],
          return_when=asyncio.FIRST_COMPLETED
      )
      return event_task in done
    finally:
      event_task.cancel()
      sleep_task.cancel()

  async def _keep_alive_manager_loop(self) -> None:
    """Centralized manager loop running heartbeats for all keep-alives."""
    logging.info("Starting centralized keep-alive manager task.")
    loop = asyncio.get_running_loop()
    while True:
      try:
        self._keep_alive_events[loop].clear()
        earliest_job, earliest_time = self._get_earliest_job(loop)
        if earliest_job is None or earliest_time is None:
          # Wait indefinitely for a new job.
          await self._keep_alive_events[loop].wait()
          continue

        now = loop.time()
        sleep_duration = earliest_time - now
        if sleep_duration > 0:
          # Wait until the next job or early wakeup by new jobs.
          if await self._wait_for_next_job(loop, sleep_duration):
            continue

        await self._run_keep_alive_job(earliest_job)

      except asyncio.CancelledError:
        logging.info("Centralized keep-alive manager task cancelled.")
        break
      except Exception:  # pylint: disable=broad-exception-caught
        # Log unexpected errors and continue running.
        logging.exception("Error in centralized keep-alive manager loop.")
        await asyncio.sleep(1.0)

  async def _run_write_keep_alive_job(
      self,
      job: _KeepAliveJob,
      stub: tiering_service_pb2_grpc.TieringServiceStub,
      now: float,
  ) -> None:
    """Executes a single write keep-alive heartbeat request."""
    try:
      request = tiering_service_pb2.ReserveKeepAliveRequest(uuid=job.asset_uuid)
      metadata = await self._get_auth_metadata()
      response = await stub.ReserveKeepAlive(request, metadata=metadata)
      job.interval = max(1.0, float(response.keep_alive_interval_seconds) * 0.8)
      job.next_run = now + job.interval
      logging.info(
          "Extended write reservation lease for asset %s", job.asset_uuid
      )
    except grpc.aio.AioRpcError as e:
      logging.warning(
          "ReserveKeepAlive failed for asset %s: %s",
          job.asset_uuid,
          e.details(),
      )
      if e.code() == grpc.StatusCode.NOT_FOUND:
        self._keep_alives.pop((job.asset_uuid, JobType.WRITE), None)
      else:
        job.next_run = now + min(5.0, job.interval)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.warning(
          "Unexpected error in write keep alive for asset %s: %s",
          job.asset_uuid,
          e,
      )
      self._keep_alives.pop((job.asset_uuid, JobType.WRITE), None)

  async def _run_prefetch_keep_alive_job(
      self,
      job: _KeepAliveJob,
      stub: tiering_service_pb2_grpc.TieringServiceStub,
      now: float,
  ) -> None:
    """Executes a single prefetch keep-alive heartbeat request."""
    try:
      request = tiering_service_pb2.PrefetchKeepAliveRequest(
          tier_path_uuid=job.tier_path_uuid
      )
      metadata = await self._get_auth_metadata()
      response = await stub.PrefetchKeepAlive(request, metadata=metadata)

      job.interval = max(1.0, float(response.keep_alive_interval_seconds) * 0.8)
      job.next_run = now + job.interval
      logging.info("Sent PrefetchKeepAlive for asset %s", job.asset_uuid)

      target_path = None
      ready = False
      for tp in response.asset.tier_paths:
        if tp.tier_path_uuid == job.tier_path_uuid:
          target_path = tp.path
          if tp.HasField("ready_at"):
            ready = True
          break

      if ready and target_path:
        loop = asyncio.get_running_loop()
        if loop in self._prefetch_futures:
          fut = self._prefetch_futures[loop].get(job.asset_uuid)
          if fut and not fut.done():
            fut.set_result(target_path)
            logging.info(
                "Prefetch completed and resolved for asset %s", job.asset_uuid
            )

    except grpc.aio.AioRpcError as e:
      logging.warning(
          "PrefetchKeepAlive failed for asset %s: %s",
          job.asset_uuid,
          e.details(),
      )
      if e.code() in (
          grpc.StatusCode.NOT_FOUND,
          grpc.StatusCode.FAILED_PRECONDITION,
          grpc.StatusCode.ABORTED,
      ):
        loop = asyncio.get_running_loop()
        if loop in self._prefetch_futures:
          fut = self._prefetch_futures[loop].get(job.asset_uuid)
          if fut and not fut.done():
            fut.set_exception(
                RuntimeError(f"Prefetch failed: {e.details()}")
            )
        self._keep_alives.pop((job.asset_uuid, JobType.PREFETCH), None)
      else:
        job.next_run = now + min(5.0, job.interval)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.warning(
          "Unexpected error in prefetch keep alive for asset %s: %s",
          job.asset_uuid,
          e,
      )
      loop = asyncio.get_running_loop()
      if loop in self._prefetch_futures:
        fut = self._prefetch_futures[loop].get(job.asset_uuid)
        if fut and not fut.done():
          fut.set_exception(e)
      self._keep_alives.pop((job.asset_uuid, JobType.PREFETCH), None)

  async def _run_keep_alive_job(self, job: _KeepAliveJob) -> None:
    """Executes a single keep-alive heartbeat request."""
    stub = self._get_or_create_stub()
    now = asyncio.get_running_loop().time()

    if job.job_type == JobType.WRITE:
      await self._run_write_keep_alive_job(job, stub, now)
    elif job.job_type == JobType.PREFETCH:
      await self._run_prefetch_keep_alive_job(job, stub, now)

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
    stub = self._get_or_create_stub()

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
    asset_uuid = asset.uuid
    interval = response.keep_alive_interval_seconds

    if not response.tier_path_uuid:
      raise RuntimeError(
          "Reserve succeeded but returned no tier_path_uuid for asset"
          f" {asset_uuid}"
      )

    # Start write keep-alive background task loop
    self._start_write_keep_alive(asset_uuid, interval)

    for tp in asset.tier_paths:
      if tp.tier_path_uuid == response.tier_path_uuid:
        return asset_uuid, tp.path

    # Stop keep-alive loop if the returned tier_path_uuid is missing from
    # asset tier paths
    self._stop_write_keep_alive(asset_uuid)
    raise RuntimeError(
        "Reserve succeeded but returned tier_path_uuid"
        f" {response.tier_path_uuid} which is not found in asset tier paths"
        f" for asset {asset_uuid}"
    )

  async def finalize(self, uuid: str) -> None:
    """Finalizes the asset, marking it stored and immutable.

    Args:
      uuid: Asset UUID to finalize.

    Raises:
      RuntimeError: If gRPC call fails.
    """
    stub = self._get_or_create_stub()

    request = tiering_service_pb2.FinalizeRequest(uuid=uuid)
    metadata = await self._get_auth_metadata()
    try:
      await stub.Finalize(request, metadata=metadata)
    except grpc.aio.AioRpcError as e:
      raise RuntimeError(
          f"Finalize RPC failed: {e.details()} ({e.code()})"
      ) from e
    finally:
      # Stop keep-alive loop inside finally, so it is stopped even on error
      self._stop_write_keep_alive(uuid)

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

    loop = asyncio.get_running_loop()
    if loop in self._prefetch_futures and uuid in self._prefetch_futures[loop]:
      return self._prefetch_futures[loop][uuid]

    stub = self._get_or_create_stub()

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

    asset = response.asset
    asset_uuid = asset.uuid
    interval = response.keep_alive_interval_seconds

    if (
        loop in self._prefetch_futures
        and asset_uuid in self._prefetch_futures[loop]
    ):
      return self._prefetch_futures[loop][asset_uuid]

    if loop not in self._prefetch_futures:
      self._prefetch_futures[loop] = {}
    future = loop.create_future()
    self._prefetch_futures[loop][asset_uuid] = future

    closest_tp = None
    if response.closest_tier_path_uuid:
      for tp in asset.tier_paths:
        if tp.tier_path_uuid == response.closest_tier_path_uuid:
          closest_tp = tp
          break
    else:
      raise RuntimeError(
          "Prefetch succeeded but returned no closest_tier_path_uuid for asset"
          f" {asset.uuid}"
      )

    if closest_tp is None:
      if loop in self._prefetch_futures:
        self._prefetch_futures[loop].pop(asset_uuid, None)
      raise RuntimeError(
          "Prefetch response did not contain closest TierPath matching "
          f"{response.closest_tier_path_uuid} for asset {asset_uuid}"
      )

    self._start_prefetch_keep_alive(
        asset_uuid, closest_tp.tier_path_uuid, interval
    )

    self._path_to_uuid[asset.path] = asset_uuid
    self._path_to_uuid[closest_tp.path] = asset_uuid

    if closest_tp.HasField("ready_at"):
      future.set_result(closest_tp.path)
    return future

  async def release(self, uuid: str) -> None:
    """Client-side release of prefetch keep-alive loop.

    Args:
      uuid: Asset UUID to release.
    """
    self._stop_prefetch_keep_alive(uuid)

  async def release_path(self, path: str) -> None:
    """Client-side release of prefetch keep-alive loop by path.

    Args:
      path: Logical path or physical Lustre path of the asset.
    """
    uuid = self._path_to_uuid.pop(path, None)
    if uuid:
      keys_to_remove = [k for k, v in self._path_to_uuid.items() if v == uuid]
      for k in keys_to_remove:
        self._path_to_uuid.pop(k, None)
      await self.release(uuid)

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

    stub = self._get_or_create_stub()

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

    stub = self._get_or_create_stub()

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
