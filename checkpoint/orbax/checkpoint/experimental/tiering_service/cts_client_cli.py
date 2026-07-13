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

"""Checkpoint Tiering Service (CTS) Client CLI Tool."""

import asyncio
import logging
import os
import shutil
import sys
from typing import Sequence

import fire
from orbax.checkpoint.experimental.tiering_service import client
from orbax.checkpoint.experimental.tiering_service.proto import tiering_service_pb2
import uvloop


class CtsClientCli:
  """CLI tool for Checkpoint Tiering Service (CTS) Client."""

  def __init__(self, server_address: str = "localhost:50051"):
    self._server_address = server_address

  def _get_id_kwargs(self, path_or_uuid: str) -> dict[str, str]:
    if len(path_or_uuid) == 36 and "-" in path_or_uuid:
      return {"uuid": path_or_uuid}
    else:
      return {"path": path_or_uuid}

  def reserve(
      self, path: str, user: str | None = None, auto_finalize: bool = False
  ) -> None:
    """Reserves an asset path on Tier 0 storage.

    Args:
      path: Unique checkpoint logical path.
      user: Optional owner user.
      auto_finalize: If True, finalizes the asset immediately after reserving.
    """

    async def _run():
      c = client.TieringClient(self._server_address)
      try:
        uuid, t0_path = await c.reserve(path, user=user)
        print("Reserve succeeded:")
        print(f"  Asset UUID: {uuid}")
        print(f"  Tier 0 Path: {t0_path}")

        if auto_finalize:
          await c.finalize(uuid)
          print(f"Auto-finalized asset UUID: {uuid}")
        else:
          print("Keep-alive task is active. Maintaining reservation lease...")
          loop = asyncio.get_running_loop()
          await loop.run_in_executor(
              None,
              input,
              "\nWrite dummy data to Tier 0 Path. Press Enter to finalize and"
              " exit...\n",
          )
          await c.finalize(uuid)
          print(f"Finalized asset UUID: {uuid}")
      finally:
        await c.close()

    asyncio.run(_run())

  def finalize(self, uuid: str) -> None:
    """Finalizes the asset, marking it stored and immutable.

    Args:
      uuid: Asset UUID to finalize.
    """

    async def _run():
      c = client.TieringClient(self._server_address)
      try:
        await c.finalize(uuid)
        print(f"Finalize succeeded for asset UUID: {uuid}")
      finally:
        await c.close()

    asyncio.run(_run())

  def prefetch(self, path_or_uuid: str) -> None:
    """Prefetches the asset to Tier 0 storage and waits for completion.

    Args:
      path_or_uuid: Logical path or asset UUID.
    """

    async def _run():
      c = client.TieringClient(self._server_address)
      try:
        print(f"Initiating prefetch for {path_or_uuid}...")
        future = await c.prefetch(**self._get_id_kwargs(path_or_uuid))
        print("Waiting for prefetch to resolve to Tier 0 path...")
        t0_path = await future
        print(f"Prefetch resolved successfully! Tier 0 path: {t0_path}")

        # Keep keep-alive running until user releases
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            input,
            "\nPrefetch lease is active. Press Enter to release and exit...\n",
        )

        is_uuid = len(path_or_uuid) == 36 and "-" in path_or_uuid
        if is_uuid:
          uuid = path_or_uuid
        else:
          assets_list = await c.info(**self._get_id_kwargs(path_or_uuid))
          if assets_list:
            uuid = assets_list[0].uuid
          else:
            print("Could not retrieve asset UUID to release.")
            return
        await c.release(uuid)
        print(f"Released prefetch for asset: {uuid}")
      finally:
        await c.close()

    asyncio.run(_run())

  def release(self, uuid: str) -> None:
    """Releases client-side prefetch keep-alive loop.

    Args:
      uuid: Asset UUID to release.
    """

    async def _run():
      c = client.TieringClient(self._server_address)
      try:
        await c.release(uuid)
        print(f"Released prefetch keep-alive for asset UUID: {uuid}")
      finally:
        await c.close()

    asyncio.run(_run())

  def delete(self, path_or_uuid: str) -> None:
    """Queues a delete job for the asset (deletes from all tiers).

    Args:
      path_or_uuid: Logical path or asset UUID to delete.
    """

    async def _run():
      c = client.TieringClient(self._server_address)
      try:
        await c.delete(**self._get_id_kwargs(path_or_uuid))
        print(f"Delete job queued successfully for {path_or_uuid}")
      finally:
        await c.close()

    asyncio.run(_run())

  def info(self, path_or_uuid: str) -> None:
    """Retrieves and prints metadata info for an asset.

    Args:
      path_or_uuid: Logical path or asset UUID.
    """

    async def _run():
      c = client.TieringClient(self._server_address)
      try:
        assets_list = await c.info(**self._get_id_kwargs(path_or_uuid))
        if not assets_list:
          print("No matching assets found.")
          return
        for i, asset in enumerate(assets_list):
          print(f"Asset #{i+1}:")
          print(f"  UUID: {asset.uuid}")
          print(f"  Path: {asset.path}")
          print(f"  User: {asset.user}")
          print(f"  State: {asset.state}")
          if asset.HasField("created_at"):
            print(f"  Created At: {asset.created_at.ToDatetime()}")
          if asset.HasField("deleted_at"):
            print(f"  Deleted At: {asset.deleted_at.ToDatetime()}")
          print("  Tier Paths:")
          for tp in asset.tier_paths:
            print(f"    - Path: {tp.path}")
            print(f"      TierPath UUID: {tp.tier_path_uuid}")
            print(
                f"      Backend: Level {tp.storage_backend.level}"
                f" ({tp.storage_backend.backend_type})"
            )
            state_name = tiering_service_pb2.TierPathState.Name(tp.state)
            clean_state = state_name.replace("TIER_PATH_STATE_", "")
            if tp.HasField("ready_at"):
              print(
                  f"      Status: {clean_state} "
                  f"(Ready At: {tp.ready_at.ToDatetime()})"
              )
            else:
              print(f"      Status: {clean_state}")
            if tp.HasField("expires_at"):
              print(f"      Expires At: {tp.expires_at.ToDatetime()}")
      finally:
        await c.close()

    asyncio.run(_run())

  def evict(self, path: str) -> None:
    """Simulates cache eviction by manually deleting a local Lustre path.

    WARNING: This command is for testing/simulation purposes only. It physically
    deletes the files/directories at the specified path on the local disk. It
    does NOT communicate with the CTS server or modify the database. This allows
    testing client recovery/prefetch behaviors when the local cache has been
    cleared.

    Args:
      path: Local file or directory path to delete.
    """
    if not os.path.exists(path):
      print(f"Error: Path {path} does not exist.")
      return
    if os.path.isdir(path):
      shutil.rmtree(path)
      print(f"Evicted directory: {path}")
    else:
      os.remove(path)
      print(f"Evicted file: {path}")


def main(argv: Sequence[str] | None = None) -> None:
  if argv is None:
    argv = sys.argv
  uvloop.install()
  try:
    asyncio.get_event_loop()
  except RuntimeError:
    loop = uvloop.new_event_loop()
    asyncio.set_event_loop(loop)
  logging.basicConfig(level=logging.WARNING)
  fire.Fire(CtsClientCli, command=argv[1:])


if __name__ == "__main__":
  main()
