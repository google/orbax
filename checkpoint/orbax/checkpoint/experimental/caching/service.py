# Copyright 2025 The Orbax Authors.
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

"""Core business logic for the Storage Service."""

from __future__ import annotations

import concurrent.futures
import dataclasses
import os
import subprocess
import threading
from typing import Type

from absl import logging
from etils import epath
from orbax.checkpoint.experimental.caching import types

StorageType = types.StorageType
StorageTransfer = types.StorageTransfer
StorageTier = types.StorageTier
AssetId = types.AssetId
AssetMetadata = types.AssetMetadata

# Default configuration for storage tiers
DEFAULT_CONFIG = {
    "storage_tier": [
        {"priority": 0, "lustre": {"path": "/lustre"}},
        {"priority": 1, "gcs": {"bucket": "cpgaffney-test-service"}},
    ]
}


def get_storage_transfer_cls(
    src_storage: StorageType, dst_storage: StorageType
) -> Type[StorageTransfer]:
  if src_storage == StorageType.LUSTRE:
    if dst_storage == StorageType.GCS:
      return LustreToGcs
  elif src_storage == StorageType.GCS:
    if dst_storage == StorageType.LUSTRE:
      return GcsToLustre

  raise ValueError(
      f"Unsupported storage transfer: {src_storage} -> {dst_storage}"
  )


def node_id() -> int:
  return int(os.environ.get("JOB_COMPLETION_INDEX", "0"))


class LustreToGcs(StorageTransfer):
  """Transfer implementation for moving data from Lustre to GCS."""

  def __init__(self, src: epath.Path, dst: epath.Path):
    self.src = src
    self.dst = dst

  def transfer(self) -> bool:
    """Moves a file from local Lustre mount to GCS."""
    logging.info("Transferring %s to %s.", self.src, self.dst)
    if self.dst.exists():
      logging.info("Destination %s already exists, skipping upload.", self.dst)
      return False
    # TODO(b/478278395) Delegate to a proper GCS client.
    subprocess.run(
        [
            "gcloud",
            "storage",
            "cp",
            "-r",
            self.src.as_posix(),
            self.dst.as_posix(),
        ],
        check=True,
        capture_output=True,
    )
    logging.info("Finished transferring %s to %s.", self.src, self.dst)
    return True

  def cleanup(self) -> None:
    logging.info("Cleaning up %s.", self.src)
    self.src.rmtree(missing_ok=False)
    logging.info("Finished cleaning up %s.", self.src)


class GcsToLustre(StorageTransfer):
  """Transfer implementation for moving data from GCS to Lustre."""

  def __init__(self, src: epath.Path, dst: epath.Path):
    self.src = src
    self.dst = dst

  def transfer(self) -> bool:
    """Moves a file from GCS to local Lustre mount."""
    logging.info("Transferring %s to %s.", self.src, self.dst)
    if self.dst.exists():
      logging.info(
          "Destination %s already exists, skipping transfer.", self.dst
      )
      return False
    self.dst.mkdir(parents=True, exist_ok=False)

    # TODO(cpgaffney) set stripe count?
    # `lfs setstripe -c 4 /lustre/path/to/destination`.
    subprocess.run(
        [
            "gcloud",
            "storage",
            "cp",
            "-r",
            self.src.as_posix(),
            self.dst.as_posix(),
        ],
        check=True,
        capture_output=True,
    )
    logging.info("Finished transferring %s to %s.", self.src, self.dst)
    return True

  def cleanup(self) -> None:
    logging.info("Cleaning up %s.", self.src)
    subprocess.run(
        [
            "gcloud",
            "storage",
            "rm",
            "-r",
            self.src.as_posix(),
        ],
        check=True,
        capture_output=True,
    )
    logging.info("Finished cleaning up %s.", self.src)


class Storages:
  """Manages storage tiers configuration."""

  def __init__(self, storages: dict[int, StorageTier]):
    self._storages = storages

  def get(self, tier: int) -> StorageTier | None:
    return self._storages.get(tier)

  @classmethod
  def from_config(cls, config) -> Storages:
    """Creates a Storages instance from a configuration dictionary."""
    storages = {}
    for tier_config in config.get("storage_tier", []):
      priority = tier_config["priority"]
      tier_type = None
      base_path = None
      if "lustre" in tier_config:
        tier_type = StorageType.LUSTRE
        base_path = tier_config["lustre"]["path"]
      elif "gcs" in tier_config:
        tier_type = StorageType.GCS
        base_path = "gs://" + tier_config["gcs"]["bucket"]
      if tier_type is None or base_path is None:
        raise ValueError(f"Invalid storage tier config: {tier_config}")

      storages[priority] = StorageTier(
          priority=priority,
          storage_type=tier_type,
          path=epath.Path(base_path),
      )

    # Validate that priorities are increasing in intervals of one
    if not storages:
      raise ValueError("No storage tiers configured.")
    sorted_priorities = sorted(storages.keys())
    if sorted_priorities[0] != 0:
      raise ValueError("Storage tier priorities must start at 0.")
    for i in range(1, len(sorted_priorities)):
      if sorted_priorities[i] - sorted_priorities[i - 1] != 1:
        raise ValueError(
            "Storage tier priorities must be increasing in intervals of one."
            f" Found gap between {sorted_priorities[i-1]} and"
            f" {sorted_priorities[i]}."
        )

    return Storages(storages)

  def to_json(self):
    return {
        priority: storage.to_dict()
        for priority, storage in self._storages.items()
    }

  def __repr__(self):
    return f"Storages({self._storages})"


class Assets:
  """Thread-safe asset management.

  # TODO(b/478278948) Persist information to databases.
  """

  def __init__(self):
    self._assets: dict[AssetId, AssetMetadata] = {}
    # TODO(b/478282309) Ensure atomicity.
    self._transfers: dict[AssetId, concurrent.futures.Future[None]] = {}
    self._thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=32)
    self._lock = threading.Lock()

  def exists(self, asset_id: AssetId) -> bool:
    with self._lock:
      return self.get(asset_id) is not None

  def get(self, asset_id: AssetId) -> AssetMetadata | None:
    with self._lock:
      return self._assets.get(asset_id)

  def set(self, asset_id: AssetId, asset_metadata: AssetMetadata):
    with self._lock:
      self._assets[asset_id] = asset_metadata

  def demote_older(
      self,
      current_asset_id: AssetId,
      current_storage: StorageTier,
      next_storage: StorageTier,
  ):
    """Demotes older assets to the next storage tier."""
    assets_to_demote = []
    with self._lock:
      for asset_id, asset in self._assets.items():
        if (
            asset.tier_id == current_storage.priority
            and asset_id.execution_id == current_asset_id.execution_id
            and asset_id.step < current_asset_id.step
        ):
          assets_to_demote.append(asset_id)
          logging.info("Starting demotion of asset: %s", asset_id)

    for asset_id in assets_to_demote:
      self.move(asset_id, current_storage, next_storage)

  def _background_move(
      self,
      asset: AssetMetadata,
      current_storage: StorageTier,
      new_storage: StorageTier,
  ):
    """Moves an asset in the background."""
    asset_id = asset.asset_id
    new_path = new_storage.asset_path(asset_id)
    transfer = get_storage_transfer_cls(
        current_storage.storage_type, new_storage.storage_type
    )(asset.path, new_path)
    assert transfer.transfer()  # This is the time-consuming bit.
    # Now the asset is in both src and dst.
    with self._lock:
      self._assets[asset_id] = dataclasses.replace(
          asset, path=new_path, tier_id=new_storage.priority
      )
    transfer.cleanup()
    logging.info("Finished move: %s @ %s", asset_id, new_path)

  def move(
      self,
      asset_id: AssetId,
      current_storage: StorageTier,
      new_storage: StorageTier,
  ):
    """Initiates a move of an asset from one storage tier to another."""
    with self._lock:
      # Move already in progress.
      if asset_id in self._transfers:
        return
      asset = self._assets.get(asset_id)
      if asset is None:
        raise KeyError(f"Asset {asset_id} not found.")
      if asset.tier_id != current_storage.priority:
        raise ValueError(
            f"Asset {asset_id} is in tier {asset.tier_id}, expected"
            f" {current_storage.priority}."
        )
      logging.info("Started move: %s @ %s", asset_id, asset.path)

      f = self._thread_pool.submit(
          self._background_move, asset, current_storage, new_storage
      )
      self._transfers[asset_id] = f

  def await_transfer(self, asset_id: AssetId):
    with self._lock:
      f = self._transfers.get(asset_id)
    if f is None:
      logging.info("No transfer in progress for asset %s. Returning.", asset_id)
      return
    logging.info("Waiting for transfer of asset %s.", asset_id)
    f.result()
    logging.info("Finished waiting for transfer of asset %s.", asset_id)
    with self._lock:
      if asset_id in self._transfers:
        del self._transfers[asset_id]

  def to_json(self):
    return {
        f"{asset_id.execution_id}-{asset_id.step}": metadata.to_dict()
        for asset_id, metadata in self._assets.items()
    }

  def __repr__(self):
    return f"Assets({self._assets})"


class StorageService:
  """Main service class managing storages and assets."""

  def __init__(self, config):
    self.storages = Storages.from_config(config)
    self.assets = Assets()

  def resolve(self, asset_id: AssetId) -> str:
    """Resolves an asset, provisioning it if necessary."""
    if asset := self.assets.get(asset_id):
      resolved_path = asset.path
      logging.info("RESOLVE: %s @ %s", asset_id, resolved_path)
      return resolved_path.as_posix()
    else:
      storage = self.storages.get(0)
      if storage is None:
        raise ValueError("Storage tier 0 not found.")
      path = storage.asset_path(asset_id)
      self.assets.set(
          asset_id,
          AssetMetadata(
              asset_id=asset_id, tier_id=0, path=path, finalized=False
          ),
      )
      logging.info("PROVISION: %s @ %s", asset_id, path)
      return path.as_posix()

  def finalize(self, asset_id: AssetId):
    """Finalizes an asset and triggers demotion of older assets."""
    if asset := self.assets.get(asset_id):
      if asset.finalized:
        raise ValueError(f"Asset {asset_id} already finalized.")
      assert asset.tier_id == 0, f"Asset {asset_id} is not in Tier 0."
      self._demote_older_assets(asset_id)
      self.assets.set(asset_id, dataclasses.replace(asset, finalized=True))
    else:
      raise ValueError(f"Asset {asset_id} not found.")

    logging.info("FINALIZE: %s marked complete.", asset_id)

  def _demote_older_assets(self, current_asset_id: AssetId):
    """Demotes assets from Tier 0 to Tier 1."""
    current_asset = self.assets.get(current_asset_id)
    assert current_asset is not None
    assert (
        current_asset.tier_id == 0
    ), f"Asset {current_asset_id} is not in Tier 0."
    current_storage_tier = self.storages.get(0)
    next_storage_tier = self.storages.get(1)
    assert current_storage_tier is not None
    assert next_storage_tier is not None
    self.assets.demote_older(
        current_asset_id, current_storage_tier, next_storage_tier
    )

  def prefetch(self, asset_id: AssetId):
    """Prefetches an asset to Tier 0."""
    if asset := self.assets.get(asset_id):
      if asset.tier_id == 0:
        logging.info("Asset %s is already in Tier 0.", asset_id)
        return

      current_storage = self.storages.get(asset.tier_id)
      # Assuming Tier 0 is the target for prefetch
      next_storage = self.storages.get(0)

      if current_storage is None or next_storage is None:
        raise ValueError("Invalid storage tiers for prefetch.")

      self.assets.move(asset_id, current_storage, next_storage)
      logging.info("PREFETCH: Scheduled %s", asset_id)
    else:
      raise ValueError(f"Asset {asset_id} not found.")

  def await_transfer(self, asset_id: AssetId):
    if not self.assets.exists(asset_id):
      raise ValueError(f"Asset {asset_id} not found.")
    self.assets.await_transfer(asset_id)
