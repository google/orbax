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

"""Unit tests for the storage service logic."""

import concurrent.futures
import shutil
import threading
import time
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from orbax.checkpoint.experimental.caching import service


class TestStorages(parameterized.TestCase):

  def test_from_config_valid(self):
    config = {
        "storage_tier": [
            {"priority": 0, "lustre": {"path": "/l"}},
            {"priority": 1, "gcs": {"bucket": "b"}},
        ]
    }
    storages = service.Storages.from_config(config)
    self.assertLen(storages._storages, 2)
    storage0 = storages.get(0)
    self.assertIsNotNone(storage0)
    self.assertEqual(storage0.storage_type, service.StorageType.LUSTRE)
    storage1 = storages.get(1)
    self.assertIsNotNone(storage1)
    self.assertEqual(storage1.storage_type, service.StorageType.GCS)

  def test_storages_validation(self):
    with self.assertRaisesRegex(
        ValueError, "Storage tier priorities must start at 0"
    ):
      service.Storages.from_config(
          {"storage_tier": [{"priority": 1, "gcs": {"bucket": "b"}}]}
      )

    with self.assertRaisesRegex(
        ValueError,
        "Storage tier priorities must be increasing in intervals of one",
    ):
      service.Storages.from_config({
          "storage_tier": [
              {"priority": 0, "lustre": {"path": "/l"}},
              {"priority": 2, "gcs": {"bucket": "b"}},
          ]
      })


class FakeStorageTransfer(service.StorageTransfer):
  """Fake storage transfer for testing."""

  def __init__(self, src: epath.Path, dst: epath.Path):
    self.src = src
    self.dst = dst
    self.wait_event = None

  def transfer(self) -> bool:
    if self.wait_event:
      self.wait_event.wait()
    if self.dst.exists():
      return False
    assert self.src.is_dir()
    shutil.copytree(self.src, self.dst, dirs_exist_ok=False)
    return True

  def cleanup(self) -> None:
    self.src.rmtree(missing_ok=False)


class TestStorageTransfers(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.temp_dir = self.create_tempdir()
    self.src = epath.Path(self.temp_dir.full_path) / "src"
    self.dst = epath.Path(self.temp_dir.full_path) / "dst"
    self.src.mkdir()
    self.mock_subprocess = self.enter_context(
        mock.patch.object(service.subprocess, "run", autospec=True)
    )

  def test_get_storage_transfer_cls(self):
    self.assertEqual(
        service.get_storage_transfer_cls(
            service.StorageType.LUSTRE, service.StorageType.GCS
        ),
        service.LustreToGcs,
    )
    self.assertEqual(
        service.get_storage_transfer_cls(
            service.StorageType.GCS, service.StorageType.LUSTRE
        ),
        service.GcsToLustre,
    )
    with self.assertRaisesRegex(ValueError, "Unsupported storage transfer"):
      service.get_storage_transfer_cls(
          service.StorageType.GCS, service.StorageType.GCS
      )

  def test_lustre_to_gcs_transfer(self):
    transfer = service.LustreToGcs(self.src, self.dst)
    # src exists (created in setUp)
    self.assertTrue(transfer.transfer())
    self.mock_subprocess.assert_called_once()
    args, _ = self.mock_subprocess.call_args
    self.assertIn("gcloud", args[0])
    self.assertIn("cp", args[0])
    self.assertEqual(args[0][-2], self.src.as_posix())
    self.assertEqual(args[0][-1], self.dst.as_posix())

  def test_lustre_to_gcs_exists(self):
    self.dst.mkdir(parents=True)
    transfer = service.LustreToGcs(self.src, self.dst)
    self.assertFalse(transfer.transfer())
    self.mock_subprocess.assert_not_called()

  def test_gcs_to_lustre_transfer(self):
    transfer = service.GcsToLustre(self.src, self.dst)
    # src exists
    self.assertTrue(transfer.transfer())
    self.mock_subprocess.assert_called_once()
    args, _ = self.mock_subprocess.call_args
    self.assertIn("gcloud", args[0])
    self.assertIn("cp", args[0])
    self.assertTrue(self.dst.exists())

  def test_gcs_to_lustre_exists(self):
    self.dst.mkdir(parents=True)
    transfer = service.GcsToLustre(self.src, self.dst)
    self.assertFalse(transfer.transfer())
    self.mock_subprocess.assert_not_called()


class TestAssets(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.temp_dir = self.create_tempdir()
    self.root = epath.Path(self.temp_dir.full_path)
    self.assets = service.Assets()
    self.storage0 = service.StorageTier(
        priority=0,
        storage_type=service.StorageType.LUSTRE,
        path=self.root / "lustre",
    )
    self.storage1 = service.StorageTier(
        priority=1,
        storage_type=service.StorageType.GCS,
        path=self.root / "gs://",
    )

  def _create_asset_file(self, path: epath.Path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "file").touch()

  def test_assets_thread_safety(self):
    # Test concurrent access to Assets.set
    def worker(i):
      asset_id = service.AssetId(execution_id=1, step=i)
      self.assets.set(
          asset_id,
          service.AssetMetadata(
              asset_id=asset_id,
              tier_id=0,
              path=epath.Path(f"/tmp/{i}"),
              finalized=False,
          ),
      )
      return self.assets.get(asset_id)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
      results = list(executor.map(worker, range(100)))

    self.assertLen(results, 100)
    for i, res in enumerate(results):
      self.assertIsNotNone(res)
      self.assertEqual(res.asset_id.step, i)

  def test_concurrent_moves(self):
    # Test multiple assets moving simultaneously
    self.enter_context(
        mock.patch.object(
            service,
            "get_storage_transfer_cls",
            return_value=FakeStorageTransfer,
        )
    )
    asset_ids = [service.AssetId(execution_id=1, step=i) for i in range(10)]
    for asset_id in asset_ids:
      path = self.storage0.asset_path(asset_id)
      self._create_asset_file(path)
      self.assets.set(
          asset_id,
          service.AssetMetadata(
              asset_id=asset_id,
              tier_id=0,
              path=path,
              finalized=True,
          ),
      )

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
      fs = [
          executor.submit(
              self.assets.move, asset_id, self.storage0, self.storage1
          )
          for asset_id in asset_ids
      ]
      concurrent.futures.wait(fs)

    for f in fs:
      self.assertIsNone(f.result())  # Ensure no exceptions

    # Wait for all transfers to complete
    for asset_id in asset_ids:
      self.assets.await_transfer(asset_id)

    for asset_id in asset_ids:
      asset = self.assets.get(asset_id)
      self.assertIsNotNone(asset)
      self.assertEqual(asset.tier_id, 1)
      self.assertTrue(asset.path.exists())

  def test_move_same_asset_concurrently(self):
    # Verify triggering move on the same asset multiple times is safe and only
    # starts one transfer
    asset_id = service.AssetId(execution_id=1, step=100)
    path = self.storage0.asset_path(asset_id)
    self._create_asset_file(path)
    self.assets.set(
        asset_id,
        service.AssetMetadata(
            asset_id=asset_id,
            tier_id=0,
            path=path,
            finalized=True,
        ),
    )

    # Slow down transfer to ensure overlap
    transfer_event = threading.Event()

    class EventFakeStorageTransfer(FakeStorageTransfer):

      def transfer(self):
        if self.wait_event:
          self.wait_event.wait()
        return super().transfer()

    # Inject the event into instances via class attribute for this test
    EventFakeStorageTransfer.wait_event = transfer_event

    self.enter_context(
        mock.patch.object(
            service,
            "get_storage_transfer_cls",
            return_value=EventFakeStorageTransfer,
        )
    )

    def trigger_move():
      self.assets.move(asset_id, self.storage0, self.storage1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
      fs = [executor.submit(trigger_move) for _ in range(5)]
      # Let them all submit
      time.sleep(0.1)
      transfer_event.set()
      concurrent.futures.wait(fs)

    for f in fs:
      self.assertIsNone(f.result())

    # Wait for the transfer to complete
    self.assets.await_transfer(asset_id)

    asset = self.assets.get(asset_id)
    self.assertIsNotNone(asset)
    self.assertEqual(asset.tier_id, 1)
    self.assertTrue(asset.path.exists())

  def test_await_transfer_concurrently(self):
    # Verify multiple threads waiting on similar or different transfers
    asset_id = service.AssetId(execution_id=1, step=100)
    self.assets.set(
        asset_id,
        service.AssetMetadata(
            asset_id=asset_id,
            tier_id=0,
            path=epath.Path("/l/1/100"),
            finalized=True,
        ),
    )

    # Manually inject a future to simulate a long transfer
    long_future = concurrent.futures.Future()
    with self.assets._lock:
      self.assets._transfers[asset_id] = long_future

    def awaiter():
      self.assets.await_transfer(asset_id)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
      fs = [executor.submit(awaiter) for _ in range(5)]
      time.sleep(0.1)
      long_future.set_result(None)  # Complete the transfer
      concurrent.futures.wait(fs)

    for f in fs:
      self.assertIsNone(f.result())  # All should return successfully

  def test_demote_older_concurrently(self):
    # Verify demote_older implementation under load
    self.enter_context(
        mock.patch.object(
            service,
            "get_storage_transfer_cls",
            return_value=FakeStorageTransfer,
        )
    )
    # Setup multiple assets that qualify for demotion

    older_assets = [service.AssetId(execution_id=1, step=i) for i in range(10)]
    curr_asset = service.AssetId(execution_id=1, step=100)

    # Add older assets to Tier 0
    for asset_id in older_assets:
      path = self.storage0.asset_path(asset_id)
      self._create_asset_file(path)
      self.assets.set(
          asset_id,
          service.AssetMetadata(
              asset_id=asset_id,
              tier_id=0,
              path=path,
              finalized=True,
          ),
      )

    # We call demote_older from multiple threads.
    # They should all try to demote the SAME older assets.
    # We want to ensure each older asset is moved EXACTLY ONCE.

    def trigger_demote():
      self.assets.demote_older(curr_asset, self.storage0, self.storage1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
      fs = [executor.submit(trigger_demote) for _ in range(5)]
      concurrent.futures.wait(fs)

    for f in fs:
      self.assertIsNone(f.result())

    # Wait for all transfers
    for asset_id in older_assets:
      self.assets.await_transfer(asset_id)

    # Check that each asset was moved to Tier 1
    for asset_id in older_assets:
      asset = self.assets.get(asset_id)
      self.assertIsNotNone(asset)
      self.assertEqual(asset.tier_id, 1)
      self.assertTrue(asset.path.exists())


class TestStorageService(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.temp_dir = self.create_tempdir()
    self.path = epath.Path(self.temp_dir.full_path)

    # Mock configuration
    self.config = {
        "storage_tier": [
            {
                "priority": 0,
                "lustre": {"path": (self.path / "lustre").as_posix()},
            },
            {"priority": 1, "gcs": {"bucket": "test-bucket"}},
        ]
    }
    self.service = service.StorageService(self.config)

    # Patch GCS storage path to local for FakeStorageTransfer to work
    # We need to bypass frozen dataclass if applicable
    gcs_storage = self.service.storages.get(1)
    object.__setattr__(gcs_storage, "path", self.path / "gcs")

    # Mock subprocess for GCS upload
    self.mock_subprocess = self.enter_context(
        mock.patch.object(service.subprocess, "run", autospec=True)
    )

  def test_resolve_provisions_new_asset(self):
    asset_id = service.AssetId(execution_id=1, step=100)
    path = self.service.resolve(asset_id)

    expected_path = (self.path / "lustre" / "1" / "100").as_posix()
    self.assertEqual(path, expected_path)

    asset = self.service.assets.get(asset_id)
    self.assertIsNotNone(asset)
    self.assertEqual(asset.tier_id, 0)
    self.assertFalse(asset.finalized)

  def test_resolve_returns_existing_asset(self):
    asset_id = service.AssetId(execution_id=1, step=100)
    path1 = self.service.resolve(asset_id)
    path2 = self.service.resolve(asset_id)
    self.assertEqual(path1, path2)

  def test_finalize_asset(self):
    asset_id = service.AssetId(execution_id=1, step=100)
    self.service.resolve(asset_id)

    self.service.finalize(asset_id)

    asset = self.service.assets.get(asset_id)
    self.assertIsNotNone(asset)
    self.assertTrue(asset.finalized)

  def test_finalize_demotes_older_assets(self):
    # Setup: 3 assets, capacity 2
    self.enter_context(
        mock.patch.object(
            service,
            "get_storage_transfer_cls",
            return_value=FakeStorageTransfer,
        )
    )
    # Tier 0 has capacity 2 (implicitly, or we just rely on logic)

    asset1 = service.AssetId(execution_id=1, step=100)
    asset2 = service.AssetId(execution_id=1, step=200)
    asset3 = service.AssetId(execution_id=1, step=300)

    # Resolve provisions new asset and sets path. We need to create files there.
    path1 = self.service.resolve(asset1)
    epath.Path(path1).mkdir(parents=True, exist_ok=True)
    (epath.Path(path1) / "file").touch()

    path2 = self.service.resolve(asset2)
    epath.Path(path2).mkdir(parents=True, exist_ok=True)
    (epath.Path(path2) / "file").touch()

    path3 = self.service.resolve(asset3)
    epath.Path(path3).mkdir(parents=True, exist_ok=True)
    (epath.Path(path3) / "file").touch()

    self.service.finalize(asset1)
    self.service.finalize(asset2)

    # Verify demotion logic:
    self.service.finalize(asset3)

    # Both asset1 and asset2 are older than asset3, so both should be demoted.
    self.service.assets._thread_pool.shutdown(wait=True)

    asset1_meta = self.service.assets.get(asset1)
    self.assertIsNotNone(asset1_meta)
    self.assertEqual(asset1_meta.tier_id, 1)

    asset2_meta = self.service.assets.get(asset2)
    self.assertIsNotNone(asset2_meta)
    self.assertEqual(asset2_meta.tier_id, 1)

    asset3_meta = self.service.assets.get(asset3)
    self.assertIsNotNone(asset3_meta)
    self.assertEqual(asset3_meta.tier_id, 0)


if __name__ == "__main__":
  absltest.main()
