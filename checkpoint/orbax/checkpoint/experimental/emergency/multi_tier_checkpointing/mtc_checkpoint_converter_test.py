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

"""Tests for converting Multi-Tier Checkpointing backups to standard Orbax checkpoints."""

import os
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import numpy as np
from orbax.checkpoint import checkpoint_manager
from orbax.checkpoint._src import asyncio_utils
from orbax.checkpoint._src.handlers import pytree_checkpoint_handler
from orbax.checkpoint.experimental.emergency.multi_tier_checkpointing import mtc_checkpoint_converter

PyTreeCheckpointHandler = pytree_checkpoint_handler.PyTreeCheckpointHandler
PyTreeSaveArgs = pytree_checkpoint_handler.PyTreeSaveArgs


class MtcCheckpointConverterTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = epath.Path(self.create_tempdir("mtc_test").full_path)
    self.backup_dir = self.test_dir / "backup"
    self.backup_dir.mkdir(parents=True, exist_ok=True)
    self.output_dir = self.test_dir / "standard_checkpoints"

  def _save_node_item(
      self, directory: epath.Path, item: dict[str, Any]
  ) -> None:
    """Helper to save a mock checkpoint item using PyTreeCheckpointHandler."""
    handler = PyTreeCheckpointHandler(
        use_ocdbt=True,
        use_zarr3=True,
    )
    save_args = PyTreeSaveArgs(item=item)
    futures = asyncio_utils.run_sync(
        handler.async_save(directory, args=save_args)
    )
    if futures:
      for f in futures:
        f.result()
    handler.finalize(directory)

  def test_parse_meta_filename(self):
    parsed = mtc_checkpoint_converter.parse_meta_filename(
        "maxtext-s1000-n2-w0.meta"
    )
    self.assertEqual(parsed, ("maxtext", 1000, 2, 0))

    # Complex job name with hyphens
    parsed_complex = mtc_checkpoint_converter.parse_meta_filename(
        "my-distributed-training-job-v2-s42-n15-w0.meta"
    )
    self.assertEqual(
        parsed_complex, ("my-distributed-training-job-v2", 42, 15, 0)
    )

    # Non-matching filenames
    self.assertIsNone(
        mtc_checkpoint_converter.parse_meta_filename("manifest.ocdbt")
    )
    self.assertIsNone(
        mtc_checkpoint_converter.parse_meta_filename("maxtext-s100.meta")
    )
    self.assertIsNone(
        mtc_checkpoint_converter.parse_meta_filename("something.data")
    )

  def test_find_data_dir(self):
    # Setup data dir
    data_dir = self.backup_dir / "D1234abcd.data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Found with 'D' prefix
    self.assertEqual(
        mtc_checkpoint_converter.find_data_dir(self.backup_dir, "D1234abcd"),
        data_dir,
    )
    # Found without 'D' prefix in search query
    self.assertEqual(
        mtc_checkpoint_converter.find_data_dir(self.backup_dir, "1234abcd"),
        data_dir,
    )

    # Missing data dir raises FileNotFoundError
    with self.assertRaises(FileNotFoundError):
      mtc_checkpoint_converter.find_data_dir(self.backup_dir, "Dnonexistent")

  def test_discover_mtc_meta_files(self):
    # Create mock .meta files for step 10 and step 20
    (self.backup_dir / "job-s10-n1-w0.meta").write_text("Dhash10_1\n")
    (self.backup_dir / "job-s10-n0-w0.meta").write_text("Dhash10_0\n")
    (self.backup_dir / "job-s20-n0-w0.meta").write_text("Dhash20_0\n")
    (self.backup_dir / "unrelated_file.txt").write_text("ignored")

    discovered = mtc_checkpoint_converter.discover_mtc_meta_files(
        self.backup_dir
    )
    self.assertEqual(sorted(discovered.keys()), [10, 20])

    # Step 10 should be sorted by node_rank (n0 before n1)
    step10_meta = discovered[10]
    self.assertLen(step10_meta, 2)
    self.assertEqual(step10_meta[0].node_rank, 0)
    self.assertEqual(step10_meta[0].data_hash, "Dhash10_0")
    self.assertEqual(step10_meta[1].node_rank, 1)
    self.assertEqual(step10_meta[1].data_hash, "Dhash10_1")

  def test_convert_mtc_step_and_standard_restore(self):
    """End-to-end test converting multi-node MTC backup and restoring."""
    step = 50

    # Node 0 shard
    node0_hash = "Dnode0hash"
    node0_dir = self.backup_dir / f"{node0_hash}.data"
    node0_state = node0_dir / "state"
    node0_tree = {
        "layer_0": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "bias_0": np.array([0.5, 0.5], dtype=np.float32),
    }
    self._save_node_item(node0_state, node0_tree)
    (node0_dir / "_CHECKPOINT_METADATA").write_text('{"version": 1}')

    # Node 1 shard
    node1_hash = "Dnode1hash"
    node1_dir = self.backup_dir / f"{node1_hash}.data"
    node1_state = node1_dir / "state"
    node1_tree = {
        "layer_1": np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
        "bias_1": np.array([1.5, 1.5], dtype=np.float32),
    }
    self._save_node_item(node1_state, node1_tree)
    if (node1_state / "ocdbt.process_0").exists():
      (node1_state / "ocdbt.process_0").rename(node1_state / "ocdbt.process_1")
    (node1_dir / "_CHECKPOINT_METADATA").write_text('{"version": 1}')

    # Write .meta files
    (self.backup_dir / f"maxtext-s{step}-n0-w0.meta").write_text(node0_hash)
    (self.backup_dir / f"maxtext-s{step}-n1-w0.meta").write_text(node1_hash)

    # Perform conversion
    converted_paths = mtc_checkpoint_converter.convert_mtc_backup(
        input_dir=self.backup_dir,
        output_dir=self.output_dir,
        step=step,
    )
    self.assertLen(converted_paths, 1)
    target_step_dir = converted_paths[0]
    self.assertEqual(target_step_dir, self.output_dir / str(step))

    # Verify directory layout of standard Orbax checkpoint
    self.assertTrue((target_step_dir / "commit_success.txt").exists())
    self.assertTrue((target_step_dir / "_CHECKPOINT_METADATA").exists())
    self.assertTrue((target_step_dir / "state" / "manifest.ocdbt").exists())
    self.assertTrue((target_step_dir / "state" / "_METADATA").exists())

    # Verify per-node subdirectories preserved for tensorstore chunk lookup
    self.assertTrue(
        (
            target_step_dir / "state" / "ocdbt.process_0" / "manifest.ocdbt"
        ).exists()
    )
    self.assertTrue(
        (
            target_step_dir / "state" / "ocdbt.process_1" / "manifest.ocdbt"
        ).exists()
    )

    # Standard Orbax CheckpointManager MUST be able to discover and restore
    mngr = checkpoint_manager.CheckpointManager(self.output_dir)
    self.assertIn(step, mngr.all_steps())
    self.assertEqual(mngr.latest_step(), step)

    # Restore item using standard PyTreeRestore (both 'state' and 'items' exist)
    restored = mngr.restore(step)
    self.assertIn("state", restored)
    self.assertIn("items", restored)
    restored_state = restored["state"]
    restored_items = restored["items"]

    # Verify arrays from node 0
    np.testing.assert_array_equal(
        restored_state["layer_0"], node0_tree["layer_0"]
    )
    np.testing.assert_array_equal(
        restored_state["bias_0"], node0_tree["bias_0"]
    )
    np.testing.assert_array_equal(
        restored_items["layer_0"], node0_tree["layer_0"]
    )

    # Verify arrays from node 1
    np.testing.assert_array_equal(
        restored_state["layer_1"], node1_tree["layer_1"]
    )
    np.testing.assert_array_equal(
        restored_state["bias_1"], node1_tree["bias_1"]
    )
    np.testing.assert_array_equal(
        restored_items["layer_1"], node1_tree["layer_1"]
    )

  def test_timestamp_backup_directory_structure(self):
    """Test converting from a top-level directory with timestamp subdirs."""
    ts_dir = self.backup_dir / "2026-09-09_18-00"
    ts_dir.mkdir(parents=True, exist_ok=True)
    step = 100

    node_hash = "Dhash100"
    data_dir = ts_dir / f"{node_hash}.data"
    state_dir = data_dir / "state"
    tree = {"param": np.array([42.0], dtype=np.float32)}
    self._save_node_item(state_dir, tree)

    (ts_dir / f"job-s{step}-n0-w0.meta").write_text(node_hash)

    # Pass the parent backup_dir (not the timestamp dir)
    converted = mtc_checkpoint_converter.convert_mtc_backup(
        input_dir=self.backup_dir,
        output_dir=self.output_dir,
    )
    self.assertLen(converted, 1)
    self.assertTrue(
        (self.output_dir / str(step) / "commit_success.txt").exists()
    )

    mngr = checkpoint_manager.CheckpointManager(self.output_dir)
    restored = mngr.restore(step)
    np.testing.assert_array_equal(restored["state"]["param"], tree["param"])

  def test_symlink_mode(self):
    """Test link (symlink) mode on local storage."""
    step = 200
    node_hash = "Dhash200"
    data_dir = self.backup_dir / f"{node_hash}.data"
    state_dir = data_dir / "state"
    tree = {"w": np.array([1.23, 4.56], dtype=np.float32)}
    self._save_node_item(state_dir, tree)
    (self.backup_dir / f"job-s{step}-n0-w0.meta").write_text(node_hash)

    converted = mtc_checkpoint_converter.convert_mtc_backup(
        input_dir=self.backup_dir,
        output_dir=self.output_dir,
    )
    self.assertLen(converted, 1)

    proc_dir = self.output_dir / str(step) / "state" / "ocdbt.process_0"
    self.assertTrue(os.path.islink(str(proc_dir)))

    mngr = checkpoint_manager.CheckpointManager(self.output_dir)
    restored = mngr.restore(step)
    np.testing.assert_array_equal(restored["state"]["w"], tree["w"])

  def test_get_default_output_dir(self):
    ts_path = epath.Path("gs://bucket/job/2026-09-09_18-00")
    self.assertEqual(
        mtc_checkpoint_converter.get_default_output_dir(ts_path),
        epath.Path("gs://bucket/job/standard_checkpoints"),
    )
    job_path = epath.Path("gs://bucket/job")
    self.assertEqual(
        mtc_checkpoint_converter.get_default_output_dir(job_path),
        epath.Path("gs://bucket/job/standard_checkpoints"),
    )

  def test_overwrite_flag_behavior(self):
    """Test that existing targets raise FileExistsError without overwrite."""
    step = 300
    node_hash = "Dhash300"
    data_dir = self.backup_dir / f"{node_hash}.data"
    state_dir = data_dir / "state"
    tree = {"v": np.array([1], dtype=np.int32)}
    self._save_node_item(state_dir, tree)
    (self.backup_dir / f"job-s{step}-n0-w0.meta").write_text(node_hash)

    # First conversion succeeds
    mtc_checkpoint_converter.convert_mtc_backup(
        input_dir=self.backup_dir,
        output_dir=self.output_dir,
    )

    # Second conversion without overwrite raises
    with self.assertRaises(FileExistsError):
      mtc_checkpoint_converter.convert_mtc_backup(
          input_dir=self.backup_dir,
          output_dir=self.output_dir,
          overwrite=False,
      )

    # Second conversion with overwrite=True succeeds
    converted = mtc_checkpoint_converter.convert_mtc_backup(
        input_dir=self.backup_dir,
        output_dir=self.output_dir,
        overwrite=True,
    )
    self.assertLen(converted, 1)

  def test_parse_gcs_path(self):
    self.assertEqual(
        mtc_checkpoint_converter._parse_gcs_path("gs://my-bucket/path/to/dir"),
        ("my-bucket", "path/to/dir"),
    )
    self.assertEqual(
        mtc_checkpoint_converter._parse_gcs_path("/gcs/my-bucket/dir/"),
        ("my-bucket", "dir"),
    )
    with self.assertRaises(ValueError):
      mtc_checkpoint_converter._parse_gcs_path("/local/path")

  def test_copy_dir_gcs_fast_path(self):
    mock_client = mock.MagicMock()
    mock_src_bucket = mock.MagicMock()
    mock_dst_bucket = mock.MagicMock()

    mock_client.bucket.side_effect = lambda name: (
        mock_src_bucket if name == "src-bucket" else mock_dst_bucket
    )

    blob1 = mock.MagicMock()
    blob1.name = "src_dir/file1.txt"
    blob2 = mock.MagicMock()
    blob2.name = "src_dir/sub/file2.txt"
    marker = mock.MagicMock()
    marker.name = "src_dir/sub_$folder$"

    mock_src_bucket.list_blobs.return_value = [blob1, blob2, marker]
    mock_dst_bucket.blob.return_value.exists.return_value = False

    with mock.patch.object(mtc_checkpoint_converter, "storage") as mock_storage:
      mock_storage.Client.return_value = mock_client
      mtc_checkpoint_converter._copy_dir_gcs(
          src=epath.Path("gs://src-bucket/src_dir"),
          dst=epath.Path("gs://dst-bucket/dst_dir"),
          overwrite=True,
          max_workers=4,
      )

    self.assertEqual(mock_src_bucket.copy_blob.call_count, 2)
    mock_src_bucket.copy_blob.assert_any_call(
        blob1, mock_dst_bucket, "dst_dir/file1.txt"
    )
    mock_src_bucket.copy_blob.assert_any_call(
        blob2, mock_dst_bucket, "dst_dir/sub/file2.txt"
    )

  def test_copy_file_gcs_fast_path(self):
    mock_client = mock.MagicMock()
    mock_src_bucket = mock.MagicMock()
    mock_dst_bucket = mock.MagicMock()

    mock_client.bucket.side_effect = lambda name: (
        mock_src_bucket if name == "src-bucket" else mock_dst_bucket
    )

    mock_src_blob = mock.MagicMock()
    mock_src_bucket.blob.return_value = mock_src_blob
    mock_dst_bucket.blob.return_value.exists.return_value = False

    with mock.patch.object(mtc_checkpoint_converter, "storage") as mock_storage:
      mock_storage.Client.return_value = mock_client
      mtc_checkpoint_converter._copy_file(
          src=epath.Path("gs://src-bucket/meta/_METADATA"),
          dst=epath.Path("gs://dst-bucket/step1/_METADATA"),
          overwrite=True,
      )

    mock_src_bucket.copy_blob.assert_called_once_with(
        mock_src_blob, mock_dst_bucket, "step1/_METADATA"
    )

  def test_copy_file_gcs_fails_fast(self):
    mock_client = mock.MagicMock()
    mock_src_bucket = mock.MagicMock()
    mock_dst_bucket = mock.MagicMock()
    mock_client.bucket.side_effect = lambda name: (
        mock_src_bucket if name == "src-bucket" else mock_dst_bucket
    )
    mock_src_blob = mock.MagicMock()
    mock_src_bucket.blob.return_value = mock_src_blob
    mock_dst_bucket.blob.return_value.exists.return_value = False
    mock_src_bucket.copy_blob.side_effect = RuntimeError("GCS network failure")

    with mock.patch.object(mtc_checkpoint_converter, "storage") as mock_storage:
      mock_storage.Client.return_value = mock_client
      with self.assertRaises(RuntimeError):
        mtc_checkpoint_converter._copy_file(
            src=epath.Path("gs://src-bucket/meta/_METADATA"),
            dst=epath.Path("gs://dst-bucket/step1/_METADATA"),
            overwrite=True,
        )

  def test_link_dir_cross_filesystem_raises(self):
    with self.assertRaises(ValueError):
      mtc_checkpoint_converter._link_dir(
          src=epath.Path("gs://src-bucket/dir"),
          dst=epath.Path("/tmp/local_dir"),
      )
    with self.assertRaises(ValueError):
      mtc_checkpoint_converter._link_dir(
          src=epath.Path("/tmp/local_dir"),
          dst=epath.Path("gs://dst-bucket/dir"),
      )

  def test_link_dir_gcs_fails_fast(self):
    with mock.patch.object(
        mtc_checkpoint_converter,
        "_copy_dir_gcs",
        side_effect=RuntimeError("GCS copy failed"),
    ):
      with self.assertRaises(RuntimeError):
        mtc_checkpoint_converter._link_dir(
            src=epath.Path("gs://src-bucket/dir"),
            dst=epath.Path("gs://dst-bucket/dir"),
        )


if __name__ == "__main__":
  absltest.main()
