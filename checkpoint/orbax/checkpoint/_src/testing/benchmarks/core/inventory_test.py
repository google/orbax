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

"""Tests for the checkpoint-directory inventory walker."""

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from orbax.checkpoint._src.testing.benchmarks.core import inventory


def _write(path: epath.Path, size: int) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_bytes(b'x' * size)


class ScanCheckpointTest(parameterized.TestCase):

  def test_empty_dir_reports_zero(self):
    d = epath.Path(self.create_tempdir().full_path)
    inv = inventory.scan_checkpoint(d)
    self.assertEqual(inv.file_count, 0)
    self.assertEqual(inv.total_bytes, 0)
    self.assertEqual(inv.small_file_count, 0)
    self.assertEqual(inv.small_file_pct, 0.0)

  def test_counts_files_and_bytes_recursively(self):
    d = epath.Path(self.create_tempdir().full_path)
    _write(d / 'a.bin', 4096)
    _write(d / 'sub' / 'b.bin', 8192)
    _write(d / 'sub' / 'deep' / 'c.bin', 1024)
    inv = inventory.scan_checkpoint(d)
    self.assertEqual(inv.file_count, 3)
    self.assertEqual(inv.total_bytes, 4096 + 8192 + 1024)

  def test_small_file_pct_is_fraction_under_1mib(self):
    d = epath.Path(self.create_tempdir().full_path)
    _write(d / 'big1.bin', 2 * 1024 * 1024)
    _write(d / 'big2.bin', 2 * 1024 * 1024)
    _write(d / 'small1.bin', 100)
    _write(d / 'small2.bin', 100)
    inv = inventory.scan_checkpoint(d)
    self.assertEqual(inv.file_count, 4)
    self.assertEqual(inv.small_file_count, 2)
    self.assertAlmostEqual(inv.small_file_pct, 0.5)

  def test_format_breakdown_counts_ocdbt_zarr_metadata(self):
    d = epath.Path(self.create_tempdir().full_path)
    _write(d / 'state' / 'ocdbt.process_0', 1024)
    _write(d / 'state' / 'ocdbt.process_1', 1024)
    _write(d / 'state' / 'manifest.ocdbt', 512)
    _write(d / 'state' / 'd' / 'leaf_chunk', 256)
    _write(d / 'state' / '_METADATA', 64)
    _write(d / 'state' / '_sharding', 32)
    _write(d / '_CHECKPOINT_METADATA', 16)
    inv = inventory.scan_checkpoint(d)
    self.assertEqual(
        inv.format['ocdbt'], 3
    )  # ocdbt.process_* and manifest.ocdbt
    self.assertEqual(
        inv.format['metadata'], 3
    )  # _METADATA + _sharding + _CHECKPOINT_METADATA
    # Remaining d/leaf_chunk counts as 'other'.
    self.assertEqual(inv.format.get('other', 0), 1)

  def test_largest_smallest_bytes_tracked(self):
    d = epath.Path(self.create_tempdir().full_path)
    _write(d / 'tiny.bin', 10)
    _write(d / 'huge.bin', 10_000_000)
    _write(d / 'mid.bin', 5_000)
    inv = inventory.scan_checkpoint(d)
    self.assertEqual(inv.largest_file_bytes, 10_000_000)
    self.assertEqual(inv.smallest_file_bytes, 10)

  def test_missing_dir_returns_empty_inventory(self):
    inv = inventory.scan_checkpoint(epath.Path('/no/such/path'))
    self.assertEqual(inv.file_count, 0)
    self.assertEqual(inv.total_bytes, 0)


if __name__ == '__main__':
  absltest.main()
