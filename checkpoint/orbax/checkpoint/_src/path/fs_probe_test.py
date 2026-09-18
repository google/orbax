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

"""Unit tests for fs_probe."""

from __future__ import annotations

import concurrent.futures

from absl.testing import absltest
from etils import epath
from orbax.checkpoint._src.path import fs_probe


class DirectoryIndexTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.root = epath.Path(self.create_tempdir().full_path)

  def _touch(self, *names: str) -> None:
    for name in names:
      (self.root / name).write_text("")

  def test_lists_immediate_children_only(self):
    self._touch("a.txt", "b.txt")
    nested = self.root / "sub"
    nested.mkdir()
    (nested / "hidden.txt").write_text("")

    index = fs_probe.index_directory(self.root)

    self.assertTrue(index.listable)
    self.assertEqual(index.names, frozenset({"a.txt", "b.txt", "sub"}))
    self.assertNotIn("hidden.txt", index.names)

  def test_missing_directory_is_not_listable(self):
    index = fs_probe.index_directory(self.root / "nope")

    self.assertFalse(index.listable)
    self.assertEmpty(index.names)
    self.assertFalse(index.has("anything"))

  def test_has_and_has_any(self):
    self._touch("manifest.ocdbt", "_METADATA")
    index = fs_probe.index_directory(self.root)

    self.assertTrue(index.has("manifest.ocdbt"))
    self.assertFalse(index.has("manifest"))
    self.assertTrue(index.has_any("missing", "_METADATA"))
    self.assertFalse(index.has_any("missing", "also_missing"))

  def test_matching_returns_sorted_prefix_hits(self):
    self._touch(
        "ocdbt.process_1",
        "ocdbt.process_0",
        "manifest.ocdbt",
    )
    index = fs_probe.index_directory(self.root)

    self.assertEqual(
        index.matching("ocdbt.process_"),
        ["ocdbt.process_0", "ocdbt.process_1"],
    )
    self.assertEmpty(index.matching("no_such_prefix"))

  def test_matching_accepts_several_prefixes(self):
    self._touch("a_one", "b_two", "c_three")
    index = fs_probe.index_directory(self.root)

    self.assertEqual(index.matching("a_", "c_"), ["a_one", "c_three"])

  def test_with_suffix(self):
    self._touch("model.safetensors", "other.safetensors", "notes.txt")
    index = fs_probe.index_directory(self.root)

    self.assertEqual(
        index.with_suffix(".safetensors"),
        ["model.safetensors", "other.safetensors"],
    )

  def test_present_preserves_candidate_order(self):
    self._touch("second", "first")
    index = fs_probe.index_directory(self.root)

    self.assertEqual(
        index.present(("first", "absent", "second")), ["first", "second"]
    )


class BatchProbeTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.root = epath.Path(self.create_tempdir().full_path)

  def test_index_directories_is_positionally_aligned(self):
    (self.root / "a").mkdir()
    (self.root / "a" / "x").write_text("")
    (self.root / "b").mkdir()

    indexes = fs_probe.index_directories(
        (self.root / "a", self.root / "missing", self.root / "b")
    )

    self.assertLen(indexes, 3)
    self.assertEqual(indexes[0].names, frozenset({"x"}))
    self.assertFalse(indexes[1].listable)
    self.assertTrue(indexes[2].listable)
    self.assertEmpty(indexes[2].names)

  def test_exists_many(self):
    (self.root / "here").write_text("")

    self.assertEqual(
        fs_probe.exists_many((self.root / "here", self.root / "gone")),
        (True, False),
    )

  def test_is_dir_many(self):
    (self.root / "dir").mkdir()
    (self.root / "file").write_text("")

    self.assertEqual(
        fs_probe.is_dir_many(
            (self.root / "dir", self.root / "file", self.root / "gone")
        ),
        (True, False, False),
    )

  def test_empty_input_does_no_work(self):
    self.assertEmpty(fs_probe.index_directories(()))
    self.assertEmpty(fs_probe.exists_many(()))
    self.assertEmpty(fs_probe.is_dir_many(()))

  def test_batch_probes_with_custom_pool(self):
    (self.root / "d").mkdir()
    (self.root / "f").write_text("")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
      indexes = fs_probe.index_directories((self.root / "d",), pool=pool)
      exists = fs_probe.exists_many((self.root / "f",), pool=pool)
      is_dir = fs_probe.is_dir_many((self.root / "d",), pool=pool)
    self.assertLen(indexes, 1)
    self.assertTrue(indexes[0].listable)
    self.assertEqual(exists, (True,))
    self.assertEqual(is_dir, (True,))


if __name__ == "__main__":
  absltest.main()
