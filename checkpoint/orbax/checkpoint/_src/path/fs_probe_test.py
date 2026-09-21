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

import unittest

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from orbax.checkpoint._src.path import fs_probe


class DirectoryIndexTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  def setUp(self):
    super().setUp()
    self.root = epath.Path(self.create_tempdir().full_path)

  def _touch(self, *names: str) -> None:
    for name in names:
      (self.root / name).write_text("")

  async def test_lists_immediate_children_only(self):
    self._touch("a.txt", "b.txt")
    nested = self.root / "sub"
    nested.mkdir()
    (nested / "hidden.txt").write_text("")

    index = await fs_probe.index_directory(self.root)

    self.assertEqual(index.path(), self.root)
    self.assertTrue(index.exists())
    self.assertTrue(index.is_directory())
    self.assertTrue(index.listable())
    self.assertEqual(index.names(), frozenset({"a.txt", "b.txt", "sub"}))
    self.assertNotIn("hidden.txt", index.names())

  async def test_missing_directory(self):
    index = await fs_probe.index_directory(self.root / "nope")

    self.assertFalse(index.exists())
    self.assertFalse(index.is_directory())
    self.assertFalse(index.listable())
    self.assertEmpty(index.names())
    self.assertFalse(index.has("anything"))

  async def test_file_as_directory(self):
    file_path = self.root / "a_file.txt"
    file_path.write_text("content")
    index = await fs_probe.index_directory(file_path)

    self.assertTrue(index.exists())
    self.assertFalse(index.is_directory())
    self.assertFalse(index.listable())
    self.assertEmpty(index.names())

  async def test_has_and_has_any(self):
    self._touch("manifest.ocdbt", "_METADATA")
    index = await fs_probe.index_directory(self.root)

    self.assertTrue(index.has("manifest.ocdbt"))
    self.assertFalse(index.has("manifest"))
    self.assertTrue(index.has_any("missing", "_METADATA"))
    self.assertFalse(index.has_any("missing", "also_missing"))

  async def test_matching_returns_sorted_prefix_hits(self):
    self._touch(
        "ocdbt.process_1",
        "ocdbt.process_0",
        "manifest.ocdbt",
    )
    index = await fs_probe.index_directory(self.root)

    self.assertEqual(
        index.matching("ocdbt.process_"),
        ["ocdbt.process_0", "ocdbt.process_1"],
    )
    self.assertEmpty(index.matching("no_such_prefix"))

  async def test_matching_accepts_several_prefixes(self):
    self._touch("a_one", "b_two", "c_three")
    index = await fs_probe.index_directory(self.root)

    self.assertEqual(index.matching("a_", "c_"), ["a_one", "c_three"])

  async def test_with_suffix(self):
    self._touch("model.safetensors", "other.safetensors", "notes.txt")
    index = await fs_probe.index_directory(self.root)

    self.assertEqual(
        index.with_suffix(".safetensors"),
        ["model.safetensors", "other.safetensors"],
    )

  async def test_present_preserves_candidate_order(self):
    self._touch("second", "first")
    index = await fs_probe.index_directory(self.root)

    self.assertEqual(
        index.present(("first", "absent", "second")), ["first", "second"]
    )


class BatchProbeTest(parameterized.TestCase, unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    super().setUp()
    self.root = epath.Path(self.create_tempdir().full_path)

  async def test_index_directories_is_positionally_aligned(self):
    (self.root / "a").mkdir()
    (self.root / "a" / "x").write_text("")
    (self.root / "b").mkdir()

    indexes = await fs_probe.index_directories(
        (self.root / "a", self.root / "missing", self.root / "b")
    )

    self.assertLen(indexes, 3)
    self.assertEqual(indexes[0].names(), frozenset({"x"}))
    self.assertTrue(indexes[0].exists())
    self.assertTrue(indexes[0].is_directory())

    self.assertFalse(indexes[1].exists())
    self.assertFalse(indexes[1].is_directory())
    self.assertFalse(indexes[1].listable())

    self.assertTrue(indexes[2].exists())
    self.assertTrue(indexes[2].is_directory())
    self.assertTrue(indexes[2].listable())
    self.assertEmpty(indexes[2].names())

  async def test_exists_many(self):
    (self.root / "here").write_text("")

    self.assertEqual(
        await fs_probe.exists_many((self.root / "here", self.root / "gone")),
        (True, False),
    )

  async def test_is_dir_many(self):
    (self.root / "dir").mkdir()
    (self.root / "file").write_text("")

    self.assertEqual(
        await fs_probe.is_dir_many(
            (self.root / "dir", self.root / "file", self.root / "gone")
        ),
        (True, False, False),
    )

  async def test_empty_input_does_no_work(self):
    self.assertEmpty(await fs_probe.index_directories(()))
    self.assertEmpty(await fs_probe.exists_many(()))
    self.assertEmpty(await fs_probe.is_dir_many(()))


if __name__ == "__main__":
  absltest.main()
