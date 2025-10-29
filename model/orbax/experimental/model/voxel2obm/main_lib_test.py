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

"""Tests for main_lib."""

import os
from unittest import mock

from absl.testing import absltest
from orbax.experimental.model import core as obm
from orbax.experimental.model.voxel2obm import main_lib
from orbax.experimental.model.voxel2obm import voxel_asset_map_pb2
from orbax.experimental.model.voxel2obm import voxel_mock


class MainLibTest(absltest.TestCase):

  def test_voxel_plan_to_obm_with_plan(self):
    # TODO(b/447200841): Replace with a real Voxel module.
    voxel_module = voxel_mock.VoxelModule()

    obm_fn = main_lib.voxel_plan_to_obm(
        voxel_module, input_signature={}, output_signature={}
    )
    self.assertEqual(obm_fn.body.proto.inlined_bytes, b"test plan")
    self.assertEqual(
        obm_fn.body.proto.mime_type, main_lib.VOXEL_PROCESSOR_MIME_TYPE
    )
    self.assertEqual(
        obm_fn.body.proto.version, main_lib.VOXEL_PROCESSOR_VERSION
    )

  def test_voxel_global_supplemental_closure(self):
    source_dir = self.create_tempdir("source")
    os.makedirs(os.path.join(source_dir, "d1"))
    os.makedirs(os.path.join(source_dir, "d2"))
    os.makedirs(os.path.join(source_dir, "d3"))
    path1 = os.path.join(source_dir, "d1", "foo.txt")
    path2 = os.path.join(source_dir, "d2", "foo.txt")
    path3 = os.path.join(source_dir, "d3", "foo.txt")
    path4 = os.path.join(source_dir, "d3", "bar.txt")
    path5 = os.path.join(source_dir, "d2", "foo_1.txt")
    path6 = os.path.join(source_dir, "d3", "foo_1.txt")
    path7 = os.path.join(source_dir, "d2", "foo_1_1.txt")
    with open(path1, "w") as f:
      f.write("foo")
    with open(path2, "w") as f:
      f.write("bar")
    with open(path3, "w") as f:
      f.write("baz")
    with open(path4, "w") as f:
      f.write("qux")
    with open(path5, "w") as f:
      f.write("quux")
    with open(path6, "w") as f:
      f.write("corge")
    with open(path7, "w") as f:
      f.write("grault")
    asset_source_paths = {path1, path2, path3, path4, path5, path6, path7}

    closure = main_lib.voxel_global_supplemental_closure(asset_source_paths)

    save_dir = self.create_tempdir("save")
    supplemental = closure(save_dir)

    expected_asset_map = voxel_asset_map_pb2.VoxelAssetsMap()
    expected_asset_map.assets[path1] = "voxel_module/assets/foo.txt"
    expected_asset_map.assets[path2] = "voxel_module/assets/foo_1.txt"
    expected_asset_map.assets[path5] = "voxel_module/assets/foo_2.txt"
    expected_asset_map.assets[path7] = "voxel_module/assets/foo_3.txt"
    expected_asset_map.assets[path4] = "voxel_module/assets/bar.txt"
    expected_asset_map.assets[path3] = "voxel_module/assets/foo_4.txt"
    expected_asset_map.assets[path6] = "voxel_module/assets/foo_5.txt"

    # Check supplemental data.
    self.assertIsInstance(supplemental, obm.GlobalSupplemental)
    self.assertEqual(supplemental.save_as, "voxel_asset_map.pb")
    self.assertEqual(
        supplemental.data.mime_type, main_lib.VOXEL_ASSET_MAP_MIME_TYPE
    )
    self.assertEqual(
        supplemental.data.version, main_lib.VOXEL_ASSET_MAP_VERSION
    )
    exported_asset_map = voxel_asset_map_pb2.VoxelAssetsMap.FromString(
        supplemental.data.inlined_bytes
    )
    self.assertEqual(exported_asset_map, expected_asset_map)

    # Check saved asset files.
    for source_path, rel_path in exported_asset_map.assets.items():
      dest_path = os.path.join(save_dir, rel_path)
      self.assertTrue(os.path.exists(dest_path))
      with open(source_path, "r") as f:
        source_content = f.read()
      with open(dest_path, "r") as f:
        dest_content = f.read()
      self.assertEqual(source_content, dest_content)


if __name__ == "__main__":
  absltest.main()
