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

import os
from unittest import mock
from absl.testing import absltest
import jaxtyping
import numpy as np
from orbax.experimental.model import core as obm
from orbax.experimental.model import jd2obm


class TestJDModule(jd2obm.JDModuleBase):

  def get_output_signature(
      self, input_signature: jaxtyping.PyTree
  ) -> jaxtyping.PyTree:
    return {
        "output": {
            "feature1": jd2obm.JDSpecBase(shape=(1, 1), dtype=np.float32),
            "feature2": jd2obm.JDSpecBase(shape=(2, 2), dtype=np.int32),
        }
    }

  def export_plan(self):
    plan_proto = mock.Mock()
    plan_proto.SerializeToString.return_value = b"test plan"
    return plan_proto


class MainLibTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.module = TestJDModule()

  def test_jd_plan_to_obm_with_plan(self):
    obm_fn = jd2obm.jd_plan_to_obm(
        self.module, input_signature={}, output_signature={}
    )
    self.assertEqual(obm_fn.body.proto.inlined_bytes, b"test plan")
    self.assertEqual(obm_fn.body.proto.mime_type, jd2obm.JD_PROCESSOR_MIME_TYPE)
    self.assertEqual(obm_fn.body.proto.version, jd2obm.JD_PROCESSOR_VERSION)

  def test_jd_module_global_supplemental_closure(self):
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

    asset_paths = {path1, path2, path3, path4, path5, path6, path7}
    self.module.set_assets(asset_paths)
    closure = jd2obm.jd_global_supplemental_closure(self.module)

    save_dir = self.create_tempdir("save")
    supplemental_map = closure(save_dir)
    self.assertIn(jd2obm.JD_ASSET_MAP_SUPPLEMENTAL_NAME, supplemental_map)
    supplemental = supplemental_map[jd2obm.JD_ASSET_MAP_SUPPLEMENTAL_NAME]

    expected_asset_map = jd2obm.jd_asset_map_pb2.JDAssetsMap()
    expected_asset_map.assets[path1] = "jd_module/assets/foo.txt"
    expected_asset_map.assets[path2] = "jd_module/assets/foo_1.txt"
    expected_asset_map.assets[path5] = "jd_module/assets/foo_2.txt"
    expected_asset_map.assets[path7] = "jd_module/assets/foo_3.txt"
    expected_asset_map.assets[path4] = "jd_module/assets/bar.txt"
    expected_asset_map.assets[path3] = "jd_module/assets/foo_4.txt"
    expected_asset_map.assets[path6] = "jd_module/assets/foo_5.txt"

    # Check supplemental data.
    self.assertIsInstance(supplemental, obm.GlobalSupplemental)
    self.assertEqual(supplemental.save_as, "jd_asset_map.pb")
    self.assertEqual(
        supplemental.data.mime_type, jd2obm.JD_ASSET_MAP_MIME_TYPE
    )
    self.assertEqual(
        supplemental.data.version, jd2obm.JD_ASSET_MAP_VERSION
    )
    exported_asset_map = jd2obm.jd_asset_map_pb2.JDAssetsMap.FromString(
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

  def test_jd_global_supplemental_closure_with_no_assets(self):
    self.module.set_assets(set())
    closure = jd2obm.jd_global_supplemental_closure(self.module)
    self.assertIsNone(closure)


if __name__ == "__main__":
  absltest.main()
