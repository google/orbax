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

from absl.testing import parameterized
from etils import epath
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.testing import multiprocess_test
from orbax.checkpoint.testing import local_path

LocalPath = local_path.LocalPath


class LocalPathTest(parameterized.TestCase, multiprocess_test.MultiProcessTest):

  def setUp(self):
    super().setUp()
    self.path = self.create_tempdir().full_path
    self.assertGreater(multihost.process_count(), 1)

  def assertPathEqual(self, p1, p2):
    self.assertEqual(p1, p2, f"{p1} != {p2}")

  @parameterized.product(input_cls=[epath.Path, str])
  def test_construction(self, input_cls):
    p = LocalPath(input_cls(self.path))
    self.assertPathEqual(
        p.path, epath.Path(self.path) / f"local_{multihost.process_index()}"
    )
    p = epath.Path(p)
    self.assertPathEqual(
        p, epath.Path(self.path) / f"local_{multihost.process_index()}"
    )

  def test_mkdir(self):
    base_path = epath.Path(self.path)
    p = LocalPath(base_path)
    p.mkdir()
    self.assertTrue((base_path / f"local_{multihost.process_index()}").exists())


if __name__ == "__main__":
  multiprocess_test.main()
