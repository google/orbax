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

from absl.testing import parameterized
from etils import epath
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.testing import multiprocess_test
from orbax.checkpoint.testing import local_path

LocalPath = local_path.LocalPath


class LocalPathTest(parameterized.TestCase, multiprocess_test.MultiProcessTest):

  def setUp(self):
    super().setUp()
    self.invalid_path = self.multiprocess_create_tempdir(name="foo")
    self.base_path = local_path.create_local_path_base(self)
    self.assertGreater(multihost.process_count(), 1)
    test_utils.sync_global_processes("LocalPathTest:setup_complete")

  def tearDown(self):
    test_utils.sync_global_processes("LocalPathTest:tests_complete")
    super().tearDown()

  def assertPathEqual(self, p1, p2):
    self.assertEqual(p1, p2, f"{p1} != {p2}")

  def assertPathExists(self, p):
    self.assertTrue(p.exists(), f"{p} does not exist.")

  @parameterized.product(input_cls=[epath.Path, str])
  def test_construction(self, input_cls):
    p = LocalPath(input_cls(self.base_path))
    self.assertPathEqual(
        p.path,
        epath.Path(self.base_path) / f"local_{multihost.process_index()}",
    )
    p = epath.Path(p)
    self.assertPathEqual(
        p, epath.Path(self.base_path) / f"local_{multihost.process_index()}"
    )

  def test_mkdir(self):
    base_path = epath.Path(self.base_path)
    p = LocalPath(base_path)
    p.mkdir(parents=False, exist_ok=False)
    self.assertPathExists(base_path / f"local_{multihost.process_index()}")

  def test_join(self):
    base_path = epath.Path(self.base_path)
    p = LocalPath(base_path)
    p.mkdir(parents=False, exist_ok=False)
    self.assertPathExists(base_path / f"local_{multihost.process_index()}")
    p /= "foobar"
    p.mkdir(parents=False, exist_ok=False)
    self.assertPathExists(
        base_path / f"local_{multihost.process_index()}" / "foobar"
    )

  def test_invalid_path(self):
    p = LocalPath(self.invalid_path)
    with self.assertRaisesRegex(
        ValueError, f"must contain {local_path._LOCAL_PATH_BASE_NAME}"
    ):
      p.exists()


if __name__ == "__main__":
  multiprocess_test.main()
