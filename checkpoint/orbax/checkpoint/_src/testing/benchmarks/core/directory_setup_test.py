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

from unittest import mock

from absl.testing import absltest
from etils import epath
import jax
from orbax.checkpoint._src.testing.benchmarks.core import directory_setup


class DirectorySetupTest(absltest.TestCase):

  def test_setup_test_directory_default_path(self):
    path = directory_setup.setup_test_directory('my_test')

    self.assertTrue(path.exists())

  def test_setup_test_directory_custom_path(self):
    temp_dir = self.create_tempdir()

    path = directory_setup.setup_test_directory(
        'my_test', base_path=temp_dir.full_path
    )

    self.assertEqual(path, epath.Path(temp_dir.full_path) / 'my_test')
    self.assertTrue(path.exists())

  def test_setup_test_directory_already_exists(self):
    temp_dir = self.create_tempdir()
    existing_path = epath.Path(temp_dir.full_path) / 'my_test'
    existing_path.mkdir()
    (existing_path / 'some_file').touch()

    path = directory_setup.setup_test_directory(
        'my_test', base_path=temp_dir.full_path
    )

    self.assertEqual(path, existing_path)
    self.assertTrue(path.exists())
    self.assertFalse((path / 'some_file').exists())

  @mock.patch.object(jax, 'process_index', return_value=1)
  def test_setup_test_directory_non_zero_process_index_does_not_exist(self, _):
    temp_dir = self.create_tempdir()

    path = directory_setup.setup_test_directory(
        'my_test', base_path=temp_dir.full_path
    )

    self.assertEqual(path, epath.Path(temp_dir.full_path) / 'my_test')
    self.assertFalse(path.exists())


if __name__ == '__main__':
  absltest.main()
