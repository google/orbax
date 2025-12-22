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

"""Tests for path utils."""

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from orbax.checkpoint._src.path import utils


class UtilsTest(parameterized.TestCase):

  def test_recursively_copy_files(self):
    src_dir = epath.Path(self.create_tempdir().full_path)
    dst_dir = epath.Path(self.create_tempdir().full_path)

    # Create test files and directories
    (src_dir / 'a').mkdir()
    (src_dir / 'a' / 'file1.txt').write_text('content1')
    (src_dir / 'b').mkdir()
    (src_dir / 'b' / 'file2.txt').write_text('content2')
    (src_dir / 'file3.txt').write_text('content3')

    utils.recursively_copy_files(src_dir, dst_dir)

    self.assertTrue((dst_dir / 'a' / 'file1.txt').exists())
    self.assertEqual((dst_dir / 'a' / 'file1.txt').read_text(), 'content1')
    self.assertTrue((dst_dir / 'b' / 'file2.txt').exists())
    self.assertEqual((dst_dir / 'b' / 'file2.txt').read_text(), 'content2')
    self.assertTrue((dst_dir / 'file3.txt').exists())
    self.assertEqual((dst_dir / 'file3.txt').read_text(), 'content3')

  def test_recursively_copy_files_with_skip_paths(self):
    src_dir = epath.Path(self.create_tempdir().full_path)
    dst_dir = epath.Path(self.create_tempdir().full_path)

    # Create test files and directories
    (src_dir / 'a').mkdir()
    (src_dir / 'a' / 'file1.txt').write_text('content1')
    (src_dir / 'a' / 'nested').mkdir()
    (src_dir / 'a' / 'nested' / 'file.txt').write_text('nested content')
    (src_dir / 'b').mkdir()
    (src_dir / 'b' / 'file2.txt').write_text('content2')
    (src_dir / 'c').mkdir()
    (src_dir / 'c' / 'file3.txt').write_text('content3')
    (src_dir / 'file4.txt').write_text('content4')

    skip_paths = ['a', 'b/file2.txt']
    utils.recursively_copy_files(src_dir, dst_dir, skip_paths=skip_paths)

    self.assertFalse((dst_dir / 'a').exists())
    self.assertFalse((dst_dir / 'a' / 'nested').exists())
    self.assertFalse((dst_dir / 'b' / 'file2.txt').exists())
    self.assertTrue((dst_dir / 'c' / 'file3.txt').exists())
    self.assertEqual((dst_dir / 'c' / 'file3.txt').read_text(), 'content3')
    self.assertTrue((dst_dir / 'file4.txt').exists())
    self.assertEqual((dst_dir / 'file4.txt').read_text(), 'content4')

  @parameterized.parameters(
      ('gs://bucket/path', 'gcs'),
      ('s3://bucket/path', 's3'),
      ('/tmp/foo/bar', 'other'),
  )
  def test_get_storage_type(self, path, expected_type):
    self.assertEqual(utils.get_storage_type(path), expected_type)


if __name__ == '__main__':
  absltest.main()
