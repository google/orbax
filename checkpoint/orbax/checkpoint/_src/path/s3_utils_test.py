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

"""Tests for s3_utils."""

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from orbax.checkpoint._src.path import s3_utils


class S3UtilsTest(parameterized.TestCase):

  @parameterized.parameters(
      ('s3://bucket/path', True),
      ('s3://bucket/path/', True),
      ('s3://bucket', True),
      ('/local/path', False),
      ('gs://bucket/path', False),
      ('/tmp/foo/bar', False),
      ('file:///path', False),
  )
  def test_is_s3_path(self, path_str, expected):
    path = epath.Path(path_str)
    self.assertEqual(s3_utils.is_s3_path(path), expected)

  @parameterized.parameters(
      ('s3://bucket/path/to/file', ('bucket', 'path/to/file/')),
      ('s3://bucket/path/', ('bucket', 'path/')),
      ('s3://bucket/path', ('bucket', 'path/')),
      ('s3://bucket/', ('bucket', '/')),
      ('s3://bucket', ('bucket', '/')),
      ('s3://my-bucket/deep/nested/path/', ('my-bucket', 'deep/nested/path/')),
  )
  def test_parse_s3_path(self, path_str, expected):
    bucket, key = s3_utils.parse_s3_path(path_str)
    self.assertEqual((bucket, key), expected)

  def test_parse_s3_path_strips_leading_slash(self):
    """Test that leading slashes are properly stripped from the path."""
    bucket, key = s3_utils.parse_s3_path('s3://bucket//path')
    self.assertEqual(bucket, 'bucket')
    self.assertEqual(key, '/path/')

  def test_parse_s3_path_adds_trailing_slash(self):
    """Test that trailing slashes are added when missing."""
    bucket, key = s3_utils.parse_s3_path('s3://bucket/path')
    self.assertTrue(key.endswith('/'))
    self.assertEqual(key, 'path/')

  def test_parse_s3_path_invalid_scheme(self):
    """Test that non-s3 schemes raise an assertion error."""
    with self.assertRaises(AssertionError):
      s3_utils.parse_s3_path('gs://bucket/path')

  def test_parse_s3_path_no_scheme(self):
    """Test that paths without schemes raise an assertion error."""
    with self.assertRaises(AssertionError):
      s3_utils.parse_s3_path('/local/path')


if __name__ == '__main__':
  absltest.main()
