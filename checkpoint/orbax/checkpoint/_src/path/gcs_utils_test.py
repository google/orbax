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

from absl.testing import absltest
from absl.testing import parameterized
from orbax.checkpoint._src.path import gcs_utils


class GcsUtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='basic',
          path='gs://my-bucket/path/to/object',
          expected=('my-bucket', 'path/to/object'),
      ),
      dict(
          testcase_name='no_trailing_slash',
          path='gs://my-bucket/path/to/object/',
          expected=('my-bucket', 'path/to/object/'),
      ),
      dict(
          testcase_name='only_bucket',
          path='gs://my-bucket/',
          expected=('my-bucket', ''),
      ),
  )
  def test_parse_gcs_path_no_trailing_slash(self, path, expected):
    bucket, path_in_bucket = gcs_utils.parse_gcs_path(
        path, add_trailing_slash=False
    )
    self.assertEqual(bucket, expected[0])
    self.assertEqual(path_in_bucket, expected[1])

  @parameterized.named_parameters(
      dict(
          testcase_name='basic',
          path='gs://my-bucket/path/to/object',
          expected=('my-bucket', 'path/to/object/'),
      ),
      dict(
          testcase_name='with_trailing_slash',
          path='gs://my-bucket/path/to/object/',
          expected=('my-bucket', 'path/to/object/'),
      ),
      dict(
          testcase_name='only_bucket',
          path='gs://my-bucket',
          expected=('my-bucket', '/'),
      ),
  )
  def test_parse_gcs_path_add_trailing_slash(self, path, expected):
    bucket, path_in_bucket = gcs_utils.parse_gcs_path(
        path, add_trailing_slash=True
    )
    self.assertEqual(bucket, expected[0])
    self.assertEqual(path_in_bucket, expected[1])

  def test_parse_gcs_path_invalid_scheme(self):
    with self.assertRaisesRegex(AssertionError, 'Unsupported scheme for GCS'):
      gcs_utils.parse_gcs_path('file://my-folder/path/to/object')

  @parameterized.named_parameters(
      dict(
          testcase_name='basic',
          path='gs://my-bucket/path/to/object',
          expected={
              'driver': 'gcs',
              'bucket': 'my-bucket',
              'path': 'path/to/object',
          },
      ),
      dict(
          testcase_name='no_path',
          path='gs://my-bucket',
          expected={'driver': 'gcs', 'bucket': 'my-bucket'},
      ),
  )
  def test_get_kvstore_for_gcs(self, path, expected):
    spec = gcs_utils.get_kvstore_for_gcs(path)
    self.assertEqual(spec, expected)


if __name__ == '__main__':
  absltest.main()
