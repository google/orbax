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

import os
from absl.testing import absltest
from orbax.experimental.model.core.python import file_utils
from orbax.experimental.model.core.python import metadata


class MetadataTest(absltest.TestCase):

  def test_save_and_load_model_version(self):
    tempdir = self.create_tempdir().full_path
    path = os.path.join(tempdir, 'model_version.txt')
    mv = metadata.ModelVersion(
        version='1', mime_type='test_mime_type', manifest_file_path='test/path'
    )
    mv.save(path)
    mv_loaded = metadata.ModelVersion.load(path)
    self.assertEqual(mv_loaded, mv)

  def test_load_with_comments_and_unknown_fields(self):
    tempdir = self.create_tempdir().full_path
    path = os.path.join(tempdir, 'model_version.txt')
    file_content = """
    # This is a comment.
    manifest_file_path: "test/path"
    version: '0.0.1'
    mime_type: test_mime_type; application/foo
    unknown_field: "some_value"
    """
    with file_utils.open_file(path, 'w') as f:
      f.write(file_content)

    mv_loaded = metadata.ModelVersion.load(path)
    mv_expected = metadata.ModelVersion(
        version='0.0.1',
        mime_type='test_mime_type; application/foo',
        manifest_file_path='test/path',
    )
    self.assertEqual(mv_loaded, mv_expected)

  def test_load_with_malformed_file(self):
    tempdir = self.create_tempdir().full_path
    path = os.path.join(tempdir, 'model_version.txt')
    file_content = """
    manifest_file_path: "test/path"
    malformed_line_no_separator
    """
    with file_utils.open_file(path, 'w') as f:
      f.write(file_content)

    with self.assertRaises(ValueError):
      metadata.ModelVersion.load(path)


if __name__ == '__main__':
  absltest.main()
