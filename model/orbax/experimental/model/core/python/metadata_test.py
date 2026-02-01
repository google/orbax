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
from absl.testing import absltest
from orbax.experimental.model.core.python import file_utils
from orbax.experimental.model.core.python import metadata


class MetadataTest(absltest.TestCase):

  def test_save_and_load(self):
    tempdir = self.create_tempdir().full_path
    path = os.path.join(tempdir, 'orbax_model_version.txt')
    mv = metadata.ModelVersion(
        version='1', mime_type='test_mime_type', manifest_file_path='test/path'
    )
    mv.save(path)
    mv_loaded = metadata.ModelVersion.load(path)
    self.assertEqual(mv_loaded, mv)

  def test_load_fails_with_unknown_fields(self):
    tempdir = self.create_tempdir().full_path
    path = os.path.join(tempdir, 'orbax_model_version.txt')
    file_content = """
    manifest_file_path: "test/path"
    version: "0.0.1"
    mime_type: "test_mime_type; application/foo"
    unknown_field: unknown_value
    """
    with file_utils.open_file(path, 'w') as f:
      f.write(file_content)

    with self.assertRaises(ValueError):
      metadata.ModelVersion.load(path)

  def test_load_fails_with_single_quoted_values(self):
    tempdir = self.create_tempdir().full_path
    path = os.path.join(tempdir, 'orbax_model_version.txt')
    file_content = """
    manifest_file_path: "test/path"
    version: '0.0.1'
    mime_type: "test_mime_type; application/foo"
    """
    with file_utils.open_file(path, 'w') as f:
      f.write(file_content)

    with self.assertRaises(ValueError):
      metadata.ModelVersion.load(path)

  def test_load_fails_with_malformed_file(self):
    tempdir = self.create_tempdir().full_path
    path = os.path.join(tempdir, 'orbax_model_version.txt')
    file_content = """
    manifest_file_path: "test/path"
    malformed_line_no_separator
    """
    with file_utils.open_file(path, 'w') as f:
      f.write(file_content)

    with self.assertRaises(ValueError):
      metadata.ModelVersion.load(path)

  def test_missing_version(self):
    tempdir = self.create_tempdir().full_path
    path = os.path.join(tempdir, 'orbax_model_version.txt')
    file_content = """
    manifest_file_path: "test/path"
    mime_type: "test_mime_type; application/foo"
    """
    with file_utils.open_file(path, 'w') as f:
      f.write(file_content)

    with self.assertRaisesRegex(ValueError, 'Version is empty'):
      metadata.ModelVersion.load(path)

  def test_missing_mime_type(self):
    tempdir = self.create_tempdir().full_path
    path = os.path.join(tempdir, 'orbax_model_version.txt')
    file_content = """
    version: "0.0.1"
    manifest_file_path: "test/path"
    """
    with file_utils.open_file(path, 'w') as f:
      f.write(file_content)

    with self.assertRaisesRegex(ValueError, 'MIME type is empty'):
      metadata.ModelVersion.load(path)

  def test_missing_manifest_file_path(self):
    tempdir = self.create_tempdir().full_path
    path = os.path.join(tempdir, 'orbax_model_version.txt')
    file_content = """
    version: "0.0.1"
    mime_type: "test_mime_type; application/foo"
    """
    with file_utils.open_file(path, 'w') as f:
      f.write(file_content)

    with self.assertRaisesRegex(ValueError, 'Manifest file path is empty'):
      metadata.ModelVersion.load(path)


if __name__ == '__main__':
  absltest.main()
