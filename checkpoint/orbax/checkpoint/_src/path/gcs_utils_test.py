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

"""Tests for gcs_utils functions."""

from unittest import mock
from absl.testing import absltest
from etils import epath
from orbax.checkpoint._src.path import gcs_utils


class GcsUtilsTest(absltest.TestCase):

  def test_rmtree_non_gcs_path(self):
    local_path = epath.Path('/tmp/some/local/dir')
    with self.assertRaisesRegex(ValueError, 'Path is not a GCS path'):
      gcs_utils.rmtree(local_path)

  @mock.patch.object(gcs_utils, 'is_hierarchical_namespace_enabled')
  @mock.patch.object(gcs_utils, 'cleanup_hns_folders')
  def test_rmtree_gcs_hns_disabled(
      self, mock_cleanup_hns_folders, mock_is_hns_enabled
  ):
    mock_is_hns_enabled.return_value = False
    gcs_path = mock.MagicMock()
    gcs_path.__str__.return_value = 'gs://bucket/dir'

    with mock.patch.object(gcs_utils, 'is_gcs_path', return_value=True):
      gcs_utils.rmtree(gcs_path)

    gcs_path.rmtree.assert_called_once()
    mock_cleanup_hns_folders.assert_not_called()

  @mock.patch.object(gcs_utils, 'is_hierarchical_namespace_enabled')
  @mock.patch.object(gcs_utils, 'cleanup_hns_folders')
  def test_rmtree_gcs_hns_enabled(
      self, mock_cleanup_hns_folders, mock_is_hns_enabled
  ):
    mock_is_hns_enabled.return_value = True
    gcs_path = mock.MagicMock(spec=epath.Path)
    gcs_path.__str__.return_value = 'gs://bucket/dir'

    with mock.patch.object(gcs_utils, 'is_gcs_path', return_value=True):
      gcs_utils.rmtree(gcs_path)

    gcs_path.rmtree.assert_called_once()
    mock_cleanup_hns_folders.assert_called_once_with(gcs_path)

  def test_cleanup_hns_folders(self):
    mock_storage_control_v2 = mock.MagicMock()
    mock_client_cls = mock_storage_control_v2.StorageControlClient
    mock_client = mock.MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.common_project_path.return_value = 'projects/_'

    # Setup a mock folder structure that list_folders returns
    # We want nested folders to ensure leaves are deleted before parents.
    # Structure:
    # /dir/
    #   |-- a/
    #   |   |-- b/
    #   |-- c/
    mock_folder_dir = mock.MagicMock()
    mock_folder_dir.name = 'projects/_/buckets/my-bucket/folders/dir/'
    mock_folder_a = mock.MagicMock()
    mock_folder_a.name = 'projects/_/buckets/my-bucket/folders/dir/a/'
    mock_folder_b = mock.MagicMock()
    mock_folder_b.name = 'projects/_/buckets/my-bucket/folders/dir/a/b/'
    mock_folder_c = mock.MagicMock()
    mock_folder_c.name = 'projects/_/buckets/my-bucket/folders/dir/c/'

    mock_client.list_folders.return_value = [
        mock_folder_dir,
        mock_folder_a,
        mock_folder_b,
        mock_folder_c,
    ]

    gcs_path = 'gs://my-bucket/dir'
    with mock.patch.dict(
        'sys.modules',
        {'google.cloud.storage_control_v2': mock_storage_control_v2},
    ):
      gcs_utils.cleanup_hns_folders(gcs_path)  # pytype: disable=wrong-arg-types

    # list_folders should have been called the right prefix
    mock_client.list_folders.assert_called_once()
    mock_storage_control_v2.ListFoldersRequest.assert_called_once_with(
        parent='projects/_/buckets/my-bucket', prefix='dir/'
    )

    # Verify the delete calls were made
    # Since it's a while loop processing leaves, it might take a few passes.
    # We just need to ensure all 4 folders were requested to be deleted.
    self.assertEqual(mock_storage_control_v2.DeleteFolderRequest.call_count, 4)
    delete_calls = mock_storage_control_v2.DeleteFolderRequest.mock_calls
    deleted_names = [call.kwargs['name'] for call in delete_calls]
    self.assertIn(
        'projects/_/buckets/my-bucket/folders/dir/a/b/', deleted_names
    )
    self.assertIn('projects/_/buckets/my-bucket/folders/dir/c/', deleted_names)
    self.assertIn('projects/_/buckets/my-bucket/folders/dir/a/', deleted_names)
    self.assertIn('projects/_/buckets/my-bucket/folders/dir/', deleted_names)


if __name__ == '__main__':
  absltest.main()
