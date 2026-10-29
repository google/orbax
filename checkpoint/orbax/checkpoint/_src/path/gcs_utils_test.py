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

import os
import types
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from google.api_core import exceptions as api_exceptions
from google.auth import exceptions as auth_exceptions
from orbax.checkpoint._src.path import gcs_utils


def _layout(
    location_type: str = 'zone', hns: bool = False
) -> types.SimpleNamespace:
  return types.SimpleNamespace(
      location_type=location_type,
      hierarchical_namespace=types.SimpleNamespace(enabled=hns),
  )


def _bucket(
    location_type: str | None = 'zone',
    storage_class: str = 'STANDARD',
    hns: bool = False,
) -> types.SimpleNamespace:
  return types.SimpleNamespace(
      location_type=location_type,
      storage_class=storage_class,
      hierarchical_namespace_enabled=hns,
  )


class GcsUtilsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    for cached_fn in (
        gcs_utils._get_bucket_info,
        gcs_utils._get_storage_control_client,
        gcs_utils._get_storage_client,
        gcs_utils.get_bucket,
    ):
      cached_fn.cache_clear()
      self.addCleanup(cached_fn.cache_clear)

  @parameterized.named_parameters(
      dict(
          testcase_name='gs_without_trailing_slash_add_true',
          path='gs://my-bucket/dir/sub',
          add_trailing_slash=True,
          expected=('my-bucket', 'dir/sub/'),
      ),
      dict(
          testcase_name='gs_with_trailing_slash_add_true',
          path='gs://my-bucket/dir/sub/',
          add_trailing_slash=True,
          expected=('my-bucket', 'dir/sub/'),
      ),
      dict(
          testcase_name='gs_without_trailing_slash_add_false',
          path='gs://my-bucket/dir/sub',
          add_trailing_slash=False,
          expected=('my-bucket', 'dir/sub'),
      ),
      dict(
          testcase_name='gs_with_trailing_slash_add_false',
          path='gs://my-bucket/dir/sub/',
          add_trailing_slash=False,
          expected=('my-bucket', 'dir/sub/'),
      ),
      dict(
          testcase_name='gs_bucket_only_add_false',
          path='gs://my-bucket',
          add_trailing_slash=False,
          expected=('my-bucket', ''),
      ),
      dict(
          testcase_name='gs_bucket_with_slash_add_false',
          path='gs://my-bucket/',
          add_trailing_slash=False,
          expected=('my-bucket', ''),
      ),
  )
  def test_parse_gcs_path(
      self,
      path: str,
      add_trailing_slash: bool,
      expected: tuple[str, str],
  ):
    self.assertEqual(
        gcs_utils.parse_gcs_path(path, add_trailing_slash=add_trailing_slash),
        expected,
    )

  def test_parse_gcs_path_invalid(self):
    with self.assertRaisesRegex(AssertionError, 'Unsupported scheme for GCS'):
      gcs_utils.parse_gcs_path('/tmp/local/path')
    with self.assertRaisesRegex(
        ValueError, 'The GCS path should contain the bucket name'
    ):
      gcs_utils.parse_gcs_path('gs://')

  @parameterized.product(
      gcs_backend=[None, 'gcs', 'gcs_grpc'],
      ckpt_path=[
          'gs://my-bucket/dir/sub',
          'gs://my-bucket/dir/sub/',
          'gs://my-bucket',
          'gs://my-bucket/',
          epath.Path('gs://my-bucket/dir/sub'),
      ],
      is_zonal=[False, True],
  )
  def test_get_kvstore_for_gcs(
      self,
      gcs_backend: str | None,
      ckpt_path: epath.PathLike,
      is_zonal: bool,
  ):
    env_override = (
        {} if gcs_backend is None else {'TENSORSTORE_GCS_BACKEND': gcs_backend}
    )
    expected_driver = gcs_backend or (
        'gcs_grpc' if is_zonal and str(ckpt_path).startswith('gs://') else 'gcs'
    )
    path_str = str(ckpt_path)
    if path_str.endswith(('my-bucket', 'my-bucket/')):
      expected_path = ''
    elif path_str.endswith('/'):
      expected_path = 'dir/sub/'
    else:
      expected_path = 'dir/sub'

    with (
        mock.patch.object(gcs_utils, 'is_zonal_bucket', return_value=is_zonal),
        mock.patch.dict(os.environ, env_override, clear=False),
    ):
      if gcs_backend is None:
        os.environ.pop('TENSORSTORE_GCS_BACKEND', None)
      self.assertEqual(
          gcs_utils.get_kvstore_for_gcs(ckpt_path),
          {
              'driver': expected_driver,
              'bucket': 'my-bucket',
              'path': expected_path,
          },
      )

  def test_get_storage_control_client_caches_none_on_import_error(self):
    with mock.patch.dict(
        'sys.modules', {'google.cloud.storage_control_v2': None}
    ):
      self.assertIsNone(gcs_utils._get_storage_control_client())
      self.assertIsNone(gcs_utils._get_storage_control_client())
      self.assertEqual(
          gcs_utils._get_storage_control_client.cache_info().hits, 1
      )

  @parameterized.named_parameters(
      ('layout_zonal_hns', _layout('zone', True), None, (True, True)),
      ('layout_regional', _layout('region', False), None, (False, False)),
      (
          'api_error_fallback_zonal',
          api_exceptions.GoogleAPICallError('err'),
          _bucket('zone', 'RAPID', True),
          (True, True),
      ),
      (
          'api_error_fallback_forbidden',
          api_exceptions.GoogleAPICallError('err'),
          api_exceptions.GoogleAPICallError('403'),
          (False, False),
      ),
      (
          'auth_error_no_fallback',
          auth_exceptions.DefaultCredentialsError('no creds'),
          None,
          (False, False),
      ),
      (
          'no_control_client_fallback_rapid',
          None,
          _bucket(None, 'RAPID', False),
          (True, False),
      ),
  )
  def test_bucket_info(self, layout, bucket, expected):
    if layout is None:
      client = None
    else:
      client = types.SimpleNamespace(
          get_storage_layout=mock.Mock(side_effect=[layout])
      )
    with (
        mock.patch.object(
            gcs_utils, '_get_storage_control_client', return_value=client
        ),
        mock.patch.object(
            gcs_utils, 'get_bucket', side_effect=[bucket]
        ) as mock_get_bucket,
    ):
      self.assertEqual(
          (
              gcs_utils.is_zonal_bucket('b'),
              gcs_utils.is_hierarchical_namespace_enabled('gs://b/d'),
          ),
          expected,
      )
      self.assertEqual(mock_get_bucket.called, bucket is not None)
    self.assertFalse(
        gcs_utils.is_hierarchical_namespace_enabled('/tmp/local/dir')
    )

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
    gcs_path.__str__.return_value = 'gs://bucket/dir'  # pyrefly: ignore[missing-attribute]

    with mock.patch.object(gcs_utils, 'is_gcs_path', return_value=True):
      gcs_utils.rmtree(gcs_path)

    self.assertEqual(
        gcs_path.rmtree.mock_calls,
        [mock.call(), mock.call(missing_ok=True)],
    )
    mock_cleanup_hns_folders.assert_not_called()

  @mock.patch.object(gcs_utils, 'is_hierarchical_namespace_enabled')
  @mock.patch.object(gcs_utils, 'cleanup_hns_folders')
  def test_rmtree_gcs_hns_enabled(
      self, mock_cleanup_hns_folders, mock_is_hns_enabled
  ):
    mock_is_hns_enabled.return_value = True
    gcs_path = mock.MagicMock(spec=epath.Path)
    gcs_path.__str__.return_value = 'gs://bucket/dir'  # pyrefly: ignore[missing-attribute]

    with mock.patch.object(gcs_utils, 'is_gcs_path', return_value=True):
      gcs_utils.rmtree(gcs_path)

    self.assertEqual(
        gcs_path.rmtree.mock_calls,
        [mock.call(), mock.call(missing_ok=True)],
    )
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
      gcs_utils.cleanup_hns_folders(gcs_path)  # pyrefly: ignore[bad-argument-type]

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

  def test_rmtree_missing_ok(self):
    gcs_path = mock.MagicMock(spec=epath.Path)
    gcs_path.rmtree.side_effect = FileNotFoundError('Directory not found')
    with (
        mock.patch.object(gcs_utils, 'is_gcs_path', return_value=True),
        mock.patch.object(
            gcs_utils, 'is_hierarchical_namespace_enabled', return_value=True
        ),
        mock.patch.object(gcs_utils, 'cleanup_hns_folders') as mock_cleanup,
    ):
      # Should not raise when missing_ok=True and should not clean up HNS.
      gcs_utils.rmtree(gcs_path, missing_ok=True)
      mock_cleanup.assert_not_called()
      with self.assertRaisesRegex(FileNotFoundError, 'Directory not found'):
        gcs_utils.rmtree(gcs_path, missing_ok=False)

  def test_split_gcs_path_and_bucket_root(self):
    prefixes = ['gs://']
    for prefix in prefixes:
      with self.subTest(prefix=prefix):
        raw_path = f'{prefix}my-bucket/dir/sub'
        self.assertEqual(
            gcs_utils.split_gcs_path(raw_path), ('my-bucket', 'dir/sub')
        )
        self.assertEqual(
            gcs_utils.gcs_bucket_root(raw_path), f'{prefix}my-bucket'
        )
        path = epath.Path(raw_path)
        self.assertEqual(
            gcs_utils.split_gcs_path(path), ('my-bucket', 'dir/sub')
        )
        self.assertEqual(
            gcs_utils.gcs_bucket_root(path),
            str(epath.Path(f'{prefix}my-bucket')),
        )

  def test_gcs_bucket_root_rejects_non_gcs_path(self):
    with self.assertRaises(ValueError):
      gcs_utils.gcs_bucket_root(epath.Path('/tmp/some/local/dir'))

  @parameterized.parameters(
      (False, 'get_bucket'),
      (True, 'bucket'),
  )
  def test_get_bucket(self, is_forbidden: bool, expected_method: str):
    mock_client = mock.MagicMock()
    if is_forbidden:
      mock_client.get_bucket.side_effect = api_exceptions.Forbidden('403')
    bucket = mock.MagicMock()
    getattr(mock_client, expected_method).return_value = bucket
    with mock.patch.object(
        gcs_utils, '_get_storage_client', return_value=mock_client
    ):
      self.assertIs(gcs_utils.get_bucket('my-bucket'), bucket)
      getattr(mock_client, expected_method).assert_called_once_with('my-bucket')


if __name__ == '__main__':
  absltest.main()
