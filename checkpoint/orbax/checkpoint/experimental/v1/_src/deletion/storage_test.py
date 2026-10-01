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

"""Tests for deletion metadata publication and existing filesystem adapters."""

import json
import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from orbax.checkpoint._src.path import gcs_utils
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.deletion import metadata
from orbax.checkpoint.experimental.v1._src.deletion import path_utils
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout
from orbax.checkpoint.experimental.v1._src.metadata import serialization as metadata_serialization


class MetadataTest(parameterized.TestCase):

  @parameterized.parameters(*checkpoint_layout.RESERVED_CHECKPOINTABLE_KEYS)
  def test_reserved_names_use_shared_layout_rules(self, name):
    with self.assertRaises(ValueError):
      path_utils.validate_checkpointable_name(name)

  def test_atomic_publication_conflict_and_permissions(self):
    directory = epath.Path(self.create_tempdir().full_path)
    path = directory / 'record'
    metadata.write_json(path, {'a': 1}, exclusive=True)
    os.chmod(path, 0o640)
    with self.assertRaises(FileExistsError):
      metadata.write_json(path, {'a': 2}, exclusive=True)
    self.assertEqual(metadata.read_json(path), {'a': 1})
    metadata.write_json(path, {'a': 3})
    self.assertEqual(metadata.read_json(path), {'a': 3})
    self.assertEqual(os.stat(path).st_mode & 0o777, 0o640)
    self.assertLen(list(directory.iterdir()), 1)

  def test_gcs_atomic_object_publication(self):
    bucket = mock.Mock()
    with mock.patch.object(gcs_utils, 'get_bucket', return_value=bucket):
      metadata.write_json(
          epath.Path('gs://bucket/checkpoint/record'), {'a': 1}, exclusive=True
      )
    bucket.blob.assert_called_once_with('checkpoint/record')
    bucket.blob.return_value.upload_from_string.assert_called_once_with(
        json.dumps({'a': 1}),
        content_type='application/json',
        if_generation_match=0,
    )

  @parameterized.parameters(False, True)
  def test_gcs_relocation_and_conflict(self, conflict):
    source, target = mock.MagicMock(), mock.MagicMock()
    source.exists.return_value = True
    target.exists.return_value = conflict
    if conflict:
      with self.assertRaises(FileExistsError):
        path_utils.delete_path(source, target)
      source.rename.assert_not_called()
    else:
      path_utils.delete_path(source, target)
      source.rename.assert_called_once_with(target)

  def test_configured_trash_destination_is_outside_checkpoint(self):
    context = context_lib.Context()
    context.deletion_options.gcs_deletion_options.todelete_full_path = 'trash'
    self.assertEqual(
        path_utils.destination(
            epath.Path('gs://bucket/run/1'), context, 'unique'
        ),
        epath.Path('gs://bucket/trash/1-unique'),
    )
    context.deletion_options.gcs_deletion_options.todelete_full_path = (
        'run/1/trash'
    )
    with self.assertRaises(ValueError):
      path_utils.destination(epath.Path('gs://bucket/run/1'), context, 'unique')


  @parameterized.parameters(
      '/elsewhere/trash/42-optimizer-unique',
      '/run/42/42-optimizer-unique',
      '/run/trash/wrong',
      'gs://bucket/trash/optimizer-unique',
  )
  def test_invalid_local_recovery_destination(self, value):
    with self.assertRaises(ValueError):
      path_utils.check_recovery_destination(
          epath.Path(value), epath.Path('/run/42'), 'optimizer', 'unique'
      )

  def test_read_visibility_is_strict_for_explicit_requests_and_read_only(self):
    path = epath.Path(self.create_tempdir().full_path)
    original = {
        'item_handlers': {
            'state': 'state_handler',
            'optimizer': 'optimizer_handler',
        }
    }
    metadata.write_json(
        metadata_serialization.checkpoint_metadata_file_path(path), original
    )
    record = {
        'version': 1,
        'path': str(path),
        'operation_id': 'a' * 32,
        'checkpointable_name': 'optimizer',
        'destination': None,
        'original_metadata': original,
        'updated_metadata': metadata.updated_metadata(original, 'optimizer'),
    }
    metadata.write_json(path / metadata.RECORD_FILENAME, record)
    before = {p.name: p.read_bytes() for p in path.iterdir()}
    with mock.patch.object(metadata.logging, 'warning') as warning:
      self.assertEqual(metadata.check_read(path, ('state',)), 'optimizer')
      warning.assert_not_called()
      self.assertEqual(metadata.check_read(path, None), 'optimizer')
      warning.assert_called_once()
    with self.assertRaises(metadata.CheckpointDeletionInProgressError):
      metadata.check_read(path, ('state', 'optimizer'))
    self.assertEqual(before, {p.name: p.read_bytes() for p in path.iterdir()})
    metadata.write_json(
        metadata_serialization.checkpoint_metadata_file_path(path),
        {'item_handlers': {}},
    )
    with self.assertRaises(metadata.CheckpointDeletionRecoveryError):
      metadata.check_read(path, ('state',))

  def test_record_disappearing_after_existence_check_is_completed_cleanup(self):
    path = epath.Path(self.create_tempdir().full_path)
    (path / metadata.RECORD_FILENAME).write_text('{}')

    def finish_cleanup(record_path):
      record_path.unlink()
      raise FileNotFoundError(record_path)

    with mock.patch.object(metadata, 'read_json', side_effect=finish_cleanup):
      self.assertIsNone(metadata.read_record(path))


if __name__ == '__main__':
  absltest.main()
