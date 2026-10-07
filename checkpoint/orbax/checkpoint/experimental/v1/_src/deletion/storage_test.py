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

import asyncio
import json
import os
import threading
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from orbax.checkpoint._src.path import async_path
from orbax.checkpoint._src.path import deleter
from orbax.checkpoint._src.path import gcs_utils
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.deletion import metadata
from orbax.checkpoint.experimental.v1._src.deletion import path_utils
from orbax.checkpoint.experimental.v1._src.deletion import validation
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout
from orbax.checkpoint.experimental.v1._src.metadata import serialization as metadata_serialization


class MetadataTest(parameterized.TestCase, unittest.IsolatedAsyncioTestCase):

  @parameterized.parameters(*checkpoint_layout.RESERVED_CHECKPOINTABLE_KEYS)
  async def test_reserved_names_use_shared_layout_rules(self, name):
    with self.assertRaises(ValueError):
      validation.validate_checkpointable_name(name)

  async def test_atomic_publication_conflict_and_permissions(self):
    directory = epath.Path(self.create_tempdir().full_path)
    path = directory / 'record'
    await metadata.write_json(path, {'a': 1}, exclusive=True)
    os.chmod(path, 0o640)
    with self.assertRaises(FileExistsError):
      await metadata.write_json(path, {'a': 2}, exclusive=True)
    self.assertEqual(json.loads(path.read_text()), {'a': 1})
    await metadata.write_json(path, {'a': 3})
    self.assertEqual(json.loads(path.read_text()), {'a': 3})
    self.assertEqual(os.stat(path).st_mode & 0o777, 0o640)
    self.assertLen(list(directory.iterdir()), 1)

  async def test_gcs_atomic_object_publication(self):
    bucket = mock.Mock()
    with mock.patch.object(gcs_utils, 'get_bucket', return_value=bucket):
      await metadata.write_json(
          epath.Path('gs://bucket/checkpoint/record'), {'a': 1}, exclusive=True
      )
    bucket.blob.assert_called_once_with('checkpoint/record')
    bucket.blob.return_value.upload_from_string.assert_called_once_with(
        json.dumps({'a': 1}),
        content_type='application/json',
        if_generation_match=0,
    )

  @parameterized.parameters(False, True)
  async def test_gcs_relocation_and_conflict(self, conflict):
    source, target = mock.MagicMock(), mock.MagicMock()
    source.exists.return_value = True
    target.exists.return_value = conflict
    if conflict:
      with self.assertRaises(FileExistsError):
        await path_utils.delete_path(source, target)
      source.rename.assert_not_called()
    else:
      await path_utils.delete_path(source, target)
      source.rename.assert_called_once_with(target)

  async def test_configured_trash_destination_is_outside_checkpoint(self):
    context = context_lib.Context()
    context.deletion_options.gcs_deletion_options.todelete_full_path = 'trash'
    self.assertEqual(
        await path_utils.destination(
            epath.Path('gs://bucket/run/1'), context, 'unique'
        ),
        epath.Path('gs://bucket/trash/1-unique'),
    )
    context.deletion_options.gcs_deletion_options.todelete_full_path = (
        'run/1/trash'
    )
    with self.assertRaises(ValueError):
      await path_utils.destination(
          epath.Path('gs://bucket/run/1'), context, 'unique'
      )

  async def test_trash_destination_preserves_gcs_prefix(self):
    context = context_lib.Context()
    context.deletion_options.gcs_deletion_options.todelete_full_path = 'trash'
    prefixes = ['gs://']
    for prefix in prefixes:
      with self.subTest(prefix=prefix):
        source = epath.Path(f'{prefix}bucket/run/1')
        expected = epath.Path(f'{prefix}bucket/trash/1-unique')
        target = await path_utils.destination(source, context, 'unique')
        assert target is not None
        self.assertEqual(str(target), str(expected))
        self.assertEqual(
            gcs_utils.gcs_bucket_root(target),
            gcs_utils.gcs_bucket_root(source),
        )

  async def test_gcs_bucket_root_rejects_non_gcs_path(self):
    with self.assertRaises(ValueError):
      gcs_utils.gcs_bucket_root(epath.Path('/tmp/run/1'))


  @parameterized.parameters(
      '/elsewhere/trash/42-optimizer-unique',
      '/run/42/42-optimizer-unique',
      '/run/trash/wrong',
      'gs://bucket/trash/optimizer-unique',
  )
  async def test_invalid_local_recovery_destination(self, value):
    with self.assertRaises(ValueError):
      await path_utils.check_recovery_destination(
          epath.Path(value), epath.Path('/run/42'), 'optimizer', 'unique'
      )

  async def test_read_visibility_is_strict_for_explicit_requests_and_read_only(
      self,
  ):
    path = epath.Path(self.create_tempdir().full_path)
    original = {
        'item_handlers': {
            'state': 'state_handler',
            'optimizer': 'optimizer_handler',
        }
    }
    await metadata.write_json(
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
    await metadata.write_json(path / metadata.RECORD_FILENAME, record)
    before = {p.name: p.read_bytes() for p in path.iterdir()}
    with mock.patch.object(metadata.logging, 'warning') as warning:
      self.assertEqual(await metadata.check_read(path, ('state',)), 'optimizer')
      warning.assert_not_called()
      self.assertEqual(await metadata.check_read(path, None), 'optimizer')
      warning.assert_called_once()
    with self.assertRaises(metadata.DeletionInProgressError):
      await metadata.check_read(path, ('state', 'optimizer'))
    self.assertEqual(before, {p.name: p.read_bytes() for p in path.iterdir()})
    await metadata.write_json(
        metadata_serialization.checkpoint_metadata_file_path(path),
        {'item_handlers': {}},
    )
    with self.assertRaises(metadata.DeletionRecoveryError):
      await metadata.check_read(path, ('state',))

  async def test_record_disappearing_after_existence_check_is_completed_cleanup(
      self,
  ):
    path = epath.Path(self.create_tempdir().full_path)
    (path / metadata.RECORD_FILENAME).write_text('{}')

    def finish_cleanup(record_path):
      record_path.unlink()
      raise FileNotFoundError(record_path)

    with mock.patch.object(metadata, '_read_json', side_effect=finish_cleanup):
      self.assertIsNone(await metadata.read_record(path))

  async def test_strict_checkpoint_metadata_read_uses_async_path(self):
    path = epath.Path(self.create_tempdir().full_path)
    metadata_path = metadata_serialization.checkpoint_metadata_file_path(path)
    with mock.patch.object(
        async_path,
        'read_text',
        new_callable=mock.AsyncMock,
        return_value='{"item_handlers": {}}',
    ) as read:
      self.assertEqual(
          await metadata.read_checkpoint_metadata(path), {'item_handlers': {}}
      )
      read.assert_awaited_once_with(metadata_path)
    with mock.patch.object(
        async_path, 'read_text', new_callable=mock.AsyncMock, return_value='[]'
    ):
      with self.assertRaisesRegex(ValueError, 'dictionary'):
        await metadata.read_checkpoint_metadata(path)
    with mock.patch.object(
        async_path,
        'read_text',
        new_callable=mock.AsyncMock,
        side_effect=PermissionError('unreadable'),
    ):
      with self.assertRaises(PermissionError):
        await metadata.read_checkpoint_metadata(path)

  async def test_atomic_write_does_not_block_event_loop(self):
    path = epath.Path(self.create_tempdir().full_path) / 'record'
    started = threading.Event()
    release = threading.Event()
    loop_thread = threading.get_ident()
    real_write = metadata._write_json

    def blocked_write(*args, **kwargs):
      self.assertNotEqual(threading.get_ident(), loop_thread)
      started.set()
      if not release.wait(timeout=5):
        raise TimeoutError('event loop failed to release writer')
      return real_write(*args, **kwargs)

    with mock.patch.object(metadata, '_write_json', side_effect=blocked_write):
      task = asyncio.create_task(metadata.write_json(path, {'a': 1}))
      try:
        self.assertTrue(await asyncio.to_thread(started.wait, 2))
        self.assertFalse(task.done())
      finally:
        release.set()
        await task
    self.assertEqual(json.loads(path.read_text()), {'a': 1})

  async def test_directory_iteration_and_removal_run_off_event_loop(self):
    path = epath.Path(self.create_tempdir().full_path) / 'checkpoint'
    path.mkdir()
    (path / 'state').mkdir()
    marker = metadata_serialization.checkpoint_metadata_file_path(path)
    marker.write_text('{}')
    loop_thread = threading.get_ident()
    real_iterdir = type(path).iterdir
    real_delete = deleter.PathDeleter.delete

    def checked_iterdir(target):
      # Check iteration too: a lazy generator must be consumed in the worker.
      self.assertNotEqual(threading.get_ident(), loop_thread)
      yield from real_iterdir(target)

    def checked_delete(instance, target, **kwargs):
      self.assertNotEqual(threading.get_ident(), loop_thread)
      return real_delete(instance, target, **kwargs)

    with (
        mock.patch.object(type(path), 'iterdir', checked_iterdir),
        mock.patch.object(deleter.PathDeleter, 'delete', checked_delete),
        mock.patch.object(
            deleter.event_tracking, 'record_delete_event'
        ) as event,
    ):
      await validation.validate_root(path, context=context_lib.Context())
      await path_utils.delete_path(path, keep_markers_until_last=True)
      event.assert_not_called()
    self.assertFalse(path.exists())

  async def test_path_checks_use_async_operations(self):
    path = epath.Path('/tmp/checkpoint')
    with mock.patch.object(
        async_path, 'is_link', new_callable=mock.AsyncMock, return_value=True
    ) as is_link:
      with self.assertRaisesRegex(ValueError, 'symlink'):
        await path_utils.check_path(path)
      is_link.assert_awaited_once_with(path)
    loop_thread = threading.get_ident()

    def checked_realpath(value):
      self.assertNotEqual(threading.get_ident(), loop_thread)
      return value

    with mock.patch.object(
        path_utils.os.path, 'realpath', side_effect=checked_realpath
    ):
      await path_utils.check_destination(epath.Path('/tmp/trash'), path)

  async def test_prepare_item_and_mutation_guard(self):
    path = epath.Path(self.create_tempdir().full_path)
    for name in ('state', 'optimizer'):
      (path / name).mkdir()
    original = {'item_handlers': {'state': 'state', 'optimizer': 'optimizer'}}
    await metadata.write_json(
        metadata_serialization.checkpoint_metadata_file_path(path), original
    )
    self.assertEqual(await validation.prepare_item(path, 'optimizer'), original)
    record = {
        'version': 1,
        'path': str(path),
        'operation_id': 'a' * 32,
        'checkpointable_name': 'optimizer',
        'destination': None,
        'original_metadata': original,
        'updated_metadata': metadata.updated_metadata(original, 'optimizer'),
    }
    await metadata.write_json(path / metadata.RECORD_FILENAME, record)
    with self.assertRaises(metadata.DeletionInProgressError):
      await metadata.ensure_available(path)
    # Empty read selection validates quietly; explicit reading names differ
    # from the one being deleted.
    with mock.patch.object(metadata.logging, 'warning') as warn:
      self.assertEqual(await metadata.check_read(path, ()), 'optimizer')
      warn.assert_not_called()
    (path / metadata.RECORD_FILENAME).unlink()
    await metadata.ensure_available(path)

  @parameterized.parameters(True, False, None, 1.0, '1')
  async def test_invalid_step_type(self, step):
    with self.assertRaises(TypeError):
      validation.validate_step(step)

  @parameterized.parameters(0, 1, 100)
  async def test_integer_step_is_valid(self, step):
    validation.validate_step(step)

  async def test_negative_step_is_invalid(self):
    with self.assertRaises(ValueError):
      validation.validate_step(-1)


if __name__ == '__main__':
  absltest.main()
