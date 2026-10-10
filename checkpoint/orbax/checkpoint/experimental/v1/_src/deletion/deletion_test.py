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

"""End-to-end deletion and recovery tests using real Orbax checkpoints."""

import concurrent.futures
import functools
import json
import os
import pathlib
import threading
from typing import Any, cast
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import numpy as np
import orbax.checkpoint as ocp_v0
from orbax.checkpoint._src.path import deleter
import orbax.checkpoint.experimental.v1 as ocp
from orbax.checkpoint.experimental.v1._src.deletion import execution
from orbax.checkpoint.experimental.v1._src.deletion import metadata
from orbax.checkpoint.experimental.v1._src.deletion import path_utils
from orbax.checkpoint.experimental.v1._src.metadata import serialization as metadata_serialization


def _delete(asynchronous: bool, path: epath.Path, **kwargs: Any) -> bool:
  """Runs `ocp.delete` or waits for `ocp.delete_async`."""
  if asynchronous:
    return ocp.delete_async(path, **kwargs).result()
  return ocp.delete(path, **kwargs)


class DeletingTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(self.create_tempdir().full_path)
    self.path = self.directory / 'checkpoint'
    self.state: Any = {'weights': np.arange(8)}

  def save(self):
    ocp.save_checkpointables(
        self.path,
        {'state': self.state, 'optimizer': {'momentum': np.ones(8)}},
        custom_metadata={'model': 'test'},
    )

  @parameterized.parameters(False, True)
  def test_whole(self, asynchronous):
    self.save()
    sibling = self.directory / 'unrelated'
    sibling.write_text('keep')
    if asynchronous:
      response = ocp.delete_async(self.path)
      self.assertIs(response.result(), True)
      self.assertIs(response.result(), True)
    else:
      self.assertIs(ocp.delete(self.path), True)
    self.assertFalse(self.path.exists())
    self.assertEqual(sibling.read_text(), 'keep')

  @parameterized.parameters(False, True)
  def test_partial_preserves_metadata_and_surviving_payload(self, asynchronous):
    self.save()
    meta_path = metadata_serialization.checkpoint_metadata_file_path(self.path)
    original = json.loads(meta_path.read_text())
    meta_path.write_text(json.dumps(original))
    before = {
        str(p.relative_to(self.path)): p.read_bytes()
        for p in pathlib.Path(str(self.path / 'state')).rglob('*')
        if p.is_file()
    }
    if asynchronous:
      self.assertIs(
          ocp.delete_async(self.path, checkpointable_name='optimizer').result(),
          True,
      )
    else:
      self.assertIs(
          ocp.delete(self.path, checkpointable_name='optimizer'), True
      )
    expected = dict(original)
    expected['item_handlers'] = dict(original['item_handlers'])
    del expected['item_handlers']['optimizer']
    self.assertEqual(json.loads(meta_path.read_text()), expected)
    self.assertFalse((self.path / 'optimizer').exists())
    self.assertFalse((self.path / metadata.RECORD_FILENAME).exists())
    self.assertEqual(
        before,
        {
            str(p.relative_to(self.path)): p.read_bytes()
            for p in pathlib.Path(str(self.path / 'state')).rglob('*')
            if p.is_file()
        },
    )
    loaded = cast(dict[str, Any], ocp.load(self.path))
    np.testing.assert_array_equal(
        loaded['weights'], self.state['weights']
    )
    self.assertEqual(
        set(ocp.checkpointables_metadata(self.path).metadata), {'state'}
    )

  @parameterized.product(asynchronous=(False, True), partial=(False, True))
  def test_already_absent_returns_false(self, asynchronous, partial):
    if partial:
      self.save()
    name = 'absent' if partial else None
    if asynchronous:
      response = ocp.delete_async(
          self.path, checkpointable_name=name, missing_ok=True
      )
      self.assertIs(response.result(), False)
      self.assertIs(response.result(), False)
    else:
      self.assertIs(
          ocp.delete(self.path, checkpointable_name=name, missing_ok=True),
          False,
      )

  @parameterized.product(
      asynchronous=(False, True), metadata_committed=(False, True)
  )
  def test_retry_cleanup_returns_true_then_false(
      self, asynchronous, metadata_committed
  ):
    self.save()

    async def interrupt(plan, whole_delete):
      del whole_delete
      await path_utils.delete_path(
          plan.path / plan.record['checkpointable_name']
      )
      if metadata_committed:
        await metadata.write_json(
            metadata_serialization.checkpoint_metadata_file_path(plan.path),
            plan.record['updated_metadata'],
        )
      raise OSError('interrupted cleanup')

    with mock.patch.object(execution, '_execute', side_effect=interrupt):
      with self.assertRaisesRegex(OSError, 'interrupted cleanup'):
        ocp.delete(self.path, checkpointable_name='optimizer')
    retry = functools.partial(
        _delete,
        asynchronous,
        self.path,
        checkpointable_name='optimizer',
        missing_ok=True,
    )
    self.assertIs(retry(), True)
    self.assertIs(retry(), False)
    self.assertEqual(set(ocp.load_checkpointables(self.path)), {'state'})

  def test_preserves_unknown_metadata_fields(self):
    self.save()
    path = metadata_serialization.checkpoint_metadata_file_path(self.path)
    original = json.loads(path.read_text())
    original['future_metadata'] = {'unknown': 123}
    path.write_text(json.dumps(original))
    ocp.delete(self.path, checkpointable_name='optimizer')
    self.assertEqual(
        json.loads(path.read_text())['future_metadata'], {'unknown': 123}
    )

  def test_legacy_flat(self):
    with ocp_v0.PyTreeCheckpointer() as ckptr:
      ckptr.save(self.path, self.state)
    with self.assertRaises((ValueError, FileNotFoundError)):
      ocp.delete(self.path, checkpointable_name='state')
    ocp.delete(self.path)
    self.assertFalse(self.path.exists())

  def test_missing(self):
    with self.assertRaises(FileNotFoundError):
      ocp.delete(self.path)
    self.assertIs(ocp.delete_async(self.path, missing_ok=True).result(), False)
    self.save()
    ocp.delete(self.path, checkpointable_name='absent', missing_ok=True)
    with self.assertRaises(FileNotFoundError):
      ocp.delete(self.path, checkpointable_name='absent')
    self.assertTrue((self.path / 'state').exists())

  @parameterized.parameters(
      '',
      '.',
      '..',
      '../state',
      '/state',
      'a/b',
      'a\\b',
      'metrics',
      'AUTO',
      '_CHECKPOINT_METADATA',
  )
  def test_invalid_name(self, name):
    self.save()
    with self.assertRaises(ValueError):
      ocp.delete(self.path, checkpointable_name=name)
    self.assertTrue((self.path / 'optimizer').exists())

  def test_checkpoint_prefix_is_not_a_reserved_namespace(self):
    name = '_CHECKPOINT_weights'
    ocp.save_checkpointables(self.path, {'state': self.state, name: self.state})
    ocp.delete(self.path, checkpointable_name=name)
    self.assertEqual(set(ocp.load_checkpointables(self.path)), {'state'})

  def test_last_item(self):
    ocp.save(self.path, self.state)
    with self.assertRaisesRegex(ValueError, 'final checkpointable'):
      ocp.delete(self.path, checkpointable_name='state')
    self.assertFalse((self.path / metadata.RECORD_FILENAME).exists())

  def test_rejects_run_and_child_paths(self):
    self.save()
    for path in (self.directory, self.path / 'state'):
      with self.assertRaises(ocp.errors.InvalidLayoutError):
        ocp.delete(path)
    self.assertTrue(self.path.exists())

  def test_partial_failure_keeps_survivors_readable_and_blocks_overwrite(self):
    self.save()
    real_write = metadata.write_json

    async def fail_metadata(path, value, **kwargs):
      if path == metadata_serialization.checkpoint_metadata_file_path(
          self.path
      ):
        raise PermissionError('injected metadata failure')
      return await real_write(path, value, **kwargs)

    with mock.patch.object(metadata, 'write_json', side_effect=fail_metadata):
      response = ocp.delete_async(self.path, checkpointable_name='optimizer')
      with self.assertRaisesRegex(PermissionError, 'injected'):
        response.result()
    self.assertFalse((self.path / 'optimizer').exists())
    self.assertTrue((self.path / metadata.RECORD_FILENAME).exists())
    loaded = cast(dict[str, Any], ocp.load(self.path))
    np.testing.assert_array_equal(
        loaded['weights'], self.state['weights']
    )
    self.assertEqual(
        set(ocp.checkpointables_metadata(self.path).metadata), {'state'}
    )
    with self.assertRaises(ocp.errors.DeletionInProgressError):
      ocp.load(self.path, checkpointable_name='optimizer')
    with self.assertRaises(ocp.errors.DeletionInProgressError):
      ocp.save(self.path, self.state, overwrite=True)
    ocp.delete(self.path, checkpointable_name='optimizer')
    loaded = cast(dict[str, Any], ocp.load(self.path))
    np.testing.assert_array_equal(
        loaded['weights'], self.state['weights']
    )
    with self.assertRaises(PermissionError):
      response.result()

  def test_whole_deletion_retry_keeps_format_marker(self):
    self.save()
    with mock.patch.object(
        path_utils, 'remove', side_effect=PermissionError('injected')
    ):
      with self.assertRaises(PermissionError):
        ocp.delete(self.path)
    self.assertTrue(
        (
            metadata_serialization.checkpoint_metadata_file_path(self.path)
        ).exists()
    )
    self.assertFalse((self.path / metadata.RECORD_FILENAME).exists())
    ocp.delete(self.path)
    self.assertFalse(self.path.exists())

  def test_record_creation_failure_does_not_remove_payload(self):
    self.save()
    with mock.patch.object(
        metadata, 'write_json', side_effect=PermissionError('record')
    ):
      with self.assertRaises(PermissionError):
        ocp.delete_async(self.path, checkpointable_name='optimizer')
    self.assertTrue((self.path / 'optimizer').exists())
    self.assertFalse((self.path / metadata.RECORD_FILENAME).exists())

  def test_corrupt_record_blocks_access_but_allows_explicit_whole_delete(self):
    self.save()
    (self.path / metadata.RECORD_FILENAME).write_text('{')
    with self.assertRaises(ocp.errors.DeletionRecoveryError):
      ocp.load(self.path)
    with self.assertRaises(ocp.errors.DeletionRecoveryError):
      ocp.delete(self.path, checkpointable_name='optimizer')
    ocp.delete(self.path)
    self.assertFalse(self.path.exists())

  def test_record_name_is_reserved_for_saving(self):
    with self.assertRaisesRegex(ValueError, 'reserved'):
      ocp.save_checkpointables(
          self.path, {metadata.RECORD_FILENAME: self.state}
      )
    self.assertFalse(self.path.exists())

  def test_recovery_after_metadata_commit_before_record_removal(self):
    self.save()

    async def interrupt(plan, whole_delete):
      del whole_delete
      await path_utils.delete_path(
          plan.path / plan.record['checkpointable_name']
      )
      await metadata.write_json(
          metadata_serialization.checkpoint_metadata_file_path(plan.path),
          plan.record['updated_metadata'],
      )
      raise OSError('interrupted before record removal')

    with mock.patch.object(execution, '_execute', side_effect=interrupt):
      with self.assertRaises(OSError):
        ocp.delete(self.path, checkpointable_name='optimizer')
    self.assertTrue((self.path / metadata.RECORD_FILENAME).exists())
    ocp.delete(self.path, checkpointable_name='optimizer')
    self.assertEqual(set(ocp.load_checkpointables(self.path)), {'state'})

  def test_changed_metadata_or_different_item_cannot_resume(self):
    self.save()
    with mock.patch.object(
        execution, '_execute', side_effect=OSError('injected')
    ):
      with self.assertRaises(OSError):
        ocp.delete(self.path, checkpointable_name='optimizer')
    with self.assertRaises(ocp.errors.DeletionRecoveryError):
      ocp.delete(self.path, checkpointable_name='state')
    path = metadata_serialization.checkpoint_metadata_file_path(self.path)
    value = json.loads(path.read_text())
    value['custom_metadata'] = {'changed': True}
    path.write_text(json.dumps(value))
    with self.assertRaises(ocp.errors.DeletionRecoveryError):
      ocp.delete(self.path, checkpointable_name='optimizer')
    self.assertTrue((self.path / 'optimizer').exists())
    ocp.delete(self.path)

  def test_record_uses_configured_permissions_and_only_checkpoint_directory(
      self,
  ):
    self.save()
    before = set(self.directory.iterdir())
    context = ocp.Context()
    context.file_options.path_permission_mode = 0o640
    with context, mock.patch.object(
        execution, '_execute', side_effect=OSError('stop')
    ):
      with self.assertRaises(OSError):
        ocp.delete(self.path, checkpointable_name='optimizer')
    self.assertEqual(set(self.directory.iterdir()), before)
    record_path = self.path / metadata.RECORD_FILENAME
    self.assertEqual(os.stat(record_path).st_mode & 0o777, 0o640)
    ocp.delete(self.path, checkpointable_name='optimizer')


  def test_symlink_target_is_rejected(self):
    self.save()
    link = pathlib.Path(str(self.directory / 'link'))
    link.symlink_to(self.path, target_is_directory=True)
    with self.assertRaisesRegex(ValueError, 'symlink'):
      ocp.delete(epath.Path(link))
    self.assertTrue(self.path.exists())

  def test_result_timeout_does_not_cancel_deletion(self):
    self.save()
    entered = threading.Event()
    release = threading.Event()
    original = execution._execute

    async def execute(*args):
      entered.set()
      if not release.wait(10):
        raise TimeoutError('test release')
      return await original(*args)

    with mock.patch.object(execution, '_execute', side_effect=execute):
      try:
        response = ocp.delete_async(self.path)
        self.assertTrue(entered.wait(10))
        with self.assertRaises(concurrent.futures.TimeoutError):
          response.result(timeout=0)
        self.assertTrue(self.path.exists())
      finally:
        release.set()
      self.assertIs(response.result(), True)
    self.assertFalse(self.path.exists())


  @parameterized.product(
      deleted_name=('state', 'optimizer'),
      legacy=(False, True),
      phase=('before_payload', 'after_payload', 'after_metadata'),
  )
  def test_interrupted_item_deletion_visibility(
      self, deleted_name, legacy, phase
  ):
    if legacy:
      self.path = self.directory / '1'
      with ocp_v0.CheckpointManager(self.directory) as manager:
        manager.save(
            1,
            args=ocp_v0.args.Composite(
                state=ocp_v0.args.PyTreeSave(self.state),
                optimizer=ocp_v0.args.PyTreeSave({'momentum': np.ones(8)}),
            ),
        )
        manager.wait_until_finished()
    else:
      self.save()
    expected = ocp.load_checkpointables(self.path)
    abstract = ocp.checkpointables_metadata(self.path)
    original_write = metadata.write_json

    async def write(path, value, **kwargs):
      if path == metadata_serialization.checkpoint_metadata_file_path(
          self.path
      ):
        if phase == 'after_metadata':
          await original_write(path, value, **kwargs)
        raise OSError('interrupted metadata update')
      return await original_write(path, value, **kwargs)

    if phase == 'before_payload':
      failure = mock.patch.object(
          execution, '_execute', side_effect=OSError('before payload')
      )
    else:
      failure = mock.patch.object(metadata, 'write_json', side_effect=write)
    with failure, self.assertRaises(OSError):
      ocp.delete(self.path, checkpointable_name=deleted_name)
    survivor = 'state' if deleted_name == 'optimizer' else 'optimizer'

    def snapshot():
      return {
          str(p.relative_to(self.path)): p.read_bytes()
          for p in pathlib.Path(str(self.path)).rglob('*')
          if p.is_file()
      }

    before = snapshot()
    # Reads must neither retry deletion nor rewrite its record or metadata.
    with mock.patch.object(
        execution,
        'start',
        side_effect=AssertionError('read triggered deletion'),
    ):
      with mock.patch.object(metadata.logging, 'warning') as warning:
        loaded = ocp.load_checkpointables(self.path)
        self.assertEqual(set(loaded), {survivor})
        warning.assert_called_once()
        self.assertIn('Excluding checkpointable', warning.call_args.args[0])
        self.assertEqual(warning.call_args.args[1], deleted_name)
      with mock.patch.object(metadata.logging, 'warning') as warning:
        self.assertEqual(
            set(ocp.checkpointables_metadata(self.path).metadata), {survivor}
        )
        warning.assert_called_once()
      with mock.patch.object(metadata.logging, 'warning') as warning:
        self.assertEqual(
            set(ocp.load_checkpointables(self.path, {survivor: None})),
            {survivor},
        )
        single = cast(
            dict[str, Any], ocp.load(self.path, checkpointable_name=survivor)
        )
        ocp.metadata(self.path, checkpointable_name=survivor)
        warning.assert_not_called()
      for key in expected[survivor]:
        np.testing.assert_array_equal(single[key], expected[survivor][key])
      for operation in (
          lambda: ocp.load(self.path, checkpointable_name=deleted_name),
          lambda: ocp.metadata(self.path, checkpointable_name=deleted_name),
          lambda: ocp.load_checkpointables(self.path, {deleted_name: None}),
          lambda: ocp.load_checkpointables(
              self.path, {'state': None, 'optimizer': None}
          ),
          lambda: ocp.load_checkpointables(self.path, abstract),
      ):
        with self.assertRaises(ocp.errors.DeletionInProgressError):
          operation()
      # AUTO discovery must never select the item, even while its files remain.
      ocp.load(self.path)
      ocp.metadata(self.path)
    self.assertEqual(snapshot(), before)
    ocp.delete(self.path, checkpointable_name=deleted_name)
    self.assertFalse((self.path / metadata.RECORD_FILENAME).exists())
    self.assertEqual(set(ocp.load_checkpointables(self.path)), {survivor})

  def test_conflicting_record_blocks_all_reads(self):
    self.save()
    with mock.patch.object(
        execution, '_execute', side_effect=OSError('injected')
    ):
      with self.assertRaises(OSError):
        ocp.delete(self.path, checkpointable_name='optimizer')
    path = metadata_serialization.checkpoint_metadata_file_path(self.path)
    current = json.loads(path.read_text())
    current['custom_metadata'] = {'unexpected': 'change'}
    path.write_text(json.dumps(current))
    for operation in (
        lambda: ocp.load(self.path, checkpointable_name='state'),
        lambda: ocp.load_checkpointables(self.path),
        lambda: ocp.checkpointables_metadata(self.path),
    ):
      with self.assertRaises(ocp.errors.DeletionRecoveryError):
        operation()
    ocp.delete(self.path)

  @parameterized.parameters(False, True)
  def test_free_paths_use_shared_physical_deleter(self, partial):
    self.save()
    name = 'optimizer' if partial else None
    target = self.path / name if name else self.path
    original = deleter.PathDeleter.delete
    with mock.patch.object(
        deleter.PathDeleter, 'delete', autospec=True, side_effect=original
    ) as delete:
      ocp.delete(self.path, checkpointable_name=name)
    self.assertIn(target, [call.args[1] for call in delete.call_args_list])
    self.assertFalse(target.exists())


if __name__ == '__main__':
  absltest.main()
