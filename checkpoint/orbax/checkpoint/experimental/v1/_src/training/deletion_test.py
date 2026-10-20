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

"""Managed deletion lifecycle, compatibility, and recovery tests."""

import concurrent.futures
import threading
import typing
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import numpy as np
from orbax.checkpoint._src.path import deleter
import orbax.checkpoint.experimental.v1 as ocp
from orbax.checkpoint.experimental.v1._src.deletion import execution
from orbax.checkpoint.experimental.v1._src.deletion import metadata
from orbax.checkpoint.experimental.v1._src.metadata import serialization as metadata_serialization
from orbax.checkpoint.experimental.v1._src.path import step as step_lib


class ManagedDeletionTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(self.create_tempdir().full_path) / 'run'
    self.state: Any = {'weights': np.arange(8)}
    self.ckptr = ocp.training.Checkpointer(
        self.directory, custom_metadata={'run': 'keep'}
    )

  def tearDown(self):
    # Tests may replace self.ckptr; close whichever instance is current.
    self.ckptr.close()
    super().tearDown()

  def assert_latest_step(self, expected_step: int):
    self.assertIsNotNone(self.ckptr.latest)
    assert self.ckptr.latest is not None
    self.assertEqual(self.ckptr.latest.step, expected_step)

  def save(self, step):
    self.ckptr.save_checkpointables(
        step,
        {'state': self.state, 'optimizer': {'momentum': np.ones(8)}},
        force=True,
        metrics={'loss': 0.5},
        custom_metadata={'model': 'test'},
    )

  def run_delete(
      self,
      asynchronous: bool,
      step: int | ocp.training.CheckpointMetadata,
      **kwargs: Any,
  ) -> bool:
    """Runs `Checkpointer.delete` or waits for `Checkpointer.delete_async`."""
    if asynchronous:
      return self.ckptr.delete_async(step, **kwargs).result()
    return self.ckptr.delete(step, **kwargs)

  def test_delete_middle_latest_final_and_reuse(self):
    for step in (1, 2, 3):
      self.save(step)
    self.ckptr.delete(2)
    self.assertEqual([c.step for c in self.ckptr.checkpoints], [1, 3])
    self.ckptr.delete_async(3).result()
    self.assert_latest_step(1)
    self.ckptr.delete(1)
    self.assertIsNone(self.ckptr.latest)
    self.assertEqual(
        self.ckptr.root_metadata().custom_metadata, {'run': 'keep'}
    )
    self.save(1)
    np.testing.assert_array_equal(
        typing.cast(dict[str, Any], self.ckptr.load(1))['weights'],
        self.state['weights'],
    )

  def test_delegate_to_v0_step_deleter(self):
    self.save(1)
    step_deleter = self.ckptr._manager._checkpoint_deleter
    with mock.patch.object(
        step_deleter, 'delete', wraps=step_deleter.delete
    ) as delete:
      self.ckptr.delete(1)
    delete.assert_called_once_with(1)

  @parameterized.parameters(False, True)
  def test_managed_paths_use_shared_physical_deleter(self, partial):
    self.save(1)
    name = 'optimizer' if partial else None
    path = self.directory / '1'
    target = path / name if name else path
    original = deleter.PathDeleter.delete
    with mock.patch.object(
        deleter.PathDeleter, 'delete', autospec=True, side_effect=original
    ) as delete:
      self.ckptr.delete(1, checkpointable_name=name)
    self.assertIn(target, [call.args[1] for call in delete.call_args_list])
    self.assertFalse(target.exists())

  @parameterized.parameters(False, True)
  def test_partial_keeps_step_and_metrics(self, asynchronous):
    self.save(1)
    self.assertIs(
        self.run_delete(asynchronous, 1, checkpointable_name='optimizer'), True
    )
    self.assert_latest_step(1)
    self.assertEqual(
        self.ckptr.checkpointables_metadata(1).metrics, {'loss': 0.5}
    )
    self.assertEqual(set(self.ckptr.load_checkpointables(1)), {'state'})

  @parameterized.parameters(False, True)
  def test_checkpoint_metadata_selects_its_step(self, asynchronous):
    self.save(1)
    self.save(2)
    latest = self.ckptr.latest
    assert latest is not None
    self.assertIs(self.run_delete(asynchronous, latest), True)
    self.assertEqual([c.step for c in self.ckptr.checkpoints], [1])

  @parameterized.parameters(None, True, -1, 1.0)
  def test_invalid_step(self, step):
    with self.assertRaises((TypeError, ValueError)):
      self.ckptr.delete(step)

  @parameterized.parameters(False, True)
  def test_missing(self, asynchronous):
    with self.assertRaises(ocp.training.errors.StepNotFoundError):
      self.run_delete(asynchronous, 9)
    self.assertIs(self.run_delete(asynchronous, 9, missing_ok=True), False)
    self.ckptr.wait()
    self.assertIsNone(self.ckptr.latest)

  def test_custom_name_and_offline_free_function(self):
    self.ckptr.close()
    self.ckptr = ocp.training.Checkpointer(
        self.directory,
        step_name_format=step_lib.standard_name_format(
            step_prefix='checkpoint', step_format_fixed_length=5
        ),
    )
    self.save(7)
    self.assertTrue((self.directory / 'checkpoint_00007').exists())
    self.ckptr.delete(7)
    self.save(8)
    self.ckptr.wait()
    ocp.delete(self.directory / 'checkpoint_00008')
    self.ckptr.reload()
    self.assertIsNone(self.ckptr.latest)

  def test_reload_after_offline_partial_failure(self):
    self.save(1)
    self.save(2)
    path = self.directory / '2'
    with mock.patch.object(
        execution, '_execute', side_effect=OSError('injected')
    ):
      with self.assertRaises(OSError):
        ocp.delete(path, checkpointable_name='optimizer')
    self.ckptr.reload()
    self.assert_latest_step(2)
    self.assertEqual(set(self.ckptr.load_checkpointables(2)), {'state'})
    ocp.delete(path, checkpointable_name='optimizer')
    self.ckptr.reload()
    self.assert_latest_step(2)
    self.save(3)

  def test_waits_for_pending_save(self):
    response = self.ckptr.save_async(1, self.state, force=True)
    assert response is not None
    self.ckptr.delete(1)
    response.result()
    self.assertFalse((self.directory / '1').exists())
    self.assertIsNone(self.ckptr.latest)

  def test_pending_whole_deletion_visibility_and_same_step_reuse(self):
    self.save(1)
    self.save(2)
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
        response = self.ckptr.delete_async(2)
        self.assertTrue(entered.wait(10))
        self.assert_latest_step(1)
        with self.assertRaises(ocp.training.errors.StepNotFoundError):
          self.ckptr.load(2)
        with self.assertRaises(concurrent.futures.TimeoutError):
          response.result(timeout=0)
      finally:
        release.set()
      # save must finish the pending deletion before reusing the same step.
      self.save(2)
    response.result()
    self.assert_latest_step(2)

  def test_failed_deletion_is_reported_once_then_retried(self):
    self.save(1)
    with mock.patch.object(
        execution, '_execute', side_effect=OSError('worker failed')
    ):
      response = self.ckptr.delete_async(1)
      with self.assertRaisesRegex(OSError, 'worker failed'):
        response.result()
    # Like v0, the step leaves the inventory when deletion starts.
    self.assertIsNone(self.ckptr.latest)
    with self.assertRaisesRegex(OSError, 'worker failed'):
      self.ckptr.wait()
    self.ckptr.wait()
    self.ckptr.reload()
    self.assert_latest_step(1)
    self.ckptr.delete(1)
    self.assertIsNone(self.ckptr.latest)
    self.assertFalse((self.directory / '1').exists())

  def test_discarded_response_is_completed_by_wait(self):
    self.save(1)
    self.ckptr.delete_async(1)
    self.ckptr.wait()
    self.assertIsNone(self.ckptr.latest)
    self.assertFalse((self.directory / '1').exists())

  @parameterized.parameters(False, True)
  def test_partial_failure_recovery(self, reopen):
    self.save(1)
    self.save(2)
    real_write = metadata.write_json

    async def write(path, value, **kwargs):
      if path.name == metadata_serialization.CHECKPOINT_METADATA_FILENAME:
        raise PermissionError('injected failure')
      return await real_write(path, value, **kwargs)

    with mock.patch.object(metadata, 'write_json', side_effect=write):
      response = self.ckptr.delete_async(2, checkpointable_name='optimizer')
      with self.assertRaises(PermissionError):
        self.ckptr.wait()
    self.assert_latest_step(2)
    if reopen:
      self.ckptr.close()
      self.ckptr = ocp.training.Checkpointer(self.directory)
      self.assert_latest_step(2)
    self.assertEqual(set(self.ckptr.load_checkpointables()), {'state'})
    with self.assertRaises(ocp.errors.DeletionInProgressError):
      self.ckptr.load_checkpointables(2, {'state': None, 'optimizer': None})
    self.ckptr.delete(2, checkpointable_name='optimizer')
    self.assert_latest_step(2)
    np.testing.assert_array_equal(
        typing.cast(dict[str, Any], self.ckptr.load(2))['weights'],
        self.state['weights'],
    )
    with self.assertRaises(PermissionError):
      response.result()


  @parameterized.parameters('state', 'optimizer')
  def test_partial_deletion_keeps_step_visible_while_worker_is_running(
      self, name
  ):
    self.save(1)
    self.save(2)
    release = threading.Event()
    entered = threading.Event()
    original = execution._execute

    async def execute(*args):
      entered.set()
      if not release.wait(10):
        raise TimeoutError('test release')
      return await original(*args)

    with mock.patch.object(execution, '_execute', side_effect=execute):
      try:
        response = self.ckptr.delete_async(2, checkpointable_name=name)
        self.assertTrue(entered.wait(10))
        self.assert_latest_step(2)
        self.assertEqual([c.step for c in self.ckptr.checkpoints], [1, 2])
        survivor = 'state' if name == 'optimizer' else 'optimizer'
        self.assertEqual(set(self.ckptr.load_checkpointables()), {survivor})
        meta = self.ckptr.checkpointables_metadata()
        assert meta.metadata is not None
        self.assertEqual(set(meta.metadata), {survivor})
        self.ckptr.load(2, checkpointable_name=survivor)
        with self.assertRaises(ocp.errors.DeletionInProgressError):
          self.ckptr.load(2, checkpointable_name=name)
        self.assertFalse(typing.cast(Any, response).done())
      finally:
        release.set()
      response.result()
    self.ckptr.wait()
    self.assert_latest_step(2)

  def test_corrupt_partial_record_rejects_reads(self):
    self.save(1)
    self.save(2)
    (self.directory / '2' / metadata.RECORD_FILENAME).write_text('{')
    self.ckptr.reload()
    self.assert_latest_step(2)
    with self.assertRaises(ocp.errors.DeletionRecoveryError):
      self.ckptr.load_checkpointables(2)
    self.ckptr.delete(2)
    self.assert_latest_step(1)


if __name__ == '__main__':
  absltest.main()
