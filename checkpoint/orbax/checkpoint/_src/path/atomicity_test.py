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

import asyncio
import concurrent.futures
import stat
import unittest
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from orbax.checkpoint import options as options_lib
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.path import atomicity
from orbax.checkpoint._src.path import atomicity_defaults
from orbax.checkpoint._src.path import atomicity_types
from orbax.checkpoint._src.path import temporary_paths
from orbax.checkpoint._src.path.snapshot import snapshot as snapshot_lib


AtomicRenameTemporaryPath = atomicity.AtomicRenameTemporaryPath
CommitFileTemporaryPath = atomicity.CommitFileTemporaryPath
TMP_DIR_SUFFIX = atomicity.TMP_DIR_SUFFIX


class AtomicRenameTemporaryPathTest(
    parameterized.TestCase,
    unittest.IsolatedAsyncioTestCase,
):

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(self.create_tempdir('ckpt').full_path)

  def test_from_final(self):
    path = self.directory / 'ckpt'
    tmp_path = AtomicRenameTemporaryPath.from_final(path)
    self.assertIn(f'ckpt{TMP_DIR_SUFFIX}', tmp_path.get().as_posix())

  async def test_create(self):
    path = self.directory / 'ckpt'
    tmp_path = AtomicRenameTemporaryPath.from_final(path)
    await tmp_path.create()
    self.assertTrue(tmp_path.get().exists())
    self.assertFalse(path.exists())

  async def test_finalize(self):
    path = self.directory / 'ckpt'
    tmp_path = AtomicRenameTemporaryPath.from_final(path)
    await tmp_path.create()
    if multihost.process_index() == 0:
      await tmp_path.finalize()
    test_utils.sync_global_processes('test_finalize')
    self.assertFalse(tmp_path.get().exists())
    self.assertTrue(path.exists())

  async def test_create_all(self):
    paths = [
        self.directory / 'ckpt1',
        self.directory / 'ckpt2',
    ]
    tmp_paths = [AtomicRenameTemporaryPath.from_final(path) for path in paths]
    with mock.patch(
        'jax.monitoring.record_event_duration_secs'
    ) as mock_record_duration:
      await atomicity.create_all(tmp_paths)
      recorded_metrics = [
          call[0][0] for call in mock_record_duration.call_args_list
      ]
      self.assertIn(
          '/jax/orbax/write/blocking_directory_creation_secs', recorded_metrics
      )
    self.assertTrue(tmp_paths[0].get().exists())
    self.assertTrue(tmp_paths[1].get().exists())
    self.assertFalse(paths[0].exists())
    self.assertFalse(paths[1].exists())

  async def test_create_paths_records_background_directory_creation(self):
    paths = [
        self.directory / 'ckpt1',
        self.directory / 'ckpt2',
    ]
    tmp_paths = [AtomicRenameTemporaryPath.from_final(path) for path in paths]
    with mock.patch(
        'jax.monitoring.record_event_duration_secs'
    ) as mock_record_duration:
      await atomicity._create_paths(  # pylint: disable=protected-access
          tmp_paths, subdirectories=['subdir']
      )
      recorded_metrics = [
          call[0][0] for call in mock_record_duration.call_args_list
      ]
      self.assertIn(
          '/jax/orbax/write/background_directory_creation_secs',
          recorded_metrics,
      )
      # Background creation must not be attributed to the blocking phase.
      self.assertNotIn(
          '/jax/orbax/write/blocking_directory_creation_secs',
          recorded_metrics,
      )
    for tmp_path in tmp_paths:
      self.assertTrue(tmp_path.get().exists())
      self.assertTrue((tmp_path.get() / 'subdir').exists())
    self.assertFalse(paths[0].exists())
    self.assertFalse(paths[1].exists())

  async def test_finalize_with_snapshot(self):
    path = self.directory / 'ckpt'
    path.mkdir(parents=True)

    (path / 'foo').write_text('bar')
    tmp_path = AtomicRenameTemporaryPath.from_final(
        path, snapshot_type=snapshot_lib.SnapshotType.IN_PLACE
    )
    self.assertIsNotNone(tmp_path._snapshot)

    await tmp_path.create()
    # pylint: disable=protected-access
    self.assertEqual(tmp_path._snapshot._source, path)
    self.assertEqual(tmp_path._snapshot._snapshot, tmp_path.get())
    # pylint: enable=protected-access

    (tmp_path.get() / 'foo').write_text('new bar')
    self.assertEqual((path / 'foo').read_text(), 'bar')
    self.assertEqual((tmp_path.get() / 'foo').read_text(), 'new bar')

    if multihost.process_index() == 0:
      await tmp_path.finalize()
    test_utils.sync_global_processes('test_finalize_with_snapshot')
    self.assertFalse(tmp_path.get().exists())
    self.assertTrue(path.exists())
    self.assertEqual((path / 'foo').read_text(), 'new bar')

  async def test_create_all_async_executes_pre_create_fn_before_create(self):
    path = self.directory / 'ckpt_pre_create'
    tmp_path = AtomicRenameTemporaryPath.from_final(path)
    events = []
    orig_create = tmp_path.create

    async def pre_create():
      events.append(('pre_create', tmp_path.get().exists()))

    async def wrapped_create():
      events.append(('create', tmp_path.get().exists()))
      return await orig_create()

    with mock.patch.object(tmp_path, 'create', side_effect=wrapped_create):
      commit_future = atomicity.create_all_async(
          [tmp_path],
          completion_signals=[],
          pre_create_fn=pre_create,
      )
      await asyncio.to_thread(commit_future.result)

    if multihost.is_primary_host(0):
      self.assertEqual(events, [('pre_create', False), ('create', False)])
      self.assertTrue(tmp_path.get().exists())


class CommitFileTemporaryPathTest(
    parameterized.TestCase,
    unittest.IsolatedAsyncioTestCase,
):

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(self.create_tempdir('ckpt').full_path)

  def test_from_final(self):
    path = self.directory / 'ckpt'
    tmp_path = CommitFileTemporaryPath.from_final(path)
    self.assertEqual(path, tmp_path.get())

  async def test_create(self):
    path = self.directory / 'ckpt'
    tmp_path = CommitFileTemporaryPath.from_final(path)
    await tmp_path.create()
    self.assertTrue(tmp_path.get().exists())
    self.assertFalse((tmp_path.get() / atomicity.COMMIT_SUCCESS_FILE).exists())

  async def test_finalize(self):
    path = self.directory / 'ckpt'
    tmp_path = CommitFileTemporaryPath.from_final(path)
    await tmp_path.create()
    if multihost.process_index() == 0:
      await tmp_path.finalize()
    test_utils.sync_global_processes('test_finalize')
    self.assertTrue(tmp_path.get().exists())
    self.assertTrue(path.exists())
    self.assertTrue((path / atomicity.COMMIT_SUCCESS_FILE).exists())

  async def test_create_all(self):
    paths = [
        self.directory / 'ckpt1',
        self.directory / 'ckpt2',
    ]
    tmp_paths = [CommitFileTemporaryPath.from_final(path) for path in paths]
    await atomicity.create_all(tmp_paths)
    self.assertTrue(tmp_paths[0].get().exists())
    self.assertTrue(tmp_paths[1].get().exists())
    self.assertTrue(paths[0].exists())
    self.assertTrue(paths[1].exists())

  async def test_finalize_with_snapshot_raises(self):
    path = self.directory / 'ckpt'
    with self.assertRaises(ValueError):
      CommitFileTemporaryPath.from_final(
          path, snapshot_type=snapshot_lib.SnapshotType.IN_PLACE
      )


class ReadOnlyTemporaryPathTest(
    parameterized.TestCase,
    unittest.IsolatedAsyncioTestCase,
):

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(self.create_tempdir().full_path)

  @parameterized.named_parameters(
      {
          'testcase_name': 'atomic_rename',
          'temporary_path_cls': AtomicRenameTemporaryPath,
      },
      {
          'testcase_name': 'commit_file',
          'temporary_path_cls': CommitFileTemporaryPath,
      },
  )
  def test_serialization(self, temporary_path_cls):
    path = self.directory / 'ckpt'
    tmp_path = temporary_path_cls.from_final(path)
    readonly_tmp_path = atomicity.ReadOnlyTemporaryPath.from_paths(
        temporary_path=tmp_path.get(), final_path=path
    )
    deserialized = atomicity.ReadOnlyTemporaryPath.from_bytes(
        readonly_tmp_path.to_bytes()
    )

    self.assertEqual(tmp_path.get(), deserialized.get())
    self.assertEqual(tmp_path.get_final(), deserialized.get_final())

  def test_from_final_raises(self):
    with self.assertRaises(NotImplementedError):
      atomicity.ReadOnlyTemporaryPath.from_final(epath.Path('/path/to/ckpt'))

  async def test_create_raises(self):
    path = atomicity.ReadOnlyTemporaryPath(
        temporary_path=epath.Path('/path/to/ckpt.orbax-checkpoint-tmp'),
        final_path=epath.Path('/path/to/ckpt'),
    )
    with self.assertRaises(NotImplementedError):
      await path.create()

  async def test_finalize_raises(self):
    path = atomicity.ReadOnlyTemporaryPath(
        temporary_path=epath.Path('/path/to/ckpt.orbax-checkpoint-tmp'),
        final_path=epath.Path('/path/to/ckpt'),
    )
    with self.assertRaises(NotImplementedError):
      await path.finalize()


class DeferredPathTest(absltest.TestCase):

  def test_set_and_get_path(self):
    dp = atomicity.DeferredPath()
    test_path = epath.Path('/test/path')
    dp.set_path(test_path)
    self.assertEqual(dp.path, test_path)

  def test_path_before_set_raises(self):
    dp = atomicity.DeferredPath()
    with self.assertRaises(ValueError):
      _ = dp.path

  def test_await_creation(self):
    dp = atomicity.DeferredPath()
    test_path = epath.Path('/test/path')
    dp.set_path(test_path)
    result = asyncio.run(dp.await_creation())
    self.assertEqual(result, test_path)

  def test_set_path_twice_raises(self):
    dp = atomicity.DeferredPath()
    dp.set_path(epath.Path('/first'))
    with self.assertRaises(concurrent.futures.InvalidStateError):
      dp.set_path(epath.Path('/second'))



if __name__ == '__main__':
  absltest.main()
