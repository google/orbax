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

import stat
import unittest
from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from orbax.checkpoint import options as options_lib
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.path import atomicity
from orbax.checkpoint._src.path import atomicity_types
from orbax.checkpoint._src.path import step as step_lib


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
      tmp_path.finalize(
      )
    test_utils.sync_global_processes('test_finalize')
    self.assertFalse(tmp_path.get().exists())
    self.assertTrue(path.exists())

  async def test_create_all(self):
    paths = [
        self.directory / 'ckpt1',
        self.directory / 'ckpt2',
    ]
    tmp_paths = [AtomicRenameTemporaryPath.from_final(path) for path in paths]
    await atomicity.create_all(tmp_paths)
    self.assertTrue(tmp_paths[0].get().exists())
    self.assertTrue(tmp_paths[1].get().exists())
    self.assertFalse(paths[0].exists())
    self.assertFalse(paths[1].exists())


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
    self.assertFalse((tmp_path.get() / step_lib._COMMIT_SUCCESS_FILE).exists())

  async def test_finalize(self):
    path = self.directory / 'ckpt'
    tmp_path = CommitFileTemporaryPath.from_final(path)
    await tmp_path.create()
    if multihost.process_index() == 0:
      tmp_path.finalize(
      )
    test_utils.sync_global_processes('test_finalize')
    self.assertTrue(tmp_path.get().exists())
    self.assertTrue(path.exists())
    self.assertTrue((path / step_lib._COMMIT_SUCCESS_FILE).exists())

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
      path.finalize(
      )



if __name__ == '__main__':
  absltest.main()
