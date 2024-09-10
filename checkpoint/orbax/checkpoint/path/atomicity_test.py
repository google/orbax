# Copyright 2024 The Orbax Authors.
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

"""Tests for atomicity.py."""

import unittest
from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from orbax.checkpoint import multihost
from orbax.checkpoint import test_utils
from orbax.checkpoint.path import atomicity
from orbax.checkpoint.path import step as step_lib

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

  @parameterized.parameters(
      ('ckpt', f'ckpt{TMP_DIR_SUFFIX}5', True),
      ('ckpt', f'ckpt{TMP_DIR_SUFFIX}11001', True),
      ('state', f'state{TMP_DIR_SUFFIX}1', True),
      ('state', f'state{TMP_DIR_SUFFIX}s', True),
      ('state', f'state{TMP_DIR_SUFFIX}', False),
      ('foo', f'{TMP_DIR_SUFFIX}12', False),
      ('foo', f'f{TMP_DIR_SUFFIX}12', False),
      ('foo', 'foo-checkpoint-tmp-2', False),
  )
  def test_match(
      self, final_name: epath.Path, tmp_name: epath.Path, result: bool
  ):
    self.assertEqual(
        result,
        AtomicRenameTemporaryPath.match(
            self.directory / tmp_name, self.directory / final_name
        ),
    )

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
      tmp_path.finalize()
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

  @parameterized.parameters(
      ('ckpt', 'ckpt', True),
      ('ckpt', 'foo', False),
  )
  def test_match(
      self, final_name: epath.Path, tmp_name: epath.Path, result: bool
  ):
    self.assertEqual(
        result,
        CommitFileTemporaryPath.match(
            self.directory / tmp_name, self.directory / final_name
        ),
    )

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
      tmp_path.finalize()
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



if __name__ == '__main__':
  absltest.main()
