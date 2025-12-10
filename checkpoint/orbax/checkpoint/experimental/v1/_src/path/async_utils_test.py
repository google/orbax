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

import asyncio
import time
import unittest
from unittest import mock
from absl.testing import absltest
from etils import epath
from orbax.checkpoint._src.path import async_path
from orbax.checkpoint._src.path import atomicity
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.path import async_utils


class AsyncUtilsTest(absltest.TestCase, unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    super().setUp()

    orig_create_paths = atomicity._create_paths

    async def _sleep_and_create_paths(
        *args,
        **kwargs,
    ):
      await asyncio.sleep(1)
      return await orig_create_paths(*args, **kwargs)

    self.enter_context(
        mock.patch.object(
            atomicity, '_create_paths', new=_sleep_and_create_paths
        )
    )
    self.directory = epath.Path(self.create_tempdir().full_path)

  async def assertExists(self, path: epath.Path):
    self.assertTrue(await async_path.exists(path))

  async def assertNotExists(self, path: epath.Path):
    self.assertFalse(await async_path.exists(path))

  def assertBetween(self, a, b, c):
    self.assertGreater(b, a)
    self.assertGreater(c, b)

  async def test_async_mkdir(self):
    await context_lib.synchronize_next_operation_id()
    tmpdir = atomicity.AtomicRenameTemporaryPath(
        self.directory / 'tmp', self.directory / 'final'
    )
    start = time.time()
    p = async_utils.start_async_mkdir(tmpdir)
    await p.await_creation()
    self.assertBetween(1, time.time() - start, 2)
    await self.assertExists(self.directory / 'tmp')

  async def test_async_mkdir_with_subdirectories(self):
    await context_lib.synchronize_next_operation_id()
    tmpdir = atomicity.AtomicRenameTemporaryPath(
        self.directory / 'tmp', self.directory / 'final'
    )
    start = time.time()
    p = async_utils.start_async_mkdir(tmpdir, ['a', 'b'])
    await p.await_creation()
    self.assertBetween(1, time.time() - start, 2)
    await self.assertExists(self.directory / 'tmp')
    await self.assertExists(self.directory / 'tmp' / 'a')
    await self.assertExists(self.directory / 'tmp' / 'b')

  async def test_async_mkdir_sequential(self):
    tmpdir1 = atomicity.AtomicRenameTemporaryPath(
        self.directory / 'tmp1', self.directory / 'final1'
    )
    tmpdir2 = atomicity.AtomicRenameTemporaryPath(
        self.directory / 'tmp2', self.directory / 'final2'
    )
    start = time.time()
    await context_lib.synchronize_next_operation_id()
    p1 = async_utils.start_async_mkdir(tmpdir1)
    await context_lib.synchronize_next_operation_id()
    p2 = async_utils.start_async_mkdir(tmpdir2)
    await p1.await_creation()
    await p2.await_creation()
    # Awaiting sequentially does not take any longer than awaiting in parallel,
    # because the operations are already in progress.
    self.assertBetween(1, time.time() - start, 2)
    await self.assertExists(self.directory / 'tmp1')
    await self.assertExists(self.directory / 'tmp2')

  async def test_async_mkdir_parallel(self):
    tmpdir1 = atomicity.AtomicRenameTemporaryPath(
        self.directory / 'tmp1', self.directory / 'final1'
    )
    tmpdir2 = atomicity.AtomicRenameTemporaryPath(
        self.directory / 'tmp2', self.directory / 'final2'
    )
    start = time.time()
    await context_lib.synchronize_next_operation_id()
    p1 = async_utils.start_async_mkdir(tmpdir1)
    await context_lib.synchronize_next_operation_id()
    p2 = async_utils.start_async_mkdir(tmpdir2)
    await asyncio.gather(p1.await_creation(), p2.await_creation())
    self.assertBetween(1, time.time() - start, 2)
    await self.assertExists(self.directory / 'tmp1')
    await self.assertExists(self.directory / 'tmp2')

  async def test_async_mkdir_with_delayed_wait(self):
    await context_lib.synchronize_next_operation_id()
    tmpdir = atomicity.AtomicRenameTemporaryPath(
        self.directory / 'tmp', self.directory / 'final'
    )
    p = async_utils.start_async_mkdir(tmpdir)
    await self.assertNotExists(self.directory / 'tmp')
    await asyncio.sleep(1)
    await p.await_creation()
    await self.assertExists(self.directory / 'tmp')


if __name__ == '__main__':
  absltest.main()
