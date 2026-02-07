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
import time
import unittest
from unittest import mock
from absl.testing import absltest
from etils import epath
from orbax.checkpoint._src.futures import future
from orbax.checkpoint._src.futures import synchronization
from orbax.checkpoint._src.path import async_path
from orbax.checkpoint._src.path import atomicity
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.path import async_utils

OperationIdGenerator = synchronization.OperationIdGenerator
AwaitableSignalsContract = future.AwaitableSignalsContract


class AsyncUtilsTest(absltest.TestCase, unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    super().setUp()

    orig_create_paths = async_utils._create_paths

    async def _sleep_and_create_paths(
        *args,
        **kwargs,
    ):
      await asyncio.sleep(1)
      return await orig_create_paths(*args, **kwargs)

    self.enter_context(
        mock.patch.object(
            async_utils,
            '_create_paths',
            new=_sleep_and_create_paths,
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
    p = async_utils.PathAwaitingCreation.build(
        tmpdir,
        [],
    )
    await p.create()
    await p.await_creation()
    self.assertBetween(1, time.time() - start, 2)
    await self.assertExists(self.directory / 'tmp')

  async def test_async_mkdir_with_subdirectories(self):
    await context_lib.synchronize_next_operation_id()
    tmpdir = atomicity.AtomicRenameTemporaryPath(
        self.directory / 'tmp', self.directory / 'final'
    )
    start = time.time()
    p = async_utils.PathAwaitingCreation.build(
        tmpdir,
        ['a', 'b'],
    )
    await p.create()
    await p.await_creation()
    self.assertBetween(1, time.time() - start, 2)
    await self.assertExists(self.directory / 'tmp')
    await self.assertExists(self.directory / 'tmp' / 'a')
    await self.assertExists(self.directory / 'tmp' / 'b')

  async def test_async_mkdir_sequential(self):
    await context_lib.synchronize_next_operation_id()
    tmpdir1 = atomicity.AtomicRenameTemporaryPath(
        self.directory / 'tmp1', self.directory / 'final1'
    )
    tmpdir2 = atomicity.AtomicRenameTemporaryPath(
        self.directory / 'tmp2', self.directory / 'final2'
    )
    start = time.time()
    p1 = async_utils.PathAwaitingCreation.build(
        tmpdir1,
        [],
    )
    p2 = async_utils.PathAwaitingCreation.build(
        tmpdir2,
        [],
    )
    await asyncio.gather(p1.create(), p2.create())
    await p1.await_creation()
    await p2.await_creation()
    # Awaiting sequentially does not take any longer than awaiting in parallel,
    # because the operations are already in progress.
    self.assertBetween(1, time.time() - start, 2)
    await self.assertExists(self.directory / 'tmp1')
    await self.assertExists(self.directory / 'tmp2')

  async def test_async_mkdir_parallel(self):
    await context_lib.synchronize_next_operation_id()
    tmpdir1 = atomicity.AtomicRenameTemporaryPath(
        self.directory / 'tmp1', self.directory / 'final1'
    )
    tmpdir2 = atomicity.AtomicRenameTemporaryPath(
        self.directory / 'tmp2', self.directory / 'final2'
    )
    start = time.time()
    p1 = async_utils.PathAwaitingCreation.build(
        tmpdir1,
        [],
    )
    p2 = async_utils.PathAwaitingCreation.build(
        tmpdir2,
        [],
    )
    await asyncio.gather(p1.create(), p2.create())
    await asyncio.gather(p1.await_creation(), p2.await_creation())
    self.assertBetween(1, time.time() - start, 2)
    await self.assertExists(self.directory / 'tmp1')
    await self.assertExists(self.directory / 'tmp2')

  async def test_async_mkdir_with_delayed_wait(self):
    await context_lib.synchronize_next_operation_id()
    tmpdir = atomicity.AtomicRenameTemporaryPath(
        self.directory / 'tmp', self.directory / 'final'
    )
    p = async_utils.PathAwaitingCreation.build(
        tmpdir,
        [],
    )
    t = asyncio.create_task(p.create())
    await self.assertNotExists(self.directory / 'tmp')
    start = time.time()
    await asyncio.sleep(1)
    await p.await_creation()
    # await_creation() should return immediately, since creation should have
    # started as soon as start_async_mkdir was called.
    self.assertBetween(1, time.time() - start, 2)
    await self.assertExists(self.directory / 'tmp')
    await t

  async def test_async_mkdir_with_no_wait(self):
    await context_lib.synchronize_next_operation_id()
    tmpdir = atomicity.AtomicRenameTemporaryPath(
        self.directory / 'tmp', self.directory / 'final'
    )
    p = async_utils.PathAwaitingCreation.build(
        tmpdir,
        [],
    )
    t = asyncio.create_task(p.create())
    await self.assertNotExists(self.directory / 'tmp')
    await asyncio.sleep(1.5)
    # It should be created even without explicitly waiting for it.
    await self.assertExists(self.directory / 'tmp')
    await t

  async def test_signals(self):
    await context_lib.synchronize_next_operation_id()
    operation_id = context_lib.get_context().operation_id()
    tmpdir = atomicity.AtomicRenameTemporaryPath(
        self.directory / 'tmp', self.directory / 'final'
    )
    expected_signals = [
        synchronization.HandlerAwaitableSignal.STEP_DIRECTORY_CREATION,
        synchronization.HandlerAwaitableSignal.ITEM_DIRECTORY_CREATION,
    ]
    p = async_utils.PathAwaitingCreation.build(
        tmpdir,
        [],
    )
    t = asyncio.create_task(p.create())
    actual_signals = (
        AwaitableSignalsContract.get_awaitable_signals_from_contract(
            operation_id
        )
    )
    self.assertEqual(len(actual_signals), len(expected_signals))
    for signal in actual_signals:
      self.assertIn(signal, expected_signals)
    with self.assertRaises(TimeoutError):
      future.wait_for_signals(
          expected_signals,
          timeout_secs=0,
          operation_id=operation_id,
      )
    start = time.time()
    await p.await_creation()
    future.wait_for_signals(
        expected_signals,
        timeout_secs=0,
        operation_id=operation_id,
    )
    self.assertBetween(1, time.time() - start, 2)
    await self.assertExists(self.directory / 'tmp')
    await t

  async def test_await_creation_without_create(self):
    await context_lib.synchronize_next_operation_id()
    tmpdir = atomicity.AtomicRenameTemporaryPath(
        self.directory / 'tmp', self.directory / 'final'
    )
    p = async_utils.PathAwaitingCreation.build(
        tmpdir,
        [],
    )
    await p.await_creation()
    await self.assertExists(self.directory / 'tmp')


if __name__ == '__main__':
  absltest.main()
