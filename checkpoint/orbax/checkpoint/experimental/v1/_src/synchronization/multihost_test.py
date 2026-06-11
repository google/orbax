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

from absl.testing import parameterized
from etils import epath
import jax
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.path import async_path
from orbax.checkpoint._src.testing import multiprocess_test
from orbax.checkpoint.experimental.v1._src.synchronization import multihost


async def primary_host_sleep_and_mkdir(path: epath.Path, seconds: int = 2):
  if multihost.process_index() == 0:
    await asyncio.sleep(seconds)
    await async_path.mkdir(path, parents=False, exist_ok=False)


class MultihostTest(
    parameterized.TestCase,
    multiprocess_test.MultiProcessTest,
    unittest.IsolatedAsyncioTestCase,
):

  def setUp(self):
    super().setUp()
    self.assertEqual(jax.device_count(), 8)
    self.assertEqual(jax.process_count(), 4)
    self.assertEqual(jax.local_device_count(), 2)

    self.tmpdir = epath.Path(self.multiprocess_create_tempdir('multihost_test'))
    test_utils.sync_global_processes('setUp')

  def tearDown(self):
    test_utils.sync_global_processes('tearDown')
    super().tearDown()

  async def test_process_errors(self):
    if multihost.process_index() == 1:
      with self.assertRaises(ValueError):
        await multihost.sync_global_processes(
            'test_process_errors_1', operation_id='op', processes={0}
        )

  async def test_sync_global_processes(self):
    path = self.tmpdir / 'dummy'
    if multihost.process_index() == 0:
      await asyncio.sleep(2)
      await async_path.mkdir(path, parents=False, exist_ok=False)
    else:
      self.assertFalse(await async_path.exists(path))
    await multihost.sync_global_processes(
        'test_sync_global_processes', operation_id='op'
    )
    self.assertTrue(await async_path.exists(path))

  async def test_sync_global_processes_partially_async(self):
    path = self.tmpdir / 'dummy'
    if multihost.process_index() == 0:
      time.sleep(2)
      path.mkdir(parents=False, exist_ok=False)
    else:
      self.assertFalse(path.exists())
    await multihost.sync_global_processes(
        'test_sync_global_processes', operation_id='op'
    )
    self.assertTrue(path.exists())

  async def test_reused_barrier_key(self):
    await multihost.sync_global_processes(
        'test_reused_barrier_key', operation_id='op'
    )
    await multihost.sync_global_processes(
        'test_reused_barrier_key', operation_id='op'
    )

  async def test_interlocking_sequential(self):
    async def foo():
      await multihost.sync_global_processes(
          'test_interlocking', operation_id='op'
      )
      await asyncio.sleep(2)

    async def bar():
      await asyncio.sleep(2)
      await multihost.sync_global_processes(
          'test_interlocking', operation_id='op'
      )

    start = time.time()
    if multihost.process_index() == 0:
      await foo()
    else:
      await bar()
    await multihost.sync_global_processes(
        'test_interlocking_final', operation_id='op'
    )
    end = time.time()
    self.assertGreaterEqual(end - start, 4)

  async def test_interlocking_different_barrier_names(self):
    async def foo():
      await multihost.sync_global_processes(
          'test_interlocking', operation_id='op'
      )
      await asyncio.sleep(2)

    async def bar():
      await asyncio.sleep(2)
      await multihost.sync_global_processes(
          'test_interlocking', operation_id='op'
      )

    start = time.time()
    if multihost.process_index() == 0:
      await foo()
      # Need to unlock the other processes, otherwise they will get stuck.
      await multihost.sync_global_processes(
          'test_interlocking', operation_id='op'
      )
    else:
      # Unlock the other process before proceeding.
      await multihost.sync_global_processes(
          'test_interlocking', operation_id='op'
      )
      await bar()
    await multihost.sync_global_processes(
        'test_interlocking_final', operation_id='op'
    )
    end = time.time()
    self.assertLess(end - start, 3)

  async def test_not_all_processes_arrived_at_barrier(self):
    if multihost.process_index() == 0:
      with self.assertRaises(TimeoutError):
        await multihost.sync_global_processes(
            'test_timeout', timeout=2, operation_id='op'
        )

  @parameterized.parameters(
      (1,),
      (5,),
  )
  async def test_sync_global_processes_background_tasks(self, num_tasks):
    paths = [self.tmpdir / f'dummy_{t}' for t in range(num_tasks)]

    async def background_fn(t):
      path = paths[t]
      await primary_host_sleep_and_mkdir(path)
      await multihost.sync_global_processes(
          f'test_sync_global_processes_{t}', operation_id='op'
      )

    async def background_fns():
      return await asyncio.gather(*[background_fn(t) for t in range(num_tasks)])

    task = asyncio.create_task(background_fns())
    exists = await asyncio.gather(*[async_path.exists(path) for path in paths])
    self.assertFalse(all(exists))
    await task
    exists = await asyncio.gather(*[async_path.exists(path) for path in paths])
    self.assertTrue(all(exists))

  async def test_sync_global_processes_background_tasks_sequential(self):
    async def fn1():
      await primary_host_sleep_and_mkdir(self.tmpdir / 'dummy1a')
      await multihost.sync_global_processes(
          'test_sync_global_processes_1', operation_id='op'
      )
      await primary_host_sleep_and_mkdir(self.tmpdir / 'dummy1b')
      await multihost.sync_global_processes(
          'test_sync_global_processes_1', operation_id='op'
      )

    async def fn2():
      path = self.tmpdir / 'dummy2'
      await primary_host_sleep_and_mkdir(path)
      await multihost.sync_global_processes(
          'test_sync_global_processes_2', operation_id='op'
      )

    async def background_fns():
      return await asyncio.gather(*[fn1(), fn2()])

    task = asyncio.create_task(background_fns())
    self.assertFalse(await async_path.exists(self.tmpdir / 'dummy1a'))
    self.assertFalse(await async_path.exists(self.tmpdir / 'dummy1b'))
    self.assertFalse(await async_path.exists(self.tmpdir / 'dummy2'))
    await asyncio.sleep(2.5)
    self.assertTrue(await async_path.exists(self.tmpdir / 'dummy1a'))
    self.assertFalse(await async_path.exists(self.tmpdir / 'dummy1b'))
    self.assertTrue(await async_path.exists(self.tmpdir / 'dummy2'))
    await asyncio.sleep(2.5)
    self.assertTrue(await async_path.exists(self.tmpdir / 'dummy1a'))
    self.assertTrue(await async_path.exists(self.tmpdir / 'dummy1b'))
    self.assertTrue(await async_path.exists(self.tmpdir / 'dummy2'))
    await task

  async def test_sequential_execution(self):

    async def sleep_and_sync(i):
      if multihost.process_index() == 0:
        await asyncio.sleep(2)
      await multihost.sync_global_processes(
          f'test_sleep_and_sync_{i}', operation_id='op'
      )

    start = time.time()
    await sleep_and_sync(0)
    await sleep_and_sync(1)
    end = time.time()
    self.assertGreaterEqual(end - start, 4)

  async def test_parallel_execution(self):

    async def sleep_and_sync(i):
      if multihost.process_index() == 0:
        await asyncio.sleep(2)
      await multihost.sync_global_processes(
          f'test_sleep_and_sync_{i}', operation_id='op'
      )

    start = time.time()
    await asyncio.gather(*[sleep_and_sync(0), sleep_and_sync(1)])
    end = time.time()
    self.assertLess(end - start, 3)

  async def test_sync_global_processes_partial(self):
    participating_processes = {0, 2}
    primary_process = 0
    non_primary_process = 1

    directory = self.tmpdir / 'testdir'
    if multihost.process_index() == primary_process:
      directory.mkdir(parents=False, exist_ok=False)
    test_utils.sync_global_processes('test_sync_global_processes_partial_setup')

    if multihost.process_index() == primary_process:
      time.sleep(2)
      (directory / 'dummy').mkdir(parents=False, exist_ok=False)
    if multihost.process_index() in participating_processes:
      await multihost.sync_global_processes(
          'test_sync_global_processes_partial_one',
          processes=participating_processes,
          operation_id='op',
      )
    if multihost.process_index() in participating_processes:
      self.assertTrue((directory / 'dummy').exists())
    else:
      self.assertFalse((directory / 'dummy').exists())

    if multihost.process_index() == primary_process:
      time.sleep(2)
      (directory / 'foo').mkdir(parents=False, exist_ok=False)
    if multihost.process_index() in participating_processes:
      await multihost.sync_global_processes(
          'test_sync_global_processes_partial_two',
          processes=participating_processes,
          operation_id='op',
      )
    if multihost.process_index() in participating_processes:
      self.assertTrue((directory / 'foo').exists())
    else:
      self.assertFalse((directory / 'foo').exists())

    await multihost.sync_global_processes(
        'test_sync_global_processes_partial_all', operation_id='op'
    )
    # If non-primary processes get past the above barrier without waiting for
    # all, then an error would happen for the primary process when trying to
    # create subdirectories.
    if multihost.process_index() == non_primary_process:
      directory.rmtree()

  async def test_different_barriers(self):
    slice1 = {0, 2}
    slice2 = {1, 3}
    primary_processes = [0, 1]

    if multihost.process_index() in primary_processes:
      # Don't sleep for slice1, but do sleep for slice2, so that when slice1
      # finishes waiting at the barrier, one file exists but the other does
      # not.
      time.sleep(3 * multihost.process_index())
      (self.tmpdir / f'dummy_{multihost.process_index()}').mkdir(
          parents=False, exist_ok=False
      )

    if multihost.process_index() in slice1:
      await multihost.sync_global_processes(
          'test_different_barriers_slice1',
          operation_id='op',
          processes=slice1,
      )
    else:
      await multihost.sync_global_processes(
          'test_different_barriers_slice2',
          operation_id='op',
          processes=slice2,
      )
    if multihost.process_index() in slice1:
      self.assertTrue((self.tmpdir / 'dummy_0').exists())
      self.assertFalse((self.tmpdir / 'dummy_1').exists())
    else:
      self.assertTrue((self.tmpdir / 'dummy_0').exists())
      self.assertTrue((self.tmpdir / 'dummy_1').exists())


if __name__ == '__main__':
  multiprocess_test.main()
