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
import concurrent.futures
import time
from absl.testing import absltest
from absl.testing import parameterized
from orbax.checkpoint.experimental.v1._src.synchronization import thread_utils


class BackgroundThreadRunnerTest(parameterized.TestCase):

  def test_successful_execution(self):
    async def target() -> int:
      await asyncio.sleep(1)
      return 42

    start = time.time()
    runner = thread_utils.BackgroundThreadRunner[int]()
    runner.run("test", target())
    result = runner.result("test")
    end = time.time()
    self.assertEqual(result, 42)
    self.assertBetween(end - start, 1, 2)

    # Calling result again should return the same thing, without running again.
    start = time.time()
    result = runner.result("test")
    end = time.time()
    self.assertEqual(result, 42)
    self.assertLess(end - start, 0.1)

    runner.close()

  def test_background_failure(self):
    async def target() -> int:
      raise ValueError("test")

    runner = thread_utils.BackgroundThreadRunner[int]()
    runner.run("test", target())
    with self.assertRaises(ValueError):
      runner.result("test")

    runner.close()

  def test_timeout(self):
    async def target():
      await asyncio.sleep(10)

    runner = thread_utils.BackgroundThreadRunner[None]()
    runner.run("test", target())
    # Can be substituted for TimeoutError in 3.11+.
    with self.assertRaises(concurrent.futures.TimeoutError):
      runner.result("test", timeout=0.5)

    runner.close()

  def test_multiple_tasks(self):
    async def target_1():
      return 42

    async def target_2():
      return 43

    runner = thread_utils.BackgroundThreadRunner[int]()
    runner.run("test_1", target_1())
    runner.run("test_2", target_2())
    result_1 = runner.result("test_1")
    result_2 = runner.result("test_2")
    self.assertEqual(result_1, 42)
    self.assertEqual(result_2, 43)

    runner.close()

  def test_operation_after_close(self):
    async def target():
      return 42

    runner = thread_utils.BackgroundThreadRunner[int]()
    runner.close()
    runner.close()  # Close twice is OK.
    with self.assertRaises(ValueError):
      runner.run("test", target())


if __name__ == "__main__":
  absltest.main()
