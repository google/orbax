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
    runner = thread_utils.BackgroundThreadRunner[int](target())
    result = runner.result()
    end = time.time()
    self.assertEqual(result, 42)
    self.assertBetween(end - start, 1, 2)

    # Calling result again should return the same thing, without running again.
    start = time.time()
    result = runner.result()
    end = time.time()
    self.assertEqual(result, 42)
    self.assertLess(end - start, 0.1)

  def test_background_failure(self):
    async def target() -> int:
      raise ValueError("test")

    runner = thread_utils.BackgroundThreadRunner[int](target())
    with self.assertRaises(ValueError):
      runner.result()

  def test_timeout(self):
    async def target():
      await asyncio.sleep(10)

    runner = thread_utils.BackgroundThreadRunner[None](target())
    # Can be substituted for TimeoutError in 3.11+.
    with self.assertRaises(concurrent.futures.TimeoutError):
      runner.result(timeout=0.5)


if __name__ == "__main__":
  absltest.main()
