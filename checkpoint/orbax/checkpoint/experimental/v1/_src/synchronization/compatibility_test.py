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
from concurrent import futures
from unittest import mock
from absl.testing import absltest
from orbax.checkpoint.experimental.v1._src.synchronization import compatibility


class CompatibilityTest(absltest.TestCase):

  def test_futures_awaitable(self):
    called = False

    async def _coro():
      nonlocal called
      called = True

    f = futures.Future()
    f.set_result(None)
    awaitable = compatibility.FuturesAwaitable([f], _coro)
    self.assertEqual(awaitable.commit_futures, [f])

    async def _test():
      await awaitable

    asyncio.run(_test())
    self.assertTrue(called)

  def test_async_futures_success(self):
    f = futures.Future()
    f.set_result(42)

    async def _test():
      await compatibility.async_futures([f])

    asyncio.run(_test())

  def test_async_futures_cancellation(self):
    mock_future = mock.MagicMock()
    mock_future.result.side_effect = asyncio.CancelledError()

    async def _test():
      await compatibility.async_futures(
          [mock_future], operation_name='test_op'
      )

    asyncio.run(_test())
    mock_future.cancel.assert_called_once()


if __name__ == '__main__':
  absltest.main()
