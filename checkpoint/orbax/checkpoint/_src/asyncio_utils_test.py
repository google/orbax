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

"""Tests for asyncio_utils module."""

import asyncio
from concurrent import futures
import functools
import threading
import timeit

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
from orbax.checkpoint._src import asyncio_utils


partial = functools.partial


async def one():
  return 1


async def add(a_coro_fn, b_coro_fn):
  a = await a_coro_fn()
  b = await b_coro_fn()
  return a + b


async def nested(a_coro_fn):
  await a_coro_fn()


async def raise_error():
  raise ValueError("test error")


async def with_run_sync(a_coro_fn):
  x = asyncio_utils.run_sync(a_coro_fn())
  y = asyncio_utils.run_sync(a_coro_fn())
  z = asyncio_utils.run_sync(a_coro_fn())
  return f"{x}{y}{z}"


class AsyncioUtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ["basic", one],
      ["nested_1_level", partial(add, one, one)],
      ["nested_2_level", partial(add, partial(add, one, one), one)],
      [
          "nested_3_level",
          partial(add, partial(add, partial(add, one, one), one), one),
      ],
      [
          "nested_4_level",
          partial(
              add,
              partial(add, partial(add, partial(add, one, one), one), one),
              one,
          ),
      ],
      [
          "nested_5_level",
          partial(
              add,
              partial(
                  add,
                  partial(add, partial(add, partial(add, one, one), one), one),
                  one,
              ),
              one,
          ),
      ],
      [
          "nested_6_level",
          partial(
              add,
              partial(
                  add,
                  partial(
                      add,
                      partial(
                          add, partial(add, partial(add, one, one), one), one
                      ),
                      one,
                  ),
                  one,
              ),
              one,
          ),
      ],
  )
  def test_run_sync_basic(self, coro_fn):
    self.assertEqual(
        asyncio.run(coro_fn()),
        asyncio_utils.run_sync(coro_fn()),
    )

  @parameterized.named_parameters(
      ["basic", raise_error],
      ["nested_1_level", partial(nested, raise_error)],
      ["nested_2_level", partial(nested, partial(nested, raise_error))],
      [
          "nested_3_level",
          partial(nested, partial(nested, partial(nested, raise_error))),
      ],
      [
          "nested_4_level",
          partial(
              nested,
              partial(nested, partial(nested, partial(nested, raise_error))),
          ),
      ],
      [
          "nested_5_level",
          partial(
              nested,
              partial(
                  nested,
                  partial(
                      nested, partial(nested, partial(nested, raise_error))
                  ),
              ),
          ),
      ],
      [
          "nested_6_level",
          partial(
              nested,
              partial(
                  nested,
                  partial(
                      nested,
                      partial(
                          nested, partial(nested, partial(nested, raise_error))
                      ),
                  ),
              ),
          ),
      ],
  )
  def test_run_sync_raising_error(self, coro_fn):
    with self.assertRaisesRegex(ValueError, "test error"):
      asyncio.run(coro_fn())
    with self.assertRaisesRegex(ValueError, "test error"):
      asyncio_utils.run_sync(coro_fn())

  @parameterized.named_parameters(
      ["basic", partial(with_run_sync, one)],
      ["nested_1_level", partial(with_run_sync, partial(with_run_sync, one))],
      [
          "nested_2_level",
          partial(
              with_run_sync, partial(with_run_sync, partial(with_run_sync, one))
          ),
      ],
      [
          "nested_3_level",
          partial(
              with_run_sync,
              partial(
                  with_run_sync,
                  partial(with_run_sync, partial(with_run_sync, one)),
              ),
          ),
      ],
      [
          "nested_4_level",
          partial(
              with_run_sync,
              partial(
                  with_run_sync,
                  partial(
                      with_run_sync,
                      partial(with_run_sync, partial(with_run_sync, one)),
                  ),
              ),
          ),
      ],
      [
          "nested_5_level",
          partial(
              with_run_sync,
              partial(
                  with_run_sync,
                  partial(
                      with_run_sync,
                      partial(
                          with_run_sync,
                          partial(with_run_sync, partial(with_run_sync, one)),
                      ),
                  ),
              ),
          ),
      ],
      [
          "nested_6_level",
          partial(
              with_run_sync,
              partial(
                  with_run_sync,
                  partial(
                      with_run_sync,
                      partial(
                          with_run_sync,
                          partial(
                              with_run_sync,
                              partial(
                                  with_run_sync, partial(with_run_sync, one)
                              ),
                          ),
                      ),
                  ),
              ),
          ),
      ],
  )
  def test_run_sync_nested(self, coro_fn):
    self.assertEqual(
        asyncio.run(coro_fn()),
        asyncio_utils.run_sync(coro_fn()),
    )

  def test_run_nested(self):
    async def with_run(a_coro_fn):
      return asyncio.run(a_coro_fn())

    with self.assertRaisesWithLiteralMatch(
        RuntimeError,
        "asyncio.run() cannot be called from a running event loop",
    ):
      asyncio.run(with_run(one))

    with self.assertRaisesWithLiteralMatch(
        RuntimeError,
        "asyncio.run() cannot be called from a running event loop",
    ):
      asyncio_utils.run_sync(with_run(one))

  def test_cancellable_success(self):
    async def foo():
      return 42

    result = asyncio_utils.run_sync(
        asyncio_utils.cancellable(foo(), "cancelled message")
    )
    self.assertEqual(result, 42)

  def test_cancellable_cancel_reraise(self):
    async def cancel_task():
      raise asyncio.CancelledError()

    with self.assertRaises(asyncio.CancelledError):
      asyncio_utils.run_sync(
          asyncio_utils.cancellable(
              cancel_task(), "cancelled message", reraise=True
          )
      )

  def test_cancellable_cancel_suppress(self):
    async def cancel_task():
      raise asyncio.CancelledError()

    result = asyncio_utils.run_sync(
        asyncio_utils.cancellable(
            cancel_task(),
            "cancelled message %s",
            process_index=1,
            reraise=False,
        )
    )
    self.assertIsNone(result)

  def test_cancellable_cancel_suppress_no_process_index(self):
    async def cancel_task():
      raise asyncio.CancelledError()

    result = asyncio_utils.run_sync(
        asyncio_utils.cancellable(
            cancel_task(), "cancelled message", reraise=False
        )
    )
    self.assertIsNone(result)

  @absltest.skip("benchmark asyncio.get_running_loop().")
  def test_benchmark_get_running_loop(self):
    def _is_event_loop_present():
      try:
        asyncio.get_running_loop()
        return True
      except RuntimeError:
        return False

    async def _async_is_event_loop_present():
      _is_event_loop_present()

    def _with_event_loop_present():
      asyncio.run(_async_is_event_loop_present())

    nel = timeit.timeit(_is_event_loop_present, number=1000)  # ~0.001s
    wel = timeit.timeit(_with_event_loop_present, number=1000)  # ~0.182s
    logging.info("time: no_event_loop=%s, with_event_loop=%s", nel, wel)

  @absltest.skip("benchmark asyncio_utils.run_sync.")
  def test_benchmark_run_sync(self):
    async def _test():
      return partial(
          add,
          partial(
              add,
              partial(
                  add,
                  partial(add, partial(add, partial(add, one, one), one), one),
                  one,
              ),
              one,
          ),
          one,
      )()

    number = 10000
    run_sync_time = timeit.timeit(
        lambda: asyncio_utils.run_sync(_test()),
        number=number,
    )  # ~1.5503s
    logging.info("time: run_sync_time=%s", run_sync_time)


class AsyncRunnerTest(absltest.TestCase):

  _TIMEOUT = 2

  def setUp(self):
    """Instantiate the runner before each test."""
    super().setUp()
    self.runner = asyncio_utils.AsyncRunner()

  def tearDown(self):
    """Ensure the runner is shut down after each test to free resources."""
    super().tearDown()
    self.runner.shutdown()

  def test_run_coroutine_success(self):
    """Test that a coroutine executes and returns the correct result."""
    async def simple_coro(value):
      await asyncio.sleep(0.1)
      return value * 2

    future = self.runner.run_coroutine(simple_coro(5))
    result = future.result(timeout=self._TIMEOUT)
    self.assertEqual(result, 10)
    self.assertTrue(future.done())

  def test_run_coroutine_exception(self):
    """Test that exceptions raised inside the coroutine propagate to the future."""
    async def failing_coro():
      await asyncio.sleep(0.1)
      raise ValueError("Something went wrong inside the async task!")

    future = self.runner.run_coroutine(failing_coro())

    # Assert that calling result() raises the embedded exception
    with self.assertRaises(ValueError):
      future.result(timeout=self._TIMEOUT)

    # Alternatively, check the exception object directly
    self.assertIsInstance(future.exception(timeout=self._TIMEOUT), ValueError)

  def test_multiple_concurrent_coroutines(self):
    """Test submitting multiple coroutines concurrently."""
    async def quick_coro(x):
      return x

    fs = [self.runner.run_coroutine(quick_coro(i)) for i in range(10)]

    # Gather results
    results = [f.result(timeout=self._TIMEOUT) for f in fs]

    self.assertEqual(results, list(range(10)))

  def test_shutdown_waits_for_tasks_to_complete(self):
    """Test that shutdown() stops the event loop and waits for tasks to finish."""
    task_can_complete = threading.Event()

    async def long_running_coro() -> None:
      await asyncio.to_thread(task_can_complete.wait)

    future = self.runner.run_coroutine(long_running_coro())
    self.assertFalse(future.done())

    # Verify by that the shutdown call blocks until we allow the coroutine
    # to complete.
    executor = self.enter_context(futures.ThreadPoolExecutor(max_workers=1))
    shutdown_future = executor.submit(self.runner.shutdown)
    self.assertFalse(shutdown_future.done())
    # Allow the coroutine to complete.
    task_can_complete.set()
    shutdown_future.result(timeout=self._TIMEOUT)
    # Coroutine should now be done.
    self.assertTrue(future.done())
    self.assertIsNone(future.result(timeout=self._TIMEOUT))

  def test_shutdown_non_blocking(self):
    """Test that shutdown(wait=False) does not wait for tasks and thread."""
    task_can_complete = threading.Event()

    async def long_running_coro() -> None:
      await asyncio.to_thread(task_can_complete.wait)

    future = self.runner.run_coroutine(long_running_coro())
    self.assertFalse(future.done())

    # Verify by that the shutdown call returns immediately.
    executor = self.enter_context(futures.ThreadPoolExecutor(max_workers=1))
    shutdown_future = executor.submit(self.runner.shutdown, wait=False)
    shutdown_future.result(timeout=self._TIMEOUT)
    # Allow the coroutine to complete, so that the other thread where
    # `task_can_complete.wait` is run (not the event loop thread) can exit.
    task_can_complete.set()

  def test_submit_after_shutdown(self):
    """Test that submitting a coroutine after shutdown raises RuntimeError."""
    self.runner.shutdown()

    async def dummy():
      pass

    with self.assertRaises(RuntimeError):
      self.runner.run_coroutine(dummy())

  def test_shutdown_duration(self):
    """Test that measures how long shutdown takes with and without active tasks."""
    # 1. Shutdown with no active tasks.
    runner = asyncio_utils.AsyncRunner()
    start_time = timeit.default_timer()
    runner.shutdown()
    duration = timeit.default_timer() - start_time
    logging.info("Shutdown with no active tasks took %f seconds.", duration)
    # Assert it is relatively fast (less than 0.5s).
    self.assertLess(duration, 0.5)

    # 2. Shutdown with a running task that takes specific time.
    runner = asyncio_utils.AsyncRunner()
    task_duration = 0.5

    async def sleeping_coro():
      await asyncio.sleep(task_duration)

    runner.run_coroutine(sleeping_coro())
    start_time = timeit.default_timer()
    runner.shutdown()
    duration = timeit.default_timer() - start_time
    logging.info(
        "Shutdown with a %f-second task took %f seconds.",
        task_duration,
        duration,
    )
    # The shutdown should block until the sleeping task is finished.
    self.assertGreaterEqual(duration, task_duration - 0.05)

  def test_shutdown_cancel_duration(self):
    """Test that measures how fast shutdown is when cancel=True."""
    runner = asyncio_utils.AsyncRunner()
    task_duration = 0.5

    async def sleeping_coro():
      await asyncio.sleep(task_duration)

    runner.run_coroutine(sleeping_coro())
    start_time = timeit.default_timer()
    # Call shutdown with cancel=True. It should cancel the task instantly.
    runner.shutdown(cancel=True)
    duration = timeit.default_timer() - start_time
    logging.info(
        "Shutdown with cancel=True and a %f-second task took %f seconds.",
        task_duration,
        duration,
    )
    # The shutdown should return almost instantly and definitely in less than
    # 1 second.
    self.assertLess(duration, 1.0)


if __name__ == "__main__":
  absltest.main()

