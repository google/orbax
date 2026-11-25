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
import contextlib
import functools
import gc
import inspect
import threading
import timeit
import warnings

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import anyio
import anyio.to_thread
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


class ThreadAnnotationTest(absltest.TestCase):
  """`run_sync` must leave the calling thread as it found it.

  `run_sync` annotates `threading.current_thread()` with `loop` and
  `main_task`. Those attribute names are not private to Orbax: anyio's worker
  threads keep their own `loop` there and read it back after the callable
  returns. Clearing ours used to clear theirs, killing the worker before it
  resolved its future and hanging the awaiting coroutine forever.
  """

  def test_absent_attributes_stay_absent(self):
    thread = threading.current_thread()
    for name in ("loop", "main_task"):
      if hasattr(thread, name):
        delattr(thread, name)

    asyncio_utils.run_sync(one())

    self.assertFalse(hasattr(thread, "loop"))
    self.assertFalse(hasattr(thread, "main_task"))

  def test_pre_existing_attributes_are_restored(self):
    thread = threading.current_thread()
    sentinel_loop = object()
    sentinel_task = object()
    setattr(thread, "loop", sentinel_loop)
    setattr(thread, "main_task", sentinel_task)
    self.addCleanup(lambda: delattr(thread, "loop"))
    self.addCleanup(lambda: delattr(thread, "main_task"))

    asyncio_utils.run_sync(one())

    self.assertIs(getattr(thread, "loop"), sentinel_loop)
    self.assertIs(getattr(thread, "main_task"), sentinel_task)

  def test_runs_inside_an_anyio_worker_thread(self):
    """End-to-end guard: this used to hang forever rather than fail."""

    def _blocking() -> int:
      return asyncio_utils.run_sync(one())

    async def _main() -> int:
      with anyio.move_on_after(30):
        return await anyio.to_thread.run_sync(_blocking)
      raise AssertionError("anyio.to_thread.run_sync did not return")

    self.assertEqual(asyncio.run(_main()), 1)


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


def _unawaited_warnings_for(
    caught: list[warnings.WarningMessage], coro_name: str
) -> list[warnings.WarningMessage]:
  return [
      w
      for w in caught
      if issubclass(w.category, RuntimeWarning) and coro_name in str(w.message)
  ]


class GatherBatchesTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("empty_list", lambda: []),
      ("empty_generator", lambda: (x for x in ())),
  )
  def test_empty_input(self, make_awaitables):
    async def _run():
      return [
          batch
          async for batch in asyncio_utils.gather_batches(make_awaitables())
      ]

    self.assertEqual(asyncio.run(_run()), [])

  def test_positional_ordering_and_partial_final_batch(self):
    async def _run():
      gate_0 = asyncio.Event()
      gate_2 = asyncio.Event()

      async def _item(idx: int) -> int:
        if idx == 0:
          await gate_0.wait()
        elif idx == 1:
          gate_0.set()
        elif idx == 2:
          await gate_2.wait()
        elif idx == 3:
          gate_2.set()
        return idx

      return [
          batch
          async for batch in asyncio_utils.gather_batches(
              (_item(i) for i in range(5)), batch_size=2
          )
      ]

    self.assertEqual(asyncio.run(_run()), [[0, 1], [2, 3], [4]])

  def test_concurrency_cap_enforced(self):
    async def _run():
      active = 0
      peak_active = 0
      events: list[tuple[str, int]] = []
      batch0_full = asyncio.Event()

      async def _item(idx: int) -> int:
        nonlocal active, peak_active
        events.append(("start", idx))
        active += 1
        peak_active = max(peak_active, active)
        if idx < 2:
          if active == 2:
            batch0_full.set()
          await batch0_full.wait()
        active -= 1
        events.append(("end", idx))
        return idx

      batches = [
          batch
          async for batch in asyncio_utils.gather_batches(
              (_item(i) for i in range(5)), batch_size=2
          )
      ]
      return batches, peak_active, events

    batches, peak_active, events = asyncio.run(_run())
    self.assertEqual(batches, [[0, 1], [2, 3], [4]])
    self.assertEqual(peak_active, 2)
    # Batch 1 must not start until all items in batch 0 have finished.
    batch0_end = max(events.index(("end", 0)), events.index(("end", 1)))
    batch1_start = min(events.index(("start", 2)), events.index(("start", 3)))
    self.assertLess(batch0_end, batch1_start)

  def test_short_circuit_generator_skips_subsequent_batches(self):
    created: list[int] = []

    async def _work(idx: int) -> int:
      return idx

    def _gen():
      for i in range(6):
        created.append(i)
        yield _work(i)

    async def _run():
      batches = []
      async with contextlib.aclosing(
          asyncio_utils.gather_batches(_gen(), batch_size=2)
      ) as it:
        async for batch in it:
          batches.append(batch)
          if len(batches) == 2:
            break
      return batches

    self.assertEqual(asyncio.run(_run()), [[0, 1], [2, 3]])
    self.assertEqual(created, [0, 1, 2, 3])

  def test_short_circuit_any_bool_condition(self):
    checked: list[int] = []

    async def _has_match(target: int | None) -> bool:
      async def _probe(idx: int) -> bool:
        checked.append(idx)
        return idx == target

      checks = (_probe(i) for i in range(10))
      async for batch in asyncio_utils.gather_batches(checks, batch_size=3):
        if any(batch):
          return True
      return False

    gc.collect()
    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("always")
      self.assertTrue(asyncio.run(_has_match(4)))
      self.assertCountEqual(checked, [0, 1, 2, 3, 4, 5])

      checked.clear()
      self.assertFalse(asyncio.run(_has_match(None)))
      self.assertCountEqual(checked, list(range(10)))
      gc.collect()

    self.assertEmpty(_unawaited_warnings_for(caught, "_probe"))

  def test_short_circuit_list_closes_remaining_coroutines(self):
    async def _work(idx: int) -> int:
      return idx

    async def _run_plain_break():
      coros = [_work(i) for i in range(6)]
      batches = []
      async for batch in asyncio_utils.gather_batches(coros, batch_size=2):
        batches.append(batch)
        break
      return batches, coros

    gc.collect()
    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("always")
      batches, coros = asyncio.run(_run_plain_break())
      self.assertEqual(batches, [[0, 1]])
      for c in coros[2:]:
        self.assertEqual(inspect.getcoroutinestate(c), inspect.CORO_CLOSED)
      del coros
      gc.collect()

    self.assertEmpty(_unawaited_warnings_for(caught, "_work"))

    async def _run_aclosing():
      coros = [_work(i) for i in range(6)]
      batches = []
      async with contextlib.aclosing(
          asyncio_utils.gather_batches(coros, batch_size=2)
      ) as it:
        async for batch in it:
          batches.append(batch)
          break
      for c in coros[2:]:
        self.assertEqual(inspect.getcoroutinestate(c), inspect.CORO_CLOSED)
      del coros
      gc.collect()
      return batches

    gc.collect()
    with warnings.catch_warnings(record=True) as caught_aclosing:
      warnings.simplefilter("always")
      self.assertEqual(asyncio.run(_run_aclosing()), [[0, 1]])

    self.assertEmpty(_unawaited_warnings_for(caught_aclosing, "_work"))

  def test_exception_return_exceptions_false_generator(self):
    created: list[int] = []
    cancelled: set[int] = set()

    async def _run():
      siblings_waiting = asyncio.Event()
      forever = asyncio.Event()
      waiting_count = 0

      async def _work(idx: int) -> int:
        nonlocal waiting_count
        if idx == 0:
          await siblings_waiting.wait()
          raise ValueError("boom")
        waiting_count += 1
        if waiting_count == 2:
          siblings_waiting.set()
        try:
          await forever.wait()
        except asyncio.CancelledError:
          cancelled.add(idx)
          raise
        return idx

      def _gen():
        for i in range(6):
          created.append(i)
          yield _work(i)

      async for _ in asyncio_utils.gather_batches(
          _gen(), batch_size=3, return_exceptions=False
      ):
        pass

    with self.assertRaisesRegex(ValueError, "boom"):
      asyncio.run(_run())

    self.assertEqual(cancelled, {1, 2})
    self.assertEqual(created, [0, 1, 2])

  def test_exception_return_exceptions_false_list_closes_remaining(self):
    cancelled: set[int] = set()
    started: set[int] = set()

    async def _run():
      siblings_waiting = asyncio.Event()
      forever = asyncio.Event()
      waiting_count = 0

      async def _work(idx: int) -> int:
        nonlocal waiting_count
        started.add(idx)
        if idx == 0:
          await siblings_waiting.wait()
          raise ValueError("boom")
        waiting_count += 1
        if waiting_count == 2:
          siblings_waiting.set()
        try:
          await forever.wait()
        except asyncio.CancelledError:
          cancelled.add(idx)
          raise
        return idx

      coros = [_work(i) for i in range(6)]
      try:
        async for _ in asyncio_utils.gather_batches(
            coros, batch_size=3, return_exceptions=False
        ):
          pass
      finally:
        for c in coros:
          self.assertEqual(inspect.getcoroutinestate(c), inspect.CORO_CLOSED)

    gc.collect()
    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("always")
      with self.assertRaisesRegex(ValueError, "boom"):
        asyncio.run(_run())
      gc.collect()

    self.assertEqual(started, {0, 1, 2})
    self.assertEqual(cancelled, {1, 2})
    self.assertEmpty(_unawaited_warnings_for(caught, "_work"))

  def test_exception_return_exceptions_true(self):
    async def _work(idx: int) -> int:
      if idx == 1:
        raise ValueError("err1")
      if idx == 3:
        raise RuntimeError("err3")
      return idx

    async def _run(flag: bool):
      return [
          batch
          async for batch in asyncio_utils.gather_batches(
              (_work(i) for i in range(5)),
              batch_size=2,
              return_exceptions=flag,
          )
      ]

    batches_literal = asyncio.run(self._gather_batches_literal_true(_work))
    batches_bool = asyncio.run(_run(True))

    for batches in (batches_literal, batches_bool):
      self.assertLen(batches, 3)
      self.assertEqual(batches[0][0], 0)
      self.assertIsInstance(batches[0][1], ValueError)
      self.assertEqual(batches[1][0], 2)
      self.assertIsInstance(batches[1][1], RuntimeError)
      self.assertEqual(batches[2], [4])

  async def _gather_batches_literal_true(self, work_fn):
    return [
        batch
        async for batch in asyncio_utils.gather_batches(
            (work_fn(i) for i in range(5)),
            batch_size=2,
            return_exceptions=True,
        )
    ]

  @parameterized.parameters(0, -1)
  def test_invalid_batch_size(self, batch_size: int):

    async def _run():
      async for _ in asyncio_utils.gather_batches([], batch_size=batch_size):
        pass

    with self.assertRaisesRegex(ValueError, "batch_size"):
      asyncio.run(_run())


class GatherBoundedTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("empty_list", lambda: []),
      ("empty_tuple", lambda: ()),
      ("empty_generator", lambda: (x for x in ())),
  )
  def test_empty_input(self, make_awaitables):
    result = asyncio.run(asyncio_utils.gather_bounded(make_awaitables()))
    self.assertEqual(result, [])

  def test_positional_ordering_preserved(self):
    async def _run():
      gate_2 = asyncio.Event()
      gate_4 = asyncio.Event()

      async def _item(idx: int) -> int:
        if idx == 0:
          await gate_2.wait()
        elif idx == 1:
          await gate_4.wait()
        elif idx == 2:
          gate_2.set()
        elif idx == 4:
          gate_4.set()
        return idx

      return await asyncio_utils.gather_bounded(
          (_item(i) for i in range(5)), max_concurrency=3
      )

    self.assertEqual(asyncio.run(_run()), [0, 1, 2, 3, 4])

  def test_max_concurrency_greater_than_input_len(self):
    async def _item(idx: int) -> int:
      return idx * 2

    result = asyncio.run(
        asyncio_utils.gather_bounded(
            [_item(i) for i in range(3)], max_concurrency=10
        )
    )
    self.assertEqual(result, [0, 2, 4])

  def test_concurrency_cap_enforced(self):
    async def _run():
      active = 0
      peak_active = 0
      first_wave_full = asyncio.Event()
      wave_started = 0

      async def _item(idx: int) -> int:
        nonlocal active, peak_active, wave_started
        active += 1
        peak_active = max(peak_active, active)
        wave_started += 1
        if wave_started <= 3:
          if active == 3:
            first_wave_full.set()
          await first_wave_full.wait()
        active -= 1
        return idx

      results = await asyncio_utils.gather_bounded(
          (_item(i) for i in range(8)), max_concurrency=3
      )
      return results, peak_active

    results, peak_active = asyncio.run(_run())
    self.assertEqual(results, list(range(8)))
    self.assertEqual(peak_active, 3)

  def test_sliding_window_does_not_block_on_slow_task(self):
    async def _run():
      item0_started = asyncio.Event()
      release_item0 = asyncio.Event()
      fast_items_done = asyncio.Event()
      completed_while_blocked: list[int] = []

      async def _item(idx: int) -> int:
        if idx == 0:
          item0_started.set()
          await release_item0.wait()
          return idx
        await item0_started.wait()
        completed_while_blocked.append(idx)
        if len(completed_while_blocked) == 4:
          fast_items_done.set()
        return idx

      gather_task = asyncio.create_task(
          asyncio_utils.gather_bounded(
              (_item(i) for i in range(5)), max_concurrency=2
          )
      )
      await fast_items_done.wait()
      self.assertEqual(completed_while_blocked, [1, 2, 3, 4])
      self.assertFalse(gather_task.done())
      release_item0.set()
      return await gather_task

    self.assertEqual(asyncio.run(_run()), [0, 1, 2, 3, 4])

  def test_exception_return_exceptions_false_generator(self):
    created: list[int] = []
    cancelled: list[int] = []

    async def _run():
      sibling_started = asyncio.Event()
      forever = asyncio.Event()

      async def _work(idx: int) -> int:
        if idx == 0:
          await sibling_started.wait()
          raise ValueError("boom")
        sibling_started.set()
        try:
          await forever.wait()
        except asyncio.CancelledError:
          cancelled.append(idx)
          raise
        return idx

      def _gen():
        for i in range(6):
          created.append(i)
          yield _work(i)

      return await asyncio_utils.gather_bounded(
          _gen(), max_concurrency=2, return_exceptions=False
      )

    with self.assertRaisesRegex(ValueError, "boom"):
      asyncio.run(_run())

    self.assertEqual(cancelled, [1])
    self.assertEqual(created, [0, 1])

  def test_exception_return_exceptions_false_list_closes_remaining(self):
    cancelled: list[int] = []
    started: set[int] = set()

    async def _run():
      sibling_started = asyncio.Event()
      forever = asyncio.Event()

      async def _work(idx: int) -> int:
        started.add(idx)
        if idx == 0:
          await sibling_started.wait()
          raise ValueError("boom")
        sibling_started.set()
        try:
          await forever.wait()
        except asyncio.CancelledError:
          cancelled.append(idx)
          raise
        return idx

      coros = [_work(i) for i in range(6)]
      try:
        return await asyncio_utils.gather_bounded(
            coros, max_concurrency=2, return_exceptions=False
        )
      finally:
        for c in coros:
          self.assertEqual(inspect.getcoroutinestate(c), inspect.CORO_CLOSED)

    gc.collect()
    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("always")
      with self.assertRaisesRegex(ValueError, "boom"):
        asyncio.run(_run())
      gc.collect()

    self.assertEqual(started, {0, 1})
    self.assertEqual(cancelled, [1])
    self.assertEmpty(_unawaited_warnings_for(caught, "_work"))

  def test_exception_return_exceptions_true(self):
    async def _run(flag: bool):
      gate_2 = asyncio.Event()

      async def _work(idx: int) -> int:
        if idx == 0:
          await gate_2.wait()
        elif idx == 1:
          raise ValueError("err1")
        elif idx == 2:
          gate_2.set()
        elif idx == 3:
          raise RuntimeError("err3")
        return idx

      return await asyncio_utils.gather_bounded(
          (_work(i) for i in range(5)),
          max_concurrency=3,
          return_exceptions=flag,
      )

    async def _run_literal_true():
      async def _work(idx: int) -> int:
        if idx == 1:
          raise ValueError("err1")
        if idx == 3:
          raise RuntimeError("err3")
        return idx

      return await asyncio_utils.gather_bounded(
          (_work(i) for i in range(5)),
          max_concurrency=3,
          return_exceptions=True,
      )

    for results in (
        asyncio.run(_run_literal_true()),
        asyncio.run(_run(True)),
    ):
      self.assertLen(results, 5)
      self.assertEqual(results[0], 0)
      self.assertIsInstance(results[1], ValueError)
      self.assertEqual(results[2], 2)
      self.assertIsInstance(results[3], RuntimeError)
      self.assertEqual(results[4], 4)

  def test_external_cancellation_propagates_and_closes_remaining(self):
    cancelled: set[int] = set()

    async def _run():
      both_started = asyncio.Event()
      forever = asyncio.Event()
      started_count = 0

      async def _work(idx: int) -> int:
        nonlocal started_count
        started_count += 1
        if started_count == 2:
          both_started.set()
        try:
          await forever.wait()
        except asyncio.CancelledError:
          cancelled.add(idx)
          raise
        return idx

      coros = [_work(i) for i in range(5)]
      task = asyncio.create_task(
          asyncio_utils.gather_bounded(
              coros, max_concurrency=2, return_exceptions=True
          )
      )
      await both_started.wait()
      task.cancel()
      try:
        with self.assertRaises(asyncio.CancelledError):
          await task
      finally:
        for c in coros:
          self.assertEqual(inspect.getcoroutinestate(c), inspect.CORO_CLOSED)

    gc.collect()
    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("always")
      asyncio.run(_run())
      gc.collect()

    self.assertEqual(cancelled, {0, 1})
    self.assertEmpty(_unawaited_warnings_for(caught, "_work"))

  @parameterized.parameters(0, -1)
  def test_invalid_max_concurrency(self, max_concurrency: int):
    with self.assertRaisesRegex(ValueError, "max_concurrency"):
      asyncio.run(
          asyncio_utils.gather_bounded([], max_concurrency=max_concurrency)
      )


if __name__ == "__main__":
  absltest.main()
