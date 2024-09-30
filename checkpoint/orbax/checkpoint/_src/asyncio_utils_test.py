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

"""Tests for asyncio_utils module."""

import asyncio
import functools
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
    # First run with enable_nest_asyncio=False, because nest_asyncio.apply
    # patches asyncio globally in a runtime. There is no way to unpatch it.
    run_time = timeit.timeit(
        lambda: asyncio_utils.run_sync(_test(), enable_nest_asyncio=False),
        number=number,
    )  # ~1.604s
    run_sync_time = timeit.timeit(
        lambda: asyncio_utils.run_sync(_test(), enable_nest_asyncio=True),
        number=number,
    )  # ~1.5503s
    logging.info(
        "time: run_sync_time=%s, run_time=%s, ratio=%s",
        run_sync_time,
        run_time,
        run_sync_time / run_time,
    )


if __name__ == "__main__":
  absltest.main()
