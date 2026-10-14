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

"""Tests for overlapping async save calls in Orbax v1."""

import asyncio
import dataclasses
import logging
import random
import time
from typing import Awaitable, Sequence

from absl.testing import absltest
from etils import epath
import orbax.checkpoint.experimental.v1 as ocp


@dataclasses.dataclass
class MockValue:
  val: int


@dataclasses.dataclass
class MockAbstractValue:
  val: int


@dataclasses.dataclass
class FailingMockValue:
  val: int


@dataclasses.dataclass
class FailingAbstractValue:
  val: int


class MockLeafHandler(
    ocp.serialization.LeafHandler[MockValue, MockAbstractValue]
):
  def __init__(self, context=None):
    pass

  async def serialize(
      self,
      params: Sequence[ocp.serialization.SerializationParam[MockValue]],
      serialization_context: ocp.serialization.SerializationContext,
  ) -> Awaitable[None]:
    # Simulate D2H blocking and other overhead on main thread.
    time.sleep(0.1)

    async def _run_background():
      # Simulate background I/O.
      await asyncio.sleep(0.5)
    return _run_background()

  async def deserialize(self, params, context):
    raise NotImplementedError()

  async def metadata(self, params, context):
    raise NotImplementedError()


class FailingLeafHandler(
    ocp.serialization.LeafHandler[FailingMockValue, FailingAbstractValue]
):
  def __init__(self, context=None):
    pass

  async def serialize(
      self,
      params: Sequence[ocp.serialization.SerializationParam[FailingMockValue]],
      serialization_context: ocp.serialization.SerializationContext,
  ) -> Awaitable[None]:
    time.sleep(0.1)

    async def _run_background():
      await asyncio.sleep(0.5)
      raise RuntimeError('Simulated Background I/O Failure')
    return _run_background()  # pytype: disable=bad-return-type

  async def deserialize(self, params, context):
    raise NotImplementedError()

  async def metadata(self, params, context):
    raise NotImplementedError()


class OverlappingAsyncSaveTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(
        self.create_tempdir('overlapping_async_save_test').full_path
    )

  def test_overlapping_save(self):
    leaf_handler_registry = ocp.serialization.StandardLeafHandlerRegistry()
    leaf_handler_registry.add(MockValue, MockAbstractValue, MockLeafHandler)

    context = ocp.Context(
        pytree_options=ocp.options.PyTreeOptions(
            leaf_handler_registry=leaf_handler_registry
        )
    )

    with context:
      # Warm up save to handle any first-time initialization overhead.
      warmup_setup_start = time.time()
      f_warmup = ocp.save_pytree_async(self.directory / 'warmup', {'x': 1})
      warmup_setup_end = time.time()
      warmup_setup_time = warmup_setup_end - warmup_setup_start
      logging.info('Warmup setup time: %s', warmup_setup_time)

      f_warmup.result()
      warmup_background_time = time.time() - warmup_setup_end
      logging.info('Warmup background time: %s', warmup_background_time)
      pytree1 = {'a': MockValue(1)}
      pytree2 = {'b': MockValue(2)}

      start_time = time.time()

      f1 = ocp.save_pytree_async(self.directory / 'ckpt1', pytree1)
      t1_setup_time = time.time()
      logging.info('save 1 setup time: %s', t1_setup_time)

      f2 = ocp.save_pytree_async(self.directory / 'ckpt2', pytree2)
      t2_setup_time = time.time()
      logging.info('save 2 setup time: %s', t2_setup_time)

      # Verify blocking behavior.
      # Each call should have blocked for at least 0.1 second (D2H).
      self.assertGreater(t1_setup_time - start_time, 0.09)
      self.assertGreater(t2_setup_time - t1_setup_time, 0.09)
      self.assertLess(t1_setup_time - start_time, 0.2)
      self.assertLess(t2_setup_time - t1_setup_time, 0.2)

      # Total time used by the two calls should be around 0.2s,
      # which is fast enough to show it didn't wait for background work.
      self.assertLess(t2_setup_time - start_time, 0.4)

      # Now wait for them to finish.
      f1.result()
      f1_finished = time.time()
      logging.info('save 1 finished time: %s', f1_finished)
      self.assertGreater(f1_finished - t1_setup_time, 0.45)
      self.assertLess(f1_finished - t1_setup_time, 0.6)

      f2.result()
      f2_finished = time.time()
      self.assertGreaterEqual(f2_finished - t2_setup_time, 0.45)
      self.assertLess(f2_finished - t2_setup_time, 0.6)

      # Total time should be around 0.1 + 0.1 + 0.5 = 0.7s.
      # If they were sequential, it would be 0.1 + 0.5 + 0.1 + 0.5 = 1.2s.
      self.assertLess(f2_finished - start_time, 1.0)

  def test_interleaved_random_stress(self):
    leaf_handler_registry = ocp.serialization.StandardLeafHandlerRegistry()
    leaf_handler_registry.add(MockValue, MockAbstractValue, MockLeafHandler)

    context = ocp.Context(
        pytree_options=ocp.options.PyTreeOptions(
            leaf_handler_registry=leaf_handler_registry
        )
    )
    with context:
      ocp.save_pytree_async(self.directory / 'warmup', {'x': 1}).result()
      num_saves = 30
      pending_futures = []
      start_time = time.time()

      completion_times = []
      for i in range(num_saves):
        path = self.directory / f'stress_{i}'
        f = ocp.save_pytree_async(path, {'data': MockValue(i)})
        pending_futures.append((i, f))

        # Randomly await some futures.
        if random.random() > 0.7 and pending_futures:
          i, fut = pending_futures.pop(0)
          fut.result()
          completion_times.append((i, time.time() - start_time))

      # Clean up remaining futures
      for i, fut in pending_futures:
        fut.result()
        completion_times.append((i, time.time() - start_time))

      total_duration = time.time() - start_time
      avg_completion = total_duration / num_saves
      self.assertLess(avg_completion, 0.5)  # We expect some overlap work < 0.6s
      logging.info('avg_completion: %s', avg_completion)
      logging.info('completion_times: %s', completion_times)

      for i in range(num_saves):
        self.assertTrue((self.directory / f'stress_{i}').exists())

  def test_failure_chain_no_deadlock(self):
    leaf_handler_registry = ocp.serialization.StandardLeafHandlerRegistry()
    leaf_handler_registry.add(MockValue, MockAbstractValue, MockLeafHandler)
    leaf_handler_registry.add(
        FailingMockValue, FailingAbstractValue, FailingLeafHandler
    )

    context = ocp.Context(
        pytree_options=ocp.options.PyTreeOptions(
            leaf_handler_registry=leaf_handler_registry
        )
    )

    with context:
      f_fail = ocp.save_pytree_async(
          self.directory / 'failing_ckpt', {'a': FailingMockValue(1)}
      )
      f_success = ocp.save_pytree_async(
          self.directory / 'healthy_ckpt', {'b': MockValue(1)}
      )
      with self.assertRaisesRegex(
          RuntimeError, 'Simulated Background I/O Failure'
      ):
        f_fail.result()

      # If there's a deadlock in the coordinator, this will hang.
      try:
        f_success.result(timeout=5)
      except RuntimeError as e:
        self.fail(
            f'Second save failed or deadlocked due to previous failure: {e}'
        )
      self.assertTrue((self.directory / 'healthy_ckpt').exists())


if __name__ == '__main__':
  absltest.main()
