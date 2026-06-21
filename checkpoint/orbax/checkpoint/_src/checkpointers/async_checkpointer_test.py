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

"""Tests for AsyncCheckpointer class."""

from __future__ import annotations

import asyncio
from concurrent import futures
import dataclasses
import time
from typing import Any, Dict, List, Optional
from unittest import mock

from absl import flags
from absl.testing import parameterized
from etils import epath
from etils.epath import typing
import jax
from orbax.checkpoint import args as args_lib
from orbax.checkpoint import checkpoint_args
from orbax.checkpoint import options as options_lib
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.checkpointers import async_checkpointer
from orbax.checkpoint._src.checkpointers import checkpointer_test_utils
from orbax.checkpoint._src.handlers import async_checkpoint_handler
from orbax.checkpoint._src.handlers import base_pytree_checkpoint_handler
from orbax.checkpoint._src.handlers import composite_checkpoint_handler
from orbax.checkpoint._src.handlers import pytree_checkpoint_handler
from orbax.checkpoint._src.logging import step_statistics
from orbax.checkpoint._src.metadata import checkpoint
from orbax.checkpoint._src.metadata import step_metadata_serialization
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.path import async_path
from orbax.checkpoint._src.path import atomicity
from orbax.checkpoint._src.path import types as path_types
from orbax.checkpoint._src.testing import multiprocess_test


Future = futures.Future[None]
AsyncCheckpointer = async_checkpointer.AsyncCheckpointer

CheckpointArgs = checkpoint_args.CheckpointArgs
AsyncCheckpointHandler = async_checkpoint_handler.AsyncCheckpointHandler
PyTreeCheckpointHandler = pytree_checkpoint_handler.PyTreeCheckpointHandler
PyTreeSaveArgs = pytree_checkpoint_handler.PyTreeSaveArgs
PyTreeRestoreArgs = pytree_checkpoint_handler.PyTreeRestoreArgs
BasePyTreeCheckpointHandler = (
    base_pytree_checkpoint_handler.BasePyTreeCheckpointHandler
)
BasePyTreeSaveArgs = base_pytree_checkpoint_handler.BasePyTreeSaveArgs
CompositeCheckpointHandler = (
    composite_checkpoint_handler.CompositeCheckpointHandler
)
CompositeOptions = composite_checkpoint_handler.CompositeOptions
CompositeArgs = composite_checkpoint_handler.CompositeArgs
register_with_handler = checkpoint_args.register_with_handler

FLAGS = flags.FLAGS


_IRRELEVANT_FUTURE_RESULT = 42


async def _async_mkdir_with_delay(path: epath.Path, *args, **kwargs):
  await asyncio.sleep(1)
  return await asyncio.to_thread(path.mkdir, *args, **kwargs)


class SleepCheckpointHandler(AsyncCheckpointHandler):
  """Returns an extra commit future that takes extra time to complete."""

  def __init__(
      self,
      handler: PyTreeCheckpointHandler,
      executor: futures.ThreadPoolExecutor,
  ):
    self._handler = handler
    self._executor = executor

  def save(self, directory: epath.Path, args: SleepSaveArgs):
    self._handler.save(directory, args=args)

  def restore(
      self, directory: str, args: Optional[SleepRestoreArgs] = None
  ) -> Any:
    return self._handler.restore(directory, args=args)

  async def async_save(
      self, directory: epath.Path, args: SleepSaveArgs
  ) -> Optional[List[Future]]:
    """Delegates to base class but adds an extra future to wait some time."""
    commit_futures = await self._handler.async_save(directory, args=args)

    def long_commit():
      time.sleep(5)  # Pretend to write data.
      return _IRRELEVANT_FUTURE_RESULT

    for _ in range(2):
      commit_futures += [self._executor.submit(long_commit)]

    return commit_futures

  def finalize(self, directory: epath.Path):
    self._handler.finalize(directory)


class DeadlockCheckpointHandler(AsyncCheckpointHandler):
  """Returns an extra commit future that takes extra time to complete."""

  def __init__(
      self,
      executor: futures.ThreadPoolExecutor,
      per_process_completion_secs: Dict[int, int],
  ):
    self._executor = executor
    self._completion_secs = per_process_completion_secs

  def save(self, directory: epath.Path, args: DeadlockSaveArgs):
    pass

  def restore(
      self, directory: str, args: Optional[DeadlockRestoreArgs] = None
  ) -> None:
    pass

  async def async_save(
      self, directory: epath.Path, args: DeadlockSaveArgs
  ) -> Optional[List[Future]]:
    """Delegates to base class but adds an extra future to wait some time."""

    def long_commit():
      time.sleep(self._completion_secs[multihost.process_index()])
      return _IRRELEVANT_FUTURE_RESULT

    return [self._executor.submit(long_commit)]

  def finalize(self, directory: epath.Path):
    pass


@register_with_handler(SleepCheckpointHandler, for_save=True)
@dataclasses.dataclass
class SleepSaveArgs(PyTreeSaveArgs):
  pass


@register_with_handler(SleepCheckpointHandler, for_restore=True)
@dataclasses.dataclass
class SleepRestoreArgs(PyTreeRestoreArgs):
  pass


@register_with_handler(DeadlockCheckpointHandler, for_save=True)
@dataclasses.dataclass
class DeadlockSaveArgs(CheckpointArgs):
  pass


@register_with_handler(DeadlockCheckpointHandler, for_restore=True)
@dataclasses.dataclass
class DeadlockRestoreArgs(CheckpointArgs):
  pass


class TimeoutCheckpointHandler(AsyncCheckpointHandler):
  """Injects delays into checkpointing to test timeouts."""

  def __init__(
      self,
      sleep_secs: dict[int, int],
      executor: futures.ThreadPoolExecutor,
      delay_in_commit: bool = False,
      delay_in_finalize: bool = False,
  ):
    self._sleep_secs = sleep_secs
    self._executor = executor
    self._delay_in_commit = delay_in_commit
    self._delay_in_finalize = delay_in_finalize

  def _get_sleep_secs(self) -> int:
    return self._sleep_secs[multihost.process_index()]

  def save(self, directory: epath.Path, args: 'TimeoutSaveArgs'):
    pass

  def restore(
      self, directory: str, args: 'TimeoutRestoreArgs' | None = None
  ) -> Any:
    pass

  async def async_save(
      self, directory: epath.Path, args: 'TimeoutSaveArgs'
  ) -> Optional[List[Future]]:
    """Returns futures that sleep for given time."""

    def long_commit():
      if self._delay_in_commit:
        time.sleep(self._get_sleep_secs())
      return _IRRELEVANT_FUTURE_RESULT

    return [
        self._executor.submit(long_commit),
        self._executor.submit(long_commit),
    ]

  def finalize(self, directory: epath.Path):
    if self._delay_in_finalize:
      time.sleep(self._get_sleep_secs())


@register_with_handler(TimeoutCheckpointHandler, for_save=True)
@dataclasses.dataclass
class TimeoutSaveArgs(CheckpointArgs):
  pass


@register_with_handler(TimeoutCheckpointHandler, for_restore=True)
@dataclasses.dataclass
class TimeoutRestoreArgs(CheckpointArgs):
  pass


class MockPathAwaitingCreation(path_types.PathAwaitingCreation):
  """Mock PathAwaitingCreation for testing."""

  def __init__(self, path: epath.Path, delay: float = 0.1):
    self._path = path
    self._delay = delay
    self._awaited = False
    self._future = futures.ThreadPoolExecutor(max_workers=1).submit(
        self._create_path
    )

  def _create_path(self):
    time.sleep(self._delay)
    self._path.mkdir(parents=True, exist_ok=True)
    return self._path

  async def await_creation(self) -> epath.Path:
    result = await asyncio.wrap_future(self._future)
    self._awaited = True
    return result

  @property
  def awaited(self) -> bool:
    return self._awaited

  @property
  def path(self) -> epath.Path:
    if not self._future.done():
      raise RuntimeError('Path not yet created')
    return self._future.result()


MockDeferredWritableTemporaryPath = test_utils.MockDeferredWritableTemporaryPath


@test_utils.barrier_compatible_test
class AsyncCheckpointerTest(
    checkpointer_test_utils.CheckpointerTestBase.Test,
    multiprocess_test.MultiProcessTest,
):

  def setUp(self):
    self.executor = futures.ThreadPoolExecutor(max_workers=2)
    super().setUp()
    if not multihost.is_runtime_to_distributed_ids_initialized():
      multihost.initialize_runtime_to_distributed_ids()

  def tearDown(self):
    self.executor.shutdown(wait=True)
    super().tearDown()

  def checkpointer(self, handler, **kwargs):
    return AsyncCheckpointer(
        SleepCheckpointHandler(handler, self.executor), **kwargs
    )

  def test_legacy_api(self):
    self.skipTest('`SleepCheckpointHandler` only uses new API.')

  def test_interleave(self):
    self.assertEqual(jax.process_count(), 2)
    # Assume that process X is denoted as PX, and thread Y as TY. Assume a
    # barrier is denoted by <step>.<module_unique_count>, where
    # module_unique_count is a global module-level counter.
    # If we have two processes, and two threads per process, we may see
    # a deadlock in the following circumstance: (P0,T0) arrives at barrier 0.1,
    # and subsequently P1,T0 arrives at barrier 0.1 as well. Later P0,T1 arrives
    # at 0.2 and P1,T1 at 0.2 and we have a deadlock. This description of how
    # the barrier is constructed no longer reflects the current implementation,
    # since the behavior was fixed to avoid this scenario.
    primary = AsyncCheckpointer(
        DeadlockCheckpointHandler(self.executor, {0: 0, 1: 6})
    )
    secondary = AsyncCheckpointer(
        DeadlockCheckpointHandler(self.executor, {0: 3, 1: 9})
    )
    primary.save(self.directory / 'primary', args=DeadlockSaveArgs())
    secondary.save(self.directory / 'secondary', args=DeadlockSaveArgs())
    primary.close()
    secondary.close()

  @parameterized.named_parameters(
      dict(
          testcase_name='all_long_commit',
          stage_to_delay='commit',
          sleep_secs={0: 20, 1: 20},
          expected_msg_primary=(
              r'ChainedFuture completed \d+/\d+ futures but timed out after'
          ),
          expected_msg_non_primary=(
              r'ChainedFuture completed \d+/\d+ futures but timed out after'
          ),
      ),
      dict(
          testcase_name='host_long_commit',
          stage_to_delay='host_commit',
          sleep_secs={0: 20, 1: 0},
          expected_msg_primary=(
              r'ChainedFuture completed \d+/\d+ futures but timed out after'
          ),
          expected_msg_non_primary=(
              'Timed out while waiting for async_write_complete barrier.'
          ),
      ),
      dict(
          testcase_name='host_long_finalize',
          stage_to_delay='host_finalize',
          sleep_secs={0: 20, 1: 0},
          expected_msg_primary='Timed out after',
          expected_msg_non_primary=(
              'Timed out while waiting for async_commit_complete barrier.'
          ),
          # Timeout exception should be raised on primary host, but b/c of
          # the atomicity, we could not enforce until after finalize.
          # On non-primary hosts, the timeout exception is raised properly at
          # the sync barrier timeout check.
          check_elapsed_time_primary=False,
      ),
  )
  def test_async_save_overall_timeout(
      self,
      stage_to_delay,
      sleep_secs,
      expected_msg_primary,
      expected_msg_non_primary,
      check_elapsed_time_primary=True,
  ):
    timeout_secs = 10
    checkpointer = AsyncCheckpointer(
        TimeoutCheckpointHandler(
            sleep_secs=sleep_secs,
            executor=self.executor,
            delay_in_commit=('commit' in stage_to_delay),
            delay_in_finalize=('finalize' in stage_to_delay),
        ),
        timeout_secs=timeout_secs,
    )

    start = time.time()
    checkpointer.save(self.directory / 'timeout', args=TimeoutSaveArgs())

    is_primary = multihost.is_primary_host(primary_host=0)
    msg = expected_msg_primary if is_primary else expected_msg_non_primary
    with self.assertRaisesRegex(TimeoutError, msg):
      checkpointer.wait_until_finished()

    check_elapsed_time = (
        check_elapsed_time_primary if is_primary else True
    )
    if check_elapsed_time:
      elapsed = time.time() - start
      self.assertAlmostEqual(elapsed, timeout_secs, delta=1.5)

  def test_async_save(self):
    """Tests that background save actually works."""
    checkpointer = self.checkpointer(PyTreeCheckpointHandler())
    checkpointer.save(self.directory, self.pytree)
    self.assertFalse(self.directory.exists())  # Not finalized yet.
    # But a tmp dir should have been created.
    self.assertNotEmpty(list(self.directory.parent.iterdir()))
    self.wait_if_async(checkpointer)
    self.assertTrue(self.directory.exists())

  @mock.patch.object(
      async_path,
      'mkdir',
      new=_async_mkdir_with_delay,
  )
  def test_async_save_with_async_directory_creation(self):
    """Tests that background save actually works with async directory creation."""
    async_options = options_lib.AsyncOptions()
    async_options.create_directories_asynchronously = True
    composite_options = CompositeOptions()
    composite_options.async_options = async_options
    handler = CompositeCheckpointHandler(
        'state', composite_options=composite_options
    )
    checkpointer = self.checkpointer(handler, async_options=async_options)

    checkpointer.save(
        self.directory,
        CompositeArgs(
            state=args_lib.StandardSave(self.pytree),
        ),
    )

    self.assertFalse(self.directory.exists())  # Not finalized yet.
    # No tmp directories created yet
    self.assertEmpty(list(self.directory.parent.iterdir()))
    self.wait_if_async(checkpointer)
    self.assertTrue(self.directory.exists())
    self.assertTrue((self.directory / 'state').exists())

  @parameterized.parameters((3,), (8,))
  def test_primary_host_background_error(self, timeout):
    def _assert_false(*args, **kwargs):
      del args, kwargs
      assert False

    handler = CompositeCheckpointHandler('state')
    checkpointer = self.checkpointer(handler)

    start = time.time()
    with (
        mock.patch.object(
            atomicity, '_create_tmp_directory', new=_assert_false
        ),
        mock.patch.object(
            multihost,
            'coordination_timeout',
            return_value=timeout,
        ),
    ):
      checkpointer.save(
          self.directory,
          CompositeArgs(
              state=args_lib.StandardSave(self.pytree),
          ),
      )
      if multihost.is_primary_host(primary_host=0):
        with self.assertRaises(AssertionError):
          self.wait_if_async(checkpointer)
        # Error should be raised quickly on primary host.
        self.assertLessEqual(time.time() - start, timeout - 1)
      else:
        with self.assertRaises(BaseException):
          self.wait_if_async(checkpointer)
        self.assertLessEqual(time.time() - start, timeout + 2)

  def test_async_step_metadata_save(self):
    class SleepyMetadataStore(checkpoint._BlockingMetadataStore):
      """MetadataStore that sleeps for 5 seconds before writing."""

      def update(self, file_path: typing.PathLike, **kwargs: Any):
        time.sleep(5)
        super().update(file_path, **kwargs)

    checkpointer = self.checkpointer(
        PyTreeCheckpointHandler(),
        checkpoint_metadata_store=SleepyMetadataStore(enable_write=True),
    )
    checkpointer.save(self.directory, self.pytree)

    self.assertFalse(self.directory.exists())  # Not finalized yet.
    # But a tmp dir should have been created.
    self.assertLen(list(self.directory.parent.iterdir()), 1)
    tmp_dir = list(self.directory.parent.iterdir())[0]
    serialized_metadata = checkpointer._metadata_store.read(
        checkpoint.step_metadata_file_path(tmp_dir)
    )
    # Only the init timestamp should be set since it is written when the tmp dir
    # is created.
    self.assertGreater(serialized_metadata['init_timestamp_nsecs'], 0)
    # The rest of the fields should be unset.
    self.assertIsNone(serialized_metadata['item_handlers'])
    self.assertEqual(serialized_metadata['metrics'], {})
    self.assertEmpty(serialized_metadata['performance_metrics'])
    self.assertIsNone(serialized_metadata['commit_timestamp_nsecs'])

    self.wait_if_async(checkpointer)
    # The directory should exist now that the async save is complete.
    self.assertTrue(self.directory.exists())
    serialized_metadata = checkpointer._metadata_store.read(
        checkpoint.step_metadata_file_path(self.directory)
    )
    step_metadata = step_metadata_serialization.deserialize(serialized_metadata)
    # All metadata fields should be set now.
    self.assertEqual(
        step_metadata.item_handlers, checkpointer._handler.typestr()
    )
    self.assertIsNone(step_metadata.item_metadata)
    self.assertEqual(step_metadata.metrics, {})
    self.assertEqual(
        step_metadata.performance_metrics, step_statistics.SaveStepStatistics()
    )
    self.assertGreater(step_metadata.init_timestamp_nsecs, 0)
    self.assertGreater(step_metadata.commit_timestamp_nsecs, 0)

  # TODO(nikhilbansall): Open source this test.


if __name__ == '__main__':
  multiprocess_test.main()
