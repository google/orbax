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

"""Tests for JsonCheckpointHandler."""

from collections import defaultdict
import threading
import time
from typing import Optional
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import jax
from orbax.checkpoint._src import asyncio_utils
from orbax.checkpoint._src.futures import future
from orbax.checkpoint._src.futures import synchronization
from orbax.checkpoint._src.handlers import json_checkpoint_handler

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()

FLAGS = flags.FLAGS

JsonCheckpointHandler = json_checkpoint_handler.JsonCheckpointHandler
JsonSaveArgs = json_checkpoint_handler.JsonSaveArgs
JsonRestoreArgs = json_checkpoint_handler.JsonRestoreArgs


class TestJsonCheckpointHandler(JsonCheckpointHandler):

  def __init__(
      self,
      filename: Optional[str] = None,
  ):
    super().__init__(filename=filename)
    self._handler = json_checkpoint_handler.JsonCheckpointHandler
    self._filename = filename or 'metadata'

  async def _save_fn(self, x, directory):
    time.sleep(5)
    await super()._save_fn(x, directory)


class JsonCheckpointHandlerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(
        self.create_tempdir(name='checkpointing_test').full_path
    )

  def test_save_restore(self):
    item = {'a': 1, 'b': {'c': 'test1', 'b': 'test2'}, 'd': 5.5}
    handler = JsonCheckpointHandler()
    handler.save(self.directory, args=JsonSaveArgs(item))
    restored = handler.restore(self.directory)
    self.assertEqual(item, restored)

  def test_save_restore_filename(self):
    item = {'a': 1, 'b': {'c': 'test1', 'b': 'test2'}, 'd': 5.5}
    handler = JsonCheckpointHandler(filename='file')
    handler.save(self.directory, args=JsonSaveArgs(item))
    restored = handler.restore(self.directory)
    self.assertEqual(item, restored)
    self.assertTrue((self.directory / 'file').exists())
    self.assertFalse((self.directory / 'metadata').exists())

  @parameterized.named_parameters(('args', True), ('legacy_item', False))
  def test_async_save_captures_item_before_returning(self, use_args):
    item = {'step': 1, 'metrics': {'loss': [0.5, 0.25]}}
    expected = {'step': 1, 'metrics': {'loss': [0.5, 0.25]}}
    handler = JsonCheckpointHandler()
    signal = synchronization.HandlerAwaitableSignal.ITEM_DIRECTORY_CREATION

    async def run_async_test():
      # Delay the write using the same contract as asynchronous directory
      # creation, so the caller can continue training before the write starts.
      with future.signals_namespace(self.id()):
        operation_id = (
            synchronization.OperationIdGenerator.get_current_operation_id()
        )
        future.AwaitableSignalsContract.add_to_awaitable_signals_contract(
            [signal]
        )
        try:
          kwargs = {'args': JsonSaveArgs(item)} if use_args else {'item': item}
          commit_futures = await handler.async_save(self.directory, **kwargs)
          try:
            self.assertFalse((self.directory / 'metadata').exists())
            item['step'] = 2
            item['metrics']['loss'].append(0.125)
            item['metrics']['accuracy'] = 0.9
          finally:
            future.set_signals([signal], operation_id=operation_id)
            for commit_future in commit_futures:
              commit_future.result(timeout=10)
        finally:
          future.AwaitableSignalsContract.remove_all_awaitable_signals(
              operation_id
          )

    asyncio_utils.run_sync(run_async_test())
    self.assertEqual(expected, handler.restore(self.directory))
    handler.close()

  @parameterized.product(
      container_kind=('dict_subclass', 'stateful_defaultdict'),
      primary=(True, False),
  )
  def test_json_ignores_nonserializable_container_state(
      self, container_kind, primary
  ):
    class TaggedDict(dict):

      def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lock = threading.Lock()

    class Factory:

      def __init__(self):
        self.lock = threading.Lock()

      def __call__(self):
        return []

    if container_kind == 'dict_subclass':
      item = TaggedDict({'a': [1]})
    else:
      item = defaultdict(Factory(), {'a': [1]})
    handler = JsonCheckpointHandler()
    with mock.patch.object(
        json_checkpoint_handler.utils, 'is_primary_host', return_value=primary
    ):
      handler.save(self.directory, args=JsonSaveArgs(item))
    if primary:
      self.assertEqual({'a': [1]}, handler.restore(self.directory))
    else:
      self.assertFalse((self.directory / 'metadata').exists())
    handler.close()

  @parameterized.named_parameters(
      ('unsupported_set', False, TypeError), ('cycle', True, ValueError)
  )
  def test_serialization_errors_are_reported_by_commit_future(
      self, cyclic, error_type
  ):
    item = {'a': {1}}
    if cyclic:
      item = {}
      item['self'] = item
    handler = JsonCheckpointHandler()
    signal = synchronization.HandlerAwaitableSignal.ITEM_DIRECTORY_CREATION

    async def run_async_test():
      with future.signals_namespace(self.id()):
        operation_id = (
            synchronization.OperationIdGenerator.get_current_operation_id()
        )
        future.AwaitableSignalsContract.add_to_awaitable_signals_contract(
            [signal]
        )
        try:
          commit_futures = await handler.async_save(
              self.directory, args=JsonSaveArgs(item)
          )
          try:
            for commit_future in commit_futures:
              with self.assertRaises(TimeoutError):
                commit_future.result(timeout=0)
          finally:
            future.set_signals([signal], operation_id=operation_id)
            for commit_future in commit_futures:
              with self.assertRaises(error_type):
                commit_future.result(timeout=10)
        finally:
          future.AwaitableSignalsContract.remove_all_awaitable_signals(
              operation_id
          )

    asyncio_utils.run_sync(run_async_test())
    self.assertFalse((self.directory / 'metadata').exists())
    handler.close()

  def test_async_save_restore(self):
    item = {'a': 1, 'b': {'c': 'test1', 'b': 'test2'}, 'd': 5.5}
    handler = TestJsonCheckpointHandler()

    async def run_async_test():
      async_futures = await handler.async_save(
          self.directory, args=JsonSaveArgs(item)
      )
      # item wasn't saved at directory right away.
      self.assertFalse((self.directory / 'metadata').exists())
      for future in async_futures:
        future.result()
      self.assertTrue((self.directory / 'metadata').exists())
      restored = handler.restore(directory=self.directory, args=None)
      self.assertEqual(item, restored)

    asyncio_utils.run_sync(run_async_test())
    handler.close()


if __name__ == '__main__':
  absltest.main()
