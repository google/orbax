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

import threading
import time
from typing import Optional
from unittest import mock

from absl import flags
from absl.testing import absltest
from etils import epath
import jax
from orbax.checkpoint._src import asyncio_utils
from orbax.checkpoint._src.handlers import json_checkpoint_handler
from orbax.checkpoint._src.path import atomicity

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


class JsonCheckpointHandlerTest(absltest.TestCase):

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

  def test_async_save_with_deferred_path(self):
    item = {'key': 'value'}
    handler = JsonCheckpointHandler(filename='custom.json')
    deferred_path = atomicity.DeferredPath()
    save_dir = self.directory / 'deferred_path_ckpt'
    await_creation_called = False
    original_await = atomicity.DeferredPath.await_creation
    set_path_lock = threading.Lock()

    async def mock_await_creation(dp_self):
      nonlocal await_creation_called
      with set_path_lock:
        if not dp_self._future_path.done():
          save_dir.mkdir(parents=True, exist_ok=True)
          dp_self.set_path(save_dir)
      await_creation_called = True
      return await original_await(dp_self)

    with mock.patch.object(
        atomicity.DeferredPath,
        'await_creation',
        mock_await_creation,
    ):

      async def run():
        futures = await handler.async_save(
            deferred_path, args=JsonSaveArgs(item)
        )
        for f in futures:
          f.result()

      asyncio_utils.run_sync(run())

    self.assertTrue(await_creation_called)
    self.assertTrue((save_dir / 'custom.json').exists())
    restored = handler.restore(save_dir)
    self.assertEqual(item, restored)


if __name__ == '__main__':
  absltest.main()
