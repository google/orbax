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

"""Tests for JsonCheckpointHandler."""

import asyncio
import time
from typing import Optional

from absl import flags
from absl.testing import absltest
from etils import epath
import jax
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

  def _save_fn(self, x, directory):
    time.sleep(5)
    return super()._save_fn(x, directory)

  def close(self):
    self._executor.shutdown(wait=True)


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

    asyncio.run(run_async_test())
    handler.close()


if __name__ == '__main__':
  absltest.main()
