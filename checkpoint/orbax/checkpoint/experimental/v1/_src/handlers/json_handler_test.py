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

import asyncio
from typing import Any, Awaitable

from absl.testing import absltest
from etils import epath
from orbax.checkpoint.experimental.v1._src.handlers import json_handler

from absl.testing import absltest


async def _run_awaitable(awaitable: Awaitable[Any]) -> Any:
  return await awaitable


class JsonHandler:
  """Wrapper around JsonHandler that can block on save and load."""

  def __init__(self, **kwargs):
    self._handler = json_handler.JsonHandler(**kwargs)

  def save(self, *args, **kwargs):
    awaitable = asyncio.run(self._handler.save(*args, **kwargs))
    return asyncio.run(_run_awaitable(awaitable))

  def load(self, *args, **kwargs):
    awaitable = asyncio.run(self._handler.load(*args, **kwargs))
    return asyncio.run(_run_awaitable(awaitable))


class JsonHandlerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(
        self.create_tempdir(name='checkpointing_test').full_path
    )

  def test_save_restore(self):
    item = {'a': 1, 'b': {'c': 'test1', 'b': 'test2'}, 'd': 5.5}
    handler = JsonHandler()
    handler.save(
        directory=self.directory,
        checkpointable=item,
    )
    restored = handler.load(self.directory)
    self.assertEqual(item, restored)

  def test_save_retore_filename(self):
    item = {'a': 1, 'b': {'c': 'test1', 'b': 'test2'}, 'd': 5.5}
    handler = JsonHandler(filename='file')
    handler.save(
        directory=self.directory,
        checkpointable=item,
    )
    restored = handler.load(self.directory)
    self.assertEqual(item, restored)
    self.assertTrue((self.directory / 'file').exists())
    self.assertFalse((self.directory / 'metadata').exists())


if __name__ == '__main__':
  googletest.main()
