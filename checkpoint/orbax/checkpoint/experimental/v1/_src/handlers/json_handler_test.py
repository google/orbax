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
from typing import Any, Awaitable, Mapping

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from orbax.checkpoint.experimental.v1._src.handlers import json_handler
from orbax.checkpoint.experimental.v1._src.path import types as path_types
from orbax.checkpoint.experimental.v1._src.testing import path_utils as path_test_utils
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types


PathAwaitingCreation = path_types.PathAwaitingCreation
PathLike = path_types.PathLike
Path = path_types.Path
Json = tree_types.JsonType


async def _run_awaitable(awaitable: Awaitable[Any]) -> Any:
  return await awaitable


class JsonHandler:
  """Wrapper around JsonHandler that can block on save and load."""

  def __init__(self, **kwargs):
    self._handler = json_handler.JsonHandler(**kwargs)

  def save(self, directory: Path, checkpointable: Json):
    path = path_test_utils.PathAwaitingCreationWrapper(directory)
    awaitable = asyncio.run(self._handler.save(path, checkpointable))
    return asyncio.run(_run_awaitable(awaitable))

  def save_async(self, directory: Path, checkpointable: Json):
    path = path_test_utils.PathAwaitingCreationWrapper(directory)
    return asyncio.run(self._handler.save(path, checkpointable))

  def load(self, path: Path, abstract_checkpointable: Json | None = None):
    awaitable = self.load_async(path, abstract_checkpointable)
    return asyncio.run(_run_awaitable(awaitable))

  def load_async(self, path: Path, abstract_checkpointable: Json | None = None):
    return asyncio.run(self._handler.load(path, abstract_checkpointable))


class JsonHandlerTest(parameterized.TestCase):

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

  @parameterized.parameters(
      ('{"one": 1, "two": {"three": "3"}, "four": [4]}', True),
      ({'one': 1, 'two': {'three': '3'}, 'four': [4]}, False),
      ('', False),
      (1, False),
      (True, False),
  )
  def test_is_handleable(self, case, expected):
    handler = json_handler.JsonHandler()
    self.assertEqual(handler.is_handleable(case), expected)


if __name__ == '__main__':
  absltest.main()
