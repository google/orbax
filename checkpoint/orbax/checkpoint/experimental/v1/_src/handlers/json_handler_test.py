# Copyright 2025 The Orbax Authors.
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

from typing import Any, Awaitable

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from orbax.checkpoint.experimental.v1._src.handlers import json_handler
from orbax.checkpoint.experimental.v1._src.path import types as path_types
from orbax.checkpoint.experimental.v1._src.testing import handler_utils as handler_test_utils
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types


PathAwaitingCreation = path_types.PathAwaitingCreation
PathLike = path_types.PathLike
Path = path_types.Path
Json = tree_types.JsonType


async def _run_awaitable(awaitable: Awaitable[Any]) -> Any:
  return await awaitable


class JsonHandlerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(
        self.create_tempdir(name='checkpointing_test').full_path
    )
    self._handler = handler_test_utils.create_test_handler(
        json_handler.JsonHandler
    )

  def test_save_restore(self):
    item = {'a': 1, 'b': {'c': 'test1', 'b': 'test2'}, 'd': 5.5}
    self._handler.save(
        directory=self.directory,
        checkpointable=item,
    )
    restored = self._handler.load(self.directory)
    self.assertEqual(item, restored)

  @parameterized.parameters(
      ('data.json',),
      ('metadata',),
      ('unrecognized.json',),
  )
  def test_supported_filenames(self, filename):
    item = {'a': 1, 'b': 'test'}
    self._handler.save(
        directory=self.directory,
        checkpointable=item,
    )
    self.assertTrue((self.directory / 'data.json').exists())
    src = self.directory / 'data.json'
    dst = self.directory / filename
    if src != dst:
      (self.directory / 'data.json').rename(self.directory / filename)
    self.assertTrue((self.directory / filename).exists())

    if filename not in json_handler._get_supported_filenames():
      with self.assertRaises(FileNotFoundError):
        self._handler.load(self.directory)
    else:
      restored = self._handler.load(self.directory)
      self.assertEqual(item, restored)

  @parameterized.parameters(
      ('{"one": 1, "two": {"three": "3"}, "four": [4]}', True),
      ({'one': 1, 'two': {'three': '3'}, 'four': [4]}, True),
      ({'a': 1, 'b': 'c'}, True),
      ({'a', 'b'}, False),
      (set([1, 2]), False),
      (object(), False),
      (lambda x: x, False),
      ('', True),
      (1, True),
      (True, True),
  )
  def test_is_handleable(self, case, expected):
    self.assertEqual(self._handler.is_handleable(case), expected)


if __name__ == '__main__':
  absltest.main()
