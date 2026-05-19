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

import sys
import unittest
from unittest import mock

from absl.testing import absltest
from orbax.checkpoint._src.engine import async_io_engine
from orbax.checkpoint._src.serialization import types

AsyncIoEngine = async_io_engine.AsyncIoEngine
BatchRequest = async_io_engine.BatchRequest


class AsyncIoEngineTest(absltest.TestCase, unittest.IsolatedAsyncioTestCase):

  def test_get_batch_memory_size_success(self):
    handler = mock.create_autospec(types.TypeHandler, instance=True)
    handler.memory_size.return_value = [(10, 20), (30, 40)]

    write_size, read_size = async_io_engine.get_batch_memory_size(
        handler, ['a', 'b']
    )
    self.assertEqual(write_size, 40)
    self.assertEqual(read_size, 60)

  def test_get_batch_memory_size_not_implemented(self):
    handler = mock.create_autospec(types.TypeHandler, instance=True)
    handler.memory_size.side_effect = NotImplementedError()

    values = ['dummy1', 'dummy2']
    expected_size = sum(sys.getsizeof(v) for v in values)

    write_size, read_size = async_io_engine.get_batch_memory_size(
        handler, values
    )
    self.assertEqual(write_size, expected_size)
    self.assertEqual(read_size, expected_size)

  def test_batch_request_validation_success(self):
    handler = mock.create_autospec(types.TypeHandler, instance=True)
    req = BatchRequest(
        handler=handler,
        keys=['k1', 'k2'],
        values=['v1', 'v2'],
        infos=[mock.Mock(), mock.Mock()],
        args=[mock.Mock(), mock.Mock()],
    )
    self.assertLen(req.values, 2)

  def test_batch_request_validation_mismatch(self):
    handler = mock.create_autospec(types.TypeHandler, instance=True)
    with self.assertRaises(AssertionError):
      BatchRequest(
          handler=handler,
          keys=['k1'],
          values=['v1', 'v2'],
          infos=[mock.Mock(), mock.Mock()],
          args=[mock.Mock(), mock.Mock()],
      )

  async def test_execute_save(self):
    engine = AsyncIoEngine()

    handler1 = mock.create_autospec(types.TypeHandler, instance=True)
    handler2 = mock.create_autospec(types.TypeHandler, instance=True)

    async def dummy_serialize1(*args, **kwargs):
      del args, kwargs
      return ['fut1', 'fut2']

    async def dummy_serialize2(*args, **kwargs):
      del args, kwargs
      return ['fut3']

    handler1.serialize.side_effect = dummy_serialize1
    handler2.serialize.side_effect = dummy_serialize2

    handler1.memory_size.return_value = [(100, 0)]
    handler2.memory_size.return_value = [(200, 0)]

    req1 = BatchRequest(
        handler=handler1,
        keys=['k1'],
        values=['v1'],
        infos=[mock.Mock()],
        args=[mock.Mock()],
    )
    req2 = BatchRequest(
        handler=handler2,
        keys=['k2'],
        values=['v2'],
        infos=[mock.Mock()],
        args=[mock.Mock()],
    )

    commit_futures, tree_memory_size = await engine.execute_save([req1, req2])

    self.assertEqual(commit_futures, [['fut1', 'fut2'], ['fut3']])
    self.assertEqual(tree_memory_size, 300)

  async def test_execute_restore(self):
    engine = AsyncIoEngine()

    handler1 = mock.create_autospec(types.TypeHandler, instance=True)
    handler2 = mock.create_autospec(types.TypeHandler, instance=True)

    async def dummy_deserialize1(*args, **kwargs):
      del args, kwargs
      return ['restored1']

    async def dummy_deserialize2(*args, **kwargs):
      del args, kwargs
      return ['restored2']

    handler1.deserialize.side_effect = dummy_deserialize1
    handler2.deserialize.side_effect = dummy_deserialize2

    handler1.memory_size.return_value = [(0, 50)]
    handler2.memory_size.return_value = [(0, 150)]

    req1 = BatchRequest(
        handler=handler1,
        keys=['k1'],
        values=['v1'],
        infos=[mock.Mock()],
        args=[mock.Mock()],
    )
    req2 = BatchRequest(
        handler=handler2,
        keys=['k2'],
        values=['v2'],
        infos=[mock.Mock()],
        args=[mock.Mock()],
    )

    deserialized_batches, tree_memory_size = await engine.execute_restore(
        [req1, req2]
    )

    self.assertEqual(deserialized_batches, [['restored1'], ['restored2']])
    self.assertEqual(tree_memory_size, 200)


if __name__ == '__main__':
  absltest.main()
