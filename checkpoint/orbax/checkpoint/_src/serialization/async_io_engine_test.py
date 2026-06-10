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
import numpy as np
from orbax.checkpoint._src.serialization import async_io_engine
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

  def test_compute_save_memory_size(self):
    handler1 = mock.create_autospec(types.TypeHandler, instance=True)
    handler2 = mock.create_autospec(types.TypeHandler, instance=True)

    # memory_size returns a list of (write_size, read_size) tuples
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

    tree_memory_size = async_io_engine.compute_save_memory_size([req1, req2])
    self.assertEqual(tree_memory_size, 300)

  def test_compute_restore_memory_size(self):
    handler1 = mock.create_autospec(types.TypeHandler, instance=True)
    handler2 = mock.create_autospec(types.TypeHandler, instance=True)

    # memory_size returns a list of (write_size, read_size) tuples
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

    deserialized_batches = [['restored1'], ['restored2']]

    tree_memory_size = async_io_engine.compute_restore_memory_size(
        [req1, req2], deserialized_batches
    )
    self.assertEqual(tree_memory_size, 200)

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

    commit_futures = await engine.execute_save([req1, req2])
    self.assertEqual(commit_futures, [['fut1', 'fut2'], ['fut3']])

    # Test the standalone memory size function
    handler1.memory_size.return_value = [(100, 0)]
    handler2.memory_size.return_value = [(200, 0)]
    tree_memory_size = async_io_engine.compute_save_memory_size([req1, req2])
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

    deserialized_batches = await engine.execute_restore([req1, req2])
    self.assertEqual(deserialized_batches, [['restored1'], ['restored2']])

    # Test the standalone memory size function
    handler1.memory_size.return_value = [(0, 50)]
    handler2.memory_size.return_value = [(0, 150)]
    tree_memory_size = async_io_engine.compute_restore_memory_size(
        [req1, req2], deserialized_batches
    )
    self.assertEqual(tree_memory_size, 200)

  @mock.patch.object(async_io_engine.jax.monitoring, 'record_scalar')
  def test_log_io_metrics_compression_ratio(self, mock_record_scalar):
    initial_ts_metrics = (
        async_io_engine.ts.experimental_collect_matching_metrics(
            '/tensorstore/'
        )
    )

    # Perform actual TensorStore write to increment bytes_written.
    ts_spec = async_io_engine.ts.Spec({
        'driver': 'zarr',
        'kvstore': {
            'driver': 'file',
            'path': self.create_tempdir().full_path,
        },
        'metadata': {
            'compressor': {'id': 'zstd'},
        },
    })
    ts_store = async_io_engine.ts.open(
        ts_spec,
        create=True,
        delete_existing=True,
        dtype=np.int32,
        shape=(10000,),
    ).result()
    ts_store.write(np.ones((10000,), dtype=np.int32)).result()

    async_io_engine.log_io_metrics(
        size=40000,
        start_time=12345.0,
        gbytes_per_sec_metric='/jax/orbax/write/gbytes_per_sec',
        initial_ts_metrics=initial_ts_metrics,
    )

    ratio_calls = [
        call
        for call in mock_record_scalar.call_args_list
        if call[0][0] == '/jax/orbax/write/compression_ratio'
    ]
    self.assertNotEmpty(ratio_calls)
    ratio = ratio_calls[0][0][1]
    # Verifies that compression actually reduced size
    self.assertGreater(ratio, 0.0)
    self.assertLess(ratio, 1.0)

    compressed_calls = [
        call
        for call in mock_record_scalar.call_args_list
        if call[0][0] == '/jax/orbax/write/compressed_gbytes'
    ]
    self.assertNotEmpty(compressed_calls)
    self.assertGreater(compressed_calls[0][0][1], 0.0)

  @mock.patch.object(async_io_engine.jax.monitoring, 'record_scalar')
  def test_log_io_metrics_compression_ratio_no_compression(
      self, mock_record_scalar
  ):
    # Capture initial_ts_metrics, but perform no TensorStore writes,
    # so compressed_bytes will be 0.
    initial_ts_metrics = (
        async_io_engine.ts.experimental_collect_matching_metrics(
            '/tensorstore/'
        )
    )

    async_io_engine.log_io_metrics(
        size=4000,
        start_time=12345.0,
        gbytes_per_sec_metric='/jax/orbax/write/gbytes_per_sec',
        initial_ts_metrics=initial_ts_metrics,
    )

    # Ensure compression_ratio was NOT recorded
    for call in mock_record_scalar.call_args_list:
      self.assertNotEqual(call[0][0], '/jax/orbax/write/compression_ratio')


if __name__ == '__main__':
  absltest.main()
