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

from unittest import mock

from absl.testing import absltest
from absl.testing import flagsaver
from etils import epath
import jax
import numpy as np
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.multihost import multislice
from orbax.checkpoint._src.testing.benchmarks import multislice_broadcast_benchmark
from orbax.checkpoint._src.testing.benchmarks.core import configs as benchmarks_configs
from orbax.checkpoint._src.testing.benchmarks.core import core as benchmarks_core
from orbax.checkpoint._src.testing.benchmarks.core import mesh_utils

MultisliceBroadcastBenchmarkOptions = (
    multislice_broadcast_benchmark.MultisliceBroadcastBenchmarkOptions
)
MultisliceBroadcastBenchmark = (
    multislice_broadcast_benchmark.MultisliceBroadcastBenchmark
)


class MultisliceBroadcastBenchmarkTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(flagsaver.flagsaver())
    for name in (
        'is_runtime_to_distributed_ids_initialized',
        'is_distributed_to_device_ids_initialized',
    ):
      self.enter_context(
          mock.patch.object(multihost, name, autospec=True, return_value=True)
      )
    # Two replicas when the devices split evenly, else one; arrays are sharded
    # within a replica. Works with however many devices the test runs on.
    num_devices = jax.device_count()
    num_replicas = 2 if num_devices % 2 == 0 else 1
    self.devices_per_replica = num_devices // num_replicas
    devices = np.array(jax.devices()).reshape(
        num_replicas, self.devices_per_replica
    )
    self.mesh = jax.sharding.Mesh(devices, ('replica', 'data'))
    # One process holds both replicas here, so pick replica 0 by hand; the
    # lookup by process would find both.
    self.enter_context(
        mock.patch.object(
            mesh_utils,
            'get_local_replica_mesh',
            autospec=True,
            return_value=jax.sharding.Mesh(devices[:1], ('replica', 'data')),
        )
    )
    self.enter_context(
        mock.patch.object(
            multislice,
            'process_spans_multiple_replicas',
            autospec=True,
            return_value=False,
        )
    )
    self.in_replica = self.enter_context(
        mock.patch.object(
            multislice, 'in_replica', autospec=True, return_value=True
        )
    )
    sharding = jax.sharding.NamedSharding(
        self.mesh, jax.sharding.PartitionSpec('data')
    )
    self.shape = (4 * num_devices, 4)
    self.pytree = {
        'a': jax.device_put(
            np.arange(np.prod(self.shape), dtype=np.float32).reshape(
                self.shape
            ),
            sharding,
        ),
    }
    self.benchmark = MultisliceBroadcastBenchmark(
        checkpoint_configs=[benchmarks_configs.CheckpointConfig(spec={})],
        options=MultisliceBroadcastBenchmarkOptions(),
    )

  def _context(self) -> benchmarks_core.TestContext:
    return benchmarks_core.TestContext(
        pytree=self.pytree,
        path=epath.Path(self.create_tempdir().full_path),
        options=MultisliceBroadcastBenchmarkOptions(replica_axis_index=0),
        mesh=self.mesh,
    )

  def test_source_replica_broadcasts_its_local_shards(self):
    broadcast_inputs = []
    real_broadcast = multislice.broadcast_one_replica_to_all

    def record_and_broadcast(in_tree, *args, **kwargs):
      # The broadcast deletes its inputs, so record them first.
      broadcast_inputs.extend(
          (arr.shape, len(arr.sharding.device_set), np.asarray(arr))
          for arr in in_tree
      )
      return real_broadcast(in_tree, *args, **kwargs)

    broadcast = self.enter_context(
        mock.patch.object(
            multislice,
            'broadcast_one_replica_to_all',
            autospec=True,
            side_effect=record_and_broadcast,
        )
    )

    # A replica can span hosts, so no host may need the whole source array.
    self.enter_context(
        mock.patch.object(
            jax,
            'device_get',
            autospec=True,
            side_effect=AssertionError('fetched the whole source array'),
        )
    )

    # test_fn checks that every replica received the source values.
    self.benchmark.test_fn(self._context())

    self.assertTrue(broadcast.call_args.kwargs['is_source'])
    self.assertLen(broadcast_inputs, 1)
    shape, num_devices, values = broadcast_inputs[0]
    self.assertEqual(shape, self.shape)
    self.assertEqual(num_devices, self.devices_per_replica)
    np.testing.assert_array_equal(values, np.asarray(self.pytree['a']))
    # The broadcast consumed a copy, so the expected array survives it.
    self.assertFalse(self.pytree['a'].is_deleted())

  def test_other_replicas_send_zeros(self):
    self.in_replica.return_value = False
    broadcast = self.enter_context(
        mock.patch.object(
            multislice,
            'broadcast_one_replica_to_all',
            autospec=True,
            return_value=((self.pytree['a'],), 1),
        )
    )

    self.benchmark.test_fn(self._context())

    (single_replica_arr,), *_ = broadcast.call_args.args
    self.assertFalse(broadcast.call_args.kwargs['is_source'])
    self.assertEqual(single_replica_arr.shape, self.shape)
    self.assertEqual(single_replica_arr.dtype, np.float32)
    self.assertLen(
        single_replica_arr.sharding.device_set, self.devices_per_replica
    )
    np.testing.assert_array_equal(
        np.asarray(single_replica_arr), np.zeros(self.shape, np.float32)
    )


if __name__ == '__main__':
  absltest.main()
