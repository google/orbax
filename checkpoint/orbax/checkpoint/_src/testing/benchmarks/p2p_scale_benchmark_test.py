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

"""Unit tests for p2p_scale_benchmark."""

import itertools
import json
import types
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import jax
import numpy as np
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.testing.benchmarks import p2p_scale_benchmark as p2p_scale
from orbax.checkpoint._src.testing.benchmarks.core import backends
from orbax.checkpoint._src.testing.benchmarks.core import configs as benchmarks_configs
from orbax.checkpoint._src.testing.benchmarks.core import core as benchmarks_core
from orbax.checkpoint.experimental.emergency import path as emergency_path

Options = p2p_scale.P2pScaleBenchmarkOptions


class ScenarioTest(parameterized.TestCase):

  def test_replica_topology(self):
    mesh = mock.MagicMock(spec=jax.sharding.Mesh)
    mesh.devices = np.array([
        [types.SimpleNamespace(process_index=p) for p in row]
        for row in [[0, 0, 1], [2, 3, 3]]
    ])
    mesh.shape_tuple = (('replica', 2), ('data', 3))
    self.assertEqual(
        p2p_scale.replica_topology(mesh, replica_axis_index=0),
        [[0, 1], [2, 3]],
    )

  @parameterized.parameters(
      dict(scenario='all_local', selection='one_host', expected=set()),
      dict(scenario='lost_hosts', selection='one_host', expected={3}),
      dict(
          scenario='lost_hosts', selection='replica_minus_one', expected={3, 4}
      ),
      # Half of 6 processes; relative ids 0, 1 and 2 each keep a copy.
      dict(scenario='lost_hosts', selection='spread', expected={3, 1, 5}),
      dict(
          scenario='lost_replica_shard', selection='one_host', expected={0, 3}
      ),
  )
  def test_lost_processes(self, scenario, selection, expected):
    options = Options(
        scenario=scenario,
        lost_hosts_selection=selection,
        lost_hosts_fraction=0.5,
    )
    self.assertEqual(
        p2p_scale.lost_processes([[0, 1, 2], [3, 4, 5]], options), expected
    )

  @parameterized.parameters(
      dict(topology=[[0, 1]], error='at least two replicas'),
      dict(topology=[[0], [0]], error='exactly one replica'),
  )
  def test_lost_processes_rejects_topology(self, topology, error):
    with self.assertRaisesRegex(ValueError, error):
      p2p_scale.lost_processes(topology, Options())

  def test_rejects_unknown_scenario(self):
    self.assertFalse(Options(scenario='x').is_valid())


class KvExchangeTest(absltest.TestCase):

  def test_simdevice_exchanges_host_values_through_kv_store(self):
    # Process 1's values are already in the store; this test is process 0.
    store = {
        'p2p_scale_kv/0_sync/1': json.dumps({'pid': 1}),
        'p2p_scale_kv/1_max/1': json.dumps([0, 3]),
    }
    client = mock.MagicMock()
    client.key_value_set.side_effect = store.__setitem__
    client.key_value_set_bytes.side_effect = store.__setitem__
    client.key_value_dir_get.side_effect = lambda prefix: [
        (k, v) for k, v in store.items() if k.startswith(prefix + '/')
    ]
    client.blocking_key_value_get_bytes.side_effect = lambda key, _: store[key]
    originals = (emergency_path.sync_global_data, multihost.global_max)
    with (
        mock.patch.object(backends, 'is_simdevice', return_value=True),
        mock.patch.object(jax, 'process_count', return_value=2),
        mock.patch.object(multihost, 'process_index', return_value=0),
        mock.patch.object(
            multihost, 'get_jax_distributed_client', return_value=client
        ),
        mock.patch.object(p2p_scale, '_kv_exchanges', itertools.count()),
    ):
      with p2p_scale._kv_host_exchanges():
        self.assertEqual(
            emergency_path.sync_global_data({'pid': 0}),
            [{'pid': 0}, {'pid': 1}],
        )
        self.assertEqual(multihost.global_max([1, 2]), [1, 3])
    self.assertEqual(
        (emergency_path.sync_global_data, multihost.global_max), originals
    )


class TestFnTest(parameterized.TestCase):
  """Runs test_fn on one process of [[0, 1], [2, 3]] with a fake manager."""

  def setUp(self):
    super().setUp()
    for name in (
        'is_runtime_to_distributed_ids_initialized',
        'is_distributed_to_device_ids_initialized',
    ):
      self.enter_context(mock.patch.object(multihost, name, return_value=True))
    self.enter_context(mock.patch.object(multihost, 'sync_global_processes'))
    self.enter_context(
        mock.patch.object(
            p2p_scale, 'replica_topology', return_value=[[0, 1], [2, 3]]
        )
    )
    mesh = jax.sharding.Mesh(np.array(jax.devices()), ('data',))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    self.mesh = mesh
    self.pytree = {'a': jax.device_put(np.arange(4.0), sharding)}

  def _run(self, scenario, process_index):
    """Runs test_fn and returns its metrics and this process's local dir."""

    local_steps = []  # Whether local/<step> existed at each restore.

    def create_manager(local_directory, **kwargs):
      del kwargs
      manager = mock.MagicMock()

      def save(step, args):
        del args
        (local_directory / str(step)).mkdir()
        (local_directory / str(step) / 'shard').write_bytes(b'x' * 32)

      def restore(step, args):
        del args
        local_steps.append((local_directory / str(step)).exists())
        return {'state': self.pytree}

      manager.save.side_effect = save
      manager.restore.side_effect = restore
      return manager

    self.enter_context(
        mock.patch.object(
            p2p_scale.p2p_manager,
            'CheckpointManager',
            side_effect=create_manager,
        )
    )
    self.enter_context(
        mock.patch.object(
            multihost, 'process_index', return_value=process_index
        )
    )
    options = Options(scenario=scenario)
    benchmark = p2p_scale.P2pScaleBenchmark(
        checkpoint_configs=[benchmarks_configs.CheckpointConfig(spec={})],
        options=options,
    )
    results = benchmark.test_fn(
        benchmarks_core.TestContext(
            pytree=self.pytree,
            path=epath.Path(self.create_tempdir().full_path),
            options=options,
            mesh=self.mesh,
        )
    ).metrics.results
    return results, local_steps

  @parameterized.parameters(
      dict(scenario='all_local', process=2, lost=False),
      # Process 2 is the first process of replica 1: the lost host.
      dict(scenario='lost_hosts', process=2, lost=True),
      dict(scenario='lost_hosts', process=1, lost=False),
      dict(scenario='lost_replica_shard', process=0, lost=True),
      dict(scenario='lost_replica_shard', process=1, lost=False),
  )
  def test_deletes_the_local_step_on_lost_hosts(self, scenario, process, lost):
    results, local_steps = self._run(scenario, process)

    self.assertEqual(local_steps, [not lost])
    self.assertEqual(results['p2p_restore::p2p/lost_host'][0], int(lost))
    self.assertEqual(results['p2p_restore::p2p/step_bytes'][0], 32)
    self.assertIn('restart_and_restore::0_basics/time_s', results)


if __name__ == '__main__':
  absltest.main()
