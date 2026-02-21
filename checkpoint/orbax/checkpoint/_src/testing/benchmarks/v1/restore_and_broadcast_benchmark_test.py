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

import json
import os

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint import v1 as ocp
from orbax.checkpoint._src.testing.benchmarks.core import configs as benchmarks_configs
from orbax.checkpoint._src.testing.benchmarks.core import core as benchmarks_core
from orbax.checkpoint._src.testing.benchmarks.v1 import restore_and_broadcast_benchmark


RestoreAndBroadcastBenchmarkOptions = (
    restore_and_broadcast_benchmark.RestoreAndBroadcastBenchmarkOptions
)
RestoreAndBroadcastBenchmark = (
    restore_and_broadcast_benchmark.RestoreAndBroadcastBenchmark
)


class RestoreAndBroadcastBenchmarkTest(parameterized.TestCase):

  def setUp(self):
    self._prev_xla_flags = os.environ.get('XLA_FLAGS')
    os.environ['XLA_FLAGS'] = (
        self._prev_xla_flags or ''
    ) + ' --xla_force_host_platform_device_count=16'
    super().setUp()
    self.assertEqual(jax.local_device_count(), 16)
    self.directory = epath.Path(self.create_tempdir().full_path)

  def tearDown(self):
    if self._prev_xla_flags is None:
      os.environ.pop('XLA_FLAGS', None)
    else:
      os.environ['XLA_FLAGS'] = self._prev_xla_flags
    super().tearDown()

  @parameterized.parameters(
      dict(
          options=RestoreAndBroadcastBenchmarkOptions(
              reference_checkpoint_path='ckpt_path',
              reference_sharding_path='sharding_path',
          ),
          expected_len=1,
      ),
  )
  def test_generate_benchmarks(self, options, expected_len):
    generator = RestoreAndBroadcastBenchmark(
        checkpoint_configs=[benchmarks_configs.CheckpointConfig(spec={})],
        options=options,
    )
    benchmarks = generator.generate()
    self.assertLen(benchmarks, expected_len)
    for benchmark in benchmarks:
      self.assertIsInstance(
          benchmark.options, RestoreAndBroadcastBenchmarkOptions
      )

  def test_get_abstract_state(self):
    # Setup real checkpoint and sharding config
    pytree = {'a': jnp.arange(32), 'b': {'c': jnp.ones((8, 8))}}
    ref_ckpt_path = self.directory / 'ref_ckpt'
    ocp.save_pytree(ref_ckpt_path, pytree)

    sharding_config = {
        'a': {
            'shape': [32],
            'dtype': 'int32',
            'sharding': {
                'mesh': {'shape': [4], 'axes': ['model']},
                'spec': ['model'],
            },
        },
        'b.c': {
            'shape': [8, 8],
            'dtype': 'float32',
            'sharding': {
                'mesh': {'shape': [4], 'axes': ['model']},
                'spec': [None, 'model'],
            },
        },
    }
    sharding_config_path = self.directory / 'sharding_config.json'
    sharding_config_path.write_text(json.dumps(sharding_config))
    global_mesh = jax.sharding.Mesh(
        np.array(jax.devices()).reshape((4, 4)), ('replica', 'model')
    )

    single_replica_abstract_pytree = (
        restore_and_broadcast_benchmark._get_single_replica_abstract_state(
            context=ocp.Context(),
            global_mesh=global_mesh,
            reference_checkpoint_path=ref_ckpt_path,
            reference_sharding_path=sharding_config_path,
        )
    )
    self.assertEqual(
        single_replica_abstract_pytree['a'].sharding.mesh.shape, {'model': 4}
    )
    self.assertEqual(
        single_replica_abstract_pytree['a'].sharding.spec,
        jax.sharding.PartitionSpec('model'),
    )
    self.assertEqual(
        single_replica_abstract_pytree['b']['c'].sharding.mesh.shape,
        {'model': 4},
    )
    self.assertEqual(
        single_replica_abstract_pytree['b']['c'].sharding.spec,
        jax.sharding.PartitionSpec(None, 'model'),
    )

    abstract_pytree = restore_and_broadcast_benchmark._get_abstract_state(
        context=ocp.Context(),
        global_mesh=global_mesh,
        single_replica_abstract_state=single_replica_abstract_pytree,
    )
    self.assertEqual(
        abstract_pytree['a'].sharding.mesh.shape, {'replica': 4, 'model': 4}
    )
    self.assertEqual(
        abstract_pytree['a'].sharding.spec, jax.sharding.PartitionSpec('model')
    )
    self.assertEqual(
        abstract_pytree['b']['c'].sharding.mesh.shape,
        {'replica': 4, 'model': 4},
    )
    self.assertEqual(
        abstract_pytree['b']['c'].sharding.spec,
        jax.sharding.PartitionSpec(None, 'model'),
    )

  def test_benchmark_test_fn(self):
    # Setup real checkpoint and sharding config
    pytree = {'a': jnp.arange(32), 'b': {'c': jnp.ones((8, 8))}}
    ref_ckpt_path = self.directory / 'ref_ckpt'
    ocp.save_pytree(ref_ckpt_path, pytree)

    sharding_config = {
        'a': {
            'shape': [32],
            'dtype': 'int32',
            'sharding': {
                'mesh': {'shape': [4], 'axes': ['model']},
                'spec': ['model'],
            },
        },
        'b.c': {
            'shape': [8, 8],
            'dtype': 'float32',
            'sharding': {
                'mesh': {'shape': [4], 'axes': ['model']},
                'spec': [None, 'model'],
            },
        },
    }
    sharding_config_path = self.directory / 'sharding_config.json'
    sharding_config_path.write_text(json.dumps(sharding_config))

    options = RestoreAndBroadcastBenchmarkOptions(
        reference_checkpoint_path=str(ref_ckpt_path),
        reference_sharding_path=str(sharding_config_path),
    )
    global_mesh = jax.sharding.Mesh(
        np.array(jax.devices()).reshape((4, 4)), ('replica', 'model')
    )

    context = benchmarks_core.TestContext(
        pytree=None,
        path=self.directory / 'test_run',
        options=options,
        mesh=global_mesh,
    )
    self.assertTrue(options.use_load_and_broadcast)
    self.assertTrue(
        options.context.array_options.loading.use_load_and_broadcast
    )

    generator = RestoreAndBroadcastBenchmark(
        checkpoint_configs=[benchmarks_configs.CheckpointConfig(spec={})],
        options=options,
    )
    result = generator.test_fn(context)
    self.assertIsInstance(result, benchmarks_core.TestResult)
    metrics = result.metrics.results
    self.assertIn('load_time_duration', metrics)


if __name__ == '__main__':
  absltest.main()
