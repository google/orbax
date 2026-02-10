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

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import jax
import jax.numpy as jnp
from orbax.checkpoint import v1 as ocp
from orbax.checkpoint._src.testing.benchmarks import v1_resharding_benchmark
from orbax.checkpoint._src.testing.benchmarks.core import configs as benchmarks_configs
from orbax.checkpoint._src.testing.benchmarks.core import core as benchmarks_core


V1ReshardingBenchmarkOptions = (
    v1_resharding_benchmark.V1ReshardingBenchmarkOptions
)
V1ReshardingBenchmark = v1_resharding_benchmark.V1ReshardingBenchmark


class V1ReshardingBenchmarkTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(self.create_tempdir().full_path)

  @parameterized.parameters(
      dict(
          options=V1ReshardingBenchmarkOptions(
              reference_checkpoint_path='ckpt_path',
              reference_sharding_path='sharding_path',
          ),
          expected_len=1,
      ),
      dict(
          options=V1ReshardingBenchmarkOptions(
              reference_checkpoint_path='ckpt_path',
              reference_sharding_path='sharding_path',
              use_ocdbt=[False, True],
          ),
          expected_len=2,
      ),
  )
  def test_generate_benchmarks(self, options, expected_len):
    generator = V1ReshardingBenchmark(
        checkpoint_configs=[benchmarks_configs.CheckpointConfig(spec={})],
        options=options,
    )
    benchmarks = generator.generate()
    self.assertLen(benchmarks, expected_len)
    for benchmark in benchmarks:
      self.assertIsInstance(benchmark.options, V1ReshardingBenchmarkOptions)

  def test_benchmark_test_fn(self):
    # Setup real checkpoint and sharding config
    pytree = {'a': jnp.arange(8), 'b': {'c': jnp.ones((4, 4))}}
    ref_ckpt_path = self.directory / 'ref_ckpt'
    ocp.save_pytree(ref_ckpt_path, pytree)

    sharding_config = {
        'a': {
            'shape': [8],
            'dtype': 'int32',
            'sharding': {
                'mesh': {'shape': [1], 'axes': ['x']},
                'spec': [None],
            },
        },
        'b.c': {
            'shape': [4, 4],
            'dtype': 'float32',
            'sharding': {
                'mesh': {'shape': [1], 'axes': ['x']},
                'spec': [None, None],
            },
        },
    }
    sharding_config_path = self.directory / 'sharding_config.json'
    sharding_config_path.write_text(json.dumps(sharding_config))

    options = V1ReshardingBenchmarkOptions(
        reference_checkpoint_path=str(ref_ckpt_path),
        reference_sharding_path=str(sharding_config_path),
    )
    generator = V1ReshardingBenchmark(
        checkpoint_configs=[benchmarks_configs.CheckpointConfig(spec={})],
        options=options,
    )

    context = benchmarks_core.TestContext(
        pytree=None,
        path=self.directory / 'test_resharding',
        options=options,
        mesh=jax.sharding.Mesh(jax.devices(), ('x',)),
    )

    result = generator.test_fn(context)

    self.assertIsInstance(result, benchmarks_core.TestResult)
    metrics = result.metrics.results
    self.assertIn('load_time_duration', metrics)


if __name__ == '__main__':
  absltest.main()
