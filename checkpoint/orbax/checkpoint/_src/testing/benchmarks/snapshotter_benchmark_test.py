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

import collections
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from orbax.checkpoint._src.testing.benchmarks import snapshotter_benchmark
from orbax.checkpoint._src.testing.benchmarks.core import configs as benchmarks_configs
from orbax.checkpoint._src.testing.benchmarks.core import core as benchmarks_core
from orbax.checkpoint.experimental.v1._src.training.pathways import snapshotter


SnapshotterOptions = snapshotter_benchmark.SnapshotterOptions
SnapshotterBenchmark = snapshotter_benchmark.SnapshotterBenchmark


class SnapshotterBenchmarkTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_snapshotter = self.enter_context(
        mock.patch.object(snapshotter, 'Snapshotter', autospec=True)
    )
    self.mock_is_pathways_backend = self.enter_context(
        mock.patch.object(
            ocp.multihost, 'is_pathways_backend', return_value=True
        )
    )
    self.mock_block_until_ready = self.enter_context(
        mock.patch.object(jax, 'block_until_ready', autospec=True)
    )
    self.mock_construct_restore_args = self.enter_context(
        mock.patch.object(
            ocp.checkpoint_utils, 'construct_restore_args', autospec=True
        )
    )

  @parameterized.parameters(
      dict(
          options=SnapshotterOptions(),
          expected_len=1,
      ),
      dict(
          options=SnapshotterOptions(
              metric_tracemalloc_enabled=[False, True]
          ),
          expected_len=2,
      ),
  )
  def test_generate_benchmarks(self, options, expected_len):
    generator = SnapshotterBenchmark(
        checkpoint_configs=[benchmarks_configs.CheckpointConfig(spec={})],
        options=options,
    )
    benchmarks = generator.generate()
    self.assertLen(benchmarks, expected_len)
    for benchmark in benchmarks:
      self.assertIsInstance(benchmark.options, SnapshotterOptions)

  def test_benchmark_test_fn(self):
    generator = SnapshotterBenchmark(
        checkpoint_configs=[benchmarks_configs.CheckpointConfig(spec={})],
        options=SnapshotterOptions(),
    )
    mock_save = self.mock_snapshotter.return_value.save_pytree
    mock_load = self.mock_snapshotter.return_value.load_pytree
    pytree = {
        'a': jnp.arange(10),
    }
    test_path = epath.Path(self.create_tempdir().full_path)
    test_options = SnapshotterOptions()
    context = benchmarks_core.TestContext(
        pytree=pytree, path=test_path, options=test_options
    )

    # needed for jax.block_until_ready(manager._snapshots[-1][0])
    self.mock_snapshotter.return_value._snapshots = collections.deque(
        [('state', 0)]
    )

    result = generator.test_fn(context)

    self.mock_snapshotter.assert_called_once()
    self.assertIsInstance(result, benchmarks_core.TestResult)
    self.assertContainsSubset(
        {
            'save_pytree_time_duration',
            'load_pytree_time_duration',
        },
        result.metrics.results.keys(),
    )
    mock_save.assert_called_once_with(step=0, state=pytree)
    mock_load.assert_called_once()
    self.mock_block_until_ready.assert_called()
    self.mock_construct_restore_args.assert_called_once()


if __name__ == '__main__':
  absltest.main()
