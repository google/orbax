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

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint import checkpoint_manager
from orbax.checkpoint._src.testing.benchmarks import checkpoint_manager_benchmark
from orbax.checkpoint._src.testing.benchmarks.core import configs as benchmarks_configs
from orbax.checkpoint._src.testing.benchmarks.core import core as benchmarks_core


CheckpointManagerBenchmarkOptions = (
    checkpoint_manager_benchmark.CheckpointManagerBenchmarkOptions
)
CheckpointManagerBenchmark = (
    checkpoint_manager_benchmark.CheckpointManagerBenchmark
)


class CheckpointManagerBenchmarkTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_checkpoint_manager_cls = self.enter_context(
        mock.patch.object(
            checkpoint_manager, 'CheckpointManager', autospec=True
        )
    )

  @parameterized.parameters(
      dict(
          options=CheckpointManagerBenchmarkOptions(train_steps=1),
          expected_len=1,
      ),
      dict(
          options=CheckpointManagerBenchmarkOptions(
              train_steps=[1, 2], max_to_keep=2
          ),
          expected_len=2,
      ),
      dict(
          options=CheckpointManagerBenchmarkOptions(
              train_steps=1, max_to_keep=[1, 2]
          ),
          expected_len=2,
      ),
  )
  def test_generate_benchmarks(self, options, expected_len):
    generator = CheckpointManagerBenchmark(
        checkpoint_configs=[benchmarks_configs.CheckpointConfig(spec={})],
        options=options,
    )
    benchmarks = generator.generate()
    self.assertLen(benchmarks, expected_len)
    for benchmark in benchmarks:
      self.assertIsInstance(
          benchmark.options, CheckpointManagerBenchmarkOptions
      )

  def test_benchmark_test_fn_succeeds(self):
    generator = CheckpointManagerBenchmark(
        checkpoint_configs=[benchmarks_configs.CheckpointConfig(spec={})],
        options=CheckpointManagerBenchmarkOptions(),
    )
    mock_cm_instance = self.mock_checkpoint_manager_cls.return_value
    mock_cm_instance.should_save.return_value = True
    mock_cm_instance.latest_step.return_value = 0

    # Mock return of restore
    pytree = {
        'a': jnp.arange(10),
    }
    json_data = {'a': 1, 'b': 'test'}
    random_key = jax.random.key(0)
    np_random_key = np.random.get_state()

    mock_cm_instance.restore.return_value = {
        'pytree': pytree,
        'json_item': json_data,
        'jax_random_key': random_key,
        'np_random_key': np_random_key,
    }

    test_path = epath.Path(self.create_tempdir().full_path)
    test_options = CheckpointManagerBenchmarkOptions(train_steps=1)
    context = benchmarks_core.TestContext(
        pytree=pytree, path=test_path, options=test_options
    )

    result = generator.test_fn(context)

    self.mock_checkpoint_manager_cls.assert_called_once()
    self.assertIsInstance(result, benchmarks_core.TestResult)
    self.assertContainsSubset(
        {'save_0', 'wait_until_finished_0', 'restore_0', 'correctness_check'},
        result.metrics.timings.keys(),
    )
    mock_cm_instance.save.assert_called_once()
    mock_cm_instance.wait_until_finished.assert_called_once()
    mock_cm_instance.restore.assert_called_once()
    mock_cm_instance.close.assert_called_once()

  def test_options_class(self):
    self.assertEqual(
        CheckpointManagerBenchmarkOptions,
        CheckpointManagerBenchmark.options_class,
    )


if __name__ == '__main__':
  absltest.main()
