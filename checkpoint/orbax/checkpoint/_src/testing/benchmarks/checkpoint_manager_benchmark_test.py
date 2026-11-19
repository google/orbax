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
_ORIGINAL_CHECKPOINT_MANAGER = checkpoint_manager.CheckpointManager


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
      dict(
          options=CheckpointManagerBenchmarkOptions(
              train_steps=2,
              overwrite_existing_step=[False, True],
              warmup_save=True,
          ),
          expected_len=2,
      ),
      dict(
          options=CheckpointManagerBenchmarkOptions(
              train_steps=2,
              overwrite_existing_step=[False, True],
              skip_sync_file_validations=[False, True],
              warmup_save=True,
          ),
          expected_len=4,
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
    test_options = CheckpointManagerBenchmarkOptions(
        train_steps=1, skip_sync_file_validations=True
    )
    context = benchmarks_core.TestContext(
        pytree=pytree, path=test_path, options=test_options
    )

    result = generator.test_fn(context)

    self.mock_checkpoint_manager_cls.assert_called_once()
    cm_options_arg = self.mock_checkpoint_manager_cls.call_args[1]['options']
    self.assertTrue(cm_options_arg.file_options.skip_sync_file_validations)
    self.assertIsInstance(result, benchmarks_core.TestResult)
    self.assertContainsSubset(
        {
            'save_0::0_basics/time_s',
            'wait_until_finished_0::0_basics/time_s',
            'restore_0::0_basics/time_s',
            'correctness_check::0_basics/time_s',
        },
        result.metrics.results.keys(),
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

  @parameterized.named_parameters(
      ('fresh_with_restore', False, True, ['0', '1', '2'], 'restore_2'),
      ('overwrite_with_restore', True, True, ['0'], 'restore_0'),
      ('overwrite_skip_restore', True, False, ['0'], None),
  )
  def test_warmup_and_overwrite_modes(
      self, overwrite, verify_restore, expected_steps, restore_key
  ):
    generator = CheckpointManagerBenchmark(
        checkpoint_configs=[benchmarks_configs.CheckpointConfig(spec={})],
        options=CheckpointManagerBenchmarkOptions(),
    )
    test_path = epath.Path(self.create_tempdir().full_path)
    options = CheckpointManagerBenchmarkOptions(
        train_steps=2,
        max_to_keep=10,
        overwrite_existing_step=overwrite,
        warmup_save=True,
        verify_restore=verify_restore,
    )
    context = benchmarks_core.TestContext(
        pytree={'a': jnp.arange(4)}, path=test_path, options=options
    )
    with mock.patch.object(
        checkpoint_manager, 'CheckpointManager', _ORIGINAL_CHECKPOINT_MANAGER
    ):
      result = generator.test_fn(context)

    metrics = result.metrics.results.keys()
    self.assertContainsSubset(
        {
            'save_0::0_basics/time_s',
            'save_1::0_basics/time_s',
            'wait_until_finished_0::0_basics/time_s',
            'wait_until_finished_1::0_basics/time_s',
        },
        metrics,
    )
    if restore_key is not None:
      self.assertContainsSubset(
          {
              f'{restore_key}::0_basics/time_s',
              'correctness_check::0_basics/time_s',
          },
          metrics,
      )
    else:
      self.assertNotIn('correctness_check::0_basics/time_s', metrics)
    self.assertIsNone(result.inventory)
    (attempt_dir,) = test_path.iterdir()
    self.assertEqual(result.path, attempt_dir)
    self.assertEqual(
        sorted(step_dir.name for step_dir in attempt_dir.iterdir()),
        expected_steps,
    )


if __name__ == '__main__':
  absltest.main()
