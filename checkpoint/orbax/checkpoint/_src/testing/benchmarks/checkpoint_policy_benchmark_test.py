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

"""Tests for Benchmark for Orbax checkpoint policies.."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import jax.numpy as jnp
import orbax.checkpoint as ocp
from orbax.checkpoint._src.checkpoint_managers import preservation_policy as preservation_policy_lib
from orbax.checkpoint._src.checkpoint_managers import save_decision_policy as save_decision_policy_lib
from orbax.checkpoint._src.handlers import pytree_checkpoint_handler
from orbax.checkpoint._src.testing.benchmarks import checkpoint_policy_benchmark
from orbax.checkpoint._src.testing.benchmarks.core import configs as benchmarks_configs
from orbax.checkpoint._src.testing.benchmarks.core import core as benchmarks_core


CheckpointPolicyBenchmarkOptions = (
    checkpoint_policy_benchmark.CheckpointPolicyBenchmarkOptions
)
checkpoint_policy_benchmark = (
    checkpoint_policy_benchmark.CheckpointPolicyBenchmark
)


class CheckpointPolicyBenchmarkTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_checkpoint_manager = self.enter_context(
        mock.patch.object(
            ocp.checkpoint_manager, 'CheckpointManager', autospec=True
        )
    )
    self.mock_handler = self.enter_context(
        mock.patch.object(
            pytree_checkpoint_handler, 'PyTreeCheckpointHandler', autospec=True
        )
    )

  @parameterized.parameters(
      dict(
          options=CheckpointPolicyBenchmarkOptions(
              num_checkpoints=1000,
          ),
          expected_len=1,
      ),
  )
  def test_generate_benchmarks(self, options, expected_len):
    generator = checkpoint_policy_benchmark(
        checkpoint_config=benchmarks_configs.CheckpointConfig(spec={}),
        options=options,
    )
    benchmarks = generator.generate()
    self.assertLen(benchmarks, expected_len)
    for benchmark in benchmarks:
      self.assertIsInstance(
          benchmark.options, CheckpointPolicyBenchmarkOptions
      )

  @parameterized.parameters(
      dict(
          options=CheckpointPolicyBenchmarkOptions(
              num_checkpoints=1000,
              preservation_policy_types=(
                  'LatestN,EveryNSeconds,EveryNSteps,CustomSteps'
              ),
              save_decision_policy_types='FixedIntervalPolicy,SpecificStepsPolicy,ContinuousCheckpointingPolicy,PreemptionCheckpointingPolicy,InitialSavePolicy',
              preservation_policy_n=100,
              preservation_policy_interval_secs=10,
              preservation_policy_interval_steps=10,
              preservation_policy_custom_steps=[1, 2, 3, 4, 5],
              save_decision_policy_interval_steps=10,
              save_decision_policy_interval_secs=10,
              save_decision_policy_custom_steps=[1, 2, 3, 4, 5],
          ),
          expected_preservation_policy=preservation_policy_lib.AnyPreservationPolicy,
          expected_save_decision_policy=save_decision_policy_lib.AnySavePolicy,
      ),
  )
  def test_get_policy(
      self, options, expected_preservation_policy, expected_save_decision_policy
  ):
    generator = checkpoint_policy_benchmark(
        checkpoint_config=benchmarks_configs.CheckpointConfig(spec={}),
        options=options,
    )
    preservation_policy, save_decision_policy = (
        generator._get_checkpoint_policies(options)
    )
    self.assertIsInstance(preservation_policy, expected_preservation_policy)
    self.assertIsInstance(save_decision_policy, expected_save_decision_policy)

  @parameterized.parameters(
      dict(
          options=CheckpointPolicyBenchmarkOptions(
              num_checkpoints=10,
              preservation_policy_types=(
                  'LatestN,EveryNSeconds,EveryNSteps,CustomSteps'
              ),
              save_decision_policy_types='ContinuousCheckpointingPolicy',
              preservation_policy_n=2,
              preservation_policy_interval_secs=10,
              preservation_policy_interval_steps=3,
              preservation_policy_custom_steps=[3, 5],
              save_decision_policy_interval_steps=10,
              save_decision_policy_interval_secs=10,
              save_decision_policy_custom_steps=[1, 2, 3, 4, 5],
          ),
      ),
  )
  def test_benchmark_test_fn(self, options):
    generator = checkpoint_policy_benchmark(
        checkpoint_config=benchmarks_configs.CheckpointConfig(spec={}),
        options=CheckpointPolicyBenchmarkOptions(),
    )
    pytree = {'a': jnp.arange(1000)}
    test_path = epath.Path(self.create_tempdir().full_path)
    context = benchmarks_core.TestContext(
        pytree=pytree,
        path=test_path,
        options=options,
    )
    generator.test_fn(context)

  def test_options_class(self):
    self.assertEqual(
        CheckpointPolicyBenchmarkOptions,
        checkpoint_policy_benchmark.options_class,
    )


if __name__ == '__main__':
  absltest.main()
