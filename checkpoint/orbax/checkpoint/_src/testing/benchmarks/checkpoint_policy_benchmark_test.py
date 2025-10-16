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

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import jax.numpy as jnp
from orbax.checkpoint._src.checkpoint_managers import preservation_policy as preservation_policy_lib
from orbax.checkpoint._src.checkpoint_managers import save_decision_policy as save_decision_policy_lib
from orbax.checkpoint._src.testing.benchmarks import checkpoint_policy_benchmark
from orbax.checkpoint._src.testing.benchmarks.core import configs as benchmarks_configs
from orbax.checkpoint._src.testing.benchmarks.core import core as benchmarks_core


CheckpointPolicyBenchmarkOptions = (
    checkpoint_policy_benchmark.CheckpointPolicyBenchmarkOptions
)
CheckpointPoliciesOptions = (
    checkpoint_policy_benchmark.CheckpointPoliciesOptions
)
PreservationPolicyOptions = (
    checkpoint_policy_benchmark.PreservationPolicyOptions
)
SaveDecisionPolicyOptions = (
    checkpoint_policy_benchmark.SaveDecisionPolicyOptions
)
CheckpointPolicyBenchmark = (
    checkpoint_policy_benchmark.CheckpointPolicyBenchmark
)


class CheckpointPolicyBenchmarkTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          options=CheckpointPolicyBenchmarkOptions(
              num_checkpoints=1000,
              checkpoint_policies_options=[
                  CheckpointPoliciesOptions(
                      preservation_policies=[
                          PreservationPolicyOptions(
                              policy_type='LatestN', n=100
                          ),
                      ],
                      save_decision_policies=[
                          SaveDecisionPolicyOptions(
                              policy_type='FixedIntervalPolicy',
                              interval_steps=20,
                          ),
                      ],
                      expected_preserve_checkpoints=[1, 2, 3, 4, 5],
                  ),
              ],
          ),
          expected_len=1,
      ),
  )
  def test_get_checkpoint_policies_returns_correct_policy_types(
      self, options, expected_len
  ):
    generator = CheckpointPolicyBenchmark(
        checkpoint_configs=[benchmarks_configs.CheckpointConfig(spec={})],
        options=options,
    )
    benchmarks = generator.generate()

    self.assertLen(benchmarks, expected_len)
    for benchmark in benchmarks:
      self.assertIsInstance(benchmark.options, CheckpointPolicyBenchmarkOptions)
      preservation_policy, save_decision_policy = (
          generator._get_checkpoint_policies(
              benchmark.options.checkpoint_policies_options
          )
      )
      self.assertIsInstance(
          preservation_policy,
          preservation_policy_lib.AnyPreservationPolicy,
      )
      self.assertIsInstance(
          save_decision_policy, save_decision_policy_lib.AnySavePolicy
      )

  @parameterized.parameters(
      dict(
          options=CheckpointPolicyBenchmarkOptions(
              num_checkpoints=100,
              checkpoint_policies_options=CheckpointPoliciesOptions(
                  preservation_policies=[
                      PreservationPolicyOptions(policy_type='LatestN', n=3),
                  ],
                  save_decision_policies=[
                      SaveDecisionPolicyOptions(
                          policy_type='FixedIntervalPolicy', interval_steps=10
                      ),
                  ],
                  expected_preserve_checkpoints=[70, 80, 90],
              ),
          ),
      ),
  )
  def test_benchmark_test_fn_success(self, options):
    generator = CheckpointPolicyBenchmark(
        checkpoint_configs=[benchmarks_configs.CheckpointConfig(spec={})],
        options=options,
    )
    pytree = {'a': jnp.arange(1000)}
    test_path = epath.Path(self.create_tempdir().full_path)
    context = benchmarks_core.TestContext(
        pytree=pytree,
        path=test_path,
        options=options,
    )
    generator.test_fn(context)


if __name__ == '__main__':
  absltest.main()
