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

"""Benchmark for Orbax checkpoint policies."""

from collections.abc import Sequence
import dataclasses
from typing import Any

from absl import logging
import orbax.checkpoint as ocp
from orbax.checkpoint._src.checkpoint_managers import preservation_policy as preservation_policy_lib
from orbax.checkpoint._src.checkpoint_managers import save_decision_policy as save_decision_policy_lib
from orbax.checkpoint._src.testing.benchmarks.core import core as benchmarks_core
from orbax.checkpoint._src.testing.benchmarks.core import metric as metric_lib


# ==============================================================================
# 1. Define the Options Dataclass
# ==============================================================================


@dataclasses.dataclass(frozen=True)
class SaveDecisionPolicyOptions:
  """Options for save decision policy."""

  policy_type: str
  interval_steps: int | None = None
  custom_steps: Sequence[int] | None = None
  minimum_interval_secs: int | None = None

  def __repr__(self) -> str:
    if self.policy_type == 'FixedIntervalPolicy':
      return f'FixedIntervalPolicy_{self.interval_steps}'
    elif self.policy_type == 'SpecificStepsPolicy':
      custom_steps_str = '_'.join(str(step) for step in self.custom_steps)
      return f'SpecificStepsPolicy_{custom_steps_str}'
    elif self.policy_type == 'ContinuousCheckpointingPolicy':
      return f'ContinuousCheckpointingPolicy_{self.minimum_interval_secs}'
    elif self.policy_type == 'InitialSavePolicy':
      return 'InitialSavePolicy'
    elif self.policy_type == 'PreemptionSavePolicy':
      return 'PreemptionSavePolicy'
    else:
      return self.policy_type

  def get_policy(self) -> save_decision_policy_lib.SaveDecisionPolicy:
    """Returns a save decision policy instance based on the options."""

    if self.policy_type == 'FixedIntervalPolicy':
      return save_decision_policy_lib.FixedIntervalPolicy(
          interval=self.interval_steps
      )
    elif self.policy_type == 'SpecificStepsPolicy':
      return save_decision_policy_lib.SpecificStepsPolicy(
          steps=self.custom_steps
      )
    elif self.policy_type == 'ContinuousCheckpointingPolicy':
      return save_decision_policy_lib.ContinuousCheckpointingPolicy(
          minimum_interval_secs=self.minimum_interval_secs
      )
    elif self.policy_type == 'InitialSavePolicy':
      return save_decision_policy_lib.InitialSavePolicy()
    elif self.policy_type == 'PreemptionSavePolicy':
      return save_decision_policy_lib.PreemptionCheckpointingPolicy()
    else:
      raise ValueError(f'Unsupported policy type: {self.policy_type}')


@dataclasses.dataclass(frozen=True)
class PreservationPolicyOptions:
  """Options for preservation policy."""

  policy_type: str
  n: int | None = None
  interval_secs: int | None = None
  interval_steps: int | None = None
  custom_steps: Sequence[int] | None = None

  def __repr__(self) -> str:
    # We only want the policy type and any primary value (n or interval_secs)
    if self.policy_type == 'LatestN':
      return f'LatestN_{self.n}'
    elif self.policy_type == 'EveryNSeconds':
      return f'EveryNSeconds_{self.interval_secs}'
    elif self.policy_type == 'EveryNSteps':
      return f'EveryNSteps_{self.interval_steps}'
    elif self.policy_type == 'CustomSteps':
      return f'CustomSteps_{"_".join(str(step) for step in self.custom_steps)}'
    else:
      return self.policy_type

  def get_policy(self) -> preservation_policy_lib.PreservationPolicy:
    """Returns a preservation policy instance based on the options."""

    if self.policy_type == 'LatestN':
      return preservation_policy_lib.LatestN(n=self.n)
    elif self.policy_type == 'EveryNSeconds':
      return preservation_policy_lib.EveryNSeconds(
          interval_secs=self.interval_secs
      )
    elif self.policy_type == 'EveryNSteps':
      return preservation_policy_lib.EveryNSteps(
          interval_steps=self.interval_steps
      )
    elif self.policy_type == 'CustomSteps':
      return preservation_policy_lib.CustomSteps(steps=self.custom_steps)
    else:
      raise ValueError(f'Unsupported policy type: {self.policy_type}')


@dataclasses.dataclass(frozen=True)
class CheckpointPoliciesOptions:
  """Options for preservation and save decision policies."""

  preservation_policies: list[PreservationPolicyOptions]
  save_decision_policies: list[SaveDecisionPolicyOptions]
  expected_preserve_checkpoints: Sequence[int] | None = None

  def __repr__(self) -> str:
    preservation_str = '_'.join(repr(p) for p in self.preservation_policies)
    save_str = '_'.join(repr(p) for p in self.save_decision_policies)
    return f'preservation_policies_{preservation_str}_save_decision_policies_{save_str}'


@dataclasses.dataclass(frozen=True)
class CheckpointPolicyBenchmarkOptions(benchmarks_core.BenchmarkOptions):
  """Configuration options for benchmarks targeting checkpoint policies.

  Attributes:
    num_checkpoints: Number of checkpoints to simulate.
    policy_type: The type of policy to benchmark: 'BestN', 'LatestN',
      'EveryNSteps', 'EveryNSeconds'.
    n: The 'n' parameter for 'BestN' and 'LatestN' policies.
    interval: The interval parameter for 'EveryNSteps' (in steps) or
      'EveryNSeconds' (in seconds).
  """

  checkpoint_policies_options: (
      CheckpointPoliciesOptions | Sequence[CheckpointPoliciesOptions]
  )
  num_checkpoints: int | Sequence[int] = 1000

  @classmethod
  def from_dict(
      cls, options_dict: dict[str, Any]
  ) -> 'CheckpointPolicyBenchmarkOptions':
    """Builds CheckpointPolicyBenchmarkOptions from a dictionary."""
    options_copy = options_dict.copy()
    cpo_list = []
    for opts in options_copy['checkpoint_policies_options']:
      ppo_list = [
          PreservationPolicyOptions(**p) for p in opts['preservation_policies']
      ]
      sdo_list = [
          SaveDecisionPolicyOptions(**s) for s in opts['save_decision_policies']
      ]
      cpo_list.append(
          CheckpointPoliciesOptions(
              preservation_policies=ppo_list,
              save_decision_policies=sdo_list,
              expected_preserve_checkpoints=opts.get(
                  'expected_preserve_checkpoints'
              ),
          )
      )
    options_copy['checkpoint_policies_options'] = cpo_list
    return cls(**options_copy)


# ==============================================================================
# 2. Implement the Benchmark Generator
# ==============================================================================
@benchmarks_core.benchmark_options(CheckpointPolicyBenchmarkOptions)
class CheckpointPolicyBenchmark(benchmarks_core.BenchmarksGenerator):
  """A generator for benchmarking CheckpointPolicy."""

  def _get_checkpoint_policies(
      self, policies_options: CheckpointPoliciesOptions
  ) -> tuple[
      preservation_policy_lib.PreservationPolicy | None,
      save_decision_policy_lib.SaveDecisionPolicy | None,
  ]:
    """Returns checkpoint policies instance based on options."""

    preservation_policies_instances = []
    for preservation_policy_options in policies_options.preservation_policies:
      preservation_policies_instances.append(
          preservation_policy_options.get_policy()
      )
    preservation_policy = preservation_policy_lib.AnyPreservationPolicy(
        policies=preservation_policies_instances
    )

    save_decision_policies_instances = []
    for save_decision_policy_options in policies_options.save_decision_policies:
      save_decision_policies_instances.append(
          save_decision_policy_options.get_policy()
      )
    save_decision_policy = save_decision_policy_lib.AnySavePolicy(
        policies=save_decision_policies_instances
    )
    return preservation_policy, save_decision_policy

  def test_fn(
      self, context: benchmarks_core.TestContext
  ) -> benchmarks_core.TestResult:
    """The core test logic for a single benchmark run."""
    metrics = metric_lib.Metrics()
    pytree = context.pytree
    save_path = context.path / 'checkpoints'
    options = context.options
    assert isinstance(options, CheckpointPolicyBenchmarkOptions)

    policies_options = options.checkpoint_policies_options
    assert isinstance(policies_options, CheckpointPoliciesOptions)

    preservation_policy, save_decision_policy = self._get_checkpoint_policies(
        policies_options
    )
    checkpointer_manager = ocp.CheckpointManager(
        save_path,
        options=ocp.CheckpointManagerOptions(
            preservation_policy=preservation_policy,
            save_decision_policy=save_decision_policy,
        ),
    )
    for step in range(options.num_checkpoints):
      with metrics.time(f'saving step {step}'):
        checkpointer_manager.save(step, args=ocp.args.PyTreeSave(pytree))
    checkpointer_manager.wait_until_finished()
    all_steps = checkpointer_manager.all_steps()
    assert len(all_steps) >= 1
    assert len(all_steps) <= options.num_checkpoints
    logging.info('all_steps: %s', all_steps)

    if policies_options.expected_preserve_checkpoints is not None:
      missing_steps = set(policies_options.expected_preserve_checkpoints) - set(
          all_steps
      )
      if missing_steps:
        raise AssertionError(
            f'Expected steps {policies_options.expected_preserve_checkpoints} '
            f'were not all found in preserved steps {all_steps}. '
            f'Missing steps: {sorted(list(missing_steps))}'
        )
    return benchmarks_core.TestResult(metrics=metrics)
