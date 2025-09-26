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
import datetime
import time
from typing import Any

import orbax.checkpoint as ocp
from orbax.checkpoint._src.checkpoint_managers import policy_checkpoint_info
from orbax.checkpoint._src.checkpoint_managers import preservation_policy as preservation_policy_lib
from orbax.checkpoint._src.checkpoint_managers import save_decision_policy as save_decision_policy_lib
from orbax.checkpoint._src.metadata import checkpoint_info
from orbax.checkpoint._src.testing.benchmarks.core import core as benchmarks_core


PolicyCheckpointInfo = policy_checkpoint_info.PolicyCheckpointInfo


# ==============================================================================
# 1. Define the Options Dataclass
# ==============================================================================
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

  num_checkpoints: int | Sequence[int] = 1000
  preservation_policy_types: str | Sequence[str] = 'BestN'
  save_decision_policy_types: str | Sequence[str] = 'EveryNSteps'
  preservation_policy_n: int | Sequence[int] = 10
  preservation_policy_interval_secs: int | Sequence[int] = 10
  preservation_policy_interval_steps: int | Sequence[int] = 10
  preservation_policy_custom_steps: Sequence[int] = (1, 2, 3, 4, 5)
  save_decision_policy_interval_steps: int | Sequence[int] = 10
  save_decision_policy_interval_secs: int | Sequence[int] = 10
  save_decision_policy_custom_steps: Sequence[int] = (1, 2, 3, 4, 5)
  save_decision_policy_min_interval_secs: int | Sequence[int] = 2


# ==============================================================================
# 2. Implement the Benchmark Generator
# ==============================================================================
@benchmarks_core.benchmark_options(CheckpointPolicyBenchmarkOptions)
class CheckpointPolicyBenchmark(benchmarks_core.BenchmarksGenerator):
  """A generator for benchmarking CheckpointPolicy."""

  def _get_checkpoint_policies(
      self, options: CheckpointPolicyBenchmarkOptions
  ) -> tuple[
      preservation_policy_lib.PreservationPolicy | None,
      save_decision_policy_lib.SaveDecisionPolicy | None,
  ]:
    """Returns checkpoint policies instance based on options."""
    preservation_policies = []
    save_decision_policies = []
    all_preservation_policies = (
        options.preservation_policy_types.split(',')
        if isinstance(options.preservation_policy_types, str)
        else options.preservation_policy_types
    )
    all_save_decision_policies = (
        options.save_decision_policy_types.split(',')
        if isinstance(options.save_decision_policy_types, str)
        else options.save_decision_policy_types
    )
    for preservation_policy_type in all_preservation_policies:
      if preservation_policy_type == 'PreserveAll':
        preservation_policies.append(preservation_policy_lib.PreserveAll())
      elif preservation_policy_type == 'LatestN':
        preservation_policies.append(
            preservation_policy_lib.LatestN(n=options.preservation_policy_n)
        )
      elif preservation_policy_type == 'EveryNSeconds':
        preservation_policies.append(
            preservation_policy_lib.EveryNSeconds(
                interval_secs=options.preservation_policy_interval_secs
            )
        )
      elif preservation_policy_type == 'EveryNSteps':
        preservation_policies.append(
            preservation_policy_lib.EveryNSteps(
                interval_steps=(
                    options.preservation_policy_interval_steps
                )
            )
        )
      elif preservation_policy_type == 'CustomSteps':
        preservation_policies.append(
            preservation_policy_lib.CustomSteps(
                steps=options.preservation_policy_custom_steps
            )
        )
      else:
        raise ValueError(
            f'Unsupported preservation_policy_type: {preservation_policy_type}'
        )
    for save_decision_policy_type in all_save_decision_policies:
      if save_decision_policy_type == 'FixedIntervalPolicy':
        save_decision_policies.append(
            save_decision_policy_lib.FixedIntervalPolicy(
                interval=options.save_decision_policy_interval_steps
            )
        )
      elif save_decision_policy_type == 'SpecificStepsPolicy':
        save_decision_policies.append(
            save_decision_policy_lib.SpecificStepsPolicy(
                steps=options.save_decision_policy_custom_steps
            )
        )
      elif save_decision_policy_type == 'ContinuousCheckpointingPolicy':
        save_decision_policies.append(
            save_decision_policy_lib.ContinuousCheckpointingPolicy(
                minimum_interval_secs=options.save_decision_policy_min_interval_secs
            )
        )
      elif save_decision_policy_type == 'PreemptionCheckpointingPolicy':
        save_decision_policies.append(
            save_decision_policy_lib.PreemptionCheckpointingPolicy()
        )
      elif save_decision_policy_type == 'InitialSavePolicy':
        save_decision_policies.append(
            save_decision_policy_lib.InitialSavePolicy()
        )
      else:
        raise ValueError(
            'Unsupported save_decision_policy_type:'
            f' {save_decision_policy_type}'
        )
    preservation_policy = None
    save_decision_policy = None
    if preservation_policies:
      preservation_policy = preservation_policy_lib.AnyPreservationPolicy(
          policies=preservation_policies
      )
    if save_decision_policies:
      save_decision_policy = save_decision_policy_lib.AnySavePolicy(
          policies=save_decision_policies
      )
    return preservation_policy, save_decision_policy

  def _get_policy_checkpoint_info(
      self, step: int, metrics: dict[str, Any] | None
  ) -> PolicyCheckpointInfo:
    """Returns PolicyCheckpointInfo instance."""
    return checkpoint_info.CheckpointInfo(
        step=step,
        metrics=metrics,
        time=datetime.datetime.now(),
    )

  def test_fn(
      self, context: benchmarks_core.TestContext
  ) -> benchmarks_core.TestResult:
    """The core test logic for a single benchmark run."""
    metrics = benchmarks_core.Metrics()
    pytree = context.pytree
    save_path = context.path / 'checkpoints'
    options = context.options
    assert isinstance(options, CheckpointPolicyBenchmarkOptions)

    handler = ocp.PyTreeCheckpointHandler()
    assert isinstance(options.save_decision_policy_types, str)
    assert isinstance(options.preservation_policy_types, str)
    preservation_policy, save_decision_policy = self._get_checkpoint_policies(
        options
    )
    checkpointer_manager = ocp.CheckpointManager(
        save_path,
        ocp.AsyncCheckpointer(handler),
        options=ocp.CheckpointManagerOptions(
            preservation_policy=preservation_policy,
            save_decision_policy=save_decision_policy,
        ),
    )
    for step in range(options.num_checkpoints):
      time.sleep(1)
      with metrics.time('save'):
        checkpointer_manager.save(step, args=ocp.args.PyTreeSave(pytree))
    print(f'all_steps_length: {len(checkpointer_manager.all_steps())}')
    return benchmarks_core.TestResult(metrics=metrics)
