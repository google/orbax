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

"""Builds `CheckpointManagerOptions` for continuous checkpointing."""

from typing import Literal, Sequence

from orbax.checkpoint import checkpoint_manager
from orbax.checkpoint._src.checkpoint_managers import preservation_policy as preservation_policy_lib
from orbax.checkpoint._src.checkpoint_managers import save_decision_policy as save_decision_policy_lib


def continuous_checkpointing_options(
    *,
    milestone_interval_steps: int,
    minimum_interval_secs: int | None = None,
    milestone_steps: Sequence[int] = (),
    max_to_keep: int | None = None,
    agreement: Literal['per_save', 'per_check'] = 'per_save',
) -> checkpoint_manager.CheckpointManagerOptions:
  """Returns `CheckpointManagerOptions` configured for continuous checkpointing.

  Continuous checkpointing saves as often as possible without blocking on an
  ongoing save. The returned options save a checkpoint when any of these holds:

  * At least `minimum_interval_secs` have passed since the last checkpoint
    (continuous checkpointing; see `agreement`).
  * The step is a multiple of `milestone_interval_steps` (milestone).
  * The step is in `milestone_steps`, e.g. the final train step.
  * A preemption was reached.
  * No checkpoint exists yet.

  `agreement` selects how hosts agree on continuous checkpointing saves:

  * `'per_save'` (default): `ScheduledContinuousCheckpointingPolicy`. During
    each save, the hosts agree once on the step of the next save. Between
    saves, each host decides locally, with no cross-host messages. It comes
    first in the policy list because it measures step time on every call.
  * `'per_check'`: `ContinuousCheckpointingPolicy`. Each check is a key-value
    exchange between hosts. It comes last, so the step-based policies (same
    answer on every host) let every host reach it in the same calls.

  The options keep the latest `max_to_keep` checkpoints and the latest
  `max_to_keep` milestone checkpoints (multiples of
  `milestone_interval_steps`).

  Call `CheckpointManager.save` (or `should_save`) on every step, or on a step
  schedule that is identical on all hosts.

  Other options can be set with `dataclasses.replace` on the result.

  Example::

    options = ocp.checkpoint_managers.continuous_checkpointing_options(
        milestone_interval_steps=1000,
        minimum_interval_secs=300,
        milestone_steps=(num_train_steps,),
        max_to_keep=3,
    )
    mngr = ocp.CheckpointManager(directory, options=options)
    for step in range(num_train_steps):
      ...
      mngr.save(step, args=ocp.args.StandardSave(state))

  Args:
    milestone_interval_steps: Always save at every multiple of this many steps.
    minimum_interval_secs: Minimum time between the previous checkpoint and a
      continuous checkpointing save. If None, a save starts as soon as the
      previous one finishes (`'per_save'` schedules it for when the previous
      write is expected to finish).
    milestone_steps: Additional steps at which to always save.
    max_to_keep: How many of the latest checkpoints, and how many of the latest
      milestone checkpoints, to keep. If None, keeps all checkpoints.
    agreement: `'per_save'` (default) for
      `ScheduledContinuousCheckpointingPolicy`, or `'per_check'` for the legacy
      `ContinuousCheckpointingPolicy`.

  Returns:
    The checkpoint manager options.

  Raises:
    ValueError: If an argument is out of range.
  """
  if milestone_interval_steps < 1:
    raise ValueError(
        'milestone_interval_steps must be >= 1, got'
        f' {milestone_interval_steps}.'
    )
  if minimum_interval_secs is not None and minimum_interval_secs < 0:
    raise ValueError(
        f'minimum_interval_secs must be >= 0, got {minimum_interval_secs}.'
    )
  if max_to_keep is not None and max_to_keep < 1:
    raise ValueError(f'max_to_keep must be >= 1, got {max_to_keep}.')
  if agreement not in ('per_save', 'per_check'):
    raise ValueError(
        f"agreement must be 'per_save' or 'per_check', got {agreement!r}."
    )

  step_policies: list[save_decision_policy_lib.SaveDecisionPolicy] = [
      save_decision_policy_lib.FixedIntervalPolicy(milestone_interval_steps)
  ]
  if milestone_steps:
    step_policies.append(
        save_decision_policy_lib.SpecificStepsPolicy(tuple(milestone_steps))
    )
  step_policies += [
      save_decision_policy_lib.PreemptionCheckpointingPolicy(),
      save_decision_policy_lib.InitialSavePolicy(),
  ]
  if agreement == 'per_save':
    policies = [
        save_decision_policy_lib.ScheduledContinuousCheckpointingPolicy(
            minimum_interval_secs=minimum_interval_secs or 0
        ),
        *step_policies,
    ]
  else:
    policies = [
        *step_policies,
        save_decision_policy_lib.ContinuousCheckpointingPolicy(
            minimum_interval_secs=minimum_interval_secs,
        ),
    ]
  preservation_policy = preservation_policy_lib.AnyPreservationPolicy([
      preservation_policy_lib.LatestN(n=max_to_keep),
      preservation_policy_lib.EveryNSteps(
          interval_steps=milestone_interval_steps, max_to_keep=max_to_keep
      ),
  ])
  return checkpoint_manager.CheckpointManagerOptions(
      enable_async_checkpointing=True,
      save_decision_policy=save_decision_policy_lib.AnySavePolicy(policies),
      preservation_policy=preservation_policy,
  )
