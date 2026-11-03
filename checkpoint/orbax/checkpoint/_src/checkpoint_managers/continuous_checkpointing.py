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
    deferral_window_steps: int | None = None,
    agreement: Literal['per_save', 'per_check'] = 'per_save',
) -> checkpoint_manager.CheckpointManagerOptions:
  """Returns `CheckpointManagerOptions` configured for continuous checkpointing.

  Continuous checkpointing saves as often as possible without blocking on an
  ongoing save. The returned options use one `ContinuousCheckpointingPolicy`
  with `target_interval_steps=milestone_interval_steps`, which saves when:

  * The step is a multiple of `milestone_interval_steps` (milestone), always.
    If a continuous save is still being written, `CheckpointManager.save`
    cancels it; the deferral window keeps the writer free, so this is rare.
  * No checkpoint exists yet.
  * No save is in progress, at least `minimum_interval_secs` have passed since
    the last checkpoint, and the step is not within `deferral_window_steps`
    before the next milestone.

  It also saves at `milestone_steps` (e.g. the final train step) and on the
  first call (`InitialSavePolicy`).

  The default deferral window (`milestone_interval_steps // 2`) pauses
  continuous saves for half of each milestone period; set
  `deferral_window_steps` to about the write time in steps to avoid that.

  There is no preemption save: with continuous checkpointing the latest
  checkpoint is recent, so a save on preemption is unnecessary.
  `CheckpointManager` still runs the JAX preemption sync check on every `save`
  and `should_save` call. To also save on preemption, build your own policy
  list with `PreemptionCheckpointingPolicy`.

  `agreement` selects how hosts agree on continuous saves (see
  `ContinuousCheckpointingPolicy`): `'per_save'` (default) exchanges messages
  only around saves, `'per_check'` on every call.

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
    milestone_interval_steps: Always save at every multiple of this many steps
      (`target_interval_steps` of `ContinuousCheckpointingPolicy`).
    minimum_interval_secs: Minimum time between the previous checkpoint and a
      continuous save. If None, a save starts as soon as the previous one
      finishes.
    milestone_steps: Additional steps at which to always save.
    max_to_keep: How many of the latest checkpoints, and how many of the latest
      milestone checkpoints, to keep. If None, keeps all checkpoints.
    deferral_window_steps: Number of steps before each milestone during which
      continuous saves are deferred. If None, `milestone_interval_steps // 2`.
    agreement: `'per_save'` (default) or `'per_check'`.

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
  if deferral_window_steps is not None and deferral_window_steps < 0:
    raise ValueError(
        f'deferral_window_steps must be >= 0, got {deferral_window_steps}.'
    )
  if agreement not in ('per_save', 'per_check'):
    raise ValueError(
        f"agreement must be 'per_save' or 'per_check', got {agreement!r}."
    )

  cc_policy = save_decision_policy_lib.ContinuousCheckpointingPolicy(
      minimum_interval_secs=minimum_interval_secs,
      target_interval_steps=milestone_interval_steps,
      deferral_window_steps=deferral_window_steps,
      agreement=agreement,
  )
  step_policies: list[save_decision_policy_lib.SaveDecisionPolicy] = []
  if milestone_steps:
    step_policies.append(
        save_decision_policy_lib.SpecificStepsPolicy(tuple(milestone_steps))
    )
  step_policies.append(save_decision_policy_lib.InitialSavePolicy())
  if agreement == 'per_save':
    # First, so that it observes every step (AnySavePolicy short-circuits).
    policies = [cc_policy, *step_policies]
  else:
    # Last: the step-based policies give the same answer on every host, so
    # every host reaches the per-check exchange in the same calls.
    policies = [*step_policies, cc_policy]
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
