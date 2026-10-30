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

from orbax.checkpoint import checkpoint_manager
from orbax.checkpoint._src.checkpoint_managers import preservation_policy as preservation_policy_lib
from orbax.checkpoint._src.checkpoint_managers import save_decision_policy as save_decision_policy_lib


def continuous_checkpointing_options(
    *,
    target_interval_steps: int,
    minimum_interval_secs: int | None = None,
    max_to_keep: int | None = None,
    deferral_window_steps: int | None = None,
) -> checkpoint_manager.CheckpointManagerOptions:
  """Returns `CheckpointManagerOptions` for continuous checkpointing.

  Saves every `minimum_interval_secs` if no write is running. Always saves at
  every multiple of `target_interval_steps` (target steps). Also starts a save
  on the first step. Like timed saves, it is cancelled if its write is still
  running at the next target step. There is no preemption save. Call `save()`
  on every step. To save the last step, call `save(step, force=True)`.

  Example::

    options = ocp.checkpoint_managers.continuous_checkpointing_options(
        target_interval_steps=1000,
        minimum_interval_secs=300,
        max_to_keep=3,
    )
    mngr = ocp.CheckpointManager(directory, options=options)
    for step in range(num_train_steps):
      ...
      mngr.save(step, args=ocp.args.StandardSave(state))
    mngr.save(num_train_steps - 1, args=..., force=True)

  Args:
    target_interval_steps: Always save at every multiple of this many steps.
    minimum_interval_secs: Minimum time between saves. If None, a save starts as
      soon as the previous one finishes.
    max_to_keep: Number of latest checkpoints, and of latest target-step
      checkpoints, to keep. If None, keeps all.
    deferral_window_steps: Skip timed saves this many steps before each target
      step. If None, timed saves are not skipped before target steps (an
      in-flight timed save is cancelled when a target step arrives).

  Returns:
    The checkpoint manager options.

  Raises:
    ValueError: If an argument is out of range.
  """
  # `ContinuousCheckpointingPolicy` validates `target_interval_steps` and
  # `deferral_window_steps`.
  if minimum_interval_secs is not None and minimum_interval_secs < 0:
    raise ValueError(
        f'minimum_interval_secs must be >= 0, got {minimum_interval_secs}.'
    )
  if max_to_keep is not None and max_to_keep < 1:
    raise ValueError(f'max_to_keep must be >= 1, got {max_to_keep}.')

  # `InitialSavePolicy` decides the same way on every host, so all hosts skip
  # or reach `ContinuousCheckpointingPolicy` together.
  save_decision_policy = save_decision_policy_lib.AnySavePolicy([
      save_decision_policy_lib.InitialSavePolicy(),
      save_decision_policy_lib.ContinuousCheckpointingPolicy(
          minimum_interval_secs=minimum_interval_secs,
          target_interval_steps=target_interval_steps,
          deferral_window_steps=deferral_window_steps,
      ),
  ])
  preservation_policy = preservation_policy_lib.AnyPreservationPolicy([
      preservation_policy_lib.LatestN(n=max_to_keep),
      preservation_policy_lib.EveryNSteps(
          interval_steps=target_interval_steps, max_to_keep=max_to_keep
      ),
  ])
  return checkpoint_manager.CheckpointManagerOptions(
      enable_async_checkpointing=True,
      save_decision_policy=save_decision_policy,
      preservation_policy=preservation_policy,
  )
