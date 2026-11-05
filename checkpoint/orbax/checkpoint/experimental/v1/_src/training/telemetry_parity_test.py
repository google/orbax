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

"""Compares the telemetry of `training.Checkpointer` and `CheckpointManager`.

Each scenario runs on both, which must record the same metrics (names, tags and
counts) and logger entries. Recorded values, such as durations, are ignored.
"""

import collections
from typing import Any
from unittest import mock

from absl.testing import absltest
from etils import epath
import jax
from jax.experimental import multihost_utils
import numpy as np
from orbax.checkpoint import args as args_lib
from orbax.checkpoint import checkpoint_manager
from orbax.checkpoint._src.logging import abstract_logger
from orbax.checkpoint._src.multihost import multihost
import orbax.checkpoint.experimental.v1 as ocp

save_decision_policies = ocp.training.save_decision_policies
preservation_policies = ocp.training.preservation_policies

# Metrics that are allowed to differ, and why.
_KNOWN_DIFFERENCES = {
    '/jax/checkpoint/write/preempt/duration_lost_secs': (
        'Not recorded by `training.Checkpointer`.'
    ),
    '/jax/orbax/write/preemption': (
        'Not recorded when `training.Checkpointer` skips an already saved step.'
    ),
    '/jax/orbax/async_checkpointer/init': (
        '`training.Checkpointer` creates a writer for each save.'
    ),
    '/jax/orbax/array_handler/init': (
        '`training.Checkpointer` creates handlers for each save and load.'
    ),
    '/jax/orbax/pytree_checkpoint_handler/init/ocdbt': (
        '`training.Checkpointer` creates handlers for each save and load.'
    ),
    '/jax/orbax/write/background_item_finalize_secs': (
        '`training.Checkpointer` also saves a `descriptor` checkpointable.'
    ),
}

_PREEMPTION_STEP = 2
_STATE = {'weights': np.arange(4)}


class _Logger(abstract_logger.AbstractLogger):

  def __init__(self):
    self.entries = []

  def log_entry(self, entry: Any):
    self.entries.append(entry)


class _V0:
  """Runs a scenario on `CheckpointManager`."""

  def __init__(self, directory, save_policy, preservation_policy, logger):
    self.manager = checkpoint_manager.CheckpointManager(
        directory,
        options=checkpoint_manager.CheckpointManagerOptions(
            save_decision_policy=save_policy,
            preservation_policy=preservation_policy,
        ),
        logger=logger,
    )

  def save(self, step: int):
    self.manager.save(
        step, args=args_lib.Composite(state=args_lib.PyTreeSave(_STATE))
    )

  def should_save(self, step: int):
    self.manager.should_save(step)

  def saving_in_progress(self):
    return mock.patch.object(
        self.manager, 'is_saving_in_progress', return_value=True
    )

  def wait(self):
    self.manager.wait_until_finished()

  def reload(self):
    self.manager.reload()

  def load(self):
    self.manager.restore(
        None, args=args_lib.Composite(state=args_lib.PyTreeRestore())
    )

  def metadata(self, step: int):
    self.manager.item_metadata(step)

  def delete(self, step: int):
    # Like `training.Checkpointer.delete`, which waits for pending saves.
    self.manager.wait_until_finished()
    self.manager.delete(step)

  def close(self):
    self.manager.close()


class _V1:
  """Runs a scenario on `training.Checkpointer`."""

  def __init__(self, directory, save_policy, preservation_policy, logger):
    self.ckptr = ocp.training.Checkpointer(
        directory,
        save_decision_policy=save_policy,
        preservation_policy=preservation_policy,
        logger=logger,
    )

  def save(self, step: int):
    self.ckptr.save_async(step, _STATE)  # pyrefly: ignore[bad-argument-type]

  def should_save(self, step: int):
    self.ckptr.should_save(step)

  def saving_in_progress(self):
    manager = self.ckptr._manager  # pylint: disable=protected-access
    return mock.patch.object(
        manager, 'is_saving_in_progress', return_value=True
    )

  def wait(self):
    self.ckptr.wait()

  def reload(self):
    self.ckptr.reload()

  def load(self):
    self.ckptr.load()

  def metadata(self, step: int):
    self.ckptr.metadata(step)

  def delete(self, step: int):
    self.ckptr.delete(step)

  def close(self):
    self.ckptr.close()


def _save_loop(runner):
  runner.save(0)
  runner.save(1)
  runner.wait()
  runner.reload()


def _load(runner):
  runner.save(0)
  runner.wait()
  runner.load()


def _metadata(runner):
  runner.save(0)
  runner.wait()
  runner.metadata(0)


def _delete(runner):
  runner.save(0)
  runner.wait()
  runner.delete(0)


def _policy_rejected(runner):
  runner.save(0)
  runner.wait()
  runner.should_save(_PREEMPTION_STEP)
  runner.save(_PREEMPTION_STEP)


def _step_already_saved(runner):
  runner.save(0)
  runner.save(4)
  runner.wait()
  runner.should_save(_PREEMPTION_STEP)


def _save_in_progress(runner):
  runner.save(0)
  runner.wait()
  with runner.saving_in_progress():
    runner.should_save(_PREEMPTION_STEP)


def _save_at_preemption(runner):
  runner.save(0)
  runner.wait()
  runner.save(_PREEMPTION_STEP)
  runner.wait()


# Scenario name: (steps, save decision policy, preservation policy).
_SCENARIOS = {
    'save_loop': (
        _save_loop,
        lambda: save_decision_policies.FixedIntervalPolicy(1),
        lambda: preservation_policies.LatestN(1),
    ),
    'load': (
        _load,
        lambda: save_decision_policies.FixedIntervalPolicy(1),
        preservation_policies.PreserveAll,
    ),
    'metadata': (
        _metadata,
        lambda: save_decision_policies.FixedIntervalPolicy(1),
        preservation_policies.PreserveAll,
    ),
    'delete': (
        _delete,
        lambda: save_decision_policies.FixedIntervalPolicy(1),
        preservation_policies.PreserveAll,
    ),
    'policy_rejected': (
        _policy_rejected,
        lambda: save_decision_policies.FixedIntervalPolicy(4),
        preservation_policies.PreserveAll,
    ),
    'step_already_saved': (
        _step_already_saved,
        lambda: save_decision_policies.FixedIntervalPolicy(4),
        preservation_policies.PreserveAll,
    ),
    'save_in_progress': (
        _save_in_progress,
        save_decision_policies.ContinuousCheckpointingPolicy,
        preservation_policies.PreserveAll,
    ),
    'minimum_interval_not_met': (
        _save_at_preemption,
        lambda: save_decision_policies.ContinuousCheckpointingPolicy(
            minimum_interval_secs=3600
        ),
        preservation_policies.PreserveAll,
    ),
    'deferred_before_target_step': (
        _save_at_preemption,
        lambda: save_decision_policies.ContinuousCheckpointingPolicy(
            target_interval_steps=4, deferral_window_steps=2
        ),
        preservation_policies.PreserveAll,
    ),
    'saved_on_preemption': (
        _save_at_preemption,
        lambda: save_decision_policies.AnySavePolicy([
            save_decision_policies.FixedIntervalPolicy(4),
            save_decision_policies.PreemptionCheckpointingPolicy(),
        ]),
        preservation_policies.PreserveAll,
    ),
}


def _normalize(value: Any, directory: epath.Path) -> Any:
  """Replaces values that differ between runs: times, rates and paths."""
  if isinstance(value, float):
    return float
  if value == str(directory):
    return '<directory>'
  return value


class TelemetryParityTest(absltest.TestCase):

  def run_scenario(self, runner_cls, run, save_policy, preservation_policy):
    """Returns the metrics and logger entries recorded by `run`."""
    directory = epath.Path(self.create_tempdir().full_path)
    logger = _Logger()
    with mock.patch.object(
        multihost_utils,
        'reached_preemption_sync_point',
        side_effect=lambda step: step == _PREEMPTION_STEP,
    ), mock.patch.object(
        # Records `/jax/orbax/write/preemption` once per step and process.
        multihost,
        '_last_logged_preemption_step',
        None,
    ), mock.patch.object(
        jax.monitoring, 'record_event'
    ) as record_event, mock.patch.object(
        jax.monitoring, 'record_event_duration_secs'
    ) as record_duration, mock.patch.object(
        jax.monitoring, 'record_scalar'
    ) as record_scalar:
      runner = runner_cls(
          directory, save_policy(), preservation_policy(), logger
      )
      try:
        run(runner)
      finally:
        runner.close()
    metrics = collections.Counter()
    for recorder in (record_event, record_duration, record_scalar):
      for call in recorder.call_args_list:
        name = call.args[0]
        if name not in _KNOWN_DIFFERENCES:
          tags = {k: _normalize(v, directory) for k, v in call.kwargs.items()}
          metrics[name, tuple(sorted(tags.items()))] += 1
    entries = [
        {k: _normalize(v, directory) for k, v in entry.items()}
        for entry in logger.entries
    ]
    return metrics, entries

  def test_same_telemetry(self):
    for name, scenario in _SCENARIOS.items():
      with self.subTest(name):
        v0_metrics, v0_entries = self.run_scenario(_V0, *scenario)
        v1_metrics, v1_entries = self.run_scenario(_V1, *scenario)
        self.assertEqual(v1_metrics, v0_metrics)
        self.assertEqual(v1_entries, v0_entries)


if __name__ == '__main__':
  absltest.main()
