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
from collections.abc import Mapping
import contextlib
import threading
from typing import Any, NamedTuple
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
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


class _Difference(NamedTuple):
  """Expected `v1 - v0` count delta for a metric and why it differs."""

  count: int
  reason: str


# Differences are given as per-scenario counts instead of static values
# because `v1` records some init/write metrics per save or load/metadata call,
# while `v0` initializes its cached writer and pytree handler once in `__init__`
# (and preemption-skip metrics only differ in scenarios that skip a save).
_WRITER_PER_SAVE_REASON = (
    '`training.Checkpointer` creates a writer for each save.'
    # +1 per completed save (`v0` and `v1` both init 1 writer in `__init__`).
)
_HANDLERS_PER_CALL_REASON = (
    '`training.Checkpointer` creates handlers for each save and load.'
    # `array_handler`: 3 * saves + 2 * reads (`v0`: 0).
    # `pytree_checkpoint_handler`: same as above minus 1 (`v0` `__init__`).
)
_DURATION_LOST_REASON = (
    'Not recorded by `training.Checkpointer`.'  # -1 per preemption skip.
)

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


def _normalize(value: Any, directory: epath.Path) -> Any:
  """Replaces values that differ between runs: times, rates and paths."""
  if isinstance(value, float):
    return float
  if value == str(directory):
    return '<directory>'
  return value


@contextlib.contextmanager
def _capture_monitoring_metrics(directory: epath.Path):
  """Records normalized jax.monitoring metrics emitted within the context."""
  metrics: collections.Counter[tuple[str, tuple[tuple[str, Any], ...]]] = (
      collections.Counter()
  )
  lock = threading.Lock()

  def record(event: str, tags: dict[str, Any]) -> None:
    normalized = tuple(
        sorted((k, _normalize(v, directory)) for k, v in tags.items())
    )
    with lock:
      metrics[event, normalized] += 1

  def on_event(event: str, **kwargs: Any) -> None:
    record(event, kwargs)

  def on_duration(event: str, duration_secs: float, **kwargs: Any) -> None:
    del duration_secs
    record(event, kwargs)

  def on_scalar(event: str, value: float | int, **kwargs: Any) -> None:
    del value
    record(event, kwargs)

  with contextlib.ExitStack() as stack:
    jax.monitoring.register_event_listener(on_event)
    stack.callback(jax.monitoring.unregister_event_listener, on_event)
    jax.monitoring.register_event_duration_secs_listener(on_duration)
    stack.callback(
        jax.monitoring.unregister_event_duration_listener, on_duration
    )
    jax.monitoring.register_scalar_listener(on_scalar)
    stack.callback(jax.monitoring.unregister_scalar_listener, on_scalar)
    yield metrics


class TelemetryParityTest(parameterized.TestCase):

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
    ), _capture_monitoring_metrics(directory) as metrics:
      runner = runner_cls(directory, save_policy, preservation_policy, logger)
      try:
        run(runner)
      finally:
        runner.close()
    entries = [
        {k: _normalize(v, directory) for k, v in entry.items()}
        for entry in logger.entries
    ]
    return metrics, entries

  @parameterized.named_parameters(
      dict(
          testcase_name='save_loop',
          run=_save_loop,
          save_policy=save_decision_policies.FixedIntervalPolicy(1),
          preservation_policy=preservation_policies.LatestN(1),
          known_differences={
              '/jax/orbax/async_checkpointer/init': _Difference(
                  2, _WRITER_PER_SAVE_REASON
              ),
              '/jax/orbax/array_handler/init': _Difference(
                  6, _HANDLERS_PER_CALL_REASON
              ),
              '/jax/orbax/pytree_checkpoint_handler/init/ocdbt': _Difference(
                  5, _HANDLERS_PER_CALL_REASON
              ),
          },
      ),
      dict(
          testcase_name='load',
          run=_load,
          save_policy=save_decision_policies.FixedIntervalPolicy(1),
          preservation_policy=preservation_policies.PreserveAll(),
          known_differences={
              '/jax/orbax/async_checkpointer/init': _Difference(
                  1, _WRITER_PER_SAVE_REASON
              ),
              '/jax/orbax/array_handler/init': _Difference(
                  5, _HANDLERS_PER_CALL_REASON
              ),
              '/jax/orbax/pytree_checkpoint_handler/init/ocdbt': _Difference(
                  4, _HANDLERS_PER_CALL_REASON
              ),
          },
      ),
      dict(
          testcase_name='metadata',
          run=_metadata,
          save_policy=save_decision_policies.FixedIntervalPolicy(1),
          preservation_policy=preservation_policies.PreserveAll(),
          known_differences={
              '/jax/orbax/async_checkpointer/init': _Difference(
                  1, _WRITER_PER_SAVE_REASON
              ),
              '/jax/orbax/array_handler/init': _Difference(
                  5, _HANDLERS_PER_CALL_REASON
              ),
              '/jax/orbax/pytree_checkpoint_handler/init/ocdbt': _Difference(
                  4, _HANDLERS_PER_CALL_REASON
              ),
          },
      ),
      dict(
          testcase_name='delete',
          run=_delete,
          save_policy=save_decision_policies.FixedIntervalPolicy(1),
          preservation_policy=preservation_policies.PreserveAll(),
          known_differences={
              '/jax/orbax/async_checkpointer/init': _Difference(
                  1, _WRITER_PER_SAVE_REASON
              ),
              '/jax/orbax/array_handler/init': _Difference(
                  3, _HANDLERS_PER_CALL_REASON
              ),
              '/jax/orbax/pytree_checkpoint_handler/init/ocdbt': _Difference(
                  2, _HANDLERS_PER_CALL_REASON
              ),
          },
      ),
      dict(
          testcase_name='policy_rejected',
          run=_policy_rejected,
          save_policy=save_decision_policies.FixedIntervalPolicy(4),
          preservation_policy=preservation_policies.PreserveAll(),
          known_differences={
              '/jax/orbax/async_checkpointer/init': _Difference(
                  1, _WRITER_PER_SAVE_REASON
              ),
              '/jax/orbax/array_handler/init': _Difference(
                  3, _HANDLERS_PER_CALL_REASON
              ),
              '/jax/orbax/pytree_checkpoint_handler/init/ocdbt': _Difference(
                  2, _HANDLERS_PER_CALL_REASON
              ),
              '/jax/checkpoint/write/preempt/duration_lost_secs': _Difference(
                  -1, _DURATION_LOST_REASON
              ),
          },
      ),
      dict(
          testcase_name='step_already_saved',
          run=_step_already_saved,
          save_policy=save_decision_policies.FixedIntervalPolicy(4),
          preservation_policy=preservation_policies.PreserveAll(),
          known_differences={
              '/jax/orbax/async_checkpointer/init': _Difference(
                  2, _WRITER_PER_SAVE_REASON
              ),
              '/jax/orbax/array_handler/init': _Difference(
                  6, _HANDLERS_PER_CALL_REASON
              ),
              '/jax/orbax/pytree_checkpoint_handler/init/ocdbt': _Difference(
                  5, _HANDLERS_PER_CALL_REASON
              ),
              '/jax/checkpoint/write/preempt/duration_lost_secs': _Difference(
                  -1, _DURATION_LOST_REASON
              ),
              '/jax/orbax/write/preemption': _Difference(
                  -1,
                  'Not recorded when `training.Checkpointer` skips an already'
                  ' saved step.',
              ),
          },
      ),
      dict(
          testcase_name='save_in_progress',
          run=_save_in_progress,
          save_policy=save_decision_policies.ContinuousCheckpointingPolicy(),
          preservation_policy=preservation_policies.PreserveAll(),
          known_differences={
              '/jax/orbax/async_checkpointer/init': _Difference(
                  1, _WRITER_PER_SAVE_REASON
              ),
              '/jax/orbax/array_handler/init': _Difference(
                  3, _HANDLERS_PER_CALL_REASON
              ),
              '/jax/orbax/pytree_checkpoint_handler/init/ocdbt': _Difference(
                  2, _HANDLERS_PER_CALL_REASON
              ),
              '/jax/checkpoint/write/preempt/duration_lost_secs': _Difference(
                  -1, _DURATION_LOST_REASON
              ),
          },
      ),
      dict(
          testcase_name='minimum_interval_not_met',
          run=_save_at_preemption,
          save_policy=save_decision_policies.ContinuousCheckpointingPolicy(
              minimum_interval_secs=3600
          ),
          preservation_policy=preservation_policies.PreserveAll(),
          known_differences={
              '/jax/orbax/async_checkpointer/init': _Difference(
                  1, _WRITER_PER_SAVE_REASON
              ),
              '/jax/orbax/array_handler/init': _Difference(
                  3, _HANDLERS_PER_CALL_REASON
              ),
              '/jax/orbax/pytree_checkpoint_handler/init/ocdbt': _Difference(
                  2, _HANDLERS_PER_CALL_REASON
              ),
              '/jax/checkpoint/write/preempt/duration_lost_secs': _Difference(
                  -1, _DURATION_LOST_REASON
              ),
          },
      ),
      dict(
          testcase_name='deferred_before_target_step',
          run=_save_at_preemption,
          save_policy=save_decision_policies.ContinuousCheckpointingPolicy(
              target_interval_steps=4, deferral_window_steps=2
          ),
          preservation_policy=preservation_policies.PreserveAll(),
          known_differences={
              '/jax/orbax/async_checkpointer/init': _Difference(
                  1, _WRITER_PER_SAVE_REASON
              ),
              '/jax/orbax/array_handler/init': _Difference(
                  3, _HANDLERS_PER_CALL_REASON
              ),
              '/jax/orbax/pytree_checkpoint_handler/init/ocdbt': _Difference(
                  2, _HANDLERS_PER_CALL_REASON
              ),
              '/jax/checkpoint/write/preempt/duration_lost_secs': _Difference(
                  -1, _DURATION_LOST_REASON
              ),
          },
      ),
      dict(
          testcase_name='saved_on_preemption',
          run=_save_at_preemption,
          save_policy=save_decision_policies.AnySavePolicy([
              save_decision_policies.FixedIntervalPolicy(4),
              save_decision_policies.PreemptionCheckpointingPolicy(),
          ]),
          preservation_policy=preservation_policies.PreserveAll(),
          known_differences={
              '/jax/orbax/async_checkpointer/init': _Difference(
                  2, _WRITER_PER_SAVE_REASON
              ),
              '/jax/orbax/array_handler/init': _Difference(
                  6, _HANDLERS_PER_CALL_REASON
              ),
              '/jax/orbax/pytree_checkpoint_handler/init/ocdbt': _Difference(
                  5, _HANDLERS_PER_CALL_REASON
              ),
          },
      ),
  )
  def test_same_telemetry(
      self,
      run,
      save_policy,
      preservation_policy,
      known_differences: Mapping[str, _Difference],
  ):
    v0_metrics, v0_entries = self.run_scenario(
        _V0, run, save_policy, preservation_policy
    )
    v1_metrics, v1_entries = self.run_scenario(
        _V1, run, save_policy, preservation_policy
    )
    actual_differences: dict[str, int] = {}
    mismatches: dict[Any, tuple[int, int]] = {}
    for key in sorted(v0_metrics.keys() | v1_metrics.keys()):
      v0_count = v0_metrics[key]
      v1_count = v1_metrics[key]
      if v1_count != v0_count:
        mismatches[key] = (v0_count, v1_count)
        name, _ = key
        self.assertNotIn(
            name,
            actual_differences,
            msg=f'Multiple tag mismatches for {name}: {mismatches}',
        )
        actual_differences[name] = v1_count - v0_count
    expected_differences = {
        name: diff.count for name, diff in known_differences.items()
    }
    self.assertEqual(
        actual_differences,
        expected_differences,
        msg=f'Mismatched metric counts (v0, v1): {mismatches}',
    )
    self.assertEqual(v1_entries, v0_entries)


if __name__ == '__main__':
  absltest.main()
