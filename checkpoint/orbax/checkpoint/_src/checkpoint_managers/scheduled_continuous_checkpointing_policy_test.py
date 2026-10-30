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

"""Tests for ScheduledContinuousCheckpointingPolicy (EXPERIMENTAL)."""

import dataclasses
import datetime
import random
from unittest import mock

from absl.testing import absltest
from orbax.checkpoint import options as options_lib
from orbax.checkpoint._src.checkpoint_managers import save_decision_policy as sdp


@dataclasses.dataclass
class _Info:
  step: int
  time: datetime.datetime
  metrics: dict[str, float] | None = None


_EPOCH = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)


def _ctx(is_saving: bool = False, preempt: bool = False) -> sdp.DecisionContext:
  return sdp.DecisionContext(
      is_saving_in_progress=is_saving,
      reached_preemption=preempt,
      multiprocessing_options=options_lib.MultiprocessingOptions(),
  )


@dataclasses.dataclass
class SimResult:
  save_steps: list[int]
  save_times: list[float]
  waits: list[float]

  @property
  def intervals(self) -> list[float]:
    return [b - a for a, b in zip(self.save_times, self.save_times[1:])]


def simulate(
    policy: sdp.SaveDecisionPolicy,
    step_time_fn,
    write_secs_fn,
    num_steps: int,
    previous: list[_Info] | None = None,
    start_step: int = 1,
) -> SimResult:
  """Simulates a training loop driving `policy` like CheckpointManager does.

  A save that starts while the previous write is running waits for it (this is
  what `CheckpointManager.save` does after `should_save` returns True).

  Args:
    policy: policy under test.
    step_time_fn: step -> seconds of compute for that step.
    write_secs_fn: save index -> seconds the background write takes.
    num_steps: number of steps.
    previous: existing checkpoints (restart case).
    start_step: first step.

  Returns:
    SimResult.
  """
  clock = [0.0]
  previous = list(previous or [])
  write_end = -1.0
  res = SimResult([], [], [])
  with mock.patch.object(sdp.time, 'monotonic', lambda: clock[0]):
    for step in range(start_step, start_step + num_steps):
      clock[0] += step_time_fn(step)
      if previous and previous[-1].step >= step:
        continue
      info = _Info(step, _EPOCH + datetime.timedelta(seconds=clock[0]))
      ctx = _ctx(is_saving=clock[0] < write_end)
      if policy.should_save(info, previous, context=ctx):
        wait = max(0.0, write_end - clock[0])
        clock[0] += wait
        sdp.notify_save_start(policy, step, context=_ctx())
        res.save_steps.append(step)
        res.save_times.append(clock[0])
        res.waits.append(wait)
        previous.append(info)
        write_end = clock[0] + write_secs_fn(len(res.save_steps))
  return res


class ScheduledPolicyTest(absltest.TestCase):

  def test_rejects_bad_args(self):
    with self.assertRaises(ValueError):
      sdp.ScheduledContinuousCheckpointingPolicy(minimum_interval_secs=-1)
    with self.assertRaises(ValueError):
      sdp.ScheduledContinuousCheckpointingPolicy(
          minimum_interval_secs=1, step_time_quantile=2.0
      )

  def test_zero_interval_schedules_after_write(self):
    p = sdp.ScheduledContinuousCheckpointingPolicy(minimum_interval_secs=0)
    res = simulate(p, lambda s: 1.0, lambda i: 5.0, num_steps=100)
    # After the first write is measured, saves follow the write duration.
    for iv in res.intervals[3:]:
      self.assertGreaterEqual(iv, 5 - 1e-9)
      self.assertLess(iv, 5 + 1 + 1e-9)

  def test_preemption_save_skips_exchange_and_clears_schedule(self):
    p = sdp.ScheduledContinuousCheckpointingPolicy(minimum_interval_secs=10)
    prev = [_Info(0, _EPOCH)]
    clock = [0.0]
    with mock.patch.object(sdp.time, 'monotonic', lambda: clock[0]):
      for step in range(1, 6):
        clock[0] = float(step)
        p.should_save(_Info(step, _EPOCH), prev, context=_ctx())
      sdp.notify_save_start(p, 5, context=_ctx())
      self.assertEqual(p.next_step, 15)
      clock[0] = 8.0
      # Pretend to be one host of several; any exchange must fail the test.
      with mock.patch.object(
          sdp.multihost, 'process_count', return_value=2
      ), mock.patch.object(
          sdp.signaling_client,
          'get_signaling_client',
          side_effect=AssertionError('no exchange on preemption saves'),
      ):
        sdp.notify_save_start(p, 8, context=_ctx(preempt=True))
    self.assertIsNone(p.next_step)
    self.assertEqual(p.schedule_history[-1], (8, None))
    # Save start time and in-flight state are recorded as usual.
    self.assertEqual(p._save_start_time, 8.0)  # pylint: disable=protected-access
    self.assertTrue(p._save_in_flight)  # pylint: disable=protected-access

  def test_bootstrap_uses_legacy_until_first_save(self):
    p = sdp.ScheduledContinuousCheckpointingPolicy(minimum_interval_secs=10)
    self.assertIsNone(p.next_step)
    # No previous checkpoints: legacy policy says save.
    self.assertTrue(p.should_save(_Info(1, _EPOCH), [], context=_ctx()))
    self.assertEqual(p.num_bootstrap_decisions, 1)

  def test_schedule_is_computed_and_decisions_are_local(self):
    p = sdp.ScheduledContinuousCheckpointingPolicy(
        minimum_interval_secs=10, step_time_quantile=0.5
    )
    prev = [_Info(0, _EPOCH)]
    clock = [0.0]
    with mock.patch.object(sdp.time, 'monotonic', lambda: clock[0]):
      for step in range(1, 6):
        clock[0] = float(step)
        p.should_save(_Info(step, _EPOCH), prev, context=_ctx())
      sdp.notify_save_start(p, 5, context=_ctx())
      self.assertEqual(p.next_step, 15)
      prev = [_Info(5, _EPOCH)]
      decisions = {}
      local_before = p.num_local_decisions
      with mock.patch.object(
          sdp.signaling_client,
          'get_signaling_client',
          side_effect=AssertionError('no messages expected'),
      ):
        for step in range(6, 16):
          clock[0] = float(step)
          decisions[step] = p.should_save(
              _Info(step, _EPOCH), prev, context=_ctx()
          )
    self.assertEqual([s for s, d in decisions.items() if d], [15])
    self.assertEqual(p.num_local_decisions - local_before, 10)

  def test_constant_step_time_meets_interval_exactly(self):
    p = sdp.ScheduledContinuousCheckpointingPolicy(minimum_interval_secs=10)
    res = simulate(p, lambda s: 0.3, lambda i: 1.0, num_steps=2000)
    steady = res.intervals[2:]
    self.assertNotEmpty(steady)
    for iv in steady:
      self.assertGreaterEqual(iv, 10 - 1e-9)
      self.assertLess(iv, 10 + 0.3 + 1e-9)
    self.assertEqual(sum(res.waits), 0.0)

  def test_jittery_step_time_drifts_later_not_earlier(self):
    rng = random.Random(0)
    times = [rng.lognormvariate(0, 0.3) for _ in range(20001)]
    p = sdp.ScheduledContinuousCheckpointingPolicy(minimum_interval_secs=30)
    res = simulate(p, lambda s: times[s], lambda i: 1.0, num_steps=20000)
    steady = res.intervals[2:]
    short = [iv for iv in steady if iv < 30]
    # With a low quantile estimate, nearly all intervals are >= the minimum,
    # and the feedback term pushes the mean above the minimum.
    self.assertLessEqual(len(short) / len(steady), 0.02)
    self.assertGreaterEqual(sum(steady) / len(steady), 30)

  def test_slow_write_waits_once_then_schedule_stretches(self):
    p = sdp.ScheduledContinuousCheckpointingPolicy(minimum_interval_secs=10)
    # Writes take 25 s, much longer than the 10 s interval.
    res = simulate(p, lambda s: 1.0, lambda i: 25.0, num_steps=300)
    waits = [w for w in res.waits if w > 0]
    # The first scheduled save waits; after that, the schedule adapts.
    self.assertLessEqual(len(waits), 2)
    for iv in res.intervals[3:]:
      self.assertGreaterEqual(iv, 25 - 1e-9)

  def test_slow_write_without_adaptation_waits_every_save(self):
    p = sdp.ScheduledContinuousCheckpointingPolicy(
        minimum_interval_secs=10, write_duration_margin=0
    )
    res = simulate(p, lambda s: 1.0, lambda i: 25.0, num_steps=300)
    self.assertGreater(sum(1 for w in res.waits if w > 0), 5)

  def test_legacy_policy_skips_instead_of_waiting(self):
    p = sdp.ContinuousCheckpointingPolicy(minimum_interval_secs=10)
    res = simulate(p, lambda s: 1.0, lambda i: 25.0, num_steps=300)
    self.assertEqual(sum(res.waits), 0.0)

  def test_preemption_and_initial_unaffected_in_any_policy(self):
    sched = sdp.ScheduledContinuousCheckpointingPolicy(
        minimum_interval_secs=1000
    )
    any_policy = sdp.AnySavePolicy(
        [sched, sdp.PreemptionCheckpointingPolicy(), sdp.InitialSavePolicy()]
    )
    # Establish a schedule far in the future.
    with mock.patch.object(sdp.time, 'monotonic', side_effect=[0.0, 1.0, 2.0]):
      sched.should_save(_Info(1, _EPOCH), [_Info(0, _EPOCH)], context=_ctx())
      sched.should_save(_Info(2, _EPOCH), [_Info(0, _EPOCH)], context=_ctx())
      sdp.notify_save_start(any_policy, 2, context=_ctx())
    self.assertEqual(sched.next_step, 1002)
    prev = [_Info(2, _EPOCH)]
    with mock.patch.object(sdp.time, 'monotonic', return_value=2.5):
      self.assertFalse(
          any_policy.should_save(_Info(3, _EPOCH), prev, context=_ctx())
      )
      self.assertTrue(
          any_policy.should_save(
              _Info(3, _EPOCH), prev, context=_ctx(preempt=True)
          )
      )
      self.assertTrue(
          any_policy.should_save(_Info(3, _EPOCH), [], context=_ctx())
      )
    # A non-scheduled save (e.g. preemption) re-anchors the schedule.
    with mock.patch.object(sdp.time, 'monotonic', return_value=3.0):
      sdp.notify_save_start(any_policy, 3, context=_ctx())
    self.assertEqual(sched.next_step, 1003)

  def test_restart_falls_back_to_legacy_then_schedules(self):
    # Previous run saved at step 100, at t=0 (simulated clock starts at 0).
    prev = [_Info(100, _EPOCH)]
    p = sdp.ScheduledContinuousCheckpointingPolicy(minimum_interval_secs=10)
    res = simulate(
        p,
        lambda s: 1.0,
        lambda i: 0.5,
        num_steps=100,
        previous=prev,
        start_step=101,
    )
    # The interval is measured from the first step after the restart (step
    # 101, t=1), not from the restored checkpoint (t=0).
    self.assertEqual(res.save_steps[0], 111)
    self.assertGreater(p.num_local_decisions, 0)
    for a, b in zip(res.save_steps[1:], res.save_steps[2:]):
      self.assertEqual(b - a, 10)

  def test_notify_ignores_other_policies(self):
    sdp.notify_save_start(sdp.FixedIntervalPolicy(5), 5, context=_ctx())
    sdp.notify_save_start(
        sdp.AnySavePolicy([sdp.InitialSavePolicy()]), 5, context=_ctx()
    )

  def test_fresh_start_schedule_set_during_bootstrap(self):
    p = sdp.ScheduledContinuousCheckpointingPolicy(minimum_interval_secs=10)
    res = simulate(p, lambda s: 1.0, lambda i: 0.5, num_steps=100)
    # Step 1 saves (no checkpoints). Then per-step agreement runs only until
    # the primary has min_samples_for_schedule samples.
    self.assertEqual(res.save_steps[:3], [1, 11, 21])
    self.assertLessEqual(p.num_bootstrap_decisions, 1 + 5)
    # The first schedule came from bootstrap, not from a save.
    first_scheduled = [s for s, n in p.schedule_history if n is not None][0]
    self.assertNotIn(first_scheduled, res.save_steps)

  def test_bootstrap_does_not_schedule_during_in_flight_write(self):
    # First write (15 s) outlasts the interval (10 s). Bootstrap must not
    # publish a schedule while it runs, so the second save never waits.
    p = sdp.ScheduledContinuousCheckpointingPolicy(minimum_interval_secs=10)
    res = simulate(p, lambda s: 1.0, lambda i: 15.0, num_steps=100)
    self.assertEqual(res.save_steps[:2], [1, 16])
    self.assertEqual(res.waits[1], 0.0)
    first_scheduled = [s for s, n in p.schedule_history if n is not None][0]
    self.assertGreaterEqual(first_scheduled, 16)

  def test_restart_bootstrap_is_short(self):
    prev = [_Info(100, _EPOCH)]
    p = sdp.ScheduledContinuousCheckpointingPolicy(minimum_interval_secs=10)
    simulate(
        p,
        lambda s: 1.0,
        lambda i: 0.5,
        num_steps=100,
        previous=prev,
        start_step=101,
    )
    # Steps 101..105: one step without a sample, then 4 samples.
    self.assertEqual(p.num_bootstrap_decisions, 5)

  def test_restart_long_after_checkpoint_does_not_save_immediately(self):
    # Restored checkpoint is older than the interval. Downtime does not count:
    # the first save comes one interval after training resumes.
    prev = [_Info(100, _EPOCH - datetime.timedelta(seconds=60))]
    p = sdp.ScheduledContinuousCheckpointingPolicy(minimum_interval_secs=10)
    res = simulate(
        p,
        lambda s: 1.0,
        lambda i: 0.5,
        num_steps=50,
        previous=prev,
        start_step=101,
    )
    # Step 101 runs at t=1; the first save is at t=11 (step 111).
    self.assertEqual(res.save_steps[0], 111)
    self.assertEqual(res.save_times[0], 11.0)
    # It came from the schedule set during bootstrap, not per-step agreement.
    self.assertEqual(p.schedule_history[0], (105, 111))
    for a, b in zip(res.save_steps, res.save_steps[1:]):
      self.assertEqual(b - a, 10)

  def test_restart_anchor_in_any_policy_with_initial_save(self):
    # Same as above inside the default composition: InitialSavePolicy does not
    # fire on restart either.
    prev = [_Info(100, _EPOCH - datetime.timedelta(seconds=3600))]
    p = sdp.AnySavePolicy([
        sdp.ScheduledContinuousCheckpointingPolicy(minimum_interval_secs=10),
        sdp.InitialSavePolicy(),
    ])
    res = simulate(
        p,
        lambda s: 1.0,
        lambda i: 0.5,
        num_steps=30,
        previous=prev,
        start_step=101,
    )
    self.assertEqual(res.save_steps[0], 111)

  def test_slow_write_duration_is_measured(self):
    p = sdp.ScheduledContinuousCheckpointingPolicy(minimum_interval_secs=10)
    simulate(p, lambda s: 1.0, lambda i: 25.0, num_steps=300)
    measured = p._last_write_secs  # pylint: disable=protected-access
    assert measured is not None
    self.assertAlmostEqual(measured, 25.0, delta=1.0)

  def test_step_time_increase_makes_saves_later_not_earlier(self):
    # Step time doubles at step 500. The schedule computed before the change
    # is too long in wall time (save late); it then re-adapts.
    p = sdp.ScheduledContinuousCheckpointingPolicy(minimum_interval_secs=20)
    res = simulate(
        p, lambda s: 0.5 if s < 500 else 1.0, lambda i: 1.0, num_steps=1500
    )
    for iv in res.intervals[1:]:
      self.assertGreaterEqual(iv, 20 - 1e-9)

  def test_step_time_decrease_makes_saves_early_once(self):
    # Step time halves at step 500. The schedule computed before the change
    # fires early in wall time once; the deficit feedback stretches the next.
    p = sdp.ScheduledContinuousCheckpointingPolicy(minimum_interval_secs=20)
    res = simulate(
        p, lambda s: 1.0 if s < 500 else 0.5, lambda i: 1.0, num_steps=2000
    )
    short = [iv for iv in res.intervals[1:] if iv < 20 - 1e-9]
    self.assertLessEqual(len(short), 1)
    if short:
      self.assertGreaterEqual(short[0], 10 - 1e-9)

  def test_any_policy_with_milestone_and_preemption(self):
    sched = sdp.ScheduledContinuousCheckpointingPolicy(minimum_interval_secs=10)
    policy = sdp.AnySavePolicy([
        sched,
        sdp.PreemptionCheckpointingPolicy(),
        sdp.SpecificStepsPolicy(steps=[55]),
    ])
    res = simulate(policy, lambda s: 1.0, lambda i: 0.5, num_steps=100)
    self.assertIn(55, res.save_steps)
    # Milestone at 55 re-anchors: next scheduled save 10 steps later.
    i = res.save_steps.index(55)
    self.assertEqual(res.save_steps[i + 1], 65)

  def test_order_in_any_policy_does_not_break_schedule(self):
    sched = sdp.ScheduledContinuousCheckpointingPolicy(minimum_interval_secs=10)
    policy = sdp.AnySavePolicy([sdp.SpecificStepsPolicy(steps=[55]), sched])
    res = simulate(policy, lambda s: 1.0, lambda i: 0.5, num_steps=100)
    self.assertIn(55, res.save_steps)
    self.assertGreater(sched.num_local_decisions, 0)
    i = res.save_steps.index(55)
    self.assertEqual(res.save_steps[i + 1], 65)


if __name__ == '__main__':
  absltest.main()
