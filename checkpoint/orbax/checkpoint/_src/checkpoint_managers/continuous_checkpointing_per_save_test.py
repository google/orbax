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

"""Simulated-clock tests for the per-save agreement of continuous checkpointing.

Drives `ContinuousCheckpointingPolicy(agreement='per_save')` with a fake clock
and fake background writes, and compares its save steps with
`agreement='per_check'`.
"""

import contextlib
import dataclasses
import datetime
import random
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from orbax.checkpoint import options as options_lib
from orbax.checkpoint._src.checkpoint_managers import save_decision_policy as sdp


@dataclasses.dataclass
class _Info:
  step: int
  time: datetime.datetime
  metrics: dict[str, float] | None = None


_EPOCH = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
# Tolerance for wall-time intervals (datetime has microsecond resolution).
_EPS = 1e-5


def _at(secs: float) -> datetime.datetime:
  return _EPOCH + datetime.timedelta(seconds=secs)


def _ctx(is_saving: bool = False, preempt: bool = False) -> sdp.DecisionContext:
  return sdp.DecisionContext(
      is_saving_in_progress=is_saving,
      reached_preemption=preempt,
      multiprocessing_options=options_lib.MultiprocessingOptions(),
  )


def _per_save(**kwargs) -> sdp.ContinuousCheckpointingPolicy:
  return sdp.ContinuousCheckpointingPolicy(agreement='per_save', **kwargs)


def _find_per_save(
    policy: sdp.SaveDecisionPolicy,
) -> sdp.ContinuousCheckpointingPolicy | None:
  policies = (
      policy.policies if isinstance(policy, sdp.AnySavePolicy) else [policy]
  )
  for p in policies:
    if (
        isinstance(p, sdp.ContinuousCheckpointingPolicy)
        and p.agreement == 'per_save'
    ):
      return p
  return None


def _exchanges(p: sdp.ContinuousCheckpointingPolicy) -> int:
  return p.num_bootstrap_decisions + p.num_planned_step_decisions


@dataclasses.dataclass
class SimResult:
  save_steps: list[int]
  save_times: list[float]
  waits: list[float]
  # Steps at which the policy made a key-value exchange in `should_save`.
  exchange_steps: list[int]
  # Saves whose write was cancelled by a later target step.
  cancelled_steps: list[int] = dataclasses.field(default_factory=list)

  @property
  def intervals(self) -> list[float]:
    return [b - a for a, b in zip(self.save_times, self.save_times[1:])]

  def exchanges_between(self, first: int, last: int) -> int:
    return sum(1 for s in self.exchange_steps if first <= s <= last)


def simulate(
    policy: sdp.SaveDecisionPolicy,
    step_time_fn,
    write_secs_fn,
    num_steps: int,
    previous: list[_Info] | None = None,
    start_step: int = 1,
    preempt_steps: tuple[int, ...] = (),
    exchange_secs: float = 0.0,
    target_interval_steps: int | None = None,
) -> SimResult:
  """Simulates a training loop driving `policy` like CheckpointManager does.

  A save that starts while the previous write is running waits for it (this is
  what `CheckpointManager.save` does after `should_save` returns True), except
  that a save at a multiple of `target_interval_steps` cancels an in-flight
  non-target write (and its checkpoint), like `CheckpointManager.save`.

  Args:
    policy: policy under test.
    step_time_fn: step -> seconds of compute for that step.
    write_secs_fn: save index -> seconds the background write takes.
    num_steps: number of steps.
    previous: existing checkpoints (restart case).
    start_step: first step.
    preempt_steps: steps at which `reached_preemption` is True.
    exchange_secs: wall time each key-value exchange in `should_save` takes.
    target_interval_steps: target steps that cancel in-flight non-target
      writes, as in `CheckpointManager.save`.

  Returns:
    SimResult.
  """
  clock = [0.0]
  previous = list(previous or [])
  write_end = -1.0
  res = SimResult([], [], [], [])
  sched = _find_per_save(policy)
  with contextlib.ExitStack() as stack:
    stack.enter_context(
        mock.patch.object(sdp.time, 'monotonic', lambda: clock[0])
    )
    if sched and exchange_secs:
      orig_exchange = sched._agreed_decision  # pylint: disable=protected-access

      def slow_exchange(*a, **k):
        clock[0] += exchange_secs
        return orig_exchange(*a, **k)

      stack.enter_context(
          mock.patch.object(sched, '_agreed_decision', slow_exchange)
      )
    for step in range(start_step, start_step + num_steps):
      clock[0] += step_time_fn(step)
      if previous and previous[-1].step >= step:
        continue
      info = _Info(step, _at(clock[0]))
      preempt = step in preempt_steps
      ctx = _ctx(is_saving=clock[0] < write_end, preempt=preempt)
      before = _exchanges(sched) if sched else 0
      decision = policy.should_save(info, previous, context=ctx)
      if sched and _exchanges(sched) > before:
        res.exchange_steps.append(step)
      if decision:
        wait = max(0.0, write_end - clock[0])
        if (
            wait > 0
            and target_interval_steps
            and step % target_interval_steps == 0
            and previous
            and previous[-1].step % target_interval_steps != 0
        ):
          # Cancel the in-flight continuous write and drop its checkpoint.
          previous.pop()
          res.cancelled_steps.append(res.save_steps[-1])
          wait = 0.0
        clock[0] += wait
        sdp.notify_save_start(policy, step, context=_ctx(preempt=preempt))
        res.save_steps.append(step)
        res.save_times.append(clock[0])
        res.waits.append(wait)
        previous.append(info)
        write_end = clock[0] + write_secs_fn(len(res.save_steps))
  return res


class PerSaveAgreementTest(parameterized.TestCase):

  def test_rejects_bad_args(self):
    with self.assertRaises(ValueError):
      _per_save(minimum_interval_secs=-1)
    with self.assertRaises(ValueError):
      _per_save(minimum_interval_secs=1, step_time_quantile=2.0)
    with self.assertRaises(ValueError):
      _per_save(minimum_interval_secs=1, write_hint_fraction=1.5)
    with self.assertRaises(ValueError):
      _per_save(minimum_interval_secs=1, write_hint_window=0)
    with self.assertRaises(ValueError):
      sdp.ContinuousCheckpointingPolicy(
          agreement='every_step'  # pyrefly: ignore[bad-argument-type]
      )

  def test_per_check_ignores_save_start_notification(self):
    p = sdp.ContinuousCheckpointingPolicy(minimum_interval_secs=10)
    sdp.notify_save_start(p, 5, context=_ctx())
    self.assertIsNone(p.next_step)
    self.assertEmpty(p.schedule_history)

  @parameterized.named_parameters(
      dict(
          testcase_name='interval_only',
          minimum_interval_secs=3,
          target_interval_steps=None,
          deferral_window_steps=None,
          write_secs=0.5,
      ),
      dict(
          testcase_name='target_and_window',
          minimum_interval_secs=3,
          target_interval_steps=10,
          deferral_window_steps=2,
          write_secs=0.5,
      ),
      dict(
          testcase_name='default_window',
          minimum_interval_secs=3,
          target_interval_steps=10,
          deferral_window_steps=None,
          write_secs=0.5,
      ),
      dict(
          testcase_name='no_interval_slow_writes',
          minimum_interval_secs=None,
          target_interval_steps=7,
          deferral_window_steps=2,
          write_secs=2.5,
      ),
      dict(
          testcase_name='writes_overrun_targets',
          minimum_interval_secs=4,
          target_interval_steps=10,
          deferral_window_steps=2,
          write_secs=13.0,
      ),
  )
  def test_same_saves_as_per_check_with_constant_step_time(
      self,
      minimum_interval_secs,
      target_interval_steps,
      deferral_window_steps,
      write_secs,
  ):
    kwargs = dict(
        minimum_interval_secs=minimum_interval_secs,
        target_interval_steps=target_interval_steps,
        deferral_window_steps=deferral_window_steps,
    )
    per_save = simulate(
        _per_save(**kwargs),
        lambda s: 1.0,
        lambda i: write_secs,
        300,
        target_interval_steps=target_interval_steps,
    )
    per_check = simulate(
        sdp.ContinuousCheckpointingPolicy(**kwargs),
        lambda s: 1.0,
        lambda i: write_secs,
        300,
        target_interval_steps=target_interval_steps,
    )
    self.assertEqual(per_save.save_steps, per_check.save_steps)
    self.assertEqual(per_save.cancelled_steps, per_check.cancelled_steps)
    # A target step cancels an in-flight continuous write; it only waits for
    # an in-flight target write. Other saves never wait.
    for s, w in zip(per_save.save_steps, per_save.waits):
      if w > 0:
        assert target_interval_steps is not None
        self.assertEqual(s % target_interval_steps, 0)

  def test_target_steps_saved_and_no_saves_in_window(self):
    p = _per_save(
        minimum_interval_secs=3,
        target_interval_steps=10,
        deferral_window_steps=2,
    )
    res = simulate(p, lambda s: 1.0, lambda i: 0.5, num_steps=200)
    for target in range(10, 201, 10):
      self.assertIn(target, res.save_steps)
    for s in res.save_steps:
      self.assertNotIn(s % 10, (8, 9), msg=f'save at {s} in window')
    # One exchange per save (the planned step) and one plan per save.
    first_plan = [s for s, n in p.schedule_history if n is not None][0]
    later_saves = [s for s in res.save_steps if s > first_plan]
    self.assertEqual(
        [s for s in res.exchange_steps if s > first_plan], later_saves
    )
    plans_at_saves = [s for s, _ in p.schedule_history if s in later_saves]
    self.assertEqual(plans_at_saves, later_saves)

  def test_target_saves_while_write_runs_like_per_check(self):
    kwargs = dict(
        minimum_interval_secs=3,
        target_interval_steps=10,
        deferral_window_steps=2,
    )
    write_secs_fn = lambda i: 9.0 if i == 3 else 0.5
    # The 3rd write (save at step 7) takes 9 s and runs over target step 10.
    per_save = simulate(
        _per_save(**kwargs),
        lambda s: 1.0,
        write_secs_fn,
        60,
        target_interval_steps=10,
    )
    per_check = simulate(
        sdp.ContinuousCheckpointingPolicy(**kwargs),
        lambda s: 1.0,
        write_secs_fn,
        60,
        target_interval_steps=10,
    )
    self.assertEqual(per_save.save_steps, per_check.save_steps)
    # Target step 10 saves although the write of step 7 runs; that write is
    # cancelled, and the save does not wait.
    self.assertIn(10, per_save.save_steps)
    self.assertEqual(per_save.cancelled_steps, [7])
    self.assertEqual(sum(per_save.waits), 0.0)

  def test_zero_interval_saves_when_write_ends(self):
    p = _per_save(minimum_interval_secs=0)
    res = simulate(p, lambda s: 1.0, lambda i: 5.0, num_steps=100)
    self.assertEqual(sum(res.waits), 0.0)
    # Each save starts on the first step after the previous write ends.
    for iv in res.intervals[3:]:
      self.assertAlmostEqual(iv, 5.0, delta=_EPS)

  def test_zero_interval_exchanges_per_write_are_few(self):
    # The write takes 50 steps. The planned step is at 0.8 x the shortest
    # recent write, so only the last ~10 steps of each write are exchanges,
    # not all 50 (as with agreement='per_check').
    p = _per_save(minimum_interval_secs=0)
    res = simulate(p, lambda s: 1.0, lambda i: 50.0, num_steps=1000)
    self.assertEqual(sum(res.waits), 0.0)
    for a, b in zip(res.save_steps[4:], res.save_steps[5:]):
      self.assertEqual(b - a, 50)
      self.assertLessEqual(res.exchanges_between(a + 1, b), 11)

  def test_preemption_save_skips_exchange_and_clears_schedule(self):
    p = _per_save(minimum_interval_secs=10)
    prev = [_Info(0, _EPOCH)]
    clock = [0.0]
    with mock.patch.object(sdp.time, 'monotonic', lambda: clock[0]):
      for step in range(1, 6):
        clock[0] = float(step)
        p.should_save(_Info(step, _at(step)), prev, context=_ctx())
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

  def test_preemption_save_rebootstraps_then_saves_after_interval(self):
    policy = sdp.AnySavePolicy([
        _per_save(minimum_interval_secs=10),
        sdp.PreemptionCheckpointingPolicy(),
    ])
    res = simulate(
        policy, lambda s: 1.0, lambda i: 0.5, num_steps=60, preempt_steps=(25,)
    )
    self.assertEqual(res.save_steps[:5], [1, 11, 21, 25, 35])
    sched = _find_per_save(policy)
    assert sched is not None
    # The preemption save cleared the schedule; re-bootstrap set a new one.
    self.assertIn((25, None), sched.schedule_history)

  def test_bootstrap_saves_without_checkpoints(self):
    p = _per_save(minimum_interval_secs=10)
    self.assertIsNone(p.next_step)
    self.assertTrue(p.should_save(_Info(1, _EPOCH), [], context=_ctx()))
    self.assertEqual(p.num_bootstrap_decisions, 1)

  def test_schedule_is_computed_and_decisions_are_local(self):
    p = _per_save(minimum_interval_secs=10, step_time_quantile=0.5)
    prev = [_Info(0, _EPOCH)]
    clock = [0.0]
    with mock.patch.object(sdp.time, 'monotonic', lambda: clock[0]):
      for step in range(1, 6):
        clock[0] = float(step)
        p.should_save(_Info(step, _at(step)), prev, context=_ctx())
      sdp.notify_save_start(p, 5, context=_ctx())
      self.assertEqual(p.next_step, 15)
      prev = [_Info(5, _at(5))]
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
              _Info(step, _at(step)), prev, context=_ctx()
          )
    self.assertEqual([s for s, d in decisions.items() if d], [15])
    # Steps 6..14 are local; step 15 (planned) asks the primary host.
    self.assertEqual(p.num_local_decisions - local_before, 9)
    self.assertEqual(p.num_planned_step_decisions, 1)
    self.assertIsNone(p.next_step)

  def test_normal_write_one_exchange_per_planned_step(self):
    p = _per_save(minimum_interval_secs=10)
    res = simulate(p, lambda s: 1.0, lambda i: 0.5, num_steps=200)
    first_scheduled = [s for s, n in p.schedule_history if n is not None][0]
    later_saves = [s for s in res.save_steps if s > first_scheduled]
    later_exchanges = [s for s in res.exchange_steps if s > first_scheduled]
    self.assertLen(later_saves, 19)
    # Exactly one exchange per save (at its planned step), none elsewhere.
    self.assertEqual(later_exchanges, later_saves)
    self.assertLen(later_saves, p.num_planned_step_decisions)

  def test_slow_write_at_planned_step_skips_then_saves_after_write(self):
    p = _per_save(minimum_interval_secs=10)
    # Saves at steps 1, 11, 21, ...; the 3rd write (step 21) takes 25 s.
    res = simulate(
        p, lambda s: 1.0, lambda i: 25.0 if i == 3 else 0.5, num_steps=100
    )
    self.assertEqual(res.save_steps[:3], [1, 11, 21])
    self.assertIn((21, 31), p.schedule_history)
    # No save at the planned step 31: the write runs until t=46.
    self.assertNotIn(31, res.save_steps)
    # Save on the first step after the write ends, without waiting.
    self.assertEqual(res.save_steps[3], 46)
    self.assertEqual(sum(res.waits), 0.0)
    # One exchange per step while the write overran its planned step.
    self.assertEqual(res.exchanges_between(22, 46), 46 - 31 + 1)

  def test_slow_write_does_not_stretch_next_interval(self):
    p = _per_save(minimum_interval_secs=10)
    res = simulate(
        p, lambda s: 1.0, lambda i: 25.0 if i == 3 else 0.5, num_steps=100
    )
    i = res.save_steps.index(46)
    # After the slow write, the interval is back to minimum_interval_secs.
    self.assertEqual(res.save_steps[i + 1 : i + 4], [56, 66, 76])
    self.assertIn((46, 56), p.schedule_history)

  def test_zero_interval_spike_write_does_not_delay_planning(self):
    p = _per_save(minimum_interval_secs=0)
    # Writes take 5 s; the 4th takes 20 s.
    res = simulate(
        p, lambda s: 1.0, lambda i: 20.0 if i == 4 else 5.0, num_steps=80
    )
    self.assertEqual(sum(res.waits), 0.0)
    i = 4  # index of the save after the spike write.
    self.assertEqual(res.save_steps[i] - res.save_steps[i - 1], 20)
    # The next planned step is 0.8 x min(recent writes) = 4 steps after the
    # save, not 0.8 x 20 = 16.
    self.assertIn(
        (res.save_steps[i], res.save_steps[i] + 4), p.schedule_history
    )
    self.assertEqual(res.save_steps[i + 1] - res.save_steps[i], 5)
    a, b = res.save_steps[i], res.save_steps[i + 1]
    self.assertEqual(res.exchanges_between(a + 1, b), 2)

  def test_constant_step_time_meets_interval_exactly(self):
    p = _per_save(minimum_interval_secs=10)
    res = simulate(p, lambda s: 0.3, lambda i: 1.0, num_steps=2000)
    steady = res.intervals[2:]
    self.assertNotEmpty(steady)
    for iv in steady:
      self.assertGreaterEqual(iv, 10 - _EPS)
      self.assertLess(iv, 10 + 0.3 + _EPS)
    self.assertEqual(sum(res.waits), 0.0)

  def test_jittery_step_time_never_saves_early(self):
    rng = random.Random(0)
    times = [rng.lognormvariate(0, 0.3) for _ in range(20001)]
    p = _per_save(minimum_interval_secs=30)
    res = simulate(p, lambda s: times[s], lambda i: 1.0, num_steps=20000)
    steady = res.intervals[2:]
    # The primary host checks the interval at the planned step, so no
    # interval is short. With the median estimate, a plan lands at most a few
    # steps late.
    for iv in steady:
      self.assertGreaterEqual(iv, 30 - _EPS)
      self.assertLessEqual(iv, 30 + 8 * max(times))
    self.assertLess(sum(steady) / len(steady), 33)
    self.assertEqual(sum(res.waits), 0.0)

  def test_slow_writes_never_wait(self):
    p = _per_save(minimum_interval_secs=10)
    res = simulate(p, lambda s: 1.0, lambda i: 25.0, num_steps=300)
    self.assertEqual(sum(res.waits), 0.0)
    # Same save steps as agreement='per_check'.
    legacy = simulate(
        sdp.ContinuousCheckpointingPolicy(minimum_interval_secs=10),
        lambda s: 1.0,
        lambda i: 25.0,
        num_steps=300,
    )
    self.assertEqual(res.save_steps, legacy.save_steps)

  def test_preemption_and_initial_unaffected_in_any_policy(self):
    sched = _per_save(minimum_interval_secs=1000)
    any_policy = sdp.AnySavePolicy(
        [sched, sdp.PreemptionCheckpointingPolicy(), sdp.InitialSavePolicy()]
    )
    # Establish a schedule far in the future. Each bootstrap call reads the
    # clock twice (before and after its exchange).
    with mock.patch.object(
        sdp.time, 'monotonic', side_effect=[0.0, 0.0, 1.0, 1.0, 2.0]
    ):
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
    # A non-scheduled save (e.g. milestone) re-anchors the schedule.
    with mock.patch.object(sdp.time, 'monotonic', return_value=3.0):
      sdp.notify_save_start(any_policy, 3, context=_ctx())
    self.assertEqual(sched.next_step, 1003)

  def test_restart_falls_back_to_per_step_then_schedules(self):
    # Previous run saved at step 100, at t=0 (simulated clock starts at 0).
    prev = [_Info(100, _EPOCH)]
    p = _per_save(minimum_interval_secs=10)
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
    p = _per_save(minimum_interval_secs=10)
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
    p = _per_save(minimum_interval_secs=10)
    res = simulate(p, lambda s: 1.0, lambda i: 15.0, num_steps=100)
    self.assertEqual(res.save_steps[:2], [1, 16])
    self.assertEqual(res.waits[1], 0.0)
    first_scheduled = [s for s, n in p.schedule_history if n is not None][0]
    self.assertGreaterEqual(first_scheduled, 16)

  def test_restart_bootstrap_is_short(self):
    prev = [_Info(100, _EPOCH)]
    p = _per_save(minimum_interval_secs=10)
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
    p = _per_save(minimum_interval_secs=10)
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
        _per_save(minimum_interval_secs=10),
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
    p = _per_save(minimum_interval_secs=10)
    simulate(p, lambda s: 1.0, lambda i: 25.0, num_steps=300)
    measured = p._last_write_secs  # pylint: disable=protected-access
    assert measured is not None
    self.assertAlmostEqual(measured, 25.0, delta=1.0)

  def test_step_time_increase_makes_saves_later_not_earlier(self):
    # Step time doubles at step 500. The schedule computed before the change
    # is too long in wall time (save late); it then re-adapts.
    p = _per_save(minimum_interval_secs=20)
    res = simulate(
        p, lambda s: 0.5 if s < 500 else 1.0, lambda i: 1.0, num_steps=1500
    )
    for iv in res.intervals[1:]:
      self.assertGreaterEqual(iv, 20 - _EPS)

  def test_step_time_decrease_never_saves_early(self):
    # Step time halves at step 500. The planned step computed before the
    # change comes early in wall time; the primary host sees that the
    # interval has not passed and publishes a new planned step instead.
    # Constant step time after the window turns over: exact plans.
    p = _per_save(minimum_interval_secs=20)
    res = simulate(
        p, lambda s: 1.0 if s < 500 else 0.5, lambda i: 1.0, num_steps=2000
    )
    for iv in res.intervals[1:]:
      self.assertGreaterEqual(iv, 20 - _EPS)
    republished = [
        (s, n)
        for s, n in p.schedule_history
        if n is not None and s not in res.save_steps
    ]
    # One schedule from bootstrap, then a few republishes per interval while
    # the step time window still holds the old, slower samples.
    self.assertLessEqual(len(republished), 20)
    # After the window turns over, plans are exact again.
    late = [s for s, _ in republished if s > 1000]
    self.assertEmpty(late)

  def test_exchange_time_is_not_counted_as_step_time(self):
    # Each exchange takes 5 s, a step 1 s. If exchange time leaked into the
    # step time estimate, planned steps would come far too early and the
    # primary host would republish many times per save.
    p = _per_save(minimum_interval_secs=30)
    res = simulate(
        p, lambda s: 1.0, lambda i: 0.5, num_steps=1000, exchange_secs=5.0
    )
    later = res.save_steps[2:]
    self.assertNotEmpty(later)
    # One exchange per save: at the planned step.
    self.assertEqual(
        res.exchanges_between(later[0] + 1, later[-1]), len(later) - 1
    )

  def test_jittery_planned_steps_converge_in_few_exchanges(self):
    rng = random.Random(1)
    times = [rng.lognormvariate(0, 0.3) for _ in range(20001)]
    p = _per_save(minimum_interval_secs=30)
    res = simulate(p, lambda s: times[s], lambda i: 1.0, num_steps=20000)
    later = [s for s in res.save_steps if s > 1000]
    # With the median estimate, a planned step usually saves; an early one
    # republishes for the remaining time. Few exchanges per save, against
    # ~30 per save for agreement='per_check'.
    self.assertLessEqual(
        res.exchanges_between(later[0] + 1, later[-1]) / (len(later) - 1), 3
    )
    # Bounded lateness: a few median step times past the minimum interval.
    steady = [
        b - a
        for a, b, s in zip(res.save_times, res.save_times[1:], res.save_steps)
        if s > 1000
    ]
    self.assertLess(sum(steady) / len(steady), 30 + 3)

  def test_any_policy_with_milestone_and_preemption(self):
    sched = _per_save(minimum_interval_secs=10)
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
    sched = _per_save(minimum_interval_secs=10)
    policy = sdp.AnySavePolicy([sdp.SpecificStepsPolicy(steps=[55]), sched])
    res = simulate(policy, lambda s: 1.0, lambda i: 0.5, num_steps=100)
    self.assertIn(55, res.save_steps)
    self.assertGreater(sched.num_local_decisions, 0)
    i = res.save_steps.index(55)
    self.assertEqual(res.save_steps[i + 1], 65)


if __name__ == '__main__':
  absltest.main()
