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

from absl.testing import absltest
from absl.testing import parameterized
from orbax.checkpoint._src.serialization import memory_regulator


class MockMemoryProfiler(memory_regulator.MemoryProfiler):

  def __init__(self):
    super().__init__()
    self.started = False
    self.blocking_time = 0.0
    self.expected_surge = 0.0

  def profiler_start(self):
    self.started = True

  def profiler_end(self):
    pass

  def get_prev_blocking_time_sec(self):
    return self.blocking_time

  def get_expected_surge_gib(self):
    return self.expected_surge


class MemoryRegulatorTest(parameterized.TestCase):

  def test_global_registries_with_callbacks(self):
    profiler = MockMemoryProfiler()
    profiler._peak_usage_bytes = int(25.5 * 1024**3)
    profiler.blocking_time = 12.0
    profiler.expected_surge = 10.0

    memory_regulator.register_memory_profiler(profiler)
    self.addCleanup(memory_regulator.register_memory_profiler, None)

    memory_regulator.profiler_start()
    self.assertTrue(profiler.started)
    memory_regulator.profiler_end()
    self.assertEqual(memory_regulator.profiler_peak_usage_gib(), 25.5)
    self.assertEqual(memory_regulator.get_prev_blocking_time_sec(), 12.0)
    self.assertEqual(memory_regulator.get_expected_surge_gib(), 10.0)

  def test_controller_initialization(self):
    controller = memory_regulator.MemoryRegulator(
        max_memory_limit_gib=80.0,
        min_memory_limit_gib=2.0,
    )
    self.assertEqual(controller.max_memory_limit_gib, 80.0)
    self.assertEqual(controller.target_ratio, 0.8)
    self.assertEqual(controller.min_memory_limit_gib, 2.0)
    self.assertEqual(controller.integral, 0.0)

  @parameterized.named_parameters(
      dict(
          testcase_name="increase",
          current_limit_gib=10.0,
          peak_memory_usage_gib=50.0,
          # Target = 64.0. error = 14.0.
          # adjustment = 0.5 * 14.0 = 7.0.
          # new_limit = 10.0 + 7.0 = 17.0.
          expected_next_limit=17.0,
      ),
      dict(
          testcase_name="decrease",
          current_limit_gib=30.0,
          peak_memory_usage_gib=74.0,
          # Target = 64.0. error = -10.0.
          # adjustment = 0.5 * -10.0 = -5.0.
          # new_limit = 30.0 - 5.0 = 25.0.
          expected_next_limit=25.0,
      ),
      dict(
          testcase_name="danger_zone_decrease",
          current_limit_gib=40.0,
          peak_memory_usage_gib=82.0,
          # Target = 64.0. error = -18.0.
          # adjustment = 0.5 * -18.0 = -9.0.
          # new_limit = 40.0 - 9.0 = 31.0.
          expected_next_limit=31.0,
      ),
  )
  def test_controller_pid(
      self, current_limit_gib, peak_memory_usage_gib, expected_next_limit
  ):
    controller = memory_regulator.MemoryRegulator(
        max_memory_limit_gib=80.0, target_ratio=0.8, kp=0.5, ki=0, kd=0
    )
    next_limit = controller.get_next_memory_limit(
        current_limit_gib=current_limit_gib,
        peak_memory_usage_gib=peak_memory_usage_gib,
        blocking_time_sec=0.0,
    )
    self.assertAlmostEqual(next_limit, expected_next_limit)

  def test_controller_feedforward_increase(self):
    controller = memory_regulator.MemoryRegulator(max_memory_limit_gib=80.0)
    controller.integral = 10.0  # Set some fake history
    controller.prev_error = 5.0

    # current limit 30, expected surge 12 -> immediate drop to 18 to make room
    new_limit = controller.get_next_memory_limit(
        current_limit_gib=30.0,
        peak_memory_usage_gib=64.0,
        blocking_time_sec=0.0,
        expected_surge_gib=12.0,
    )

    # History should be PRESERVED (frozen) during surge.
    self.assertEqual(controller.integral, 10.0)
    self.assertEqual(controller.prev_error, 5.0)
    self.assertLess(new_limit, 30.0)

  def test_initialization_validation(self):
    with self.assertRaisesRegex(
        ValueError, "max_memory_limit_gib must be positive"
    ):
      memory_regulator.MemoryRegulator(max_memory_limit_gib=-1.0)
    with self.assertRaisesRegex(
        ValueError, "min_memory_limit_gib must be positive"
    ):
      memory_regulator.MemoryRegulator(
          max_memory_limit_gib=80.0, min_memory_limit_gib=0
      )
    with self.assertRaisesRegex(
        ValueError, "min_memory_limit_gib must be less than target memory"
    ):
      memory_regulator.MemoryRegulator(
          max_memory_limit_gib=10.0, target_ratio=0.5, min_memory_limit_gib=6.0
      )

  def test_danger_zone_with_positive_derivative(self):
    # Target = 64.0. Max = 80.0.
    controller = memory_regulator.MemoryRegulator(
        max_memory_limit_gib=80.0,
        min_memory_limit_gib=10.0,
        target_ratio=0.8,
        kp=0.5,
        ki=0,
        kd=10.0,
    )
    # Step 1: Huge OOM.
    # Peak = 120. error = 64-120 = -56.
    _ = controller.get_next_memory_limit(40.0, 120.0, 0.0)

    # Step 2: Still over hard limit (80), but Peak is recovering.
    # New peak = 85. error = 64-85 = -21.
    # prev_error was -56.
    # d_term = 10 * (-21 - (-56)) = 10 * 35 = 350.
    # p_term = 0.5 * -21 = -10.5.
    # base_adjustment = 350 - 10.5 = 339.5.
    # adjustment is capped at max_error_gib (-5.0).
    next_limit = controller.get_next_memory_limit(20.0, 85.0, 0.0)
    # 20.0 - 5.0 = 15.0
    self.assertAlmostEqual(next_limit, 15.0)

  def test_integral_windup_clamping(self):
    # ki=1.0 to see integral effect clearly.
    controller = memory_regulator.MemoryRegulator(
        max_memory_limit_gib=80.0, target_ratio=0.8, kp=0, ki=1.0, kd=0
    )
    # Target = 64. error = 64 - 10 = 54.
    # windup limit is 50.0.
    _ = controller.get_next_memory_limit(30.0, 10.0, 0.0)
    self.assertEqual(controller.integral, 50.0)

    # Opposite direction.
    _ = controller.get_next_memory_limit(
        30.0, 200.0, 0.0
    )  # error = 64 - 200 = -136
    self.assertEqual(controller.integral, -50.0)

  def test_surge_history_preservation(self):
    controller = memory_regulator.MemoryRegulator(
        max_memory_limit_gib=80.0, kd=1.0
    )
    # Target = 64. Peak = 90. error = -26.
    _ = controller.get_next_memory_limit(30.0, 90.0, 0.0)
    self.assertAlmostEqual(controller.prev_error, -26.0)

    # Surge happens.
    _ = controller.get_next_memory_limit(
        30.0, 90.0, 0.0, expected_surge_gib=10.0
    )
    # prev_error should be FROZEN (not updated) during surge.
    self.assertAlmostEqual(controller.prev_error, -26.0)

  def test_surge_resumption_level(self):
    controller = memory_regulator.MemoryRegulator(
        max_memory_limit_gib=80.0, kp=0.5
    )
    # Target = 64.0.
    # Baseline: Current limit 30.0. Peak 64.0 (error=0).
    # next_limit = 30.0 + 0.5*0 = 30.0.
    limit = controller.get_next_memory_limit(30.0, 64.0, 0.0)
    self.assertEqual(limit, 30.0)

    # Surge starts: expected_surge = 10.0.
    # adjustment should be 0.0 (bypassed).
    # surge_delta = 10.0 - 0.0 = 10.0.
    # next_limit = 30.0 + 0.0 - 10.0 = 20.0.
    limit = controller.get_next_memory_limit(
        30.0, 64.0, 0.0, expected_surge_gib=10.0
    )
    self.assertEqual(limit, 20.0)

    # Surge continues: expected_surge = 10.0.
    # adjustment = 0.0. surge_delta = 10.0 - 10.0 = 0.0.
    # next_limit = 20.0 + 0.0 - 0.0 = 20.0.
    limit = controller.get_next_memory_limit(
        20.0, 64.0, 0.0, expected_surge_gib=10.0
    )
    self.assertEqual(limit, 20.0)

    # Surge ends: expected_surge = 0.0.
    # adjustment = 0.5 * (64.0 - 64.0) = 0.0.
    # surge_delta = 0.0 - 10.0 = -10.0.
    # next_limit = 20.0 + 0.0 - (-10.0) = 30.0.
    limit = controller.get_next_memory_limit(
        20.0, 64.0, 0.0, expected_surge_gib=0.0
    )
    self.assertEqual(limit, 30.0)

  def test_consecutive_surges(self):
    controller = memory_regulator.MemoryRegulator(
        max_memory_limit_gib=80.0, kp=0.5
    )

    limit = controller.get_next_memory_limit(
        30.0, 64.0, 0.0, expected_surge_gib=10.0
    )
    self.assertEqual(limit, 20.0)  # 30 - 10 = 20.

    limit = controller.get_next_memory_limit(
        20.0, 64.0, 0.0, expected_surge_gib=15.0
    )
    self.assertEqual(limit, 15.0)  # 30 - 15 = 15.

    limit = controller.get_next_memory_limit(
        15.0, 64.0, 0.0, expected_surge_gib=15.0
    )
    self.assertEqual(limit, 15.0)  # 30 - 15 = 15.

    limit = controller.get_next_memory_limit(
        15.0, 64.0, 0.0, expected_surge_gib=0.0
    )
    self.assertEqual(limit, 30.0)

  def test_adjustment_during_surge(self):
    controller = memory_regulator.MemoryRegulator(
        max_memory_limit_gib=80.0, kp=0.5, ki=0, kd=0
    )
    # Target = 64.0.
    # Current limit 30.0. Surge 10.0.
    # Peak Memory Usage = 74.0. error = 64.0 - 74 = -10.
    # PID adjustment = 0.5 * -10 = -5.
    # Surge delta = 10 - 0 = 10.
    limit = controller.get_next_memory_limit(
        30.0, 74.0, 0.0, expected_surge_gib=10.0
    )
    # next_limit = 30 + (-5) - 10 = 15.
    self.assertEqual(limit, 15.0)


if __name__ == "__main__":
  absltest.main()
