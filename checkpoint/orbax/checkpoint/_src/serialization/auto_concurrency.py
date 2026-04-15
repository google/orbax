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

"""Manages automatic concurrency for Orbax Checkpoint operations.

This module provides a mechanism to dynamically adjust resource limits (e.g.,
maximum concurrent memory usage) based on system feedback, such as peak memory
usage and blocking times. It uses a PID-like controller and a registry for
profiling functions.
"""

from typing import Callable

# Global registry for callbacks
_START_MEMORY_PROFILER_FN = None
_STOP_AND_GET_PEAK_MEMORY_PROFILER_GIB_FN = None
_GET_PREVIOUS_BLOCKING_TIME_SEC_FN = None
_GET_EXPECTED_SURGE_GB_FN = None


def register_start_memory_profiler_fn(fn: Callable[[], None]):
  global _START_MEMORY_PROFILER_FN
  _START_MEMORY_PROFILER_FN = fn


def register_stop_and_get_peak_memory_profiler_gib_fn(fn: Callable[[], float]):
  global _STOP_AND_GET_PEAK_MEMORY_PROFILER_GIB_FN
  _STOP_AND_GET_PEAK_MEMORY_PROFILER_GIB_FN = fn


def register_get_previous_blocking_time_sec_fn(fn: Callable[[], float]):
  global _GET_PREVIOUS_BLOCKING_TIME_SEC_FN
  _GET_PREVIOUS_BLOCKING_TIME_SEC_FN = fn


def register_get_expected_surge_gb_fn(fn: Callable[[], float]):
  global _GET_EXPECTED_SURGE_GB_FN
  _GET_EXPECTED_SURGE_GB_FN = fn


def start_memory_profiler():
  if _START_MEMORY_PROFILER_FN:
    _START_MEMORY_PROFILER_FN()


def stop_and_get_peak_memory_profiler_gib() -> float:
  if _STOP_AND_GET_PEAK_MEMORY_PROFILER_GIB_FN:
    return _STOP_AND_GET_PEAK_MEMORY_PROFILER_GIB_FN()
  return 0.0


def get_previous_blocking_time_sec() -> float:
  if _GET_PREVIOUS_BLOCKING_TIME_SEC_FN:
    return _GET_PREVIOUS_BLOCKING_TIME_SEC_FN()
  return 0.0


def get_expected_surge_gb() -> float:
  if _GET_EXPECTED_SURGE_GB_FN:
    return _GET_EXPECTED_SURGE_GB_FN()
  return 0.0


class MaxConcurrentMemoryController:
  """Controls max concurrent d2h memory usage based on system feedback.

  This controller uses a PID-like mechanism, combined with custom logic,
  to adjust a memory limit. It aims to keep memory usage within a target ratio
  of the host limit, while also considering performance (blocking time) and
  preemptive adjustments for expected memory surges.
  """

  def __init__(
      self,
      host_limit_gib: float,
      target_ratio: float = 0.80,
      kp: float = 0.4,
      ki: float = 0.05,
      kd: float = 0.1,
      kb: float = 3.0,
      min_memory_limit_gib: float = 1.0,
  ):
    self.target_ratio = target_ratio
    self.kp = kp
    self.ki = ki
    self.kd = kd
    self.kb = kb

    self.integral = 0.0
    self.prev_error = 0.0

    self.host_limit_gib = host_limit_gib
    self.min_memory_limit = min_memory_limit_gib
    self.max_memory_limit = host_limit_gib * 0.80
    self.integral_windup_limit = 50.0

  def get_next_memory_limit(
      self,
      current_limit_gib: float,
      peak_rss_gib: float,
      blocking_time_sec: float,
      expected_surge_gib: float = 0.0,
  ) -> float:
    """Returns the next memory limit, factoring in preemptive known surges."""
    # --- PREEMPTIVE FEED-FORWARD OVERRIDE ---
    if expected_surge_gib > 0:
      # 1. Forcefully drop the current limit to make room for the surge
      current_limit_gib = max(
          self.min_memory_limit, current_limit_gib - expected_surge_gib
      )
      # 2. Erase PID history so it doesn't fight our manual drop
      self.integral = 0.0
      self.prev_error = 0.0

    effective_host_limit = self.max_memory_limit
    target_mem_gib = effective_host_limit * self.target_ratio

    error_gib = target_mem_gib - peak_rss_gib
    max_error_gib = effective_host_limit - peak_rss_gib

    # --- STANDARD PID MATH ---
    p_term = self.kp * error_gib

    self.integral += error_gib
    self.integral = max(
        -self.integral_windup_limit,
        min(self.integral_windup_limit, self.integral),
    )
    i_term = self.ki * self.integral

    d_term = self.kd * (error_gib - self.prev_error)
    self.prev_error = error_gib

    base_adjustment = p_term + i_term + d_term

    # --- CUSTOM LOGIC ---
    if max_error_gib < 0:
      # Prioritize memory space
      adjustment = base_adjustment * 2.0
    else:
      # Otherwise, boost for performance
      adjustment = base_adjustment
      if blocking_time_sec > 0:
        boost = self.kb * blocking_time_sec  # Linear regression
        adjustment += boost
        # Clamp adjustment to prevent overshooting the target
        adjustment = min(adjustment, max_error_gib)

    # Apply and clamp to hardware limits
    new_limit_gib = current_limit_gib + adjustment
    new_limit_gib = max(
        self.min_memory_limit, min(self.max_memory_limit, new_limit_gib)
    )

    return new_limit_gib
