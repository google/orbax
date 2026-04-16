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

"""Manages memory limits for checkpoint operations via PID control and profiling."""

from __future__ import annotations

import abc
import dataclasses

from absl import logging
import humanize

# CONSTANT
_BYTES_TO_GIB = 1024.0**3


class MemoryProfiler(abc.ABC):
  """A memory profiler providing feedback for memory regulation."""

  def __init__(self):
    self._peak_usage_bytes = 0

  @property
  def peak_usage_bytes(self) -> int:
    return self._peak_usage_bytes

  @abc.abstractmethod
  def profiler_start(self) -> None:
    """Starts the memory profiler."""
    raise NotImplementedError

  @abc.abstractmethod
  def profiler_end(self) -> None:
    """Stops the profiler."""
    raise NotImplementedError

  @property
  def peak_usage_gib(self) -> float:
    """Peak memory usage in GiB."""
    return self.peak_usage_bytes / _BYTES_TO_GIB

  @abc.abstractmethod
  def get_prev_blocking_time_sec(self) -> float:
    """Returns the previous iteration's blocking time in seconds."""
    raise NotImplementedError

  @abc.abstractmethod
  def get_expected_surge_gib(self) -> float:
    """Returns the expected memory surge for the next iteration in GiB."""
    raise NotImplementedError

  @property
  def total_memory_gib(self) -> float:
    """Total system capacity in GiB, if available."""
    raise NotImplementedError


_profiler: MemoryProfiler | None = None


def register_memory_profiler(profiler: MemoryProfiler | None) -> None:
  global _profiler
  _profiler = profiler


def profiler_start() -> None:
  if _profiler:
    _profiler.profiler_start()


def profiler_end() -> None:
  if _profiler:
    _profiler.profiler_end()


def profiler_peak_usage_gib() -> float:
  if _profiler:
    return _profiler.peak_usage_gib
  return 0.0


def get_prev_blocking_time_sec() -> float:
  if _profiler:
    return _profiler.get_prev_blocking_time_sec()
  return 0.0


def get_expected_surge_gib() -> float:
  if _profiler:
    return _profiler.get_expected_surge_gib()
  return 0.0


def get_total_memory_gib() -> float:
  if _profiler:
    return _profiler.total_memory_gib
  return 0.0


@dataclasses.dataclass
class MemoryRegulator:
  """Regulates maximum concurrent memory usage using a PID controller based on peak memory usage feedback.

  For setting up the coefficients, we have the following guidelines:

  | Coefficient | Suggested Range | Justification |
  | :--- | :--- | :--- |
  | kp | 0.30 - 0.60 | A moderate Kp (~0.4) safely scales the limit (e.g.,
  opening by 4 GiB for a 10 GiB gap). |
  | ki | <= 0.08 | Must be kept low to mitigate integral windup. |
  | kd | 0.10 - 0.30 | Acts as a brake against the rate of growth. A higher Kd
  (e.g., 0.2) can overpower Kp during rapid memory spikes, capping the limit to
  ensure a soft landing at the target. |

  Attributes:
    max_memory_limit_gib: The maximum host memory limit in GiB
    target_ratio: The target ratio of host memory limit to use for peak memory
    min_memory_limit_gib: The minimum memory limit in GiB allowed for regulation
    kp: Proportional coefficient
    ki: Integral coefficient
    kd: Derivative coefficient
    integral: Integral term accumulated over time
    prev_error: Error term from the previous step
    integral_windup_limit: Upper and lower bounds for the integral term to
      prevent windup
  """

  max_memory_limit_gib: float
  target_ratio: float = 0.80
  min_memory_limit_gib: float = 10.0
  kp: float = 0.4
  ki: float = 0.05
  kd: float = 0.1

  integral: float = dataclasses.field(init=False)
  prev_error: float = dataclasses.field(init=False)
  _prev_expected_surge_gib: float = dataclasses.field(init=False)
  integral_windup_limit: float = dataclasses.field(init=False)

  def __post_init__(self):
    """Post-initialization validation and field setup."""
    self.integral = 0.0
    self.prev_error = 0.0
    self._prev_expected_surge_gib = 0.0
    self.integral_windup_limit = 50.0

    if self.max_memory_limit_gib <= 0:
      raise ValueError(
          'max_memory_limit_gib must be positive, got'
          f' {self.max_memory_limit_gib}'
      )
    if self.min_memory_limit_gib <= 0:
      raise ValueError(
          'min_memory_limit_gib must be positive, got'
          f' {self.min_memory_limit_gib}'
      )
    if (
        self.min_memory_limit_gib
        >= self.max_memory_limit_gib * self.target_ratio
    ):
      raise ValueError(
          'min_memory_limit_gib must be less than target memory ('
          f'{self.max_memory_limit_gib * self.target_ratio} GiB)'
      )

  def get_next_memory_limit(
      self,
      *,
      current_limit_gib: float,
      peak_memory_usage_gib: float,
      blocking_time_sec: float,  # pylint: disable=unused-argument
      expected_surge_gib: float = 0.0,
      total_memory_gib: float,
  ) -> float:
    """Calculates the next memory limit using PID control and expected surge data.

    The PID controller adjusts the memory limit based on feedback from
    `peak_memory_usage_gib` to guide usage towards
    `effective_host_limit * target_ratio`.

    If `expected_surge_gib` is positive, it signals an anticipated temporary
    increase in memory consumption. The regulator preemptively reduces the
    memory limit by this amount to create headroom and prevent potential OOMs.
    During such a surge, PID integral and error history are not updated, and
    the PID controller is prevented from increasing the limit. When the
    surge passes, `expected_surge_gib` should be reset to 0, and the memory
    limit will be restored.

    Args:
      current_limit_gib: The current memory limit in GiB.
      peak_memory_usage_gib: The peak memory usage observed in GiB since the
        last adjustment.
      blocking_time_sec: The time in seconds that consumers were blocked waiting
        for memory in the last interval. Currently unused.
      expected_surge_gib: The anticipated memory surge in GiB. If 0, no surge
        is expected.
      total_memory_gib: The total system memory capacity in GiB.
    Returns:
      The calculated memory limit for the next interval in GiB.
    """
    effective_host_limit = total_memory_gib

    target_mem_gib = effective_host_limit * self.target_ratio

    error_gib = target_mem_gib - peak_memory_usage_gib
    max_error_gib = effective_host_limit - peak_memory_usage_gib

    # --- STANDARD PID MATH ---
    p_term = self.kp * error_gib

    i_term = self.ki * self.integral
    d_term = self.kd * (error_gib - self.prev_error)

    # Update history ONLY if not in an expected surge.
    # This preserves history for when the surge ends.
    if expected_surge_gib == 0:
      self.integral += error_gib
      self.integral = max(
          -self.integral_windup_limit,
          min(self.integral_windup_limit, self.integral),
      )
      self.prev_error = error_gib

    base_adjustment = p_term + i_term + d_term

    # --- CUSTOM LOGIC ---
    if max_error_gib < 0:
      # Prioritize memory space.
      # Force a reduction if we are over the hard limit, even if
      # the PID controller suggests an increase (e.g. due to recovery).
      # We take the more aggressive of either the PID drop or the raw overflow.
      adjustment = min(max_error_gib, base_adjustment)
    else:
      adjustment = base_adjustment

    # Apply and clamp to hardware limits
    # Bypass PID adjustment if in an active surge to honor the manual drop.
    if expected_surge_gib > 0:
      # If in a surge, we allow the PID to reduce the limit further (negative
      # adjustment) but not increase it. This prevents "double counting"
      # the surge headroom while still allowing throttling.
      adjustment = min(0.0, adjustment)

    # Surge delta handles immediate jump down and up.
    surge_delta = expected_surge_gib - self._prev_expected_surge_gib
    self._prev_expected_surge_gib = expected_surge_gib

    new_limit_gib = current_limit_gib + adjustment - surge_delta
    new_limit_gib = max(
        self.min_memory_limit_gib, min(self.max_memory_limit_gib, new_limit_gib)
    )

    return new_limit_gib

  def update_limit_bytes(self, current_limit_bytes: int) -> int:
    """Calculates the next memory limit in bytes, using profiler inputs."""
    peak_usage_gib = profiler_peak_usage_gib()
    blocking_time_sec = get_prev_blocking_time_sec()
    expected_surge_gib = get_expected_surge_gib()

    total_memory_gib = get_total_memory_gib()
    current_limit_gib = current_limit_bytes / (1024**3)
    next_limit_gib = self.get_next_memory_limit(
        current_limit_gib=current_limit_gib,
        peak_memory_usage_gib=peak_usage_gib,
        blocking_time_sec=blocking_time_sec,
        expected_surge_gib=expected_surge_gib,
        total_memory_gib=total_memory_gib,
    )
    next_limit_bytes = int(next_limit_gib * 1024**3)
    logging.info(
        'MemoryRegulated: Updated device_host_concurrent_bytes to %s'
        ' (peak=%f GiB, total=%f GiB)',
        humanize.naturalsize(next_limit_bytes, binary=True),
        peak_usage_gib,
        total_memory_gib,
    )
    return next_limit_bytes

