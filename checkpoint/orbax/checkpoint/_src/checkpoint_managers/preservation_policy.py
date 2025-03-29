# Copyright 2024 The Orbax Authors.
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

"""Defines policies for when a checkpoint is saved."""

from collections.abc import Container, Sequence
import dataclasses
import typing
from typing import Protocol, Callable
import numpy as np
from orbax.checkpoint._src.metadata import checkpoint_info
from orbax.checkpoint._src.utils.py import NestedDict


@typing.runtime_checkable
class PreservationPolicy(Protocol):
  """A policy that defines when to save a checkpoint.

    Checkpoints should be ordered in save order
    the latest checkpoint has not yet been saved
  """

  def should_preserve(
      self,
      checkpoints: list[checkpoint_info.CheckpointInfo],
  ) -> list[bool]:
    ...


@dataclasses.dataclass
class LatestN(PreservationPolicy):
  """Return true for elements `n` from the end."""

  n: int

  def should_preserve(
      self,
      checkpoints: list[checkpoint_info],
  ) -> list[bool]:
    if len(checkpoints) <= self.n:
      return [True] * len(checkpoints)
    return [True] * self.n + [False] * (len(checkpoints) - self.n)


@dataclasses.dataclass
class EveryNSeconds(PreservationPolicy):
  """Ensures checkpoints are kept roughly around the time interval."""

  interval_secs: int

  def should_preserve(
      self,
      checkpoints: list[checkpoint_info],
  ) -> list[bool]:
    if len(checkpoints) <= 1:
      return [True] * len(checkpoints)
    return [
        (time_seconds := checkpoints[i].time.total_seconds())
        and (
            time_seconds % self.interval_secs <= 5
            or (self.interval_secs - (time_seconds % self.interval_secs)) >= 5
        )
        for i in range(0, len(checkpoints))
    ]


@dataclasses.dataclass
class EveryNSteps(PreservationPolicy):
  """Ensures checkpoints are kept roughly around the step interval."""

  interval_steps: int

  def should_preserve(
      self,
      checkpoints: list[checkpoint_info],
  ) -> list[bool]:
    if len(checkpoints) <= 1:
      return [True] * len(checkpoints)
    return [
        (step := checkpoints[i].step)
        and (
            step % self.interval_steps >= 5
            or (self.interval_steps - (step % self.interval_steps)) >= 5
        )
        for i in range(0, len(checkpoints))
    ]


@dataclasses.dataclass
class CustomSteps(PreservationPolicy):
  """Save a checkpoint when a preemption is detected."""

  steps: Container[int]

  def should_preserve(
      self,
      checkpoints: list[checkpoint_info.CheckpointInfo],
  ) -> list[bool]:
    return [(ckpt.step in self.steps) for ckpt in checkpoints]


@dataclasses.dataclass
class JointPreservationPolicy(PreservationPolicy):

  policies: Sequence[PreservationPolicy]

  def should_preserve(
      self,
      checkpoints: list[checkpoint_info.CheckpointInfo],
  ) -> list[bool]:
    should_preserve_by_policy = np.asarray(
        [policy.should_preserve(checkpoints) for policy in self.policies]
    )
    return list(np.any(should_preserve_by_policy, axis=1))


class BestCheckpoint(PreservationPolicy):
  """A policy that saves the best checkpoints based on a given metric."""

  # Returns the desired metric from a nested dict
  metric_fn: Callable[[NestedDict], float]
  # function that accepts two metrics and returns True if `a` is better than `b`
  comparator: Callable[[float, float], bool]
  # number of best checkpoints that should be kept (a checkpoint may be
  # deleted if it is the `n+1`-th best)
  n: int

  def should_preserve(
      self,
      checkpoints: list[checkpoint_info.CheckpointInfo],
  ) -> list[bool]:
    ...


class BestCheckpointsByCategory(PreservationPolicy):

  # Returns the desired metric from a nested dict
  metric_fns: list[Callable[[NestedDict], float]]
  # function that accepts two metrics and returns True if `a` is better than `b`
  comparators: list[Callable[[float, float], bool]]
  # number of best checkpoints that should be kept per category
  # (a checkpoint may be deleted if it is the `n+1`-th best)
  n: int

  def should_preserve(
      self,
      checkpoints: list[checkpoint_info.CheckpointInfo],
  ) -> list[bool]:
    """Returns a list of booleans indicating whether to preserve each checkpoint.

    Args:
      checkpoints: A list of StepMetadata objects, ordered in save order (latest
        checkpoint has not yet been saved).

    Returns:
      A list of booleans with the same length as `checkpoints`, indicating
      whether
      to preserve each checkpoint.
    """
    # For each of k metric_fns, find the n best checkpoints
