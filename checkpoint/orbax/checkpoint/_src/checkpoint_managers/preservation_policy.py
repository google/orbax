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

import dataclasses
import datetime
import typing
from typing import Any, Callable, Dict, Protocol, Sequence
import numpy as np
from orbax.checkpoint import options as options_lib
from orbax.checkpoint._src.metadata import checkpoint_info

NestedDict = Dict[str, Any]
PyTree = Any


@dataclasses.dataclass(kw_only=True)
class PreservationContext:
  """Additional properties for making a save decision."""
  multiprocessing_options: options_lib.MultiprocessingOptions


@typing.runtime_checkable
class PreservationPolicy(Protocol):
  """A policy that defines when to save a checkpoint.

    Checkpoints should be ordered in save order
    the latest checkpoint has not yet been saved
  """

  def should_preserve(
      self,
      checkpoints: Sequence[checkpoint_info.CheckpointInfo],
      *,
      context: PreservationContext,
  ) -> list[bool]:
    ...


@dataclasses.dataclass
class LatestN(PreservationPolicy):
  """Return true for elements `n` from the end."""

  n: int | None = None

  def should_preserve(
      self,
      checkpoints: Sequence[checkpoint_info.CheckpointInfo],
      *,
      context: PreservationContext,
  ) -> list[bool]:
    if self.n is None or len(checkpoints) <= self.n:
      return [True] * len(checkpoints)
    return [False] * (len(checkpoints) - self.n) + [True] * self.n


@dataclasses.dataclass
class EveryNSeconds(PreservationPolicy):
  """Ensures checkpoints are kept roughly around the time interval."""

  interval_secs: datetime.timedelta | None = None
  # TODO(abhisekar): UTC timezone.

  def should_preserve(
      self,
      checkpoints: Sequence[checkpoint_info.CheckpointInfo],
      *,
      context: PreservationContext,
  ) -> list[bool]:
    if not checkpoints:
      return []
    interval_preserved_checkpoints = [checkpoints[0]]
    for info in checkpoints[1:]:
      if info.time.timestamp() >= (
          interval_preserved_checkpoints[-1].time.timestamp()
          + self.interval_secs.total_seconds() if self.interval_secs else 0
      ):
        interval_preserved_checkpoints.append(info)
    return [ckpt in interval_preserved_checkpoints for ckpt in checkpoints]


@dataclasses.dataclass
class EveryNSteps(PreservationPolicy):
  """Ensures checkpoints are kept roughly around the step interval."""

  interval_steps: int | None = None

  def should_preserve(
      self,
      checkpoints: Sequence[checkpoint_info.CheckpointInfo],
      *,
      context: PreservationContext,
  ) -> list[bool]:
    return [ckpt.step % self.interval_steps == 0 for ckpt in checkpoints]


@dataclasses.dataclass
class CustomSteps(PreservationPolicy):
  """Save a checkpoint when a preemption is detected."""

  steps: list[int]

  def should_preserve(
      self,
      checkpoints: Sequence[checkpoint_info.CheckpointInfo],
      *,
      context: PreservationContext,
  ) -> list[bool]:
    return [ckpt.step in self.steps for ckpt in checkpoints]


@dataclasses.dataclass
class JointPreservationPolicy(PreservationPolicy):
  """Applies multiple preservation policies and preserves if any policy preserves."""

  policies: Sequence[PreservationPolicy]

  def should_preserve(
      self,
      checkpoints: Sequence[checkpoint_info.CheckpointInfo],
      *,
      context: PreservationContext,
  ) -> list[bool]:
    should_preserve_by_policy = np.asarray([
        policy.should_preserve(checkpoints, context=context)
        for policy in self.policies
    ])
    return np.any(should_preserve_by_policy, axis=0).tolist()


@dataclasses.dataclass(kw_only=True)
class BestN(PreservationPolicy):
  """A policy that saves the best checkpoints based on a given metric."""

  best_fn: Callable[[PyTree], float] | None = None
  reverse: bool | None = False
  n: int | None = None
  keep_checkpoints_without_metrics: bool | None = True

  def should_preserve(
      self,
      checkpoints: Sequence[checkpoint_info.CheckpointInfo],
      *,
      context: PreservationContext,
  ) -> list[bool]:
    if self.n is None or len(checkpoints) <= self.n:
      return [True] * len(checkpoints)
    if self.n == 0:
      return [False] * len(checkpoints)
    all_checkpoints = [(i, cp) for i, cp in enumerate(checkpoints)]
    indexed_checkpoints_without_metrics = [
        (i, info) for (i, info) in all_checkpoints if info.metrics is None
    ]
    indexed_checkpoints_with_metrics = [
        (i, info) for (i, info) in all_checkpoints if info.metrics is not None
    ]
    if self.best_fn is not None:
      indexed_checkpoints_with_metrics = sorted(
          indexed_checkpoints_with_metrics,
          key=lambda item: self.best_fn(item[1].metrics),
          reverse=(self.reverse),
      )
      all_checkpoints = (
          indexed_checkpoints_without_metrics
          + indexed_checkpoints_with_metrics
      )
    else:
      indexed_checkpoints_without_metrics = []
    preserve_indices = set()
    preserve_indices.update(item[0] for item in all_checkpoints[-self.n :])
    if self.keep_checkpoints_without_metrics:
      preserve_indices.update(
          item[0] for item in indexed_checkpoints_without_metrics
      )
    preserve_flags = [i in preserve_indices for i in range(len(checkpoints))]
    return preserve_flags
