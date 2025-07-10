# Copyright 2025 The Orbax Authors.
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

"""Defines policies for when a checkpoint is preserved."""

import dataclasses
import datetime
from typing import Any, Callable, Dict, Protocol, Sequence, Set
from absl import logging
import numpy as np
from orbax.checkpoint._src.checkpoint_managers import policy_checkpoint_info


NestedDict = Dict[str, Any]
PyTree = Any
PolicyCheckpointInfo = policy_checkpoint_info.PolicyCheckpointInfo


@dataclasses.dataclass(kw_only=True)
class PreservationContext:
  """Additional properties for making a save decision."""


class PreservationPolicy(Protocol):
  """A policy that defines when checkpoints should be preserved.
  """

  def should_preserve(
      self,
      checkpoints: Sequence[PolicyCheckpointInfo],
      *,
      context: PreservationContext,
  ) -> Sequence[bool]:
    """Indicates which checkpoints should be preserved.."""
    ...


def _log_preservation_decision(
    policy_name: str,
    checkpoints: Sequence[PolicyCheckpointInfo],
    should_preserve_list: Sequence[bool],
):
  """Logs preservation decisions."""
  if logging.vlog_is_on(1):
    for i, checkpoint in enumerate(checkpoints):
      if should_preserve_list[i]:
        logging.vlog(
            1,
            f" {policy_name}: Preserving checkpoint at step"
            f" {checkpoint.step}).",
        )
      else:
        logging.vlog(
            1,
            f" {policy_name}: Not preserving checkpoint at step"
            f" {checkpoint.step}).",
        )


@dataclasses.dataclass
class PreserveAll(PreservationPolicy):
  """Preserves all checkpoints."""

  def should_preserve(
      self,
      checkpoints: Sequence[PolicyCheckpointInfo],
      *,
      context: PreservationContext,
  ) -> Sequence[bool]:
    result = [True] * len(checkpoints)
    _log_preservation_decision(
        "PreserveAll",
        checkpoints,
        result
    )
    return result


@dataclasses.dataclass
class LatestN(PreservationPolicy):
  """Preserves the last n checkpoints. Preserves all checkpoint if n is None."""

  n: int | None = None

  def should_preserve(
      self,
      checkpoints: Sequence[PolicyCheckpointInfo],
      *,
      context: PreservationContext,
  ) -> Sequence[bool]:
    if self.n is None or len(checkpoints) <= self.n:
      result = [True] * len(checkpoints)
    else:
      result = [False] * (len(checkpoints) - self.n) + [True] * self.n
    _log_preservation_decision(
        f"LatestN (n={self.n})",
        checkpoints,
        result
    )
    return result


@dataclasses.dataclass
class EveryNSeconds(PreservationPolicy):
  """Ensures checkpoints are preserved at least after the time interval."""

  interval_secs: int

  def should_preserve(
      self,
      checkpoints: Sequence[PolicyCheckpointInfo],
      *,
      context: PreservationContext,
  ) -> Sequence[bool]:
    if not checkpoints:
      return []
    last_preserved_checkpoint = checkpoints[0]
    result = [True]
    for info in checkpoints[1:]:
      if info.time - last_preserved_checkpoint.time >= datetime.timedelta(
          seconds=self.interval_secs
      ):
        last_preserved_checkpoint = info
        result.append(True)
      else:
        result.append(False)
    _log_preservation_decision(
        f"EveryNSeconds (interval_secs={self.interval_secs})",
        checkpoints,
        result
    )
    return result


@dataclasses.dataclass
class EveryNSteps(PreservationPolicy):
  """Ensures checkpoints are preserved at least every N steps."""

  interval_steps: int

  def should_preserve(
      self,
      checkpoints: Sequence[PolicyCheckpointInfo],
      *,
      context: PreservationContext,
  ) -> Sequence[bool]:
    if self.interval_steps == 0:
      raise ValueError("interval_steps must not be 0.")
    result = [ckpt.step % self.interval_steps == 0 for ckpt in checkpoints]
    _log_preservation_decision(
        f"EveryNSteps (interval_steps={self.interval_steps})",
        checkpoints,
        result
    )
    return result


@dataclasses.dataclass
class CustomSteps(PreservationPolicy):
  """Preserves checkpoints at the given steps."""

  steps: dataclasses.InitVar[Sequence[int]]
  _steps_set: Set[int] = dataclasses.field(init=False, repr=False)

  def __post_init__(self, steps_init: Sequence[int]):
    """Initializes the internal set of steps after the object is created."""
    self._steps_set = set(steps_init)

  def should_preserve(
      self,
      checkpoints: Sequence[PolicyCheckpointInfo],
      *,
      context: PreservationContext,
  ) -> Sequence[bool]:
    result = [ckpt.step in self._steps_set for ckpt in checkpoints]
    _log_preservation_decision(
        f"CustomSteps (steps={self._steps_set})",
        checkpoints,
        result
    )
    return result


@dataclasses.dataclass
class AnyPreservationPolicy(PreservationPolicy):
  """Applies multiple preservation policies and preserves if any policy preserves."""

  policies: Sequence[PreservationPolicy]

  def should_preserve(
      self,
      checkpoints: Sequence[PolicyCheckpointInfo],
      *,
      context: PreservationContext,
  ) -> Sequence[bool]:
    logging.vlog(
        1, "AnyPreservationPolicy: Preserving if any child policy preserves."
    )
    should_preserve_by_policy = np.asarray([
        policy.should_preserve(checkpoints, context=context)
        for policy in self.policies
    ])
    return np.any(should_preserve_by_policy, axis=0).tolist()


@dataclasses.dataclass(kw_only=True)
class BestN(PreservationPolicy):
  """A policy that preserves the best checkpoints based on a best_fn.

  get_metric_fn:
    A function that accepts a nested tree of metrics and returns a scalar value
    representing the value used for ranking checkpoints.
  reverse:
    If False (default), checkpoints are sorted in ascending order, according to
    the best_fn. If True, checkpoints are sorted in descending order. Same as
    the semantics of built-in sorted() function.
  n:
    The number of checkpoints to preserve. If None, all checkpoints are
    preserved. If 0, no checkpoints are preserved.
  """

  get_metric_fn: Callable[[PyTree], float]
  reverse: bool = False
  n: int | None = None
  keep_checkpoints_without_metrics: bool = True

  def should_preserve(
      self,
      checkpoints: Sequence[PolicyCheckpointInfo],
      *,
      context: PreservationContext,
  ) -> Sequence[bool]:
    if self.n is None or len(checkpoints) <= self.n:
      return [True] * len(checkpoints)
    if self.n == 0:
      return [False] * len(checkpoints)
    indexed_checkpoints_with_metrics = [
        (i, info)
        for (i, info) in [(i, cp) for i, cp in enumerate(checkpoints)]
        if info.metrics is not None
    ]
    indexed_checkpoints_without_metrics = [
        (i, info)
        for (i, info) in [(i, cp) for i, cp in enumerate(checkpoints)]
        if info.metrics is None
    ]
    indexed_checkpoints_with_metrics = sorted(
        indexed_checkpoints_with_metrics,
        key=lambda item: self.get_metric_fn(item[1].metrics),
        reverse=self.reverse,
    )
    preserve_indices = [
        i for i, _ in indexed_checkpoints_with_metrics[-self.n :]
    ]
    if self.keep_checkpoints_without_metrics:
      preserve_indices += [i for i, _ in indexed_checkpoints_without_metrics]
    preserve_indices_set = set(preserve_indices)
    preserve_flags = [
        i in preserve_indices_set for i in range(len(checkpoints))
    ]
    _log_preservation_decision(
        f"BestN (n={self.n})",
        checkpoints,
        preserve_flags
    )
    return preserve_flags
