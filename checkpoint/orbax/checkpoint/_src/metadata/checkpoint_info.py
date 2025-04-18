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

"""Step level checkpoint metadata.

It is different from `path.step.Metadata` and `metadata.StepMetadata`.

TODO(niketkb): Figure out how to merge this with `path.step.Metadata`,
`metadata.StepMetadata` and `save_decision_policy_lib.StepInfo`.
"""

from __future__ import annotations

import dataclasses
import datetime
import threading
from typing import Any, Callable, Generic, Sequence, TypeVar

from absl import logging
import jax
from orbax.checkpoint._src import threading as threading_lib

PyTree = Any

_T = TypeVar('_T')


@dataclasses.dataclass
class CheckpointInfo:
  """Metadata about a checkpoint."""

  step: int
  time: datetime.datetime
  metrics: PyTree | None

  def __post_init__(self):
    # Users may provide step as a jax.Array.
    if isinstance(self.step, jax.Array):
      self.step = int(self.step)

  def __str__(self) -> str:
    return f'Checkpoint[step={self.step} | time={self.time}]'

  def __eq__(self, other: CheckpointInfo) -> bool:
    return self.step == other.step

  def __hash__(self) -> int:
    return self.step


@dataclasses.dataclass
class _LazyField(Generic[_T]):
  """A field that is lazily initialized."""

  initializer: Callable[[], _T]
  is_initialized: bool = dataclasses.field(default=False)
  _value: _T = dataclasses.field(init=False)
  _lock: threading.RLock = dataclasses.field(default_factory=threading.RLock)

  @property
  def value(self) -> _T:
    """Returns the value of the field."""
    with self._lock:
      if not self.is_initialized:
        self._value = self.initializer()
        self.is_initialized = True
      return self._value

  @value.setter
  def value(self, val: _T):
    """Sets the value of the field."""
    with self._lock:
      self._value = val
      self.is_initialized = True


class LazyCheckpointInfo(CheckpointInfo):
  """Metadata about a checkpoint, but some fields are lazily initialized."""

  def __init__(
      self,
      *,
      step: int,
      time_initializer: Callable[[], datetime.datetime],
      metrics_initializer: Callable[[], PyTree | None],
  ):
    self._time = _LazyField[datetime.datetime](initializer=time_initializer)
    self._metrics = _LazyField[PyTree | None](initializer=metrics_initializer)
    super().__init__(
        step=step,
        time=datetime.datetime.min,  # dummy not-None required value.
        metrics=None,
    )
    # Reset because super().__init__ sets it to True.
    self._time.is_initialized = False
    self._metrics.is_initialized = False

  @property
  def time(self) -> datetime.datetime:
    """Returns time of the checkpoint.

    The value is lazily loaded from `time_provider` if not already set. If the
    value is set, then it is returned.
    """
    return self._time.value

  @time.setter
  def time(self, value: datetime.datetime):
    """Sets time of the checkpoint."""
    self._time.value = value

  @property
  def metrics(self) -> PyTree | None:
    """Returns metrics of the checkpoint.

    The value is lazily loaded from `metrics_provider` if not already set. If
    the value is set, then it is returned.
    """
    return self._metrics.value

  @metrics.setter
  def metrics(self, value: PyTree | None):
    """Sets metrics of the checkpoint."""
    self._metrics.value = value

  def __str__(self) -> str:
    return f'LazyCheckpoint[step={self.step}]'


class CheckpointInfos:
  """Thread-safe list of `CheckpointInfo`.

  It does not gurantee thread-safety for individual CheckpointInfo.

  See this doc for container thread-safety:
  https://docs.python.org/3/faq/library.html#what-kinds-of-global-value-mutation-are-thread-safe
  """

  def __init__(
      self,
      checkpoint_infos: Sequence[CheckpointInfo] | None = None,
      timeout_sec=5.0,
  ):
    """Initializes thread-safe list of `CheckpointInfo`."""
    self._lock = threading_lib.TimeoutRLock(timeout_sec)
    with self._lock:
      self._checkpoint_infos: list[CheckpointInfo] = list(
          checkpoint_infos or []
      )

  def set(self, checkpoint_infos: Sequence[CheckpointInfo]) -> CheckpointInfos:
    """Sets `checkpoint_infos` to `self` and returns it."""
    with self._lock:
      self._checkpoint_infos = list(checkpoint_infos)
      logging.vlog(
          1,
          'CheckpointInfos.set: steps=%s',
          [info.step for info in self._checkpoint_infos],
      )
    return self

  # Support indexing.
  def __getitem__(self, index):
    return self._checkpoint_infos[index]

  def __iter__(self):
    with self._lock:
      for info in self._checkpoint_infos:
        yield info

  def __len__(self):
    return len(self._checkpoint_infos)

  def latest(self) -> CheckpointInfo | None:
    """Returns the latest CheckpointInfo if any."""
    with self._lock:
      return self._checkpoint_infos[-1] if self._checkpoint_infos else None

  def empty(self) -> bool:
    """Returns true if the list of checkpoint infos is empty."""
    return not self._checkpoint_infos

  def size(self) -> int:
    """Returns the number of CheckpointInfo."""
    return len(self._checkpoint_infos)

  def delete_if(self, select_fn: Callable[[CheckpointInfo], bool]) -> None:
    """Deletes all CheckpointInfo that match `select_fn`."""
    with self._lock:
      retained_infos = []
      deleted_steps = []
      for info in self._checkpoint_infos:
        if select_fn(info):
          deleted_steps.append(info.step)
        else:
          retained_infos.append(info)
      self.set(retained_infos)
      logging.vlog(
          1, 'CheckpointInfos.delete_if: deleted steps=%s', deleted_steps
      )

  def append(self, checkpoint_info: CheckpointInfo) -> None:
    """Appends a CheckpointInfo."""
    self._checkpoint_infos.append(checkpoint_info)

  def __repr__(self):
    with self._lock:
      steps = [info.step for info in self._checkpoint_infos]
      return f'CheckpointInfos(size={self.size()}, steps={steps})'
