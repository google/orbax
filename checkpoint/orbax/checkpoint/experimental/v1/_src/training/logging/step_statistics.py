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

"""Step statistics for training.Checkpointer.

Note that the exact fields are subject to change as needed and should not be
relied upon. Please reach out if there are additional fields you are interested
in collecting.
"""

import dataclasses

# Note:`*_end_time` fields are not required here as it can be calculated by
# *_start_time + *_duration_secs


@dataclasses.dataclass
class SaveStepStatistics:
  """Attributes for save step statistics.

  Attributes:
    step: The step number.
    event_type: The event type.
    directory: The directory of the checkpoint.
    reached_preemption: Whether the event reached preemption.
    preemption_received_at: The time when preemption was received.
    synchronous: Whether the event is synchronous.
    wait_for_prev_start_time: The start time of waiting for previous checkpoint.
    wait_for_prev_duration_secs: The duration of waiting for previous
      checkpoint.
    checkpointer_blocking_start_time: The start time of blocking time introduced
      by checkpointer.
    checkpointer_blocking_duration_secs: The duration of blocking time
      introduced by checkpointer.
    get_old_steps_start_time: The start time of getting old steps.
    get_old_steps_duration_secs: The duration of getting old steps.
    checkpoint_manager_blocking_start_time: The start time of checkpoint manager
      blocking section.
    checkpoint_manager_blocking_duration_secs: The duration of checkpoint
      manager blocking section.
  """

  step: int | None = None
  event_type: str | None = "save"
  directory: str | None = None
  reached_preemption: bool | None = False
  preemption_received_at: float | None = None
  synchronous: bool | None = False
  wait_for_prev_start_time: float | None = None
  wait_for_prev_duration_secs: float | None = None
  checkpointer_blocking_start_time: float | None = None
  checkpointer_blocking_duration_secs: float | None = None
  get_old_steps_start_time: float | None = None
  get_old_steps_duration_secs: float | None = None
  checkpoint_manager_blocking_start_time: float | None = None
  checkpoint_manager_blocking_duration_secs: float | None = None


@dataclasses.dataclass
class LoadStepStatistics:
  """Attributes for loading step statistics.

  Attributes:
    step: The step number.
    event_type: The event type.
    directory: The directory of the checkpoint.
    checkpointer_start_time: The start time of loading the checkpoint, while
      using the checkpointer.
    checkpointer_duration_secs: The total duration for loading the checkpoint,
      while using the checkpointer.
  """

  step: int | None = None
  event_type: str | None = "load"
  directory: str | None = None
  checkpointer_start_time: float | None = None
  checkpointer_duration_secs: float | None = None
