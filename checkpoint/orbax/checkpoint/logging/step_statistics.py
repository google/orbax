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

"""Step statistics."""

import dataclasses
from typing import Optional


@dataclasses.dataclass
class StepStatistics:
  """Attributes.

  Attributes:
    step: The step number.
    event_type: The event type.
    checkpoint_manager_blocking_start_time: The start time of checkpoint
      manager blocking section.
    checkpoint_manager_blocking_end_time: The end time of checkpoint manager
      blocking section.
    directory: The directory of the checkpoint.
    reached_preemption: Whether the event reached preemption.
    preemption_received_at: The time when preemption was received.
    wait_for_prev_start_time: The start time of waiting for previous checkpoint.
    wait_for_prev_end_time: The end time of waiting for previous checkpoint.
    checkpoint_blocking_start_time: The start time of checkpointing.
    checkpoint_blocking_end_time: The end time of checkpointing.
    get_old_steps_start_time: The start time of getting old steps.
    get_old_steps_end_time: The end time of getting old steps.
    synchronous: Whether the event is synchronous.
    wait_for_prev_duration_secs: The duration of waiting for previous
      checkpoint.
    checkpoint_blocking_duration_secs: The duration of checkpoint which is
      blocking.
    get_old_steps_duration_secs: The duration of getting old steps.
    checkpoint_manager_blocking_duration_secs: The duration of checkpoint
      manager blocking section.
  """

  step: Optional[int] = None
  event_type: Optional[str] = None
  directory: Optional[str] = None
  checkpoint_manager_blocking_start_time: Optional[float] = None
  checkpoint_manager_blocking_end_time: Optional[float] = None
  reached_preemption: Optional[bool] = False
  preemption_received_at: Optional[float] = None
  wait_for_prev_start_time: Optional[float] = None
  wait_for_prev_end_time: Optional[float] = None
  checkpoint_blocking_start_time: Optional[float] = None
  checkpoint_blocking_end_time: Optional[float] = None
  get_old_steps_start_time: Optional[float] = None
  get_old_steps_end_time: Optional[float] = None
  synchronous: Optional[bool] = False
  wait_for_prev_duration_secs: Optional[float] = None
  checkpoint_blocking_duration_secs: Optional[float] = None
  get_old_steps_duration_secs: Optional[float] = None
  checkpoint_manager_blocking_duration_secs: Optional[float] = None
