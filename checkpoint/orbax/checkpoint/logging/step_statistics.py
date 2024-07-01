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
    start_time: The start time of the event.
    end_time: The end time of the event.
    reached_preemption: Whether the event reached preemption.
    preemption_received_at: The time when preemption was received.
    wait_for_prev_start_time: The start time of waiting for previous checkpoint.
    wait_for_prev_end_time: The end time of waiting for previous checkpoint.
    checkpoint_start_time: The start time of checkpointing.
    checkpoint_end_time: The end time of checkpointing.
    get_old_steps_start_time: The start time of getting old steps.
    get_old_steps_end_time: The end time of getting old steps.
    synchronous: Whether the event is synchronous.
    persistent_storage: Whether the event is for persistent storage.
    broadcast_start_time: The start time of broadcasting(Restore).
    broadcast_end_time: The end time of broadcasting(Restore).
    is_restoring_slice: Whether the event is for restoring a slice.
    restore_start_time: The start time of restoring.
    restore_end_time: The end time of restoring.
  """

  step: Optional[int] = None
  event_type: Optional[str] = None
  start_time: Optional[float] = None
  end_time: Optional[float] = None
  reached_preemption: Optional[bool] = False
  preemption_received_at: Optional[float] = None
  wait_for_prev_start_time: Optional[float] = None
  wait_for_prev_end_time: Optional[float] = None
  checkpoint_start_time: Optional[float] = None
  checkpoint_end_time: Optional[float] = None
  get_old_steps_start_time: Optional[float] = None
  get_old_steps_end_time: Optional[float] = None
  synchronous: Optional[bool] = False
  persistent_storage: Optional[bool] = True
  broadcast_start_time: Optional[float] = None
  broadcast_end_time: Optional[float] = None
  is_restoring_slice: Optional[bool] = False
  restore_start_time: Optional[float] = None
  restore_end_time: Optional[float] = None
