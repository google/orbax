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

"""Save decision policies for P2P checkpointing."""

import dataclasses
from typing import Sequence

from orbax.checkpoint.checkpoint_managers import save_decision_policy as save_decision_policy_lib


@dataclasses.dataclass
class OffsetFixedIntervalPolicy(save_decision_policy_lib.SaveDecisionPolicy):
  """Checkpoint at a fixed interval with a given offset.

  Evaluates to True whenever (step - offset) is an exact multiple of the
  configured `interval` and step is >= offset.
  """

  interval: int
  offset: int = 0

  def should_save(
      self,
      step: save_decision_policy_lib.PolicyCheckpointInfo,
      previous_steps: Sequence[save_decision_policy_lib.PolicyCheckpointInfo],
      *,
      context: save_decision_policy_lib.DecisionContext,
  ) -> bool:
    del previous_steps
    del context
    if step.step < self.offset:
      return False
    return (step.step - self.offset) % self.interval == 0
