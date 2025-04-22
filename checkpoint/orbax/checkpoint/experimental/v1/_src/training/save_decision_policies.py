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

"""Save decision policies for Checkpointer."""

from orbax.checkpoint._src.checkpoint_managers import save_decision_policy

SaveDecisionPolicy = save_decision_policy.SaveDecisionPolicy

ContinuousCheckpointingPolicy = (
    save_decision_policy.ContinuousCheckpointingPolicy
)
FixedIntervalPolicy = save_decision_policy.FixedIntervalPolicy
InitialSavePolicy = save_decision_policy.InitialSavePolicy
PreemptionCheckpointingPolicy = (
    save_decision_policy.PreemptionCheckpointingPolicy
)
SpecificStepsPolicy = save_decision_policy.SpecificStepsPolicy
AnySavePolicy = save_decision_policy.AnySavePolicy

DecisionContext = save_decision_policy.DecisionContext
