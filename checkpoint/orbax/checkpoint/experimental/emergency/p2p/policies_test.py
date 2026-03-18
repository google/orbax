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

from unittest import mock
from absl.testing import absltest
from orbax.checkpoint.checkpoint_managers import save_decision_policy as save_decision_policy_lib
from orbax.checkpoint.experimental.emergency.p2p import policies


class PoliciesTest(absltest.TestCase):

  def test_offset_fixed_interval_policy(self):
    policy = policies.OffsetFixedIntervalPolicy(interval=10, offset=0)
    context = mock.Mock(spec=save_decision_policy_lib.DecisionContext)

    step_0 = mock.Mock(
        spec=save_decision_policy_lib.PolicyCheckpointInfo, step=0
    )
    self.assertTrue(policy.should_save(step_0, [], context=context))

    step_5 = mock.Mock(
        spec=save_decision_policy_lib.PolicyCheckpointInfo, step=5
    )
    self.assertFalse(policy.should_save(step_5, [], context=context))

    step_10 = mock.Mock(
        spec=save_decision_policy_lib.PolicyCheckpointInfo, step=10
    )
    self.assertTrue(policy.should_save(step_10, [], context=context))

  def test_offset_fixed_interval_policy_with_offset(self):
    policy = policies.OffsetFixedIntervalPolicy(interval=50, offset=1)
    context = mock.Mock(spec=save_decision_policy_lib.DecisionContext)

    step_0 = mock.Mock(
        spec=save_decision_policy_lib.PolicyCheckpointInfo, step=0
    )
    self.assertFalse(policy.should_save(step_0, [], context=context))

    step_1 = mock.Mock(
        spec=save_decision_policy_lib.PolicyCheckpointInfo, step=1
    )
    self.assertTrue(policy.should_save(step_1, [], context=context))

    step_50 = mock.Mock(
        spec=save_decision_policy_lib.PolicyCheckpointInfo, step=50
    )
    self.assertFalse(policy.should_save(step_50, [], context=context))

    step_51 = mock.Mock(
        spec=save_decision_policy_lib.PolicyCheckpointInfo, step=51
    )
    self.assertTrue(policy.should_save(step_51, [], context=context))

    step_101 = mock.Mock(
        spec=save_decision_policy_lib.PolicyCheckpointInfo, step=101
    )
    self.assertTrue(policy.should_save(step_101, [], context=context))


if __name__ == '__main__':
  absltest.main()
