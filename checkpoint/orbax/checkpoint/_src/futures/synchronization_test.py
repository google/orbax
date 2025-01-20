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

from absl.testing import absltest
from orbax.checkpoint._src.futures import synchronization
from orbax.checkpoint._src.multihost import multihost


HandlerAwaitableSignalBarrierKeyGenerator = (
    synchronization.HandlerAwaitableSignalBarrierKeyGenerator
)


class HandlerAwaitableSignalBarrierKeyGeneratorTest(absltest.TestCase):

  def test_get_unique_barrier_key_without_operation_id_raises_error(self):
    step_directory_creation_signal = (
        synchronization.HandlerAwaitableSignal.STEP_DIRECTORY_CREATION
    )
    HandlerAwaitableSignalBarrierKeyGenerator._operation_id = None

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "_operation_id is not initialized. Please call `next_operation_id()`"
        " first.",
    ):
      HandlerAwaitableSignalBarrierKeyGenerator.get_unique_barrier_key(
          step_directory_creation_signal
      )

  def test_get_unique_barrier_key(self):
    step_directory_creation_signal = (
        synchronization.HandlerAwaitableSignal.STEP_DIRECTORY_CREATION
    )
    expected_barrier_key_0 = multihost.unique_barrier_key(
        step_directory_creation_signal.value, suffix="0"
    )
    expected_barrier_key_1 = multihost.unique_barrier_key(
        step_directory_creation_signal.value, suffix="1"
    )

    HandlerAwaitableSignalBarrierKeyGenerator.next_operation_id()
    barrier_key_0 = (
        HandlerAwaitableSignalBarrierKeyGenerator.get_unique_barrier_key(
            step_directory_creation_signal
        )
    )
    HandlerAwaitableSignalBarrierKeyGenerator.next_operation_id()
    barrier_key_1 = (
        HandlerAwaitableSignalBarrierKeyGenerator.get_unique_barrier_key(
            step_directory_creation_signal
        )
    )

    self.assertEqual(barrier_key_0, expected_barrier_key_0)
    self.assertEqual(barrier_key_1, expected_barrier_key_1)


if __name__ == "__main__":
  absltest.main()
