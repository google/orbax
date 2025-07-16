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

import itertools
from absl.testing import absltest
from orbax.checkpoint._src.futures import synchronization


OperationIdGenerator = synchronization.OperationIdGenerator


class OperationIdGeneratorTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    OperationIdGenerator._operation_id_counter = itertools.count()
    OperationIdGenerator._operation_id = next(
        OperationIdGenerator._operation_id_counter
    )

  def test_get_operation_id(self):
    OperationIdGenerator.next_operation_id()
    operation_id_1 = OperationIdGenerator.get_current_operation_id()

    OperationIdGenerator.next_operation_id()
    operation_id_2 = OperationIdGenerator.get_current_operation_id()

    self.assertEqual(operation_id_1, "1")
    self.assertEqual(operation_id_2, "2")


class OpTrackerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    OperationIdGenerator._operation_id_counter = itertools.count()
    OperationIdGenerator._operation_id = next(
        OperationIdGenerator._operation_id_counter
    )

  def test_op_tracker_factory(self):
    tracker_1 = synchronization.OpTrackerFactory.create_tracker("test_tracker")
    tracker_2 = synchronization.OpTrackerFactory.create_tracker("test_tracker")
    self.assertEqual(tracker_1._operation_id, "1")
    self.assertEqual(tracker_2._operation_id, "2")
    self.assertEqual(
        OperationIdGenerator.get_current_operation_id(),
        "2",
    )

  def test_op_tracker_get_in_progress_ids(self):
    tracker = synchronization.OpTrackerFactory.create_tracker("test_tracker")
    self.assertEmpty(tracker.get_in_progress_ids())
    tracker.start()
    self.assertSameElements(tracker.get_in_progress_ids(), [0])
    tracker.complete()
    self.assertEmpty(tracker.get_in_progress_ids())


if __name__ == "__main__":
  absltest.main()
