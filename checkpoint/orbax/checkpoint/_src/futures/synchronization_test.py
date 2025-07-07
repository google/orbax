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
from unittest import mock
from absl.testing import absltest
from orbax.checkpoint._src.futures import synchronization


OperationIdGenerator = synchronization.OperationIdGenerator
ThreadSaveMultiHostValueHolder = synchronization.ThreadSaveMultiHostValueHolder


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


class ThreadSaveMultiHostValueHolderTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.thread_save_barrier_sync_fn = mock.MagicMock()
    self.value_holder = ThreadSaveMultiHostValueHolder(
        value="initial_value",
        thread_save_barrier_sync_fn=self.thread_save_barrier_sync_fn,
    )

  def test_get_value(self):
    self.assertEqual(self.value_holder.get_value(), "initial_value")

  def test_set_value(self):
    new_value = "new_value"
    self.value_holder.set_value(new_value)
    self.assertEqual(self.value_holder.get_value(), new_value)
    self.assertEqual(self.thread_save_barrier_sync_fn.call_count, 2)


if __name__ == "__main__":
  absltest.main()
