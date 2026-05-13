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
from absl.testing import parameterized
from orbax.checkpoint._src.serialization import jax_array_transfer_tracker


class StopLoopError(Exception):
  pass


class DictKey:

  def __init__(self, key: str):
    self.key = key

  def __eq__(self, other):
    return isinstance(other, DictKey) and self.key == other.key

  def __hash__(self):
    return hash(self.key)

  def __repr__(self):
    return f"DictKey(key={self.key!r})"


class JaxArrayTransferTrackerTest(parameterized.TestCase):

  def test_finish_transfer_twice_no_error(self):
    tracker = jax_array_transfer_tracker.TransferTracker()

    tracker.register_batch([("key1",)])
    # Finish the transfer the first time
    tracker.finish_transfer(("key1",))
    # Finish the transfer the second time, should not raise any error
    tracker.finish_transfer(("key1",))

  def test_prefix_matching(self):
    tracker = jax_array_transfer_tracker.TransferTracker()

    tracker.register_batch([("key1", "key2")])
    # Finish sub-key
    tracker.finish_transfer(("key1", "key2"))
    # This should complete the prefix as well
    self.assertNotIn(("key1",), tracker.in_flight_counts)

  def test_wait_for_transfer(self):
    tracker = jax_array_transfer_tracker.TransferTracker()
    tracker.register_batch([("key1", "key2"), ("key1", "key3")])
    tracker.finish_transfer(("key1", "key2"))

    with self.subTest("test_prefix_completed"):
      with mock.patch.object(tracker.condition, "wait") as mock_wait:
        tracker.wait_for_transfer(("key1", "key2"))
        mock_wait.assert_not_called()
      self.assertNotIn(("key1", "key2"), tracker.in_flight_counts)

    with self.subTest("test_prefix_not_completed_child"):
      with mock.patch.object(tracker.condition, "wait") as mock_wait:
        mock_wait.side_effect = StopLoopError
        with self.assertRaises(StopLoopError):
          tracker.wait_for_transfer(("key1", "key3"))
        mock_wait.assert_called_once()

    with self.subTest("test_prefix_not_completed_parent"):
      with mock.patch.object(tracker.condition, "wait") as mock_wait:
        mock_wait.side_effect = StopLoopError
        with self.assertRaises(StopLoopError):
          tracker.wait_for_transfer(("key1",))
        mock_wait.assert_called_once()
      self.assertIn(("key1",), tracker.in_flight_counts)

    with self.subTest("test_not_tracked_prefix"):
      tracker.finish_transfer(("key1", "key4"))
      with mock.patch.object(tracker.condition, "wait") as mock_wait:
        tracker.wait_for_transfer(("key1", "key4"))
        mock_wait.assert_not_called()
      self.assertNotIn(("key1", "key4"), tracker.in_flight_counts)

  def test_nested_dict_keys(self):
    dk = DictKey
    tracker = jax_array_transfer_tracker.TransferTracker()

    key1 = (dk("a"), dk("b"), dk("c"), dk("f"))
    key2 = (dk("a"), dk("b"), dk("c"), dk("g"))
    key3 = (dk("a"), dk("b"), dk("cd"), dk("h"))
    key4 = (dk("a"), dk("b"), dk("ce"), dk("i"))

    tracker.register_batch([key1, key2, key3, key4])

    # Check prefix counts
    prefix_primary = ("a",)
    prefix_secondary = ("a", "b")
    prefix_tertiary = ("a", "b", "c")

    self.assertEqual(tracker.in_flight_counts[prefix_primary], 4)
    self.assertEqual(tracker.in_flight_counts[prefix_secondary], 4)
    self.assertEqual(tracker.in_flight_counts[prefix_tertiary], 2)

    # Finish key1
    tracker.finish_transfer(key1)
    self.assertEqual(tracker.in_flight_counts[prefix_primary], 3)
    self.assertEqual(tracker.in_flight_counts[prefix_secondary], 3)
    self.assertEqual(tracker.in_flight_counts[prefix_tertiary], 1)

    # Finish key2
    tracker.finish_transfer(key2)
    self.assertEqual(tracker.in_flight_counts[prefix_primary], 2)
    self.assertEqual(tracker.in_flight_counts[prefix_secondary], 2)
    self.assertNotIn(prefix_tertiary, tracker.in_flight_counts)

    # Finish key3 and key4
    tracker.finish_transfer(key3)
    tracker.finish_transfer(key4)
    self.assertNotIn(prefix_primary, tracker.in_flight_counts)

  def test_none_keypath(self):
    tracker = jax_array_transfer_tracker.TransferTracker()
    # Check registering a None keypath, should not raise TypeError or any error
    tracker.register_batch([None])
    self.assertEmpty(tracker.in_flight_counts)

    # Check finishing None keypath, should not raise TypeError or any error
    tracker.finish_transfer(None)
    self.assertEmpty(tracker.in_flight_counts)


if __name__ == "__main__":
  absltest.main()
