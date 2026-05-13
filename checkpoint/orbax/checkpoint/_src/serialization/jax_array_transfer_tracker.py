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

"""Tracks in-flight D2H transfers using prefix reference counting."""

import collections
import threading
from typing import Any

from absl import logging


def _to_string_tuple(key_tuple: tuple[Any, ...] | None) -> tuple[str, ...]:
  """Converts a tuple of JAX keys or strings to a tuple of raw strings."""
  if key_tuple is None:
    return ()

  def _to_str(key: Any) -> str:
    if isinstance(key, str):
      return key
    if hasattr(key, 'key'):
      return str(key.key)
    if hasattr(key, 'idx'):
      return str(key.idx)
    return str(key)

  return tuple(_to_str(k) for k in key_tuple)


class TransferTracker:
  """Tracks in-flight D2H transfers using prefix reference counting.

  O(1) lookup, O(tuple_length) update.
  """

  def __init__(self):
    self.in_flight_counts = {}
    self.lock = threading.Lock()
    self.condition = threading.Condition(self.lock)

  def __deepcopy__(self, memo):
    """Support deepcopying the TransferTracker (constructs new locks)."""
    cls = self.__class__
    result = cls.__new__(cls)
    memo[id(self)] = result
    result.in_flight_counts = dict(self.in_flight_counts)
    result.lock = threading.Lock()
    result.condition = threading.Condition(result.lock)
    return result

  def __getstate__(self):
    """Support pickling TransferTracker (serializes counts)."""
    return {'in_flight_counts': dict(self.in_flight_counts)}

  def __setstate__(self, state):
    """Support unpickling TransferTracker (reconstructs locks)."""
    self.in_flight_counts = state['in_flight_counts']
    self.lock = threading.Lock()
    self.condition = threading.Condition(self.lock)

  def register_batch(self, keys: list[tuple[Any, ...] | None]):
    """Registers a batch of keys for transfer tracking.

    Args:
      keys: A list of keys to register for transfer tracking.
    """
    # Compute prefix counts
    local_counts = collections.defaultdict(int)
    for key in keys:
      str_key = _to_string_tuple(key)
      # Generate all prefixes: e.g., (A,), (A, B), (A, B, C)
      for i in range(1, len(str_key) + 1):
        local_counts[str_key[:i]] += 1

    with self.lock:
      for prefix, count in local_counts.items():
        current = self.in_flight_counts.get(prefix, 0)
        self.in_flight_counts[prefix] = current + count

    logging.info('Registered batch of %d keys', len(keys))

  def finish_transfer(self, key_tuple: tuple[Any, ...] | None):
    """Finishes a transfer for a single key.

    Args:
      key_tuple: The key tuple of the finished transfer.
    """
    str_key = _to_string_tuple(key_tuple)

    with self.lock:
      for i in range(1, len(str_key) + 1):
        prefix = str_key[:i]
        if prefix not in self.in_flight_counts:
          continue
        new_count = self.in_flight_counts[prefix] - 1

        if new_count <= 0:
          logging.info('Transfer for prefix %s is done', prefix)
          if prefix in self.in_flight_counts:
            del self.in_flight_counts[prefix]
        else:
          self.in_flight_counts[prefix] = new_count

      self.condition.notify_all()

  def wait_for_transfer(self, prefix_tuple: tuple[Any, ...] | None):
    """Waits until all transfers under a specific prefix are complete.

    Args:
      prefix_tuple: The prefix tuple of the transfer to wait for.
    """
    str_prefix = _to_string_tuple(prefix_tuple)

    with self.condition:
      while str_prefix in self.in_flight_counts:
        self.condition.wait()
