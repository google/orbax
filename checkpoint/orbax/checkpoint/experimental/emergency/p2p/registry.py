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

"""Domain objects for P2P peer registry management."""

import collections
from collections.abc import Iterator

from orbax.checkpoint.experimental.emergency.p2p import protocol


# Type Aliases for clarity
Step = int
ProcessIndex = int


class PeerRegistry:
  """A strict registry that allows only ONE peer per (Step, ProcessIndex)."""

  def __init__(self):
    # Mapping: Step -> ProcessIndex -> PeerInfo
    self._store: dict[Step, dict[ProcessIndex, protocol.PeerDiscoveryInfo]] = (
        collections.defaultdict(dict)
    )

  def clear(self):
    self._store.clear()

  def register_peer(self, peer: protocol.PeerDiscoveryInfo):
    """Registers a peer.

    Existing entries for the same process index will be overwritten. The caller
    is responsible for validating and avoiding unintended overwrites.

    Args:
      peer: The PeerDiscoveryInfo object to register.
    """
    for step in peer.steps:
      # Simple assignment. If a 'zombie' node sends data later, it overwrites.
      self._store[step][peer.process_index] = peer

  def get_peer(
      self, step: Step, process_index: ProcessIndex
  ) -> protocol.PeerDiscoveryInfo | None:
    """Returns the single peer holding this shard, or None."""
    if step not in self._store:
      return None
    return self._store[step].get(process_index)

  def get_shard_map(
      self, step: Step
  ) -> dict[ProcessIndex, protocol.PeerDiscoveryInfo]:
    """Returns the full map of available shards for a given step."""
    return self._store.get(step, {})

  def iter_steps(self) -> Iterator[Step]:
    return iter(self._store.keys())

  def has_step(self, step: Step) -> bool:
    return step in self._store
