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

from absl.testing import absltest
from orbax.checkpoint.experimental.emergency.p2p import protocol
from orbax.checkpoint.experimental.emergency.p2p import registry


class PeerRegistryTest(absltest.TestCase):

  def test_register_and_get_peer(self):
    reg = registry.PeerRegistry()
    peer_info = protocol.PeerDiscoveryInfo(
        process_index=0, steps=[1, 2], ip='addr0', port=0
    )
    reg.register_peer(peer_info)

    self.assertEqual(reg.get_peer(1, 0), peer_info)
    self.assertEqual(reg.get_peer(2, 0), peer_info)
    self.assertIsNone(reg.get_peer(3, 0))
    self.assertIsNone(reg.get_peer(1, 1))

  def test_register_overwrite(self):
    reg = registry.PeerRegistry()
    peer_info1 = protocol.PeerDiscoveryInfo(
        process_index=0, steps=[1], ip='addr0', port=0
    )
    reg.register_peer(peer_info1)
    self.assertEqual(reg.get_peer(1, 0), peer_info1)

    peer_info2 = protocol.PeerDiscoveryInfo(
        process_index=0, steps=[1], ip='addr1', port=0
    )
    reg.register_peer(peer_info2)
    self.assertEqual(reg.get_peer(1, 0), peer_info2)

  def test_clear(self):
    reg = registry.PeerRegistry()
    peer_info = protocol.PeerDiscoveryInfo(
        process_index=0, steps=[1], ip='addr0', port=0
    )
    reg.register_peer(peer_info)
    self.assertIsNotNone(reg.get_peer(1, 0))
    reg.clear()
    self.assertIsNone(reg.get_peer(1, 0))

  def test_get_shard_map(self):
    reg = registry.PeerRegistry()
    peer0 = protocol.PeerDiscoveryInfo(
        process_index=0, steps=[1, 2], ip='addr0', port=0
    )
    peer1 = protocol.PeerDiscoveryInfo(
        process_index=1, steps=[1], ip='addr1', port=0
    )
    reg.register_peer(peer0)
    reg.register_peer(peer1)

    shard_map_1 = reg.get_shard_map(1)
    self.assertDictEqual(shard_map_1, {0: peer0, 1: peer1})

    shard_map_2 = reg.get_shard_map(2)
    self.assertDictEqual(shard_map_2, {0: peer0})

    shard_map_3 = reg.get_shard_map(3)
    self.assertDictEqual(shard_map_3, {})

  def test_iter_steps_and_has_step(self):
    reg = registry.PeerRegistry()
    peer0 = protocol.PeerDiscoveryInfo(
        process_index=0, steps=[1, 3], ip='addr0', port=0
    )
    peer1 = protocol.PeerDiscoveryInfo(
        process_index=1, steps=[1, 2], ip='addr1', port=0
    )
    reg.register_peer(peer0)
    reg.register_peer(peer1)

    self.assertTrue(reg.has_step(1))
    self.assertTrue(reg.has_step(2))
    self.assertTrue(reg.has_step(3))
    self.assertFalse(reg.has_step(4))

    steps = list(reg.iter_steps())
    self.assertLen(steps, 3)
    self.assertIn(1, steps)
    self.assertIn(2, steps)
    self.assertIn(3, steps)


if __name__ == '__main__':
  absltest.main()
