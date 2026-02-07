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
import jax
import numpy as np
from orbax.checkpoint.experimental.emergency.p2p import peer_selector


class MockDevice:

  def __init__(self, process_index):
    self.process_index = process_index

  def __repr__(self):
    return f'MockDevice(pi={self.process_index})'


def get_mock_mesh(replicas=2, processes_per_replica=2, devices_per_process=2):
  p = 0
  devs_list = []
  for _ in range(replicas):
    replica_devs = []
    for _ in range(processes_per_replica):
      for _ in range(devices_per_process):
        replica_devs.append(MockDevice(p))
      p += 1
    devs_list.append(replica_devs)
  devices = np.array(devs_list)
  return mock.Mock(
      spec=jax.sharding.Mesh, devices=devices, shape_tuple=devices.shape
  )


class PeerSelectorTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.mesh = get_mock_mesh()

  @parameterized.named_parameters(
      dict(
          testcase_name='replica_axis_0',
          replica_axis_index=0,
          devices=np.array([
              [MockDevice(0), MockDevice(1)],
              [MockDevice(2), MockDevice(3)],
          ]),
          expected_map={
              0: {'replica': 0, 'relative': 0},
              1: {'replica': 0, 'relative': 1},
              2: {'replica': 1, 'relative': 0},
              3: {'replica': 1, 'relative': 1},
          },
      ),
      dict(
          testcase_name='replica_axis_1',
          replica_axis_index=1,
          devices=np.array([
              [MockDevice(0), MockDevice(2)],
              [MockDevice(1), MockDevice(3)],
          ]),
          expected_map={
              0: {'replica': 0, 'relative': 0},
              1: {'replica': 0, 'relative': 1},
              2: {'replica': 1, 'relative': 0},
              3: {'replica': 1, 'relative': 1},
          },
      ),
  )
  def test_build_topology_map(
      self, replica_axis_index, devices, expected_map
  ):
    mesh = mock.Mock(
        spec=jax.sharding.Mesh, devices=devices, shape_tuple=devices.shape
    )
    selector = peer_selector.PeerSelector(
        mesh, replica_axis_index=replica_axis_index, raw_metadata_list=[]
    )
    self.assertEqual(selector._process_map, expected_map)

  @parameterized.named_parameters(
      dict(
          testcase_name='1rep_4proc_1dev',
          replicas=1,
          processes_per_replica=4,
          devices_per_process=1,
          expected_map={
              0: {'replica': 0, 'relative': 0},
              1: {'replica': 0, 'relative': 1},
              2: {'replica': 0, 'relative': 2},
              3: {'replica': 0, 'relative': 3},
          },
      ),
      dict(
          testcase_name='4rep_1proc_1dev',
          replicas=4,
          processes_per_replica=1,
          devices_per_process=1,
          expected_map={
              0: {'replica': 0, 'relative': 0},
              1: {'replica': 1, 'relative': 0},
              2: {'replica': 2, 'relative': 0},
              3: {'replica': 3, 'relative': 0},
          },
      ),
      dict(
          testcase_name='2rep_1proc_4dev',
          replicas=2,
          processes_per_replica=1,
          devices_per_process=4,
          expected_map={
              0: {'replica': 0, 'relative': 0},
              1: {'replica': 1, 'relative': 0},
          },
      ),
  )
  def test_build_topology_map_various_topologies(
      self, replicas, processes_per_replica, devices_per_process, expected_map
  ):
    mesh = get_mock_mesh(
        replicas=replicas,
        processes_per_replica=processes_per_replica,
        devices_per_process=devices_per_process,
    )
    selector = peer_selector.PeerSelector(
        mesh, replica_axis_index=0, raw_metadata_list=[]
    )
    self.assertEqual(selector._process_map, expected_map)

  def test_get_latest_complete_step(self):
    selector = peer_selector.PeerSelector(
        self.mesh, replica_axis_index=0, raw_metadata_list=[]
    )
    self.assertIsNone(selector.get_latest_complete_step())

    metadata = [
        {'process_index': 0, 'steps': [1], 'ip': 'ip0', 'port': 1234},
        {'process_index': 1, 'steps': [1], 'ip': 'ip1', 'port': 1234},
        {'process_index': 2, 'steps': [1], 'ip': 'ip2', 'port': 1234},
        {'process_index': 3, 'steps': [1], 'ip': 'ip3', 'port': 1234},
    ]
    selector = peer_selector.PeerSelector(
        self.mesh, replica_axis_index=0, raw_metadata_list=metadata
    )

    # For step 1, p0(Rep0,Rel0), p1(Rep0,Rel1), p2(Rep1,Rel0), p3(Rep1,Rel1)
    # have data. All relative_ids {0,1} are present, so step is complete.
    self.assertEqual(selector.get_latest_complete_step(), 1)

    # For step 2, p0(Rep0,Rel0) and p2(Rep1,Rel0) have data.
    # Only relative_id 0 is present, need {0,1} for step to be complete.
    new_metadata = [
        {'process_index': 0, 'steps': [2], 'ip': 'ip0', 'port': 1234},
        {'process_index': 2, 'steps': [2], 'ip': 'ip2', 'port': 1234},
    ]
    selector = peer_selector.PeerSelector(
        self.mesh, replica_axis_index=0, raw_metadata_list=new_metadata
    )
    self.assertIsNone(selector.get_latest_complete_step())

    # If 1 and 3 also have step 2, it becomes complete.
    new_metadata_2 = [
        {'process_index': 0, 'steps': [2], 'ip': 'ip0', 'port': 1234},
        {'process_index': 1, 'steps': [2], 'ip': 'ip1', 'port': 1234},
        {'process_index': 2, 'steps': [2], 'ip': 'ip2', 'port': 1234},
        {'process_index': 3, 'steps': [2], 'ip': 'ip3', 'port': 1234},
    ]
    selector = peer_selector.PeerSelector(
        self.mesh, replica_axis_index=0, raw_metadata_list=new_metadata_2
    )
    self.assertEqual(selector.get_latest_complete_step(), 2)

    # p0(Rep0,Rel0),p2(Rep1,Rel0) have step 1; p1(Rep0,Rel1),p3(Rep1,Rel1) have
    # step 2. Step 1 only has Rel0, Step 2 only has Rel1. Neither are complete.
    new_metadata_3 = [
        {'process_index': 0, 'steps': [1], 'ip': 'ip0', 'port': 1234},
        {'process_index': 1, 'steps': [2], 'ip': 'ip1', 'port': 1234},
        {'process_index': 2, 'steps': [1], 'ip': 'ip2', 'port': 1234},
        {'process_index': 3, 'steps': [2], 'ip': 'ip3', 'port': 1234},
    ]
    selector = peer_selector.PeerSelector(
        self.mesh, replica_axis_index=0, raw_metadata_list=new_metadata_3
    )
    self.assertIsNone(selector.get_latest_complete_step())

    # p0(Rep0,Rel0),p1(Rep0,Rel1) have step 1; p2(Rep1,Rel0),p3(Rep1,Rel1) have
    # step 2. Both steps 1 and 2 are complete. Should return latest complete
    # step: 2.
    new_metadata_4 = [
        {'process_index': 0, 'steps': [1], 'ip': 'ip0', 'port': 1234},
        {'process_index': 1, 'steps': [1], 'ip': 'ip1', 'port': 1234},
        {'process_index': 2, 'steps': [2], 'ip': 'ip2', 'port': 1234},
        {'process_index': 3, 'steps': [2], 'ip': 'ip3', 'port': 1234},
    ]
    selector = peer_selector.PeerSelector(
        self.mesh, replica_axis_index=0, raw_metadata_list=new_metadata_4
    )
    self.assertEqual(selector.get_latest_complete_step(), 2)

    # p0(Rep0,Rel0), p1(Rep0,Rel1), p2(Rep1,Rel0), p3(Rep1,Rel1) have step 1,
    # so step 1 is complete.
    # p0 also has step 2, but step 2 is not complete because only Rel0 is
    # present.
    # Should return latest complete step: 1.
    new_metadata_5 = [
        {'process_index': 0, 'steps': [1, 2], 'ip': 'ip0', 'port': 1234},
        {'process_index': 1, 'steps': [1], 'ip': 'ip1', 'port': 1234},
        {'process_index': 2, 'steps': [1], 'ip': 'ip2', 'port': 1234},
        {'process_index': 3, 'steps': [1], 'ip': 'ip3', 'port': 1234},
    ]
    selector = peer_selector.PeerSelector(
        self.mesh, replica_axis_index=0, raw_metadata_list=new_metadata_5
    )
    self.assertEqual(selector.get_latest_complete_step(), 1)

  def test_get_source_peer_no_step(self):
    selector = peer_selector.PeerSelector(
        self.mesh, replica_axis_index=0, raw_metadata_list=[]
    )
    self.assertIsNone(selector.get_source_peer(1, 0))

  def test_get_source_peer_direct_hit(self):
    metadata = [
        {'process_index': 0, 'steps': [1], 'ip': 'ip0', 'port': 1234},
    ]
    selector = peer_selector.PeerSelector(
        self.mesh, replica_axis_index=0, raw_metadata_list=metadata
    )
    source = selector.get_source_peer(1, 0)
    self.assertIsNotNone(source)
    self.assertEqual(source.process_index, 0)

  def test_get_source_peer_topology_match(self):
    metadata = [
        # p2(Rep1,Rel0) has step 1
        {'process_index': 2, 'steps': [1], 'ip': 'ip2', 'port': 1234},
    ]
    selector = peer_selector.PeerSelector(
        self.mesh, replica_axis_index=0, raw_metadata_list=metadata
    )

    # p0(Rep0,Rel0) needs step 1 data. p2 has it.
    source = selector.get_source_peer(1, 0)
    self.assertIsNotNone(source)
    self.assertEqual(source.process_index, 2)

    # p1(Rep0,Rel1) needs step 1 data. No one has it.
    self.assertIsNone(selector.get_source_peer(1, 1))

  def test_get_source_peer_load_balancing(self):
    # 3 replicas, 2 processes per replica.
    # p0(Rep0,Rel0), p1(Rep0,Rel1), p2(Rep1,Rel0), p3(Rep1,Rel1),
    # p4(Rep2,Rel0), p5(Rep2,Rel1)
    mesh = get_mock_mesh(replicas=3, processes_per_replica=2)

    # See who p0 gets data from if it's missing for step 1.
    metadata_p0_missing = [
        {'process_index': 2, 'steps': [1], 'ip': 'ip2', 'port': 1234},
        {'process_index': 4, 'steps': [1], 'ip': 'ip4', 'port': 1234},
    ]
    selector = peer_selector.PeerSelector(
        mesh, replica_axis_index=0, raw_metadata_list=metadata_p0_missing
    )
    self.assertIsNone(selector._registry.get_peer(1, 0))

    source = selector.get_source_peer(1, 0)  # p0(Rep0,Rel0) needs Rel0 data
    # candidates for p0(Rep0,Rel0) are p2(Rep1,Rel0), p4(Rep2,Rel0)
    # p2 is Rep1, p4 is Rep2.
    # sorted candidates based on replica_id: [p2, p4]
    # target p0 is in Rep0. idx = 0 % 2 = 0.
    # should return p2.
    self.assertIsNotNone(source)
    self.assertEqual(source.process_index, 2)

    # See who p2 gets data from if it's missing for step 1.
    metadata_p2_missing = [
        {'process_index': 0, 'steps': [1], 'ip': 'ip0', 'port': 1234},
        {'process_index': 4, 'steps': [1], 'ip': 'ip4', 'port': 1234},
    ]
    selector = peer_selector.PeerSelector(
        mesh, replica_axis_index=0, raw_metadata_list=metadata_p2_missing
    )
    self.assertIsNone(selector._registry.get_peer(1, 2))

    source = selector.get_source_peer(1, 2)  # p2(Rep1,Rel0) needs Rel0 data
    # candidates for p2(Rep1,Rel0) are p0(Rep0,Rel0), p4(Rep2,Rel0)
    # sorted: [p0, p4]
    # target p2 is in Rep1. idx = 1 % 2 = 1.
    # should return p4
    self.assertIsNotNone(source)
    self.assertEqual(source.process_index, 4)

  def test_visualize_topology(self):
    metadata = [
        {'process_index': 0, 'steps': [1], 'ip': 'ip0.0', 'port': 1234},
        {'process_index': 3, 'steps': [1], 'ip': 'ip3.3', 'port': 1234},
    ]
    selector = peer_selector.PeerSelector(
        self.mesh, replica_axis_index=0, raw_metadata_list=metadata
    )
    output = selector.visualize_topology(1)
    self.assertIn('P2P TOPOLOGY VISUALIZATION (Step: 1)', output)
    self.assertIn('Physical Topology Map', output)
    self.assertIn('Data Availability Matrix', output)
    self.assertIn('Recovery Plan', output)
    self.assertRegex(output, r'0\s*\|\s*0\s*\|\s*0')
    self.assertRegex(output, r'1\s*\|\s*0\s*\|\s*1')
    self.assertRegex(output, r'2\s*\|\s*1\s*\|\s*0')
    self.assertRegex(output, r'3\s*\|\s*1\s*\|\s*1')
    self.assertIn('[OK] P0', output)
    self.assertIn('[--] P1', output)
    self.assertIn('[--] P2', output)
    self.assertIn('[OK] P3', output)
    # p1(Rep0,Rel1) needs Rel1 data, p3(Rep1,Rel1) has it.
    self.assertIn('P1 (Rep 0)', output)
    self.assertIn('P3 (Rep 1) @ ..3', output)
    # p2(Rep1,Rel0) needs Rel0 data, p0(Rep0,Rel0) has it.
    self.assertIn('P2 (Rep 1)', output)
    self.assertIn('P0 (Rep 0) @ ..0', output)


if __name__ == '__main__':
  absltest.main()
