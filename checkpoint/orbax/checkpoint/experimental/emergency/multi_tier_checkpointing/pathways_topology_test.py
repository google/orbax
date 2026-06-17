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

"""Tests for Pathways MTC topology helpers."""

from __future__ import annotations

import dataclasses
from unittest import mock

from absl.testing import absltest
import numpy as np
from orbax.checkpoint.experimental.emergency.multi_tier_checkpointing import (
    pathways_topology,
)


@dataclasses.dataclass(frozen=True)
class _FakeDevice:
  id: int
  virtual_task_index: int | None = None
  slice_index: int | None = None


@dataclasses.dataclass(frozen=True)
class _FakeShard:
  data: np.ndarray


@dataclasses.dataclass(frozen=True)
class _FakeRankArray:
  addressable_shards: tuple[_FakeShard, ...]
  is_fully_addressable: bool = False


class PathwaysTopologyTest(absltest.TestCase):

  def test_from_devices_orders_workers_slice_major(self):
    devices = [
        _FakeDevice(id=74, virtual_task_index=1, slice_index=1),
        _FakeDevice(id=72, virtual_task_index=0, slice_index=1),
        _FakeDevice(id=2, virtual_task_index=1, slice_index=0),
        _FakeDevice(id=0, virtual_task_index=0, slice_index=0),
        _FakeDevice(id=1, virtual_task_index=0, slice_index=0),
    ]

    topology = pathways_topology.Topology.from_devices(devices)  # pytype: disable=wrong-arg-types

    self.assertEqual(
        topology.distributed_to_device_ids, ((0, 1), (2,), (72,), (74,))
    )
    self.assertEqual([worker.rank for worker in topology.workers], [0, 1, 2, 3])
    self.assertEqual(
        [worker.key for worker in topology.workers],
        [(0, 0), (1, 0), (0, 1), (1, 1)],
    )

  def test_peer_ranks_by_worker_rank(self):
    devices = [
        _FakeDevice(id=0, virtual_task_index=0, slice_index=0),
        _FakeDevice(id=10, virtual_task_index=1, slice_index=0),
        _FakeDevice(id=72, virtual_task_index=0, slice_index=1),
        _FakeDevice(id=82, virtual_task_index=1, slice_index=1),
    ]

    peers = pathways_topology.Topology.from_devices(  # pytype: disable=wrong-arg-types
        devices
    ).peer_ranks_by_worker_rank(
        2
    )

    self.assertEqual(peers, ((2,), (3,), (0,), (1,)))

  def test_peer_ranks_by_worker_rank_matches_task_ids(self):
    devices = [
        _FakeDevice(id=0, virtual_task_index=0, slice_index=0),
        _FakeDevice(id=10, virtual_task_index=10, slice_index=0),
        _FakeDevice(id=72, virtual_task_index=0, slice_index=1),
        _FakeDevice(id=82, virtual_task_index=10, slice_index=1),
    ]

    peers = pathways_topology.Topology.from_devices(  # pytype: disable=wrong-arg-types
        devices
    ).peer_ranks_by_worker_rank(
        2
    )

    self.assertEqual(peers, ((2,), (3,), (0,), (1,)))

  def test_topology_rejects_mismatched_task_sets_across_slices(self):
    devices = [
        _FakeDevice(id=0, virtual_task_index=0, slice_index=0),
        _FakeDevice(id=10, virtual_task_index=10, slice_index=0),
        _FakeDevice(id=72, virtual_task_index=0, slice_index=1),
        _FakeDevice(id=83, virtual_task_index=11, slice_index=1),
    ]

    topology = pathways_topology.Topology.from_devices(devices)  # pytype: disable=wrong-arg-types

    with self.assertRaisesRegex(ValueError, 'identical task sets'):
      topology.validate_num_slices(2)

  def test_remap_distributed_device_ids(self):
    source_devices = [
        _FakeDevice(id=10, virtual_task_index=1, slice_index=0),
        _FakeDevice(id=0, virtual_task_index=0, slice_index=0),
        _FakeDevice(id=1, virtual_task_index=0, slice_index=0),
        _FakeDevice(id=72, virtual_task_index=0, slice_index=1),
    ]
    target_devices = [
        _FakeDevice(id=110),
        _FakeDevice(id=100),
        _FakeDevice(id=101),
        _FakeDevice(id=172),
    ]
    topology = pathways_topology.Topology.from_devices(source_devices)  # pytype: disable=wrong-arg-types

    remapped = topology.remap_distributed_device_ids(
        source_devices, target_devices  # pytype: disable=wrong-arg-types
    )

    self.assertEqual(remapped, [[100, 101], [110], [172]])

  def test_topology_rejects_non_divisible_num_slices(self):
    devices = [
        _FakeDevice(id=0, virtual_task_index=0, slice_index=0),
        _FakeDevice(id=1, virtual_task_index=1, slice_index=0),
        _FakeDevice(id=2, virtual_task_index=2, slice_index=0),
    ]

    topology = pathways_topology.Topology.from_devices(devices)  # pytype: disable=wrong-arg-types

    with self.assertRaisesRegex(ValueError, 'num_workers must be divisible'):
      topology.validate_num_slices(2)

  def test_worker_cpu_devices_selects_representatives(self):
    devices = [
        _FakeDevice(id=10, virtual_task_index=1, slice_index=0),
        _FakeDevice(id=0, virtual_task_index=0, slice_index=0),
        _FakeDevice(id=1, virtual_task_index=0, slice_index=0),
        _FakeDevice(id=73, virtual_task_index=0, slice_index=1),
        _FakeDevice(id=72, virtual_task_index=0, slice_index=1),
    ]
    cpu_devices = (mock.Mock(id=100), mock.Mock(id=101), mock.Mock(id=102))
    topology = pathways_topology.Topology.from_devices(devices)  # pytype: disable=wrong-arg-types

    with mock.patch.object(
        pathways_topology.colocated_transport,
        'unique_colocated_cpu_devices',
        return_value=cpu_devices,
    ) as mock_unique_cpu_devices:
      result = topology.worker_cpu_devices()

    self.assertEqual(result, cpu_devices)
    mock_unique_cpu_devices.assert_called_once_with((
        devices[1],  # worker (task=0, slice=0), first in input order.
        devices[0],  # worker (task=1, slice=0).
        devices[3],  # worker (task=0, slice=1), first in input order.
    ))

  def test_worker_rank_array_uses_host_rank_values(self):
    worker_cpu_devices = (
        mock.Mock(id=100),
        mock.Mock(id=101),
        mock.Mock(id=102),
    )

    with mock.patch.object(
        pathways_topology.jax.sharding, 'Mesh', return_value='mesh'
    ), mock.patch.object(
        pathways_topology.jax.sharding, 'NamedSharding', return_value='sharding'
    ), mock.patch.object(
        pathways_topology.jax, 'make_array_from_callback'
    ) as mock_make_array:
      pathways_topology.worker_rank_array(worker_cpu_devices)  # pytype: disable=wrong-arg-types

    shape, sharding, callback = mock_make_array.call_args.args
    self.assertEqual(shape, (3,))
    self.assertEqual(sharding, 'sharding')
    self.assertEqual(mock_make_array.call_args.kwargs['dtype'], np.int32)
    np.testing.assert_array_equal(callback(None), np.asarray([0, 1, 2]))
    np.testing.assert_array_equal(callback(np.s_[0:3]), np.asarray([0, 1, 2]))
    np.testing.assert_array_equal(callback(np.s_[1:2]), np.asarray([1]))

  def test_worker_rank_from_array_rejects_multiple_values(self):
    with self.assertRaisesRegex(ValueError, 'exactly one logical worker rank'):
      pathways_topology.worker_rank_from_array(np.asarray([1, 2]))

  def test_worker_rank_from_array_uses_single_addressable_shard(self):
    rank_array = _FakeRankArray(
        addressable_shards=(_FakeShard(np.asarray([7], dtype=np.int32)),)
    )

    self.assertEqual(pathways_topology.worker_rank_from_array(rank_array), 7)  # pytype: disable=wrong-arg-types

  def test_worker_rank_prefers_single_fully_addressable_shard(self):
    rank_array = _FakeRankArray(
        addressable_shards=(_FakeShard(np.asarray([7], dtype=np.int32)),),
        is_fully_addressable=True,
    )

    self.assertEqual(pathways_topology.worker_rank_from_array(rank_array), 7)  # pytype: disable=wrong-arg-types

  def test_worker_rank_from_array_rejects_no_addressable_shards(self):
    rank_array = _FakeRankArray(addressable_shards=())

    with self.assertRaisesRegex(ValueError, 'one addressable'):
      pathways_topology.worker_rank_from_array(rank_array)  # pytype: disable=wrong-arg-types

  def test_worker_rank_from_array_rejects_multiple_addressable_shards(self):
    rank_array = _FakeRankArray(
        addressable_shards=(
            _FakeShard(np.asarray([1], dtype=np.int32)),
            _FakeShard(np.asarray([2], dtype=np.int32)),
        )
    )

    with self.assertRaisesRegex(ValueError, 'one addressable'):
      pathways_topology.worker_rank_from_array(rank_array)  # pytype: disable=wrong-arg-types

  def test_worker_rank_from_array_rejects_non_scalar_addressable_shard(self):
    rank_array = _FakeRankArray(
        addressable_shards=(_FakeShard(np.asarray([1, 2], dtype=np.int32)),)
    )

    with self.assertRaisesRegex(ValueError, 'exactly one logical worker rank'):
      pathways_topology.worker_rank_from_array(rank_array)  # pytype: disable=wrong-arg-types


if __name__ == '__main__':
  absltest.main()
