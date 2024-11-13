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
from absl.testing import parameterized
import jax
import numpy as np
from orbax.checkpoint._src.serialization import replica_slices


def is_pow_of_two(n):
  while n > 1:
    n, rem = divmod(n, 2)
    if rem == 1:
      return False
  return True


def make_multi_device_array(shape, partitioned):
  """Creates a partially- or fully-replicated array."""
  num_devices = len(jax.devices())
  assert num_devices >= 4
  assert is_pow_of_two(num_devices)
  mesh = jax.sharding.Mesh(
      np.asarray(jax.devices()).reshape((2, num_devices // 2)),
      ('x', 'y'),
  )
  if partitioned:
    # partially-replicated (partitioned dimension 0 along mesh axis x)
    spec = jax.sharding.PartitionSpec('x')
    num_partitions = 2
    num_replicas = num_devices // 2
  else:
    # fully-replicated
    spec = jax.sharding.PartitionSpec()
    num_partitions = 1
    num_replicas = num_devices
  sharding = jax.sharding.NamedSharding(mesh, spec)

  key = jax.random.PRNGKey(0)
  x = jax.random.normal(jax.random.PRNGKey(0), shape)
  data = jax.device_put(x, sharding)

  return data, num_partitions, num_replicas


@parameterized.product(partitioned=[False, True])
class ReplicaSlicesTest(parameterized.TestCase):

  def test_get_replica_slices_single_replica(self, partitioned):
    arr, num_partitions, num_replicas = make_multi_device_array(
        (64, 64),
        partitioned=partitioned,
    )

    # Using an addressable replica_id yields that replica.
    for replica_id in range(num_replicas):
      rslices = replica_slices.get_replica_slices(
          arr,
          replica_id=replica_id
      ).replica_slices
      self.assertEqual(len(rslices), num_partitions)
      for rslice in rslices:
        self.assertEqual(rslice.replica_id, replica_id)

    # Omitting replica_id yields _some_ replica.
    rslices = replica_slices.get_replica_slices(
        arr,
        replica_id=None
    ).replica_slices
    self.assertEqual(len(rslices), num_partitions)
    for rslice in rslices:
      self.assertEqual(rslice.replica_id, rslices[0].replica_id)

    # Using an unaddressable replica_id yields nothing.
    rslices = replica_slices.get_replica_slices(
        arr,
        replica_id=-1,
    ).replica_slices
    self.assertEqual(len(rslices), 0)

  def test_transfer(self, partitioned):
    arr, num_partitions, num_replicas = make_multi_device_array(
        (64, 64),
        partitioned=partitioned,
    )
    replica0_shards = [
        shard
        for shard in arr.addressable_shards
        if shard.replica_id == 0
    ]

    rslices = replica_slices.transfer_arrays_to_host(
        [arr],
        replica_id=0
    )[0].replica_slices
    self.assertEqual(len(rslices), num_partitions)
    self.assertEqual(len(rslices), len(replica0_shards))

    index_start = lambda x: x.index[0].start or 0
    rslices = sorted(rslices, key=index_start)
    replica0_shards = sorted(replica0_shards, key=index_start)

    for rslice, replica0_shard in zip(rslices, replica0_shards):
      self.assertTrue(rslice.is_on_host)
      self.assertIsInstance(rslice.data, np.ndarray)
      self.assertEqual(rslice.index, replica0_shard.index)
      np.testing.assert_array_equal(rslice.data, replica0_shard.data)


if __name__ == '__main__':
  absltest.main()
