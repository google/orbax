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


def make_multi_device_array(shape, partitioned, pinned_host=False):
  """Creates a partially- or fully-replicated array."""
  num_devices = len(jax.devices())
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
  if pinned_host:
    sharding = sharding.with_memory_kind('pinned_host')

  x = jax.random.normal(jax.random.PRNGKey(0), shape)
  data = jax.device_put(x, sharding)

  return data, num_partitions, num_replicas


class ReplicaSlicesTest(parameterized.TestCase):

  @parameterized.product(partitioned=[False, True])
  def test_get_replica_slices_single_replica(self, partitioned):
    if jax.device_count() < 4:
      self.skipTest('Not enough devices to test.')
    arr, num_partitions, num_replicas = make_multi_device_array(
        (64, 64),
        partitioned=partitioned,
    )

    # Using an addressable replica_id yields that replica.
    for replica_id in range(num_replicas):
      rslices = replica_slices.get_replica_slices(
          arr,
          replica_id=replica_id,
          use_replica_parallel=False,
          min_slice_bytes_for_replica_parallel=None,
          max_replicas_for_replica_parallel=None,
      )
      self.assertLen(rslices.replica_slices, num_partitions)

    # Omitting replica_id yields _some_ replica.
    rslices = replica_slices.get_replica_slices(
        arr,
        replica_id=None,
        use_replica_parallel=False,
        min_slice_bytes_for_replica_parallel=None,
        max_replicas_for_replica_parallel=None,
    )
    self.assertLen(rslices.replica_slices, num_partitions)

    # Using an unaddressable replica_id yields nothing.
    rslices = replica_slices.get_replica_slices(
        arr,
        replica_id=-1,
        use_replica_parallel=False,
        min_slice_bytes_for_replica_parallel=None,
        max_replicas_for_replica_parallel=None,
    ).replica_slices
    self.assertEmpty(rslices)

  @parameterized.parameters([
      ((64, 64), 0),
      ((13, 64), 1),
      ((13, 11), None),
  ])
  def test_get_replica_slices_replica_parallel(self, shape, expected_axis):
    if len(jax.devices()) < 4:
      self.skipTest('Test requires multiple devices.')
    arr, _, num_replicas = make_multi_device_array(shape, partitioned=False)

    rslices = replica_slices.get_replica_slices(
        arr, replica_id=0, use_replica_parallel=True,
        min_slice_bytes_for_replica_parallel=None,
        max_replicas_for_replica_parallel=None,
    ).replica_slices
    if expected_axis is None:
      # Replica-parallel expected to fail. Fall back to a single replica owning
      # the entire shard.
      self.assertLen(rslices, 1)
      self.assertIsNone(rslices[0].slice_args)
    else:
      # Replica-parallel expected to succeed. Every replica owns some data.
      # We're running on a single host, so all replicas' shards are addressable.
      self.assertLen(rslices, num_replicas)
      for rslice in rslices:
        self.assertTrue(rslice.slice_args)
        self.assertEqual(rslice.slice_args.axis, expected_axis)

  @parameterized.product(
      shape=[2 * 768, 2 * 1024,],
      partitioned=[False, True],
  )
  def test_get_replica_slices_pinned_host_passes(self, shape, partitioned):
    if len(jax.devices()) < 4:
      self.skipTest('Test requires multiple devices.')
    if jax.devices()[0].platform == "cpu":
      self.skipTest('Test requires devices that support pinned host memory.')
    arr, _, _ = make_multi_device_array(
        shape, partitioned=partitioned, pinned_host=True
    )
    rslices = replica_slices.get_replica_slices(
        arr, replica_id=0, use_replica_parallel=True,
        min_slice_bytes_for_replica_parallel=None,
        max_replicas_for_replica_parallel=None,
    ).replica_slices
    for rslice in rslices:
      _ = np.array(rslice.data())  # check this does not hang


  @parameterized.parameters([
    ((64, 64), False, 4, 0),
    ((64, 64), False, 2, 0),
    ((64, 64), True, 2, 0),
    ((13, 64), False, 4, 1),
    ((13, 64), False, 2, 1),
  ])
  def test_get_replica_slices_above_max_replicas_successful(
    self, shape, partitioned, max_num_replicas, expected_axis
  ):
    # If we don't have at least 8 devices, then the test's max_num_replicas 
    # might become larger than the num_replicas produced by 
    # make_multi_device_array.
    if len(jax.devices()) < 8:
      self.skipTest('Test requires >= 8 devices.')

    arr, num_partitions, num_replicas = make_multi_device_array(
        shape,
        partitioned=partitioned,
    )

    assert max_num_replicas <= num_replicas

    rslices = replica_slices.get_replica_slices(
        arr, replica_id=0, use_replica_parallel=True,
        min_slice_bytes_for_replica_parallel=None,
        max_replicas_for_replica_parallel=max_num_replicas,
    ).replica_slices

    # Replica-parallel expected to succeed.
    self.assertLen(rslices, num_partitions * max_num_replicas)

    for rslice in rslices:
      self.assertTrue(rslice.slice_args)
      self.assertEqual(rslice.slice_args.axis, expected_axis)


  @parameterized.product(
    shape=[(64, 64), (32, 16), (8, 8)],
    partitioned=[False, True]
  )
  def test_get_replica_slices_above_max_replicas_unsuccessful(
    self, shape, partitioned
  ):
    arr, num_partitions, num_replicas = make_multi_device_array(
        shape,
        partitioned=partitioned,
    )

    # make_multi_device_array guarantees that num_replicas is a power of
    # 2, so max_num_replicas computed below will either be 0 or odd. In
    # either case, we expect replica parallel saving to fail.
    max_num_replicas = (num_replicas // 2) - 1
    assert max_num_replicas < num_replicas

    rslices = replica_slices.get_replica_slices(
        arr, replica_id=0, use_replica_parallel=True,
        min_slice_bytes_for_replica_parallel=None,
        max_replicas_for_replica_parallel=max_num_replicas,
    ).replica_slices

    # Replica-parallel expected to fail.
    self.assertLen(rslices, num_partitions * 1)
    self.assertIsNone(rslices[0].slice_args)
  

  @parameterized.parameters([
    ((64, 64), False, 0),
    ((32, 8), True, 0),
    ((13, 64), False, 1),
  ])
  def test_get_replica_slices_large_max_replicas_successful(
    self, shape, partitioned, expected_axis
  ):
    # If we don't have at least 4 devices, then num_replicas might become
    # equal to 1, breaking the test. 
    if len(jax.devices()) < 4:
      self.skipTest('Test requires >= 4 devices.')

    arr, num_partitions, num_replicas = make_multi_device_array(
        shape,
        partitioned=partitioned,
    )

    max_num_replicas = 2 * num_replicas
    assert max_num_replicas > num_replicas

    rslices = replica_slices.get_replica_slices(
        arr, replica_id=0, use_replica_parallel=True,
        min_slice_bytes_for_replica_parallel=None,
        max_replicas_for_replica_parallel=max_num_replicas,
    ).replica_slices

    # Replica-parallel expected to succeed.
    self.assertLen(rslices, num_partitions * num_replicas)
    for rslice in rslices:
      self.assertTrue(rslice.slice_args)
      self.assertEqual(rslice.slice_args.axis, expected_axis)


  @parameterized.parameters([
    ((64, 64), False, 4, 0, 0),
    ((64, 64), False, 2, 0, 0),
    ((64, 64), True, 2, 0, 0),
    ((13, 64), False, 4, 1, 0),
    ((13, 64), False, 2, 1, 0),
    ((64, 64), False, 2, 0, 1),
    ((13, 64), False, 2, 1, 1),
  ])
  def test_get_replica_slices_above_min_bytes(
      self, 
      shape,
      partitioned,
      max_num_replicas,
      expected_axis,
      min_bytes_decrement,
  ):
    # If we don't have at least 8 devices, then the test's max_num_replicas 
    # might become larger than the num_replicas produced by 
    # make_multi_device_array.
    if len(jax.devices()) < 8:
      self.skipTest('Test requires >= 8 devices.')
    
    arr, num_partitions, num_replicas = make_multi_device_array(
        shape,
        partitioned=partitioned,
    )

    assert max_num_replicas <= num_replicas

    expected_bytes_per_slice = arr.addressable_shards[0].data.nbytes // max_num_replicas
    min_bytes_per_slice = expected_bytes_per_slice - min_bytes_decrement
    assert min_bytes_per_slice <= expected_bytes_per_slice

    rslices = replica_slices.get_replica_slices(
        arr, replica_id=0, use_replica_parallel=True,
        min_slice_bytes_for_replica_parallel=min_bytes_per_slice,
        max_replicas_for_replica_parallel=max_num_replicas,
    ).replica_slices

    # Replica-parallel expected to succeed.
    self.assertLen(rslices, num_partitions * max_num_replicas)
    for rslice in rslices:
      self.assertTrue(rslice.slice_args)
      self.assertEqual(rslice.slice_args.axis, expected_axis)
      self.assertEqual(rslice.data().nbytes, expected_bytes_per_slice)


  @parameterized.parameters([
    ((64, 64), False, 4, 1),
    ((64, 64), False, 2, 1),
    ((64, 64), True, 2, 1),
    ((13, 64), False, 4, 1),
    ((13, 64), False, 2, 1),
    ((64, 64), True, 2, 1000),
    ((13, 64), False, 2, 1000),
  ])
  def test_get_replica_slices_below_min_bytes(
    self,
    shape,
    partitioned,
    max_num_replicas,
    min_bytes_increment,
  ):
    # If we don't have at least 8 devices, then the test's max_num_replicas 
    # might become larger than the num_replicas produced by 
    # make_multi_device_array.
    if len(jax.devices()) < 8:
      self.skipTest('Test requires >= 8 devices.')
    
    arr, num_partitions, num_replicas = make_multi_device_array(
        shape,
        partitioned=partitioned,
    )

    assert max_num_replicas <= num_replicas

    expected_bytes_per_slice = arr.addressable_shards[0].data.nbytes // max_num_replicas
    min_bytes_per_slice = expected_bytes_per_slice + min_bytes_increment
    assert min_bytes_per_slice > expected_bytes_per_slice

    rslices = replica_slices.get_replica_slices(
        arr, replica_id=0, use_replica_parallel=True,
        min_slice_bytes_for_replica_parallel=min_bytes_per_slice,
        max_replicas_for_replica_parallel=max_num_replicas,
    ).replica_slices

    # Replica-parallel expected to fail.
    self.assertLen(rslices, 1 * num_partitions)
    self.assertIsNone(rslices[0].slice_args)


  @parameterized.product(
      partitioned=[False, True],
      use_replica_parallel=[False, True],
  )
  def test_transfer(self, partitioned, use_replica_parallel):
    # If we don't have at least 4 devices, then num_replicas might become
    # equal to 1, breaking the test.
    if jax.device_count() < 4:
      self.skipTest('Test requires >= 4 devices.')
    
    arr, num_partitions, num_replicas = make_multi_device_array(
        (64, 64),
        partitioned=partitioned,
    )

    rslices = replica_slices.transfer_arrays_to_host(
        [arr],
        replica_id=0,
        use_replica_parallel=use_replica_parallel,
        min_slice_bytes_for_replica_parallel=None,
        max_replicas_for_replica_parallel=None,
    )[0]

    # Replica slices cover every element of the original array exactly once, and
    # combining all the replica slices is equivalent to the original array.
    combined_rslices = np.zeros(arr.shape) * np.nan
    for rslice in rslices.replica_slices:
      self.assertTrue(rslice.is_on_host)
      self.assertTrue(np.all(np.isnan(combined_rslices[rslice.index])))
      combined_rslices[rslice.index] = rslice.data()
    self.assertFalse(np.any(np.isnan(arr)))
    self.assertFalse(np.any(np.isnan(combined_rslices)))
    np.testing.assert_array_equal(combined_rslices, arr)

    if use_replica_parallel:
      # With replica-parallel we transfer each of the `num_partitions` shards
      # as `num_replicas` slices.
      self.assertLen(rslices.replica_slices, num_partitions * num_replicas)
    else:
      # With single-replica we transfer a single slice for each shard.
      self.assertLen(rslices.replica_slices, num_partitions)


if __name__ == '__main__':
  absltest.main()
