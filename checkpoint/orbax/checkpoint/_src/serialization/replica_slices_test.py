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


def _normalize_index(index, shape):
  """Returns `index` as a hashable tuple of (start, stop) pairs."""
  return tuple(
      s.indices(size)[:2] for s, size in zip(index, shape, strict=True)
  )


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
      )
      self.assertLen(rslices.replica_slices, num_partitions)

    # Omitting replica_id yields _some_ replica.
    rslices = replica_slices.get_replica_slices(
        arr,
        replica_id=None,
        use_replica_parallel=False,
    )
    self.assertLen(rslices.replica_slices, num_partitions)

    # Using an unaddressable replica_id yields nothing.
    rslices = replica_slices.get_replica_slices(
        arr,
        replica_id=-1,
        use_replica_parallel=False,
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
        arr, replica_id=0, use_replica_parallel=True
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
    arr, _, _ = make_multi_device_array(
        shape, partitioned=partitioned, pinned_host=True
    )
    rslices = replica_slices.get_replica_slices(
        arr, replica_id=0, use_replica_parallel=True
    ).replica_slices
    for rslice in rslices:
      _ = np.array(rslice.data())  # check this does not hang

  @parameterized.parameters([False, True])
  def test_get_replica_slices_above_max_replicas_successful(
      self, halve_num_replicas
  ):
    # If we don't have at least 4 devices, then for
    # max_num_replicas_factor = 0.5, max_num_replicas becomes = 1,
    # causing replica parallel saving to fail.
    if len(jax.devices()) < 4:
      self.skipTest('Test requires >= 4 devices.')

    arr, _, num_replicas = make_multi_device_array(
        (64, 64),
        partitioned=False,
    )

    max_num_replicas = num_replicas
    if halve_num_replicas:
      max_num_replicas = max_num_replicas // 2

    rslices = replica_slices.get_replica_slices(
        arr,
        replica_id=0,
        use_replica_parallel=True,
        min_slice_bytes_for_replica_parallel=None,
        max_replicas_for_replica_parallel=max_num_replicas,
    ).replica_slices

    # Replica-parallel expected to succeed.
    self.assertLen(rslices, max_num_replicas)

    for rslice in rslices:
      self.assertTrue(rslice.slice_args)

  @parameterized.product(
      shape=[(64, 64), (32, 16), (8, 8)], partitioned=[False, True]
  )
  def test_get_replica_slices_above_max_replicas_unsuccessful(
      self, shape, partitioned
  ):
    # If we don't have at least 4 devices, then for
    # max_num_replicas_factor = 0.5, max_num_replicas becomes = 1,
    # causing replica parallel saving to fail.
    if len(jax.devices()) < 4:
      self.skipTest('Test requires >= 4 devices.')
    arr, num_partitions, num_replicas = make_multi_device_array(
        shape,
        partitioned=partitioned,
    )

    # make_multi_device_array guarantees that num_replicas is a power of
    # 2, so max_num_replicas computed below will either be 0 or odd. In
    # either case, we expect replica parallel saving to fail.
    max_num_replicas = (num_replicas // 2) - 1

    rslices = replica_slices.get_replica_slices(
        arr,
        replica_id=0,
        use_replica_parallel=True,
        min_slice_bytes_for_replica_parallel=None,
        max_replicas_for_replica_parallel=max_num_replicas,
    ).replica_slices

    # Replica-parallel expected to fail.
    self.assertLen(rslices, num_partitions * 1)
    for rslice in rslices:
      self.assertIsNone(rslice.slice_args)

  def test_get_replica_slices_large_max_replicas_successful(self):
    # If we don't have at least 4 devices, then for
    # max_num_replicas_factor = 0.5, max_num_replicas becomes = 1,
    # causing replica parallel saving to fail.
    if len(jax.devices()) < 4:
      self.skipTest('Test requires >= 4 devices.')

    arr, _, num_replicas = make_multi_device_array(
        (64, 64),
        partitioned=False,
    )

    max_num_replicas = 2 * num_replicas

    rslices = replica_slices.get_replica_slices(
        arr,
        replica_id=0,
        use_replica_parallel=True,
        min_slice_bytes_for_replica_parallel=None,
        max_replicas_for_replica_parallel=max_num_replicas,
    ).replica_slices

    # Replica-parallel expected to succeed.
    self.assertLen(rslices, num_replicas)
    for rslice in rslices:
      self.assertTrue(rslice.slice_args)

  @parameterized.parameters([0, 1])
  def test_get_replica_slices_above_min_bytes(self, min_bytes_decrement):
    # If we don't have at least 4 devices, then for
    # max_num_replicas_factor = 0.5, max_num_replicas becomes = 1,
    # causing replica parallel saving to fail.
    if len(jax.devices()) < 4:
      self.skipTest('Test requires >= 4 devices.')

    arr, _, num_replicas = make_multi_device_array(
        (64, 64),
        partitioned=False,
    )

    max_num_replicas = 2
    assert max_num_replicas <= num_replicas

    expected_bytes_per_slice = (
        arr.addressable_shards[0].data.nbytes // max_num_replicas
    )
    min_bytes_per_slice = expected_bytes_per_slice - min_bytes_decrement
    assert min_bytes_per_slice <= expected_bytes_per_slice

    rslices = replica_slices.get_replica_slices(
        arr,
        replica_id=0,
        use_replica_parallel=True,
        min_slice_bytes_for_replica_parallel=min_bytes_per_slice,
        max_replicas_for_replica_parallel=max_num_replicas,
    ).replica_slices

    # Replica-parallel expected to succeed.
    self.assertLen(rslices, max_num_replicas)
    for rslice in rslices:
      self.assertTrue(rslice.slice_args)
      self.assertEqual(rslice.data().nbytes, expected_bytes_per_slice)

  def test_get_replica_slices_below_min_bytes(self):
    # If we don't have at least 4 devices, then for
    # max_num_replicas_factor = 0.5, max_num_replicas becomes = 1,
    # causing replica parallel saving to fail.
    if len(jax.devices()) < 4:
      self.skipTest('Test requires >= 4 devices.')

    arr, _, num_replicas = make_multi_device_array(
        (64, 64),
        partitioned=False,
    )

    max_num_replicas = 2
    assert max_num_replicas <= num_replicas

    expected_bytes_per_slice = (
        arr.addressable_shards[0].data.nbytes // max_num_replicas
    )
    min_bytes_per_slice = expected_bytes_per_slice + 1
    assert min_bytes_per_slice > expected_bytes_per_slice

    rslices = replica_slices.get_replica_slices(
        arr,
        replica_id=0,
        use_replica_parallel=True,
        min_slice_bytes_for_replica_parallel=min_bytes_per_slice,
        max_replicas_for_replica_parallel=max_num_replicas,
    ).replica_slices

    # Replica-parallel expected to fail.
    self.assertLen(rslices, 1)
    for rslice in rslices:
      self.assertIsNone(rslice.slice_args)

  @parameterized.product(
      partitioned=[False, True],
      use_replica_parallel=[False, True],
  )
  def test_transfer(self, partitioned, use_replica_parallel):
    if jax.device_count() < 4:
      self.skipTest('Not enough devices to test.')
    arr, num_partitions, num_replicas = make_multi_device_array(
        (64, 64),
        partitioned=partitioned,
    )

    rslices = replica_slices.transfer_arrays_to_host(
        [arr],
        replica_id=0,
        use_replica_parallel=use_replica_parallel,
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

  @parameterized.parameters([
      ((64, 64), False),
      ((64, 64), True),
      ((13, 64), False),
  ])
  def test_get_replica_parallel_sharding(self, shape, partitioned):
    if jax.device_count() < 4:
      self.skipTest('Test requires multiple devices.')
    arr, _, _ = make_multi_device_array(shape, partitioned=partitioned)

    sharding = replica_slices.get_replica_parallel_sharding(arr, replica_id=0)

    self.assertIsNotNone(sharding)
    # Each replica holds the slice that it saves with replica-parallel saving.
    self.assertEqual(
        sharding.shard_shape(arr.shape),
        replica_slices.get_replica_slices(
            arr, replica_id=0, use_replica_parallel=True
        ).local_shape,
    )
    # The sharding has no replicas, and every device's shard is a slice of its
    # current shard.
    current_indices = arr.sharding.devices_indices_map(arr.shape)
    new_indices = sharding.devices_indices_map(arr.shape)
    self.assertLen(
        {_normalize_index(index, arr.shape) for index in new_indices.values()},
        jax.device_count(),
    )
    for device, index in new_indices.items():
      for (start, stop), (current_start, current_stop) in zip(
          _normalize_index(index, arr.shape),
          _normalize_index(current_indices[device], arr.shape),
      ):
        self.assertBetween(start, current_start, current_stop)
        self.assertBetween(stop, current_start, current_stop)

  def test_get_replica_parallel_sharding_unsupported(self):
    if jax.device_count() < 4:
      self.skipTest('Test requires multiple devices.')
    arr, _, num_replicas = make_multi_device_array((64, 64), partitioned=False)
    unsupported = {
        'local_checkpointing': (arr, dict(replica_id=None)),
        'other_replica_id': (arr, dict(replica_id=1)),
        'below_min_slice_bytes': (
            arr,
            dict(replica_id=0, min_slice_bytes_for_replica_parallel=arr.nbytes),
        ),
        'subset_of_replicas': (
            arr,
            dict(
                replica_id=0,
                max_replicas_for_replica_parallel=num_replicas // 2,
            ),
        ),
        'no_divisible_axis': (
            make_multi_device_array((13, 11), partitioned=False)[0],
            dict(replica_id=0),
        ),
        'pinned_host': (
            make_multi_device_array(
                (64, 1024), partitioned=False, pinned_host=True
            )[0],
            dict(replica_id=0),
        ),
        'not_replicated': (
            jax.device_put(
                np.arange(64 * jax.device_count()),
                jax.sharding.NamedSharding(
                    jax.sharding.Mesh(jax.devices(), ('x',)),
                    jax.sharding.PartitionSpec('x'),
                ),
            ),
            dict(replica_id=0),
        ),
    }
    for name, (unsupported_arr, kwargs) in unsupported.items():
      with self.subTest(name):
        self.assertIsNone(
            replica_slices.get_replica_parallel_sharding(
                unsupported_arr, **kwargs
            )
        )

  @parameterized.parameters([False, True])
  def test_slice_replicas_on_device(self, partitioned):
    if jax.device_count() < 4:
      self.skipTest('Test requires multiple devices.')
    arr, num_partitions, num_replicas = make_multi_device_array(
        (64, 64), partitioned=partitioned
    )
    sharding = replica_slices.get_replica_parallel_sharding(arr, replica_id=0)

    (sliced,) = replica_slices.slice_replicas_on_device([arr], [sharding])

    self.assertEqual(sliced.sharding, sharding)
    np.testing.assert_array_equal(np.asarray(sliced), np.asarray(arr))
    # Each device slices its own shard, without communication between devices.
    compiled_text = (
        replica_slices._get_reshard_fn(sharding)  # pylint: disable=protected-access
        .lower(arr)
        .compile()
        .as_text()
    )
    for collective in (
        'all-gather',
        'all-reduce',
        'all-to-all',
        'collective-permute',
        'reduce-scatter',
    ):
      self.assertNotIn(collective, compiled_text)
    # Saving the sliced array with replica id 0 saves the same slices as saving
    # the original array replica-parallel.
    expected, actual = (
        replica_slices.transfer_arrays_to_host(
            [a], replica_id=0, use_replica_parallel=True
        )[0]
        for a in (arr, sliced)
    )
    self.assertEqual(actual.local_shape, expected.local_shape)
    self.assertLen(actual.replica_slices, num_partitions * num_replicas)
    expected_data = {
        _normalize_index(rslice.index, arr.shape): rslice.data()
        for rslice in expected.replica_slices
    }
    actual_data = {
        _normalize_index(rslice.index, arr.shape): rslice.data()
        for rslice in actual.replica_slices
    }
    self.assertCountEqual(actual_data, expected_data)
    for index, data in actual_data.items():
      np.testing.assert_array_equal(data, expected_data[index])

  def test_slice_replicas_on_device_keeps_arrays_without_sharding(self):
    if jax.device_count() < 4:
      self.skipTest('Test requires multiple devices.')
    arr, _, _ = make_multi_device_array((64, 64), partitioned=False)

    (result,) = replica_slices.slice_replicas_on_device([arr], [None])

    self.assertIs(result, arr)

  def test_nbytes_no_slicing(self):
    rslice = replica_slices.ReplicaSlice(
        index=(slice(None),),
        unsliced_data=np.zeros((2, 3), dtype=np.float32),
        slice_args=None,
    )
    rslices = replica_slices.ReplicaSlices(
        global_shape=(2, 3),
        local_shape=(2, 3),
        sharding=jax.sharding.NamedSharding(
            jax.sharding.Mesh(np.array(jax.devices()), ('x',)),
            jax.sharding.PartitionSpec(),
        ),
        dtype=np.dtype(np.float32),
        is_on_host=True,
        replica_slices=[rslice],
    )
    expected_nbytes = 2 * 3 * np.dtype(np.float32).itemsize
    self.assertEqual(rslices.nbytes, expected_nbytes)

  def test_nbytes_with_slicing(self):
    rslice = replica_slices.ReplicaSlice(
        index=(slice(None),),
        unsliced_data=jax.numpy.zeros((4, 5), dtype=np.int16),
        slice_args=replica_slices.SliceArgs(
            start_index=1, limit_index=3, axis=0
        ),
    )
    rslices = replica_slices.ReplicaSlices(
        global_shape=(4, 5),
        local_shape=(4, 5),
        sharding=jax.sharding.NamedSharding(
            jax.sharding.Mesh(np.array(jax.devices()), ('x',)),
            jax.sharding.PartitionSpec(),
        ),
        dtype=np.dtype(np.int16),
        is_on_host=False,  # Set to False to allow slice_args
        replica_slices=[rslice],
    )
    # Shape of the slice is (2, 5)
    expected_nbytes = 2 * 5 * np.dtype(np.int16).itemsize
    self.assertEqual(rslices.nbytes, expected_nbytes)

  def test_nbytes_multiple_slices(self):
    rslice1 = replica_slices.ReplicaSlice(
        index=(slice(None),),
        unsliced_data=jax.numpy.zeros((4, 5), dtype=np.int16),
        slice_args=replica_slices.SliceArgs(
            start_index=0, limit_index=2, axis=0
        ),
    )
    rslice2 = replica_slices.ReplicaSlice(
        index=(slice(None),),
        unsliced_data=jax.numpy.zeros((4, 5), dtype=np.int16),
        slice_args=replica_slices.SliceArgs(
            start_index=2, limit_index=4, axis=0
        ),
    )
    rslices = replica_slices.ReplicaSlices(
        global_shape=(4, 5),
        local_shape=(4, 5),
        sharding=jax.sharding.NamedSharding(
            jax.sharding.Mesh(np.array(jax.devices()), ('x',)),
            jax.sharding.PartitionSpec(),
        ),
        dtype=np.dtype(np.int16),
        is_on_host=False,  # Set to False to allow slice_args
        replica_slices=[rslice1, rslice2],
    )
    # Shape of each slice is (2, 5)
    expected_nbytes = (2 * 5 * np.dtype(np.int16).itemsize) * 2
    self.assertEqual(rslices.nbytes, expected_nbytes)

  def test_nbytes_mixed_slicing(self):
    rslice1 = replica_slices.ReplicaSlice(
        index=(slice(None),),
        unsliced_data=jax.numpy.zeros((4, 5), dtype=np.int16),  # jax.Array
        slice_args=None,
    )
    rslice2 = replica_slices.ReplicaSlice(
        index=(slice(None),),
        unsliced_data=jax.numpy.zeros((4, 5), dtype=np.int16),  # jax.Array
        slice_args=replica_slices.SliceArgs(
            start_index=0, limit_index=2, axis=1
        ),
    )
    rslices = replica_slices.ReplicaSlices(
        global_shape=(4, 5),
        local_shape=(4, 5),
        sharding=jax.sharding.NamedSharding(
            jax.sharding.Mesh(np.array(jax.devices()), ('x',)),
            jax.sharding.PartitionSpec(),
        ),
        dtype=np.dtype(np.int16),
        is_on_host=False,  # Set to False to allow slice_args
        replica_slices=[rslice1, rslice2],
    )
    # bytes for rslice1: (4 * 5)
    # bytes for rslice2: (4 * 2)
    expected_nbytes = (4 * 5 * np.dtype(np.int16).itemsize) + (
        4 * 2 * np.dtype(np.int16).itemsize
    )
    self.assertEqual(rslices.nbytes, expected_nbytes)

if __name__ == '__main__':
  absltest.main()
