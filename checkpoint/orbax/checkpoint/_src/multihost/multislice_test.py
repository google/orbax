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

import functools
import string
from typing import List, Optional, Tuple, cast
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import jax
import numpy as np
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.multihost import multislice

FLAGS = flags.FLAGS
jax.config.update('jax_enable_x64', True)


class MockDevice:

  def __init__(self, process_index, device_id):
    self.process_index = process_index
    self.id = device_id


def setup_pytree(mesh):
  """Creates a numpy PyTree for testing."""
  dtype = jax.numpy.bfloat16
  bytes_per_type = 2
  leaf_sizes = (64, 512)
  pytree = {
      'c': {
          'a': jax.numpy.array(
              np.arange(leaf_sizes[0]).reshape((leaf_sizes[0] // 8, 8)),
              dtype=dtype,
          ),
          'e': jax.numpy.array(
              np.arange(leaf_sizes[1]).reshape((leaf_sizes[1] // 32, 32)),
              dtype=dtype,
          ),
      },
  }

  mesh_tree = {
      'c': {
          'a': mesh,
          'e': mesh,
      },
  }
  mesh_axes = jax.sharding.PartitionSpec('axis1', 'axis2')
  axes_tree = {
      'c': {
          'a': mesh_axes,
          'e': mesh_axes,
      },
  }

  pytree = jax.tree.map(
      test_utils.create_sharded_array, pytree, mesh_tree, axes_tree
  )
  memory = bytes_per_type * sum(leaf_sizes) // len(jax.devices())
  return pytree, memory


def setup_replica_sharded_arrays(
    arrays: List[jax.Array],
    mesh_shape: Tuple[int, ...],
    is_replica_first: Optional[bool] = True,
):
  """Creates a tuple of sharded arrays for testing."""
  devices = jax.devices()
  devices = np.asarray(devices)

  dim = len(mesh_shape)
  axis_names = list(string.ascii_lowercase)
  mesh = jax.sharding.Mesh(devices.reshape(mesh_shape), axis_names[-dim:])
  if is_replica_first:
    mesh_axes = jax.sharding.PartitionSpec(None, axis_names[-dim + 1 :])
  else:
    mesh_axes = jax.sharding.PartitionSpec(axis_names[-dim:-1], None)

  sharded_arrs = [
      test_utils.create_sharded_array(arr, mesh, mesh_axes) for arr in arrays
  ]
  return sharded_arrs, mesh, mesh_axes


class UtilsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(
        self.create_tempdir(name='checkpointing_test').full_path
    )

  def tearDown(self):
    test_utils.sync_global_processes(
        'SingleReplicaArrayHandlerTest:tests_complete'
    )
    super().tearDown()

  def test_tree_memory_per_device(self):
    mesh = jax.sharding.Mesh(
        np.reshape(jax.devices(), (len(jax.devices()) // 2, 2)),
        ('axis1', 'axis2'),
    )
    tree, expected_tree_memory = setup_pytree(mesh)
    tree_memory = multislice.tree_memory_per_device(tree)
    self.assertEqual(expected_tree_memory, tree_memory)

  def test_get_leaf_memory_per_device(self):
    """Test get_leaf_memory_per_device from utils.py."""
    key = jax.random.PRNGKey(0)
    shape = (100, 100)
    dtype = jax.numpy.bfloat16
    arr = jax.random.uniform(key, shape, dtype=dtype)
    mesh = jax.sharding.Mesh(
        np.reshape(jax.devices(), (len(jax.devices()) // 2, 2)),
        ('axis1', 'axis2'),
    )
    sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec('axis1', 'axis2')
    )
    arr = jax.device_put(arr, sharding)
    expected_size = arr.size // jax.device_count() * arr.itemsize
    self.assertEqual(expected_size, multislice.get_leaf_memory_per_device(arr))

  def test_get_device_memory_real_hardware(self):
    if jax.device_count() > 0 and jax.devices()[0].platform == 'tpu':
      memory = multislice.get_device_memory()
      self.assertIsInstance(memory, int)
      self.assertGreater(memory, 0)

  def test_get_device_memory_tpuv3(self):
    if jax.device_count() > 0 and jax.devices()[0].device_kind == 'TPU v3':
      memory = multislice.get_device_memory()
      self.assertIsInstance(memory, int)
      self.assertEqual(memory, 16624107520)

  def test_number_of_broadcasts(self):
    array_size = 16000
    arr = [
        np.arange(array_size * 8).reshape((8, array_size)) * 1,
        np.arange(array_size * 16).reshape((16, array_size)) * 2,
        np.arange(2 * array_size * 8).reshape((8 * 2, array_size)) * 3,
        np.arange(3 * array_size * 16).reshape((16, 3 * array_size)) * 4,
    ]
    arrays, mesh, mesh_axes = setup_replica_sharded_arrays(
        arr, (2, len(jax.devices()) // 2)
    )
    replica_axis_index = 0

    mem_per_leafs = [multislice.get_leaf_memory_per_device(a) for a in arrays]
    # Slightly increase the memory limit for the edge case of when the
    # subtree has exactly the half of the total memory.
    memory_offset = 10
    broadcast_memory_limit_bytes = sum(mem_per_leafs) // 2 + memory_offset

    args = [
        test_utils.create_single_replica_restore_args(
            arr,
            mesh,
            mesh_axes,
        )
        for arr in arrays
    ]

    shardings = [arg.sharding for arg in args]
    single_replica_shardings = [arg.single_replica_sharding for arg in args]

    @functools.partial(
        jax.jit, static_argnums=0, out_shardings=tuple(single_replica_shardings)
    )
    def create_zeros(shape_dtype_tup):
      return jax.tree.map(
          lambda sd: jax.numpy.zeros(sd.shape, dtype=sd.dtype), shape_dtype_tup
      )

    _, primary_replica_pids = multislice.get_primary_replica_ids_and_pids(
        replica_axis_idx=replica_axis_index,
        mesh=shardings[0].mesh,  # pytype: disable=attribute-error
        primary_replica_id=0,
    )
    is_in_primary_replica = multihost.process_index() in primary_replica_pids
    if is_in_primary_replica:
      deserialized = arrays
    else:
      shape_dtype = [
          jax.ShapeDtypeStruct(arg.global_shape, arg.dtype) for arg in args
      ]
      deserialized = create_zeros(tuple(shape_dtype))

    deserialized = tuple(deserialized)
    global_mesh = cast(jax.sharding.NamedSharding, shardings[0])
    _, num_broadcasts = multislice.broadcast_one_replica_to_all(
        deserialized,
        global_mesh.mesh,
        replica_axis_index,
        is_in_primary_replica,
        memory_limit_bytes=broadcast_memory_limit_bytes,
    )
    self.assertEqual(num_broadcasts, 2)


class ProcessReplicaAssignmentTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_jax_process_count = self.enter_context(
        mock.patch.object(jax, 'process_count', autospec=True)
    )
    self.mock_jax_device_count = self.enter_context(
        mock.patch.object(jax, 'device_count', autospec=True)
    )
    self.mock_jax_local_device_count = self.enter_context(
        mock.patch.object(jax, 'local_device_count', autospec=True)
    )
    self.mock_jax_devices = self.enter_context(
        mock.patch.object(jax, 'devices', autospec=True)
    )
    self.mock_unique_processes = self.enter_context(
        mock.patch.object(
            multihost, 'unique_processes_from_devices', autospec=True
        )
    )

    def side_effect(devices):
      return set([d.process_index for d in devices])

    self.mock_unique_processes.side_effect = side_effect

  def test_process_spans_multiple_replicas_returns_false(
      self,
  ):
    self.mock_jax_process_count.return_value = 2
    self.mock_jax_device_count.return_value = 4
    self.mock_jax_local_device_count.return_value = 2
    mock_devices = [
        MockDevice(0, 0),
        MockDevice(0, 1),
        MockDevice(1, 2),
        MockDevice(1, 3),
    ]
    self.mock_jax_devices.return_value = mock_devices
    devices = np.asarray(mock_devices, dtype=object)
    mesh = jax.sharding.Mesh(devices.reshape((2, 2)), ('replica', 'data'))

    self.assertFalse(
        multislice.process_spans_multiple_replicas(mesh, replica_axis_index=0)
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='replica_axis_0',
          replica_axis_index=0,
          axis_names=('replica', 'data'),
      ),
      dict(
          testcase_name='replica_axis_1',
          replica_axis_index=1,
          axis_names=('data', 'replica'),
      ),
  )
  def test_process_spans_multiple_replicas_returns_true(
      self,
      replica_axis_index,
      axis_names,
  ):
    self.mock_jax_process_count.return_value = 1
    self.mock_jax_device_count.return_value = 4
    self.mock_jax_local_device_count.return_value = 4
    mock_devices = [
        MockDevice(0, 0),
        MockDevice(0, 1),
        MockDevice(0, 2),
        MockDevice(0, 3),
    ]
    self.mock_jax_devices.return_value = mock_devices
    devices = np.asarray(mock_devices, dtype=object)
    mesh = jax.sharding.Mesh(devices.reshape((2, 2)), axis_names)

    self.assertTrue(
        multislice.process_spans_multiple_replicas(
            mesh, replica_axis_index=replica_axis_index
        )
    )


if __name__ == '__main__':
  absltest.main()
