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
from orbax.checkpoint._src.arrays import sharding as sharding_lib


def _get_devices(n):
  devices = []
  for i in range(n):
    d = mock.create_autospec(jax.Device, instance=True, spec_set=True)
    d.id = i
    d.platform = 'cpu'
    d.device_kind = 'cpu'
    devices.append(d)
  return devices


class ShardingTest(parameterized.TestCase):

  @parameterized.parameters(
      {
          'shape': (64, 32),
          'num_devices': 8,
          'shard_shape': (8, 32),
          'pspec': ('a', None),
          'mesh_shape': (8,),
      },
      {
          'shape': (5, 7),
          'num_devices': 8,
          'shard_shape': (5, 7),
          'pspec': (None, None),
          'mesh_shape': (8,),
      },
      {
          'shape': (16, 8),
          'num_devices': 8,
          'shard_shape': (2, 8),
          'pspec': ('a', None),
          'mesh_shape': (8,),
      },
      {
          'shape': (16, 7),
          'num_devices': 8,
          'shard_shape': (2, 7),
          'pspec': ('a', None),
          'mesh_shape': (8,),
      },
      {
          'shape': (12, 6, 3),
          'num_devices': 12,
          'shard_shape': (1, 6, 3),
          'pspec': ('a', None, None),
          'mesh_shape': (12,),
      },
      {
          'shape': (8, 8),
          'num_devices': 16,
          'shard_shape': (8, 8),
          'pspec': (None, None),
          'mesh_shape': (16,),
      },
      {
          'shape': (8, 8),
          'num_devices': 4,
          'shard_shape': (8, 2),
          'pspec': (None, 'a'),
          'mesh_shape': (4,),
      },
      {
          'shape': (10, 9),
          'num_devices': 6,
          'shard_shape': (5, 3),
          'pspec': ('a', 'b'),
          'mesh_shape': (2, 3),
      },
      {
          'shape': (),
          'num_devices': 8,
          'shard_shape': (),
          'pspec': (),
          'mesh_shape': (8,),
      },
      {
          'shape': (2, 3),
          'num_devices': 1,
          'shard_shape': (2, 3),
          'pspec': (None, None),
          'mesh_shape': (),
      },
  )
  def test_construct_maximal_shardings(
      self, shape, num_devices, shard_shape, pspec, mesh_shape
  ):
    s = jax.ShapeDtypeStruct(shape=shape, dtype=np.int32)
    devices = _get_devices(num_devices)

    shardings = sharding_lib.construct_maximal_shardings(
        {'a': s}, devices=devices
    )

    self.assertIn('a', shardings)
    sharding: jax.sharding.NamedSharding = shardings['a']

    self.assertEqual(sharding.shard_shape(s.shape), shard_shape)
    self.assertEqual(sharding.spec, jax.sharding.PartitionSpec(*pspec))
    self.assertEqual(sharding.mesh.devices.shape, mesh_shape)  # pytype: disable=attribute-error


if __name__ == '__main__':
  absltest.main()
