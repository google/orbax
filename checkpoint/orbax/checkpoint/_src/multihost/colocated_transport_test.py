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
import jax
import jax.experimental.colocated_python as cp
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint._src.metadata import sharding as sharding_metadata
from orbax.checkpoint._src.multihost import colocated_transport
from orbax.checkpoint._src.serialization import type_handlers


class ColocatedTransportTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.arr = jax.device_put(
        np.arange(jax.device_count()),
        jax.sharding.NamedSharding(
            jax.sharding.Mesh(np.array(jax.devices()), ('d',)),
            jax.sharding.PartitionSpec('d'),
        ),
    )
    self.mock_device_put = self.enter_context(
        mock.patch(
            'jax.device_put', side_effect=lambda x, d, may_alias=False: x
        )
    )
    self.mock_cp_devices = self.enter_context(
        mock.patch.object(cp, 'colocated_cpu_devices', autospec=True)
    )

    def colocated_cpu_devices_side_effect(arg):
      if isinstance(arg, jax.sharding.Mesh):
        return jax.sharding.Mesh(np.array(jax.devices()), arg.axis_names)
      return list(arg)

    self.mock_cp_devices.side_effect = colocated_cpu_devices_side_effect

  def test_unique_colocated_cpu_devices_dedupes(self):
    cpu0 = mock.Mock(id=0)
    cpu1 = mock.Mock(id=1)
    self.mock_cp_devices.side_effect = None
    self.mock_cp_devices.return_value = [cpu0, cpu0, cpu1, cpu1]

    result = colocated_transport.unique_colocated_cpu_devices(jax.devices())

    self.assertEqual(result, (cpu0, cpu1))

  def test_colocated_cpu_sharding_named_sharding(self):
    cpu_sharding = colocated_transport.colocated_cpu_sharding(self.arr.sharding)
    self.assertIsInstance(cpu_sharding, jax.sharding.NamedSharding)
    self.mock_cp_devices.assert_called_once_with(self.arr.sharding.mesh)

  def test_transform_tree_shardings(self):
    restore_args = type_handlers.ArrayRestoreArgs(sharding=self.arr.sharding)
    sharding_meta = sharding_metadata.NamedShardingMetadata.from_jax_sharding(
        self.arr.sharding
    )
    metadata_args = type_handlers.ArrayRestoreArgs(sharding=sharding_meta)
    sds = jax.ShapeDtypeStruct(
        self.arr.shape, self.arr.dtype, sharding=self.arr.sharding
    )

    transformed = colocated_transport.transform_tree_shardings({
        'array': self.arr,
        'sharding': self.arr.sharding,
        'spec': sds,
        'restore_args': restore_args,
        'metadata_args': metadata_args,
    })

    self.assertIs(transformed['array'], self.arr)
    self.assertIsInstance(transformed['sharding'], jax.sharding.NamedSharding)
    self.assertIsInstance(transformed['spec'], jax.ShapeDtypeStruct)
    self.assertIsInstance(
        transformed['restore_args'].sharding, jax.sharding.NamedSharding
    )
    self.assertIsInstance(
        transformed['metadata_args'].sharding,
        sharding_metadata.ShardingMetadata,
    )

  def test_to_final_specs_uses_target_sharding(self):
    specs = jax.ShapeDtypeStruct(
        self.arr.shape, self.arr.dtype, sharding=self.arr.sharding
    )

    result = colocated_transport.to_final_specs(self.arr, specs)

    self.assertIs(result, self.arr)
    self.mock_device_put.assert_called_once_with(
        self.arr, specs.sharding, may_alias=True
    )

  def test_make_array_placeholder(self):
    spec = jax.ShapeDtypeStruct(
        (4,),
        jnp.int32,
        sharding=jax.sharding.SingleDeviceSharding(jax.devices()[0]),
    )

    arr = colocated_transport.make_array_placeholder(spec)

    self.assertEqual(arr.shape, (4,))
    self.assertEqual(arr.dtype, jnp.int32)

  def test_make_scalar_array_like_uses_callback_not_device_put(self):
    self.mock_device_put.reset_mock()
    sentinel = object()

    with mock.patch(
        'jax.make_array_from_callback', return_value=sentinel
    ) as mock_cb:
      result = colocated_transport.make_scalar_array_like(
          7, self.arr, dtype=jnp.int32
      )

    self.assertIs(result, sentinel)
    mock_cb.assert_called_once()
    self.mock_device_put.assert_not_called()


if __name__ == '__main__':
  absltest.main()
