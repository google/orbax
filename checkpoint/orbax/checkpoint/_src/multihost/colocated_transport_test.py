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

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring,too-few-public-methods

from unittest import mock

from absl.testing import absltest
import jax
import jax.experimental.colocated_python as cp
from jax.experimental.colocated_python import serialization as cp_serialization
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint._src.metadata import sharding as sharding_metadata
from orbax.checkpoint._src.multihost import colocated_transport
from orbax.checkpoint._src.serialization import type_handlers


class ColocatedTransportTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    colocated_transport._get_cpu_device_map.cache_clear()  # pylint: disable=protected-access
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

  def tearDown(self):
    colocated_transport._get_cpu_device_map.cache_clear()  # pylint: disable=protected-access
    super().tearDown()

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

    transformed = colocated_transport.transform_tree_shardings(
        {
            'array': self.arr,
            'sharding': self.arr.sharding,
            'spec': sds,
            'restore_args': restore_args,
            'metadata_args': metadata_args,
        }
    )

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

  def test_shape_dtype_struct_for_array_preserves_array_metadata(self):
    spec = colocated_transport.shape_dtype_struct_for_array(self.arr)

    self.assertEqual(spec.shape, self.arr.shape)
    self.assertEqual(spec.dtype, self.arr.dtype)
    self.assertIs(spec.sharding, self.arr.sharding)

  def test_zeros_like_spec(self):
    spec = jax.ShapeDtypeStruct(
        (4,),
        jnp.int32,
        sharding=jax.sharding.SingleDeviceSharding(jax.devices()[0]),
    )

    arr = colocated_transport.zeros_like_spec(spec)

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

  def test_extract_pjrt_ifrt_device_id_from_repr(self):
    fake_cpu = mock.Mock()
    fake_cpu.__repr__ = mock.Mock(
        return_value='CpuDevice(id=0)[PjRtIFRTDeviceId=120]'
    )

    result = colocated_transport._extract_pjrt_ifrt_device_id(  # pylint: disable=protected-access
        fake_cpu
    )

    self.assertEqual(result, 120)

  def test_device_platform_falls_back_to_unknown(self):

    class _DeviceWithoutPlatform:
      pass

    self.assertEqual(
        colocated_transport._device_platform(_DeviceWithoutPlatform()),  # pytype: disable=wrong-arg-types # pylint: disable=protected-access
        'unknown',
    )

  def test_all_backend_devices_uses_default_client_to_enumerate_devices(self):
    default_client = mock.Mock()
    cpu0 = mock.Mock(id=0, platform='cpu', client=default_client)
    cpu1 = mock.Mock(id=1, platform='cpu', client=default_client)
    default_client._get_all_devices.return_value = (cpu0, cpu1)  # pylint: disable=protected-access
    local0 = mock.Mock(client=default_client)
    local1 = mock.Mock(client=default_client)

    with mock.patch.object(
        cp_serialization.xb,
        'local_devices',
        return_value=(local0, local1),
    ), mock.patch.object(cp_serialization.xb, 'backends', return_value={}):
      result = colocated_transport._all_backend_devices()  # pylint: disable=protected-access

    self.assertEqual(result, (cpu0, cpu1))

  def test_all_backend_devices_dedupes_by_explicit_device_identity(self):
    default_client = mock.Mock()
    local_cpu = mock.Mock(id=0, platform='cpu', client=default_client)
    duplicate_cpu = mock.Mock(id=0, platform='cpu', client=default_client)
    other_cpu = mock.Mock(id=1, platform='cpu', client=default_client)
    default_client._get_all_devices.return_value = (local_cpu,)  # pylint: disable=protected-access
    local_device = mock.Mock(client=default_client)
    backend = mock.Mock()
    backend._get_all_devices.return_value = (duplicate_cpu, other_cpu)  # pylint: disable=protected-access

    with mock.patch.object(
        cp_serialization.xb,
        'local_devices',
        return_value=(local_device,),
    ), mock.patch.object(
        cp_serialization.xb, 'backends', return_value={'cpu': backend}
    ):
      result = colocated_transport._all_backend_devices()  # pylint: disable=protected-access

    self.assertEqual(result, (local_cpu, other_cpu))

  def test_get_cpu_device_map_supports_ifrt_and_local_ids(self):
    local_device = mock.Mock(id=0, platform='cpu', client=object())
    local_device.__repr__ = mock.Mock(
        return_value='CpuDevice(id=0)[PjRtIFRTDeviceId=34]'
    )

    with mock.patch.object(
        colocated_transport,
        '_all_backend_devices',
        return_value=(local_device,),
    ):
      device_map = colocated_transport._get_cpu_device_map()  # pylint: disable=protected-access

    self.assertIs(device_map[34], local_device)
    self.assertIs(device_map[0], local_device)

  def test_get_cpu_device_map_rejects_ifrt_local_id_collision(self):
    ifrt_device = mock.Mock(id=1, platform='cpu', client=object())
    ifrt_device.__repr__ = mock.Mock(
        return_value='CpuDevice(id=1)[PjRtIFRTDeviceId=0]'
    )
    local_id_device = mock.Mock(id=0, platform='cpu', client=object())
    local_id_device.__repr__ = mock.Mock(return_value='CpuDevice(id=0)')

    with mock.patch.object(
        colocated_transport,
        '_all_backend_devices',
        return_value=(ifrt_device, local_id_device),
    ):
      with self.assertRaisesRegex(ValueError, 'CPU device id 0 is ambiguous'):
        colocated_transport._get_cpu_device_map()  # pylint: disable=protected-access

  def test_get_cpu_device_map_caches_backend_scan(self):
    cpu = mock.Mock(id=0, platform='cpu', client=object())
    cpu.__repr__ = mock.Mock(return_value='CpuDevice(id=0)')

    with mock.patch.object(
        colocated_transport, '_all_backend_devices', return_value=(cpu,)
    ) as all_backend_devices:
      first_map = colocated_transport._get_cpu_device_map()  # pylint: disable=protected-access
      second_map = colocated_transport._get_cpu_device_map()  # pylint: disable=protected-access

    self.assertIs(first_map, second_map)
    all_backend_devices.assert_called_once()

  def test_get_cpu_device_map_returns_immutable_mapping(self):
    cpu = mock.Mock(id=0, platform='cpu', client=object())
    cpu.__repr__ = mock.Mock(return_value='CpuDevice(id=0)')

    with mock.patch.object(
        colocated_transport, '_all_backend_devices', return_value=(cpu,)
    ):
      device_map = colocated_transport._get_cpu_device_map()  # pylint: disable=protected-access

    with self.assertRaises(TypeError):
      device_map[1] = cpu  # pytype: disable=unsupported-operands

  def test_normalize_mesh_to_colocated_cpu_remaps_non_cpu_devices(self):
    cpu0 = mock.Mock(platform='cpu')
    cpu1 = mock.Mock(platform='cpu')
    tpu0 = mock.Mock(platform='tpu')
    tpu1 = mock.Mock(platform='tpu')

    class _FakeMesh:
      devices = np.array([tpu0, tpu1], dtype=object)
      axis_names = ('d',)
      axis_types = None

    mesh = _FakeMesh()

    with mock.patch.object(
        colocated_transport,
        '_to_serializable_cpu_device',
        side_effect=[cpu0, cpu1],
    ):
      cpu_mesh = colocated_transport._normalize_mesh_to_colocated_cpu(  # pytype: disable=wrong-arg-types # pylint: disable=protected-access
          mesh
      )

    self.assertEqual(cpu_mesh.axis_names, mesh.axis_names)
    self.assertEqual(tuple(cpu_mesh.devices.flat), (cpu0, cpu1))

  def test_install_pathways_colocated_serialization_patch_is_idempotent(self):
    original_reduce_mesh = cp_serialization._reduce_mesh  # pylint: disable=protected-access
    original_get_cpu_device_map = cp_serialization._get_cpu_device_map  # pylint: disable=protected-access

    colocated_transport.install_pathways_colocated_serialization_patch()
    patched_reduce_mesh = cp_serialization._reduce_mesh  # pylint: disable=protected-access
    patched_get_cpu_device_map = cp_serialization._get_cpu_device_map  # pylint: disable=protected-access
    colocated_transport.install_pathways_colocated_serialization_patch()

    self.assertIsNot(original_reduce_mesh, patched_reduce_mesh)
    self.assertIs(
        patched_reduce_mesh,
        cp_serialization._reduce_mesh,  # pylint: disable=protected-access
    )
    self.assertIsNot(original_get_cpu_device_map, patched_get_cpu_device_map)
    self.assertIs(
        patched_get_cpu_device_map,
        cp_serialization._get_cpu_device_map,  # pylint: disable=protected-access
    )


if __name__ == '__main__':
  absltest.main()
