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
import jax.experimental.colocated_python as cp
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint._src.metadata import sharding as sharding_metadata
from orbax.checkpoint._src.multihost import dispatchers
from orbax.checkpoint._src.serialization import type_handlers



def _get_mock_dispatcher_array():
  mesh = jax.sharding.Mesh(np.array(jax.devices()), ('d',))
  sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('d'))
  return jax.device_put(np.arange(jax.device_count()), sharding)


class DispatchersTest(parameterized.TestCase):

  def test_get_dummy_input_array(self):
    devices = jax.devices()
    arr = dispatchers.get_dummy_input_array(devices)
    self.assertEqual(arr.shape, ())
    self.assertEqual(arr.dtype, jnp.bool)
    self.assertTrue(arr.sharding.is_fully_replicated)
    self.assertCountEqual(list(arr.devices()), devices)

  def test_get_dummy_input_array_from_result_specs(self):
    arr = _get_mock_dispatcher_array()
    result_specs = jax.ShapeDtypeStruct(
        arr.shape, arr.dtype, sharding=arr.sharding
    )
    dummy = dispatchers._get_dummy_input_array_from_result_specs(result_specs)
    self.assertEqual(dummy.shape, ())
    self.assertEqual(dummy.dtype, jnp.bool)
    self.assertTrue(dummy.sharding.is_fully_replicated)
    self.assertCountEqual(list(dummy.devices()), arr.devices())

  def test_make_dummy_result_array(self):
    arr = _get_mock_dispatcher_array()
    dummy = dispatchers._make_dummy_result_array(arr)
    self.assertIsInstance(dummy, jax.Array)
    self.assertEqual(dummy.shape, ())
    self.assertEqual(dummy.dtype, jnp.bool)
    self.assertTrue(dummy.sharding.is_fully_replicated)
    self.assertCountEqual(list(dummy.devices()), arr.devices())

  def test_make_dummy_result_array_abstract(self):
    arr = _get_mock_dispatcher_array()
    dummy = dispatchers._make_dummy_result_array(arr, abstract=True)
    self.assertIsInstance(dummy, jax.ShapeDtypeStruct)
    self.assertEqual(dummy.shape, ())
    self.assertEqual(dummy.dtype, jnp.bool)
    self.assertTrue(dummy.sharding.is_fully_replicated)
    self.assertCountEqual(list(dummy.sharding.device_set), arr.devices())


class ColocatedPythonDispatcherTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.arr = _get_mock_dispatcher_array()

    self.mock_block_until_ready = self.enter_context(
        mock.patch('jax.block_until_ready')
    )
    self.mock_device_put = self.enter_context(
        mock.patch(
            'jax.device_put', side_effect=lambda x, d, may_alias=False: x
        )
    )

    self.mock_cp_colocated_python = self.enter_context(
        mock.patch.object(cp, 'colocated_python', autospec=True)
    )
    self.mock_cp_devices = self.enter_context(
        mock.patch.object(cp, 'colocated_cpu_devices', autospec=True)
    )

    def cp_decorator(f):
      def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

      wrapper.specialize = mock.MagicMock(return_value=wrapper)
      return wrapper

    self.mock_cp_colocated_python.side_effect = cp_decorator

    def colocated_cpu_devices_side_effect(arg):
      if isinstance(arg, jax.sharding.Mesh):
        return jax.sharding.Mesh(np.array(jax.devices()), arg.axis_names)
      else:
        return list(arg)

    self.mock_cp_devices.side_effect = colocated_cpu_devices_side_effect

  def test_device_to_host(self):
    dispatcher = dispatchers.ColocatedPythonDispatcher()
    cpu_arr = dispatcher.device_to_host(self.arr)

    self.assertIs(cpu_arr, self.arr)
    self.mock_cp_devices.assert_called_once_with(self.arr.sharding.mesh)
    self.mock_device_put.assert_called_once_with(
        self.arr, mock.ANY, may_alias=True
    )

  def test_to_colocated_python_copies_array(self):
    dispatcher = dispatchers.ColocatedPythonDispatcher()
    cpu_arr = dispatcher.to_colocated_python(self.arr)

    self.assertIs(cpu_arr, self.arr)
    self.mock_cp_devices.assert_called_once_with(self.arr.sharding.mesh)
    self.mock_device_put.assert_called_once_with(
        self.arr, mock.ANY, may_alias=True
    )

  def test_colocated_cpu_sharding_single_device_sharding(self):
    dispatcher = dispatchers.ColocatedPythonDispatcher()
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    cpu_sharding = dispatcher._colocated_cpu_sharding(sharding)
    self.assertIsInstance(cpu_sharding, jax.sharding.SingleDeviceSharding)
    self.mock_cp_devices.assert_called_once_with(list(sharding.device_set))

  def test_colocated_cpu_sharding_named_sharding(self):
    dispatcher = dispatchers.ColocatedPythonDispatcher()
    sharding = self.arr.sharding
    cpu_sharding = dispatcher._colocated_cpu_sharding(sharding)
    self.assertIsInstance(cpu_sharding, jax.sharding.NamedSharding)
    self.mock_cp_devices.assert_called_once_with(sharding.mesh)

  def test_colocated_cpu_sharding_unsupported_sharding(self):
    dispatcher = dispatchers.ColocatedPythonDispatcher()
    sharding = mock.Mock()
    del sharding.mesh
    with self.assertRaisesRegex(TypeError, 'Sharding type'):
      dispatcher._colocated_cpu_sharding(sharding)

  def test_convert_array_restore_args_with_mesh(self):
    dispatcher = dispatchers.ColocatedPythonDispatcher()
    args = type_handlers.ArrayRestoreArgs(mesh=self.arr.sharding.mesh)
    converted_args = dispatcher._convert_array_restore_args(args)
    self.assertIsInstance(converted_args.mesh, jax.sharding.Mesh)
    self.mock_cp_devices.assert_called_once_with(args.mesh)

  def test_convert_array_restore_args_with_sharding(self):
    dispatcher = dispatchers.ColocatedPythonDispatcher()
    args = type_handlers.ArrayRestoreArgs(sharding=self.arr.sharding)
    converted_args = dispatcher._convert_array_restore_args(args)
    self.assertIsInstance(converted_args.sharding, jax.sharding.NamedSharding)
    self.mock_cp_devices.assert_called_once_with(args.sharding.mesh)

  def test_convert_array_restore_args_with_sharding_metadata(self):
    dispatcher = dispatchers.ColocatedPythonDispatcher()
    sharding_meta = sharding_metadata.NamedShardingMetadata.from_jax_sharding(
        self.arr.sharding
    )
    args = type_handlers.ArrayRestoreArgs(sharding=sharding_meta)
    converted_args = dispatcher._convert_array_restore_args(args)
    self.assertIsInstance(
        converted_args.sharding, sharding_metadata.ShardingMetadata
    )
    self.mock_cp_devices.assert_called_once_with(self.arr.sharding.mesh)

  def test_convert_single_replica_restore_args(self):
    dispatcher = dispatchers.ColocatedPythonDispatcher()
    args = type_handlers.SingleReplicaArrayRestoreArgs(
        single_replica_sharding=self.arr.sharding
    )
    converted_args = dispatcher._convert_single_replica_restore_args(args)
    self.assertIsInstance(
        converted_args.single_replica_sharding, jax.sharding.NamedSharding
    )
    assert args.single_replica_sharding is not None
    self.mock_cp_devices.assert_called_once_with(
        args.single_replica_sharding.mesh
    )

  def test_transform_pytree_shardings(self):
    dispatcher = dispatchers.ColocatedPythonDispatcher()
    restore_args = type_handlers.ArrayRestoreArgs(sharding=self.arr.sharding)
    sds = jax.ShapeDtypeStruct(
        self.arr.shape, self.arr.dtype, sharding=self.arr.sharding
    )
    pytree = {
        'a': self.arr.sharding,
        'b': sds,
        'c': restore_args,
        'd': self.arr,
        'e': 1,
    }
    transformed_pytree = dispatcher._transform_pytree_shardings(pytree)
    self.assertIsInstance(transformed_pytree['a'], jax.sharding.NamedSharding)
    self.assertIsInstance(transformed_pytree['b'], jax.ShapeDtypeStruct)
    self.assertIsInstance(
        transformed_pytree['b'].sharding, jax.sharding.NamedSharding
    )
    self.assertIsInstance(
        transformed_pytree['c'], type_handlers.ArrayRestoreArgs
    )
    self.assertIsInstance(
        transformed_pytree['c'].sharding, jax.sharding.NamedSharding
    )
    self.assertIs(transformed_pytree['d'], self.arr)
    self.assertEqual(transformed_pytree['e'], 1)

  def test_to_final_specs(self):
    dispatcher = dispatchers.ColocatedPythonDispatcher()
    specs = jax.ShapeDtypeStruct(
        self.arr.shape, self.arr.dtype, sharding=self.arr.sharding
    )
    result = dispatcher._to_final_specs(self.arr, specs)
    self.assertIs(result, self.arr)
    self.mock_device_put.assert_called_once_with(
        self.arr, specs.sharding, may_alias=True
    )

  def test_dispatch_without_result_specs_discards_result(self):
    fn = mock.MagicMock(return_value=self.arr + 1)
    dispatcher = dispatchers.ColocatedPythonDispatcher()

    result = dispatcher.dispatch(
        fn, input_arrays=self.arr, func_kwargs={'a': 1}
    )

    fn.assert_called_once_with(self.arr, a=1)
    self.mock_cp_colocated_python.assert_called_once()
    self.assertEqual(self.mock_device_put.call_count, 2)
    self.assertEqual(result.shape, ())
    self.assertEqual(result.dtype, jnp.bool)
    self.assertTrue(result.sharding.is_fully_replicated)
    self.assertCountEqual(list(result.devices()), self.arr.devices())

  def test_dispatch_with_result_specs_returns_result(self):
    fn = mock.MagicMock(return_value=self.arr)
    dispatcher = dispatchers.ColocatedPythonDispatcher()
    specs = jax.ShapeDtypeStruct(
        self.arr.shape, self.arr.dtype, sharding=self.arr.sharding
    )

    result = dispatcher.dispatch(
        fn, input_arrays=self.arr, result_specs=specs, func_args=(1,)
    )

    fn.assert_called_once_with(self.arr, 1)
    self.mock_cp_colocated_python.assert_called_once()
    self.assertEqual(self.mock_device_put.call_count, 2)
    self.assertIs(result, self.arr)

  def test_dispatch_no_input(self):
    fn = mock.MagicMock()
    dispatcher = dispatchers.ColocatedPythonDispatcher()
    dispatcher.dispatch(fn)
    fn.assert_called_once_with()
    self.mock_cp_colocated_python.assert_called_once()
    self.assertEqual(self.mock_device_put.call_count, 3)




if __name__ == '__main__':
  absltest.main()
