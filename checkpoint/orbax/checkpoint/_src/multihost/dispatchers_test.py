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

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.experimental.colocated_python as cp
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint._src.multihost import dispatchers



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


class ColocatedPythonDispatcherTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.arr = _get_mock_dispatcher_array()

    self.mock_block_until_ready = self.enter_context(
        mock.patch('jax.block_until_ready')
    )
    self.mock_device_put = self.enter_context(
        mock.patch('jax.device_put', side_effect=lambda arr, _: arr)
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
      else:  # devices
        return jax.devices()

    self.mock_cp_devices.side_effect = colocated_cpu_devices_side_effect

  def test_to_colocated_python_copies_array(self):
    dispatcher = dispatchers.ColocatedPythonDispatcher()
    cpu_arr = dispatcher.to_colocated_python(self.arr)

    self.assertIs(cpu_arr, self.arr)
    self.mock_cp_devices.assert_called_once()
    self.mock_device_put.assert_called_once()

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
    # device_put called in to_colocated_python and _to_final_specs
    self.assertEqual(self.mock_device_put.call_count, 2)
    self.assertIs(result, self.arr)




if __name__ == '__main__':
  absltest.main()
