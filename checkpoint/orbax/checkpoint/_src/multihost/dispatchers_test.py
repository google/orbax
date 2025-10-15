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
from orbax.checkpoint._src.futures import future as future_lib
from orbax.checkpoint._src.multihost import dispatchers



def _get_mock_dispatcher_array():
  mesh = jax.sharding.Mesh(np.array(jax.devices()), ('d',))
  sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('d'))
  return jax.device_put(np.arange(jax.device_count()), sharding)


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

    self.mock_cp_result = mock.MagicMock()

    self.mock_cp_colocated_python = self.enter_context(
        mock.patch.object(cp, 'colocated_python', autospec=True)
    )
    self.mock_cp_devices = self.enter_context(
        mock.patch.object(cp, 'colocated_cpu_devices', autospec=True)
    )

    def cp_decorator(f):
      def wrapper(*args, **kwargs):
        f(*args, **kwargs)
        return self.mock_cp_result

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
    cpu_arr = dispatchers.to_colocated_python(self.arr)

    self.assertIs(cpu_arr, self.arr)
    self.mock_cp_devices.assert_called_once()
    self.mock_device_put.assert_called_once()

  def test_get_abstract_dummy_result_returns_abstract_result(self):
    dummy = dispatchers.get_abstract_dummy_result([self.arr])

    self.assertLen(dummy, 1)
    self.assertEqual(dummy[0].shape, ())
    self.assertEqual(dummy[0].dtype, jnp.bool)
    self.assertTrue(dummy[0].sharding.is_fully_replicated)

  def test_dispatch_devices_executes_function(self):
    fn = mock.MagicMock()
    dispatcher = dispatchers.ColocatedPythonDispatcher()

    future = dispatcher.dispatch_devices(fn)
    future.result()

    fn.assert_called_once()
    self.mock_cp_devices.assert_called_once()
    self.mock_block_until_ready.assert_called_once_with(self.mock_cp_result)

  def test_dispatch_arrays_executes_function_with_arrays(self):
    fn = mock.MagicMock()
    dispatcher = dispatchers.ColocatedPythonDispatcher()

    future = dispatcher.dispatch_arrays(fn, [self.arr], metadata={'key': 1})
    future.result()

    fn.assert_called_once_with([self.arr], metadata={'key': 1})
    self.mock_block_until_ready.assert_called_once_with(self.mock_cp_result)


class DirectDispatcherTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.arr = _get_mock_dispatcher_array()

  def test_dispatch_devices_executes_function(self):
    fn = mock.MagicMock()
    dispatcher = dispatchers.DirectDispatcher()

    future = dispatcher.dispatch_devices(fn)
    future.result()

    fn.assert_called_once()
    self.assertIsInstance(future, future_lib.NoopFuture)

  def test_dispatch_arrays_executes_function_with_arrays(self):
    fn = mock.MagicMock()
    dispatcher = dispatchers.DirectDispatcher()

    future = dispatcher.dispatch_arrays(fn, [self.arr], metadata={'key': 1})
    future.result()

    fn.assert_called_once_with([self.arr], metadata={'key': 1})
    self.assertIsInstance(future, future_lib.NoopFuture)




if __name__ == '__main__':
  absltest.main()
