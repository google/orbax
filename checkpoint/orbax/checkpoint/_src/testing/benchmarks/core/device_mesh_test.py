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
from jax.experimental import mesh_utils
import numpy as np
from orbax.checkpoint._src.testing.benchmarks.core import configs
from orbax.checkpoint._src.testing.benchmarks.core import device_mesh


class FakeDevice:

  def __init__(self, device_id):
    self.id = device_id

  def __str__(self):
    return f"FakeDevice(id={self.id})"


class DeviceMeshTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_devices = self.enter_context(
        mock.patch.object(jax, "devices", autospec=True)
    )
    self.mock_create_hybrid_device_mesh = self.enter_context(
        mock.patch.object(
            mesh_utils, "create_hybrid_device_mesh", autospec=True
        )
    )
    self.mock_num_slices = self.enter_context(
        mock.patch.object(device_mesh, "_num_slices", autospec=True)
    )
    self.mock_process_count = self.enter_context(
        mock.patch.object(jax, "process_count", autospec=True)
    )

  def test_create_mesh_success(self):
    self.mock_devices.return_value = [FakeDevice(device_id=i) for i in range(8)]
    self.mock_num_slices.return_value = 2
    config = configs.MeshConfig(
        mesh_axes=["x", "y"],
        ici_parallelism={"x": 2, "y": 2},  # num_devices_per_granule = 4
        dcn_parallelism={"y": 2},  # num_slices = 2
        allow_split_physical_axes=True,
        process_is_granule=False,
    )
    expected_ici_shape = [2, 2]
    expected_dcn_shape = [1, 2]
    mock_mesh_array = np.arange(8).reshape((2, 4))
    self.mock_create_hybrid_device_mesh.return_value = mock_mesh_array

    mesh = device_mesh.create_mesh(config)

    self.mock_create_hybrid_device_mesh.assert_called_once_with(
        expected_ici_shape,
        expected_dcn_shape,
        self.mock_devices.return_value,
        allow_split_physical_axes=True,
        process_is_granule=False,
    )
    self.assertIsInstance(mesh, jax.sharding.Mesh)
    self.assertEqual(mesh.axis_names, ("x", "y"))
    np.testing.assert_array_equal(mesh.devices, mock_mesh_array)

  def test_create_mesh_device_divisibility_error_raises_error(self):
    self.mock_devices.return_value = [FakeDevice(device_id=i) for i in range(7)]
    self.mock_num_slices.return_value = 2
    config = configs.MeshConfig(
        mesh_axes=["x", "y"],
        ici_parallelism={"x": 2, "y": 2},  # num_devices_per_granule = 4
        dcn_parallelism={"y": 2},  # num_slices = 2
    )

    # 7 total devices, num_slices = 2. Fails.
    with self.assertRaisesRegex(ValueError, "must be divisible by num_slices"):
      device_mesh.create_mesh(config)

  def test_create_mesh_invalid_dcn_parallelism_raises_error(self):
    self.mock_devices.return_value = [FakeDevice(device_id=i) for i in range(8)]
    self.mock_num_slices.return_value = 2
    config = configs.MeshConfig(
        mesh_axes=["x", "y"],
        ici_parallelism={"x": 2, "y": 2},
        dcn_parallelism={"y": 3},  # Product is 3, != num_slices = 2
    )

    with self.assertRaisesRegex(
        ValueError, "The product of DCN parallelism values"
    ):
      device_mesh.create_mesh(config)

  def test_create_mesh_invalid_ici_parallelism_raises_error(self):
    self.mock_devices.return_value = [FakeDevice(device_id=i) for i in range(8)]
    self.mock_num_slices.return_value = 2
    config = configs.MeshConfig(
        mesh_axes=["x", "y"],
        ici_parallelism={
            "x": 2,
            "y": 3,
        },  # Product is 6, != num_devices_per_granule = 4
        dcn_parallelism={"y": 2},  # num_slices = 2
    )

    with self.assertRaisesRegex(
        ValueError, "The product of ICI parallelism values"
    ):
      device_mesh.create_mesh(config)

  def test_create_mesh_success_process_is_granule(self):
    self.mock_devices.return_value = [FakeDevice(device_id=i) for i in range(8)]
    self.mock_process_count.return_value = 2
    config = configs.MeshConfig(
        mesh_axes=["x", "y"],
        ici_parallelism={"x": 2, "y": 2},  # num_devices_per_granule = 4
        dcn_parallelism={"y": 2},  # process_count = 2
        allow_split_physical_axes=True,
        process_is_granule=True,
    )
    expected_ici_shape = [2, 2]
    expected_dcn_shape = [1, 2]
    mock_mesh_array = np.arange(8).reshape((2, 4))
    self.mock_create_hybrid_device_mesh.return_value = mock_mesh_array

    mesh = device_mesh.create_mesh(config)

    self.mock_create_hybrid_device_mesh.assert_called_once_with(
        expected_ici_shape,
        expected_dcn_shape,
        self.mock_devices.return_value,
        allow_split_physical_axes=True,
        process_is_granule=True,
    )
    self.assertIsInstance(mesh, jax.sharding.Mesh)
    self.assertEqual(mesh.axis_names, ("x", "y"))
    np.testing.assert_array_equal(mesh.devices, mock_mesh_array)

  def test_process_create_mesh_device_divisibility_error_raises_error(self):
    self.mock_devices.return_value = [FakeDevice(device_id=i) for i in range(7)]
    self.mock_process_count.return_value = 2
    config = configs.MeshConfig(
        mesh_axes=["x", "y"],
        ici_parallelism={"x": 2, "y": 2},  # num_devices_per_granule = 4
        dcn_parallelism={"y": 2},  # process_count = 2
        process_is_granule=True,
    )

    # 7 total devices, process_count = 2. Fails.
    with self.assertRaisesRegex(
        ValueError, "must be divisible by process_count"
    ):
      device_mesh.create_mesh(config)

  def test_process_create_mesh_invalid_dcn_parallelism_raises_error(self):
    self.mock_devices.return_value = [FakeDevice(device_id=i) for i in range(8)]
    self.mock_process_count.return_value = 2
    config = configs.MeshConfig(
        mesh_axes=["x", "y"],
        ici_parallelism={"x": 2, "y": 2},
        dcn_parallelism={"y": 3},  # Product is 3, != process_count = 2
        process_is_granule=True,
    )

    with self.assertRaisesRegex(
        ValueError, "The product of DCN parallelism values"
    ):
      device_mesh.create_mesh(config)

  def test_process_create_mesh_invalid_ici_parallelism_raises_error(self):
    self.mock_devices.return_value = [FakeDevice(device_id=i) for i in range(8)]
    self.mock_process_count.return_value = 2
    config = configs.MeshConfig(
        mesh_axes=["x", "y"],
        ici_parallelism={
            "x": 2,
            "y": 3,
        },  # Product is 6, != num_devices_per_granule = 4
        dcn_parallelism={"y": 2},  # process_count = 2
        process_is_granule=True,
    )

    with self.assertRaisesRegex(
        ValueError, "The product of ICI parallelism values"
    ):
      device_mesh.create_mesh(config)


if __name__ == "__main__":
  absltest.main()
