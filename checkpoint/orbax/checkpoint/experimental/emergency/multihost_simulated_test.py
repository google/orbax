# Copyright 2024 The Orbax Authors.
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

"""Emergency checkpointing utils for multihost / multislice without real accelerators."""

import math

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from orbax.checkpoint.experimental.emergency import multihost as emergency_multihost


def create_fake_global_mesh(mesh_shape, axis_names, device_ids):
  size = math.prod(mesh_shape)
  if len(device_ids) != size:
    raise ValueError(
        f"Mesh shape is {mesh_shape} but only {len(device_ids)} devices are"
        " specified."
    )
  devices = [FakeDevice(device_id) for device_id in device_ids]
  mesh_devices = np.array(devices).reshape(mesh_shape)
  global_mesh = jax.sharding.Mesh(mesh_devices, axis_names)
  return global_mesh


class FakeDevice:
  """Fake device for testing."""

  def __init__(self, device_id):
    self.id = device_id

  def __repr__(self):
    return f"FakeDevice({self.id})"

  def __eq__(self, value):
    return self.id == value.id

  def __hash__(self):
    return self.id


# Same as multihost_test.py, but allows us to test without real accelerators.
# We cannot run Forge tests with large number of chips that are representative
# of real workloads, so we use this test to add new test cases from problematic
# runs as reproducers.
class MultihostSimulatedTest(parameterized.TestCase, absltest.TestCase):

  @parameterized.named_parameters(
      dict(
          # TODO(cpgaffney): Add test cases for other topologies.
          testcase_name="df_4x4",
          mesh_shape=(16, 2),
          axis_names=["data", "model"],
          previous_flattened_mesh_device_ids=[0, 1, 8, 9, 2, 3, 10, 11,
                                              16, 17, 24, 25, 18, 19, 26, 27,
                                              4, 5, 12, 13, 6, 7, 14, 15,
                                              20, 21, 28, 29, 22, 23, 30, 31],
          # Note that each process has a discontiguous range of device ids.
          # (e.g. process 0 has ids 0-3, 8-11 instead of 0-7).
          previous_distributed_to_device_ids=[[0, 1, 2, 3, 8, 9, 10, 11],
                                              [4, 5, 6, 7, 12, 13, 14, 15],
                                              [16, 17, 18, 19, 24, 25, 26, 27],
                                              [20, 21, 22, 23, 28, 29, 30, 31]
                                             ],
          # Swap distributed process 1 and 2.
          current_distributed_to_device_ids=[[0, 1, 2, 3, 8, 9, 10, 11],
                                             [16, 17, 18, 19, 24, 25, 26, 27],
                                             [4, 5, 6, 7, 12, 13, 14, 15],
                                             [20, 21, 22, 23, 28, 29, 30, 31]
                                            ],
          expected_flattened_mesh_device_ids=[
              0, 1, 8, 9, 2, 3, 10, 11,
              # This mesh row is swapped (compared to the original
              # `previous_flattened_mesh_device_ids`)
              4, 5, 12, 13, 6, 7, 14, 15,
              16, 17, 24, 25, 18, 19, 26, 27,
              20, 21, 28, 29, 22, 23, 30, 31],
      ),
  )
  def test_consistent_restore_mesh(
      self,
      mesh_shape,
      axis_names,
      previous_flattened_mesh_device_ids,
      previous_distributed_to_device_ids,
      current_distributed_to_device_ids,
      expected_flattened_mesh_device_ids,
  ):
    num_devices = math.prod(mesh_shape)
    devices = [FakeDevice(device_id) for device_id in range(num_devices)]
    input_mesh = create_fake_global_mesh(
        mesh_shape, axis_names, previous_flattened_mesh_device_ids
    )
    transformed_mesh = emergency_multihost.consistent_restore_mesh(
        devices,
        input_mesh,
        previous_flattened_mesh_device_ids,
        previous_distributed_to_device_ids,
        current_distributed_to_device_ids,
    )
    expected_mesh_devices = np.array([
        FakeDevice(device_id)
        for device_id in expected_flattened_mesh_device_ids
    ]).reshape(mesh_shape)

    logging.info("transformed_mesh.devices: %s", transformed_mesh.devices)
    logging.info("expected_mesh_devices: %s", expected_mesh_devices)
    np.testing.assert_array_equal(
        transformed_mesh.devices, expected_mesh_devices
    )

if __name__ == "__main__":
  absltest.main()
