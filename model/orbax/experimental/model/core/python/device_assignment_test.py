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

import os
from absl.testing import absltest
import jax
from jax.experimental import mesh_utils
from jax.experimental.topologies import get_topology_desc
import numpy as np
from orbax.experimental.model.core.python import device_assignment
from .testing.pybase import parameterized


class DeviceAssignmentTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='with_all_fields',
          assignment=device_assignment.DeviceAssignment(
              id=5, coords=(1, 2), core_on_chip=3
          ),
          expected_id=5,
          expected_coords=(1, 2),
          expected_core_on_chip=3,
      ),
      dict(
          testcase_name='with_defaults',
          assignment=device_assignment.DeviceAssignment(id=0),
          expected_id=0,
          expected_coords=None,
          expected_core_on_chip=None,
      ),
  )
  def test_device_assignment(
      self, assignment, expected_id, expected_coords, expected_core_on_chip
  ):
    self.assertEqual(assignment.id, expected_id)
    self.assertEqual(assignment.coords, expected_coords)
    self.assertEqual(assignment.core_on_chip, expected_core_on_chip)

  def test_jax_mesh_to_obm_device_assignment_by_coords_cpu(self):
    def _create_mesh() -> jax.sharding.Mesh:
      os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'
      devices = mesh_utils.create_device_mesh((2, 2, 2))
      return jax.sharding.Mesh(devices, ('b', 'x', 'y'))

    mesh = _create_mesh()
    result = device_assignment.jax_mesh_to_obm_device_assignment_by_coords(mesh)

    expected = [
        device_assignment.DeviceAssignment(id=i, coords=None, core_on_chip=None)
        for i in range(8)
    ]

    self.assertEqual(result, expected)

  def test_jax_mesh_to_obm_device_assignment_by_coords_tpu(self):
    def _create_mesh() -> jax.sharding.Mesh:
      devices = np.array(get_topology_desc('df=4x2').devices).reshape(4, 4)
      return jax.sharding.Mesh(devices, ('x', 'y'))

    mesh = _create_mesh()
    result = device_assignment.jax_mesh_to_obm_device_assignment_by_coords(mesh)

    expected = [
        device_assignment.DeviceAssignment(
            id=0, coords=[0, 0, 0], core_on_chip=0
        ),
        device_assignment.DeviceAssignment(
            id=1, coords=[0, 0, 0], core_on_chip=1
        ),
        device_assignment.DeviceAssignment(
            id=2, coords=[1, 0, 0], core_on_chip=0
        ),
        device_assignment.DeviceAssignment(
            id=3, coords=[1, 0, 0], core_on_chip=1
        ),
        device_assignment.DeviceAssignment(
            id=8, coords=[0, 1, 0], core_on_chip=0
        ),
        device_assignment.DeviceAssignment(
            id=9, coords=[0, 1, 0], core_on_chip=1
        ),
        device_assignment.DeviceAssignment(
            id=10, coords=[1, 1, 0], core_on_chip=0
        ),
        device_assignment.DeviceAssignment(
            id=11, coords=[1, 1, 0], core_on_chip=1
        ),
        device_assignment.DeviceAssignment(
            id=4, coords=[2, 0, 0], core_on_chip=0
        ),
        device_assignment.DeviceAssignment(
            id=5, coords=[2, 0, 0], core_on_chip=1
        ),
        device_assignment.DeviceAssignment(
            id=6, coords=[3, 0, 0], core_on_chip=0
        ),
        device_assignment.DeviceAssignment(
            id=7, coords=[3, 0, 0], core_on_chip=1
        ),
        device_assignment.DeviceAssignment(
            id=12, coords=[2, 1, 0], core_on_chip=0
        ),
        device_assignment.DeviceAssignment(
            id=13, coords=[2, 1, 0], core_on_chip=1
        ),
        device_assignment.DeviceAssignment(
            id=14, coords=[3, 1, 0], core_on_chip=0
        ),
        device_assignment.DeviceAssignment(
            id=15, coords=[3, 1, 0], core_on_chip=1
        ),
    ]

    self.assertEqual(result, expected)


if __name__ == '__main__':
  absltest.main()
