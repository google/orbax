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

from absl.testing import absltest
from orbax.experimental.model.core.protos import manifest_pb2
from orbax.experimental.model.core.python import manifest_util
from orbax.experimental.model.core.python.device_assignment import DeviceAssignment


class ManifestUtilTest(absltest.TestCase):

  def test_build_device_assignment_by_coords_proto(self):
    # Test with an empty list of device assignments.
    device_assignments = []
    proto = manifest_util.build_device_assignment_by_coords_proto(
        device_assignments
    )
    self.assertEqual(proto, manifest_pb2.DeviceAssignmentByCoords())

    # Test with a list of fully populated device assignments.
    device_assignments = [
        DeviceAssignment(id=0, coords=[0, 1], core_on_chip=0),
        DeviceAssignment(id=1, coords=[1, 0], core_on_chip=1),
    ]
    proto = manifest_util.build_device_assignment_by_coords_proto(
        device_assignments
    )

    self.assertLen(proto.devices, 2)
    # Check first device.
    self.assertEqual(proto.devices[0].id, 0)
    self.assertSequenceEqual(proto.devices[0].coords, [0, 1])
    self.assertEqual(proto.devices[0].core_on_chip, 0)
    # Check second device.
    self.assertEqual(proto.devices[1].id, 1)
    self.assertSequenceEqual(proto.devices[1].coords, [1, 0])
    self.assertEqual(proto.devices[1].core_on_chip, 1)

    # Test with a list of device assignments with optional fields None.
    device_assignments = [
        DeviceAssignment(id=0),
        DeviceAssignment(id=1),
        DeviceAssignment(id=2),
    ]
    proto = manifest_util.build_device_assignment_by_coords_proto(
        device_assignments
    )
    self.assertLen(proto.devices, 3)
    # Check devices where optional fields were None.
    for i, device in enumerate(proto.devices):
      self.assertEqual(device.id, i)
      self.assertSequenceEqual(device.coords, [])
      self.assertEqual(device.core_on_chip, 0)  # Proto default


if __name__ == '__main__':
  absltest.main()
